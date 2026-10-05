"""RePlAce global placement loop.

    min_v  f(v) = W(v) + lambda * D(v)

W is the WA wirelength, D the electrostatic density energy. Starting from a
wirelength-only initial placement, Nesterov's method minimizes f while the
density penalty factor lambda grows and the WA smoothing parameter gamma
shrinks, spreading objects until the density overflow reaches the target.

With dynamic step size adaptation (-ds), a trial placement first records the
HPWL curve, which then sets how fast lambda may grow in the actual placement.
With the local density function (-ld), objects in overflowed bins get extra,
per-object weighted density forces:  f + sum_i Delta_i * D_local.
"""

import math
import time
import warnings
from dataclasses import dataclass, field

import torch

from .density import Density, bin_count, make_fillers
from .design import Design
from .dynamic_step import DynamicStepSize, TransitionPoints, find_transition_points
from .initial import initial_place
from .nesterov import Nesterov
from .wirelength import hpwl, wa_wirelength, wirelength


@dataclass
class PlacerConfig:
    target_density: float = 1.0
    target_overflow: float = 0.1  # stop once the density overflow tau is this low
    max_iters: int = 3000
    initial_place_rounds: int = 20
    bins: int | None = None  # bins per side; None picks it from the design
    # lambda_0 = init_density_penalty * |grad W|_1 / |grad D|_1. RePlAce uses
    # 8e-5; with it our runs spend their first ~200 iterations growing lambda
    # before cells start to spread (unlike the paper's Fig. 5), and -ds can stall
    # in its slow first phase. 100x larger gives the same HPWL in fewer iterations.
    init_density_penalty: float = 8e-3
    # Per-iteration lambda multiplier range [cof_min, cof_max] (Algorithm 2).
    cof_min: float = 0.95
    cof_max: float = 1.05
    # Reference HPWL increment per iteration (Algorithm 2). ePlace uses 3.5e5 on
    # ISPD-2005 designs, about 0.075 * #nets * bin width on ADAPTEC1; we use
    # that ratio to scale it to any design when left as None.
    delta_hpwl_ref: float | None = None
    # Dynamic step size adaptation (-ds): trial placement + Eq. 10 for cof_max.
    dynamic_step: bool = False
    # The trial stops at overflow <= initial overflow / trial_overflow_divisor.
    trial_overflow_divisor: float = 2.5
    # Constraint-oriented local density function (-ld). alpha scales the
    # per-bin penalty nu_j = exp(alpha * overflow_j / bin area); beta the growth
    # of the per-object coefficients Delta_i. Both start at these values (beta
    # relative to the initial lambda) and grow with lambda's multiplier.
    local_density: bool = False
    ld_alpha: float = 1e-12
    ld_beta: float = 1.0
    seed: int = 0
    log_every: int = 0  # print progress every N iterations (0: silent)


@dataclass
class PlaceResult:
    pos: torch.Tensor  # (n, 2) final object centers
    filler_pos: torch.Tensor  # (f, 2)
    filler_size: torch.Tensor  # (f, 2)
    initial_pos: torch.Tensor  # (n, 2) object centers when the Nesterov loop starts
    initial_filler_pos: torch.Tensor  # (f, 2)
    hpwl: float
    overflow: float
    iterations: int  # actual placement only
    initial_hpwl: float
    time_initial: float  # seconds in initial placement
    time_global: float  # seconds in the Nesterov loop(s), including the trial
    bins: int
    history: list[dict] = field(default_factory=list)
    trial_iterations: int = 0  # -ds only
    trial_history: list[dict] = field(default_factory=list)
    transition_points: TransitionPoints | None = None


def wa_gamma(overflow: float, bin_size: float) -> float:
    """WA smoothing gamma scheduled by overflow (RePlAce): coarse (40 bins)
    while everything overlaps, fine (0.4 bins) near the target overflow."""
    tau = min(max(overflow, 0.1), 1.0)
    return 4.0 * bin_size * 10 ** ((tau - 0.1) * 20 / 9 - 1)


def lambda_multiplier(delta_hpwl: float, delta_ref: float, cof_min: float, cof_max: float) -> float:
    """Algorithm 2: grow lambda fast when HPWL decreases, slower the more it increases."""
    p = delta_hpwl / delta_ref
    return cof_max if p < 0 else max(cof_min, cof_max ** (1 - p))


class _Problem:
    """The objective over movable objects followed by fillers, plus everything
    needed to start a Nesterov run from the initial placement."""

    def __init__(self, design: Design, cfg: PlacerConfig):
        self.design = design
        dtype = design.pos.dtype
        xl, yl, xh, yh = design.die

        # Wirelength-only initial placement of the real objects.
        t0 = time.perf_counter()
        self.base_pos = initial_place(design, rounds=cfg.initial_place_rounds)
        self.time_initial = time.perf_counter() - t0
        self.initial_hpwl = hpwl(design, self.base_pos)

        # Fillers start uniformly spread over the die.
        self.mov_idx = design.movable.nonzero().squeeze(1)
        nm = self.num_movable = len(self.mov_idx)
        self.filler_size = make_fillers(design, cfg.target_density)
        filler_pos = torch.rand(len(self.filler_size), 2, dtype=dtype) * torch.tensor([xh - xl, yh - yl], dtype=dtype)
        filler_pos += torch.tensor([xl, yl], dtype=dtype)
        sizes = torch.cat([design.size[self.mov_idx], self.filler_size])

        self.bins = cfg.bins or bin_count(design, cfg.target_density)
        self.density = Density(design, sizes, nm, cfg.target_density, self.bins)
        self.bin_size = self.density.bin_size.mean().item()
        self.lo = torch.tensor([xl, yl], dtype=dtype) + sizes / 2
        self.hi = torch.tensor([xh, yh], dtype=dtype) - sizes / 2

        # Quadratic placement puts objects with identical connectivity at exactly
        # the same spot, where their own density peak exerts no force to separate
        # them. A random kick of up to half a bin breaks such ties.
        kick = (torch.rand(nm, 2, dtype=dtype) - 0.5) * self.density.bin_size
        self.x0 = self.project(torch.cat([self.base_pos[self.mov_idx] + kick, filler_pos]))
        _, self.initial_overflow = self.density(self.x0)

        # Jacobi-style preconditioner (ePlace-MS): diag Hessian ~ #pins + lambda * area.
        num_pins = torch.bincount(design.pin_obj, minlength=design.num_objects)[self.mov_idx].to(dtype)
        self.num_pins = torch.cat([num_pins, num_pins.new_zeros(len(self.filler_size))])
        self.area = sizes.prod(1)

        self.delta_hpwl_ref = cfg.delta_hpwl_ref or 0.075 * design.num_nets * self.bin_size
        self.init_density_penalty = cfg.init_density_penalty
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        """Initial gamma from the overflow, initial lambda balancing the
        wirelength and density gradients."""
        x = self.x0.clone().requires_grad_()
        energy, overflow = self.density(x)
        self.gamma = wa_gamma(overflow, self.bin_size)
        (g_wl,) = torch.autograd.grad(wa_wirelength(self.design, self.full_pos(x), self.gamma), x)
        (g_d,) = torch.autograd.grad(energy, x)
        self.lam = self.init_density_penalty * g_wl.abs().sum().item() / g_d.abs().sum().clamp(min=1e-30).item()
        # Local density state (-ld): Delta_i starts at 0 (Eq. 3).
        self.alpha = self.cfg.ld_alpha
        self.beta = self.cfg.ld_beta * self.lam
        self.delta = torch.zeros(len(self.x0), dtype=self.x0.dtype)

    def update_penalties(self, multiplier: float, stats: dict) -> None:
        """After an iteration: scale lambda (and alpha, beta, which grow at the
        same rate), and accumulate each object's local density coefficient."""
        self.lam *= multiplier
        if self.cfg.local_density:
            self.delta += self.beta * stats["bin_overflow"]  # Eq. 3 / Algorithm 1
            self.alpha = min(self.alpha * multiplier, 1e5)  # cap as in the original code
            self.beta *= multiplier

    def full_pos(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_pos.index_put((self.mov_idx,), x[: self.num_movable])

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(x, self.lo), self.hi)

    def evaluate(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Preconditioned gradient of f at x plus HPWL/overflow/potential statistics."""
        x = x.detach().requires_grad_()
        pos = self.full_pos(x)
        ld = self.cfg.local_density
        energy, overflow, *local = self.density(x, self.alpha if ld else None)
        wl, exact_hpwl = wirelength(self.design, pos, self.gamma)
        (grad,) = torch.autograd.grad(wl + self.lam * energy, x)
        stats = {"hpwl": exact_hpwl, "overflow": overflow, "energy": energy.item()}
        if ld:
            # Eq. 9: each object's local density gradient scaled by its Delta_i.
            local_grad, stats["bin_overflow"] = local[0]
            grad = grad + self.delta[:, None] * local_grad
        precond = (self.num_pins + self.lam * self.area).clamp(min=1.0)
        return grad / precond[:, None], stats


def _nesterov_run(prob: _Problem, cfg: PlacerConfig, stop_overflow: float, schedule: DynamicStepSize | None, tag: str):
    """One Nesterov placement from the initial placement until the overflow
    reaches stop_overflow. Returns the optimizer and the per-iteration history."""
    prob.reset()
    opt = Nesterov(prob.x0, prob.evaluate, prob.project, initial_move=0.1 * prob.bin_size)
    prev_hpwl = opt.stats["hpwl"]
    scalars = lambda stats: {k: v for k, v in stats.items() if isinstance(v, float)}  # noqa: E731
    history = [dict(iter=0, **scalars(opt.stats), lam=prob.lam, gamma=prob.gamma, cof_max=cfg.cof_max)]
    for it in range(1, cfg.max_iters + 1):
        opt.step()
        cur_hpwl, overflow = opt.stats["hpwl"], opt.stats["overflow"]
        cof_max = schedule.cof_max(cur_hpwl, opt.stats["energy"]) if schedule else cfg.cof_max
        history.append(dict(iter=it, **scalars(opt.stats), lam=prob.lam, gamma=prob.gamma, cof_max=cof_max))
        if cfg.log_every and it % cfg.log_every == 0:
            print(f"  {tag} iter {it:5d}  hpwl {cur_hpwl:.4e}  overflow {overflow:.3f}  lambda {prob.lam:.3e}  cof_max {cof_max:.4f}")
        if overflow <= stop_overflow:
            break
        if math.isnan(cur_hpwl):
            raise FloatingPointError("placement diverged")
        prob.gamma = wa_gamma(overflow, prob.bin_size)
        prob.update_penalties(lambda_multiplier(cur_hpwl - prev_hpwl, prob.delta_hpwl_ref, cfg.cof_min, cof_max), opt.stats)
        prev_hpwl = cur_hpwl
    else:
        warnings.warn(f"{tag}: overflow {overflow:.3f} > {stop_overflow:.3f} after max_iters={cfg.max_iters}")
    return opt, history


def global_place(design: Design, cfg: PlacerConfig = PlacerConfig()) -> PlaceResult:
    torch.manual_seed(cfg.seed)
    prob = _Problem(design, cfg)
    t0 = time.perf_counter()

    schedule, trial, tps = None, [], None
    if cfg.dynamic_step:
        # Trial placement (tGP) with the default lambda schedule, stopped once
        # the overflow is well below its initial value (Section IV-B).
        stop = prob.initial_overflow / cfg.trial_overflow_divisor
        _, trial = _nesterov_run(prob, cfg, max(stop, cfg.target_overflow), None, "trial")
        tps = find_transition_points([h["hpwl"] for h in trial])
        schedule = DynamicStepSize(tps, [h["hpwl"] for h in trial], [h["energy"] for h in trial])

    opt, history = _nesterov_run(prob, cfg, cfg.target_overflow, schedule, "place")
    return PlaceResult(
        pos=prob.full_pos(opt.v),
        filler_pos=opt.v[prob.num_movable :],
        filler_size=prob.filler_size,
        initial_pos=prob.full_pos(prob.x0),
        initial_filler_pos=prob.x0[prob.num_movable :],
        hpwl=opt.stats["hpwl"],
        overflow=opt.stats["overflow"],
        iterations=len(history) - 1,
        initial_hpwl=prob.initial_hpwl,
        time_initial=prob.time_initial,
        time_global=time.perf_counter() - t0,
        bins=prob.bins,
        history=history,
        trial_iterations=max(len(trial) - 1, 0),
        trial_history=trial,
        transition_points=tps,
    )
