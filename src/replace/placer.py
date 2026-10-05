"""RePlAce global placement loop.

    min_v  f(v) = W(v) + lambda * D(v)

W is the WA wirelength, D the electrostatic density energy. Starting from a
wirelength-only initial placement, Nesterov's method minimizes f while the
density penalty factor lambda grows and the WA smoothing parameter gamma
shrinks, spreading objects until the density overflow reaches the target.
"""

import math
import time
from dataclasses import dataclass, field

import torch

from .density import Density, bin_count, make_fillers
from .design import Design
from .initial import initial_place
from .nesterov import Nesterov
from .wirelength import hpwl, wa_wirelength


@dataclass
class PlacerConfig:
    target_density: float = 1.0
    target_overflow: float = 0.1  # stop once the density overflow tau is this low
    max_iters: int = 3000
    initial_place_rounds: int = 20
    bins: int | None = None  # bins per side; None picks it from the design
    # lambda_0 = init_density_penalty * |grad W|_1 / |grad D|_1  (RePlAce default)
    init_density_penalty: float = 8e-5
    # Per-iteration lambda multiplier range [cof_min, cof_max] (Algorithm 2).
    cof_min: float = 0.95
    cof_max: float = 1.05
    # Reference HPWL increment per iteration (Algorithm 2). ePlace uses 3.5e5 on
    # ISPD-2005 designs, about 0.075 * #nets * bin width on ADAPTEC1; we use
    # that ratio to scale it to any design when left as None.
    delta_hpwl_ref: float | None = None
    seed: int = 0
    log_every: int = 0  # print progress every N iterations (0: silent)


@dataclass
class PlaceResult:
    pos: torch.Tensor  # (n, 2) final object centers
    filler_pos: torch.Tensor  # (f, 2)
    filler_size: torch.Tensor  # (f, 2)
    hpwl: float
    overflow: float
    iterations: int
    initial_hpwl: float
    time_initial: float  # seconds in initial placement
    time_global: float  # seconds in the Nesterov loop (incl. setup)
    bins: int
    history: list[dict] = field(default_factory=list)


def wa_gamma(overflow: float, bin_size: float) -> float:
    """WA smoothing gamma scheduled by overflow (RePlAce): coarse (40 bins)
    while everything overlaps, fine (0.4 bins) near the target overflow."""
    tau = min(max(overflow, 0.1), 1.0)
    return 4.0 * bin_size * 10 ** ((tau - 0.1) * 20 / 9 - 1)


def lambda_multiplier(delta_hpwl: float, delta_ref: float, cof_min: float, cof_max: float) -> float:
    """Algorithm 2: grow lambda fast when HPWL decreases, slower the more it increases."""
    p = delta_hpwl / delta_ref
    return cof_max if p < 0 else max(cof_min, cof_max ** (1 - p))


def global_place(design: Design, cfg: PlacerConfig = PlacerConfig()) -> PlaceResult:
    torch.manual_seed(cfg.seed)
    dtype = design.pos.dtype
    xl, yl, xh, yh = design.die

    # 1) Wirelength-only initial placement of the real objects.
    t0 = time.perf_counter()
    base_pos = initial_place(design, rounds=cfg.initial_place_rounds)
    time_initial = time.perf_counter() - t0
    initial_hpwl = hpwl(design, base_pos)

    # 2) Optimization variables: movable objects followed by fillers, which
    #    start uniformly spread over the die.
    t0 = time.perf_counter()
    mov_idx = design.movable.nonzero().squeeze(1)
    nm = len(mov_idx)
    filler_size = make_fillers(design, cfg.target_density)
    filler_pos = torch.rand(len(filler_size), 2, dtype=dtype) * torch.tensor([xh - xl, yh - yl], dtype=dtype)
    filler_pos += torch.tensor([xl, yl], dtype=dtype)
    sizes = torch.cat([design.size[mov_idx], filler_size])

    bins = cfg.bins or bin_count(design, cfg.target_density)
    density = Density(design, sizes, nm, cfg.target_density, bins)
    bin_size = density.bin_size.mean().item()
    lo = torch.tensor([xl, yl], dtype=dtype) + sizes / 2
    hi = torch.tensor([xh, yh], dtype=dtype) - sizes / 2

    # Quadratic placement puts objects with identical connectivity at exactly
    # the same spot, where their own density peak exerts no force to separate
    # them. A random kick of up to half a bin breaks such ties.
    kick = (torch.rand(nm, 2, dtype=dtype) - 0.5) * density.bin_size.to(dtype)
    x0 = torch.cat([base_pos[mov_idx] + kick, filler_pos])
    x0 = torch.minimum(torch.maximum(x0, lo), hi)

    # Jacobi-style preconditioner (ePlace-MS): diag Hessian ~ #pins + lambda * area.
    num_pins = torch.bincount(design.pin_obj, minlength=design.num_objects)[mov_idx].to(dtype)
    num_pins = torch.cat([num_pins, num_pins.new_zeros(len(filler_size))])
    area = sizes.prod(1)

    def full_pos(x: torch.Tensor) -> torch.Tensor:
        return base_pos.index_put((mov_idx,), x[:nm])

    def project(x: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(x, lo), hi)

    def evaluate(x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Preconditioned gradient of f at x plus HPWL/overflow statistics."""
        x = x.detach().requires_grad_()
        pos = full_pos(x)
        energy, overflow = density(x)
        f = wa_wirelength(design, pos, gamma) + lam * energy
        (grad,) = torch.autograd.grad(f, x)
        precond = (num_pins + lam * area).clamp(min=1.0)
        return grad / precond[:, None], {"hpwl": hpwl(design, pos), "overflow": overflow, "energy": energy.item()}

    # 3) Initial gamma and lambda (balance wirelength and density gradients).
    x = x0.clone().requires_grad_()
    energy, overflow = density(x)
    gamma = wa_gamma(overflow, bin_size)
    (g_wl,) = torch.autograd.grad(wa_wirelength(design, full_pos(x), gamma), x)
    (g_d,) = torch.autograd.grad(energy, x)
    lam = cfg.init_density_penalty * g_wl.abs().sum().item() / g_d.abs().sum().clamp(min=1e-30).item()
    delta_ref = cfg.delta_hpwl_ref or 0.075 * design.num_nets * bin_size

    # 4) Nesterov loop.
    opt = Nesterov(x0, evaluate, project, initial_move=0.1 * bin_size)
    prev_hpwl = opt.stats["hpwl"]
    history = []
    for it in range(1, cfg.max_iters + 1):
        opt.step()
        cur_hpwl, overflow = opt.stats["hpwl"], opt.stats["overflow"]
        history.append(dict(iter=it, hpwl=cur_hpwl, overflow=overflow, lam=lam, gamma=gamma, step=opt.step_length, energy=opt.stats["energy"]))
        if cfg.log_every and it % cfg.log_every == 0:
            print(f"  iter {it:5d}  hpwl {cur_hpwl:.4e}  overflow {overflow:.3f}  lambda {lam:.3e}  gamma {gamma:.3g}")
        if overflow <= cfg.target_overflow:
            break
        if math.isnan(cur_hpwl):
            raise FloatingPointError("placement diverged")
        gamma = wa_gamma(overflow, bin_size)
        lam *= lambda_multiplier(cur_hpwl - prev_hpwl, delta_ref, cfg.cof_min, cfg.cof_max)
        prev_hpwl = cur_hpwl

    return PlaceResult(
        pos=full_pos(opt.v),
        filler_pos=opt.v[nm:],
        filler_size=filler_size,
        hpwl=opt.stats["hpwl"],
        overflow=opt.stats["overflow"],
        iterations=len(history),
        initial_hpwl=initial_hpwl,
        time_initial=time_initial,
        time_global=time.perf_counter() - t0,
        bins=bins,
        history=history,
    )
