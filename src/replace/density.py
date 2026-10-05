"""Electrostatic density penalty (ePlace), the global density function of RePlAce.

Objects are positive charges whose charge equals their area. The placement
region is split into an M x M grid of bins; the bin charge density rho drives
a Poisson equation  laplacian(phi) = -rho  with Neumann boundary conditions,
solved spectrally with a 2-D DCT. The penalty is the system's potential
energy  N = 1/2 * sum(rho * phi).

Filler cells (movable charges without pins) absorb the whitespace so that the
equilibrium state is a uniform density at the target value.
"""

import math

import torch

from .design import Design

# Objects spanning at most this many bins per axis (standard cells, fillers)
# touch a window of WINDOW consecutive bins; their overlaps are computed only
# there. Larger objects (macros) use dense overlaps with every bin.
SMALL_SPAN = 4
WINDOW = SMALL_SPAN + 1


def bin_count(design: Design, target_density: float) -> int:
    """Bins per side, M (a power of two), giving roughly one average movable
    object per bin at the target density (as in RePlAce)."""
    mov = design.movable
    avg_area = design.size[mov].prod(1).mean().item()
    die_area = design.die_width * design.die_height
    ideal = math.sqrt(die_area * target_density / avg_area)
    return int(2 ** min(max(round(math.log2(ideal)), 4), 10))


def make_fillers(design: Design, target_density: float) -> torch.Tensor:
    """(f, 2) filler sizes. Filler size is the average of the middle 80% of
    movable standard cells (by area); total filler area fills the whitespace
    up to the target density (ePlace)."""
    std = design.size[design.movable & ~design.macro]
    std = std[std.prod(1).argsort()]
    lo, hi = int(0.1 * len(std)), max(int(0.9 * len(std)), int(0.1 * len(std)) + 1)
    filler = std[lo:hi].mean(0)

    die_area = design.die_width * design.die_height
    fixed_area = design.size[design.fixed].prod(1).sum().item()
    movable_area = design.size[design.movable].prod(1).sum().item()
    filler_area = target_density * (die_area - fixed_area) - movable_area
    count = max(int(filler_area / filler.prod().item()), 0)
    return filler.expand(count, 2).clone()


def dct_matrix(m: int, dtype=torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix: D[u, i] = s_u cos(pi * u * (i + 1/2) / m)."""
    u = torch.arange(m, dtype=torch.float64)[:, None]
    i = torch.arange(m, dtype=torch.float64)[None, :]
    d = torch.cos(math.pi * u * (i + 0.5) / m) * math.sqrt(2.0 / m)
    d[0] /= math.sqrt(2.0)
    return d.to(dtype)


def dst_matrix(m: int, dtype=torch.float32) -> torch.Tensor:
    """Sine counterpart of `dct_matrix`: S[u, i] = s_u sin(pi * u * (i + 1/2) / m)."""
    u = torch.arange(m, dtype=torch.float64)[:, None]
    i = torch.arange(m, dtype=torch.float64)[None, :]
    return (torch.sin(math.pi * u * (i + 0.5) / m) * math.sqrt(2.0 / m)).to(dtype)


def overlap(center: torch.Tensor, half: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """1-D overlap length between intervals [center +- half] (k,) and bins [lo, hi] (M,) or (k, M)."""
    return (torch.minimum(center[:, None] + half[:, None], hi) - torch.maximum(center[:, None] - half[:, None], lo)).clamp(min=0)


def overlap_slope(center: torch.Tensor, half: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Derivative of `overlap` w.r.t. the center: +1 where only the interval's
    right end lies inside the bin, -1 where only its left end does."""
    c, h = center[:, None], half[:, None]
    inside = overlap(center, half, lo, hi) > 0
    return ((c + h < hi).to(c.dtype) - (c - h > lo).to(c.dtype)) * inside


class Density:
    """Density penalty for a fixed set of movable objects (cells, then fillers).

    Calling it with object centers returns (energy, overflow); energy
    back-propagates the electrostatic gradient through autograd.
    """

    def __init__(self, design: Design, sizes: torch.Tensor, num_cells: int, target_density: float, bins: int):
        xl, yl, xh, yh = design.die
        m = bins
        self.bins = m
        self.num_cells = num_cells  # objects [0, num_cells) are real cells, the rest fillers
        self.target_density = target_density
        self.origin = torch.tensor([xl, yl], dtype=sizes.dtype)
        self.bin_size = torch.tensor([(xh - xl) / m, (yh - yl) / m], dtype=sizes.dtype)
        self.bin_area = self.bin_size.prod().item()
        edges_x = torch.linspace(xl, xh, m + 1, dtype=sizes.dtype)
        edges_y = torch.linspace(yl, yh, m + 1, dtype=sizes.dtype)
        self.lo = (edges_x[:-1], edges_y[:-1])
        self.hi = (edges_x[1:], edges_y[1:])

        # Local smoothing (ePlace): objects smaller than ~1.4 bins are stretched
        # to sqrt(2) bins per side with proportionally reduced charge density,
        # so their density map (and force) varies smoothly with position.
        stretched = torch.maximum(sizes, math.sqrt(2) * self.bin_size)
        self.half = stretched / 2
        # Each object's per-unit-area weight, so the map is in units of density.
        self.weight = sizes.prod(1) / stretched.prod(1) / self.bin_area
        self.movable_area = sizes[:num_cells].prod(1).sum().item()
        small = (stretched <= SMALL_SPAN * self.bin_size).all(1)
        self.small, self.large = small.nonzero().squeeze(1), (~small).nonzero().squeeze(1)

        # Fixed objects: exact overlap area, constant over the whole placement.
        f = design.fixed
        fx = overlap(design.pos[f, 0], design.size[f, 0] / 2, self.lo[0], self.hi[0])
        fy = overlap(design.pos[f, 1], design.size[f, 1] / 2, self.lo[1], self.hi[1])
        fixed_area = fx.T @ fy
        self.free_area = (self.bin_area - fixed_area).clamp(min=0)
        # Fixed area counts at the target density so that total charge is
        # uniform at the target when the placement is spread (ePlace-MS).
        self.fixed_rho = target_density * fixed_area / self.bin_area

        # Spectral Poisson solver. With phi = sum a_uv cos(w_u x) cos(w_v y),
        # laplacian(phi) = -rho gives a_uv(phi) = a_uv(rho) / (w_u^2 + w_v^2).
        # The (0, 0) term (mean density) is dropped: the system is neutralized.
        # The field E = -grad(phi) has the same coefficients times w_u (or w_v)
        # on sine instead of cosine basis functions.
        self.dct, self.dst = dct_matrix(m, sizes.dtype), dst_matrix(m, sizes.dtype)
        w_u = math.pi * torch.arange(m, dtype=torch.float64) / (xh - xl)
        w_v = math.pi * torch.arange(m, dtype=torch.float64) / (yh - yl)
        w2 = w_u[:, None] ** 2 + w_v[None, :] ** 2
        w2[0, 0] = 1.0
        inv = 1.0 / w2
        inv[0, 0] = 0.0
        self.inv_w2 = inv.to(sizes.dtype)
        self.wu_w2 = (w_u[:, None] * inv).to(sizes.dtype)
        self.wv_w2 = (w_v[None, :] * inv).to(sizes.dtype)

    def _window(self, center: torch.Tensor, half: torch.Tensor, axis: int):
        """Bin indices, lower bin edges (both (k, WINDOW)) of the WINDOW bins
        starting at each small object's first bin along one axis."""
        size = self.bin_size[axis]
        first = ((center - half - self.origin[axis]) / size).floor().long().clamp(0, self.bins - 1)
        idx = first[:, None] + torch.arange(WINDOW)
        # Bins past the die edge get zero overlap; clamp their index to stay valid.
        return idx.clamp(max=self.bins - 1), self.origin[axis] + idx * size

    def _footprints(self, centers: torch.Tensor, slopes: bool = False) -> dict:
        """Per-axis overlaps of every object with the bins: small objects over
        their WINDOW bins ("sx", "sy", flattened 2-D bin indices "flat"), large
        ones over all bins ("lx", "ly"). Rectangle/bin overlap area factorizes
        into x-overlap * y-overlap; x-overlaps include the object's weight.
        With slopes, also the derivatives of the x/y-overlaps w.r.t. the center."""
        s, big, m = self.small, self.large, self.bins
        fp = {}
        for axis, name in ((0, "x"), (1, "y")):
            c, h = centers[s, axis], self.half[s, axis]
            idx, lo = self._window(c, h, axis)
            hi = lo + self.bin_size[axis]
            fp["s" + name], fp["s_idx" + name] = overlap(c, h, lo, hi), idx
            cb, hb = centers[big, axis], self.half[big, axis]
            fp["l" + name] = overlap(cb, hb, self.lo[axis], self.hi[axis])
            if slopes:
                fp["sd" + name] = overlap_slope(c, h, lo, hi)
                fp["ld" + name] = overlap_slope(cb, hb, self.lo[axis], self.hi[axis])
        fp["flat"] = fp.pop("s_idxx")[:, :, None] * m + fp.pop("s_idxy")[:, None, :]
        return fp

    def _gather(self, fp: dict, field: torch.Tensor, x: str = "x", y: str = "y", weighted: bool = True) -> torch.Tensor:
        """(k,) per object: sum over bins of weight * x-part * y-part * field,
        where the parts are the overlaps (or slopes, x="dx"...) in `fp`."""
        s, big = self.small, self.large
        out = field.new_empty(len(self.half))
        sx, lx = fp["s" + x], fp["l" + x]
        if weighted:
            sx, lx = sx * self.weight[s, None], lx * self.weight[big, None]
        out[s] = ((sx[:, :, None] * fp["s" + y][:, None, :]) * field.flatten()[fp["flat"]]).sum((1, 2))
        out[big] = ((lx @ field) * fp["l" + y]).sum(1)
        return out

    def _map(self, fp: dict, small_mask: torch.Tensor, big_mask: torch.Tensor) -> torch.Tensor:
        """(M, M) density map of the selected objects."""
        m = self.bins
        s, big = self.small, self.large
        area = (fp["sx"] * self.weight[s, None])[small_mask][:, :, None] * fp["sy"][small_mask][:, None, :]
        rho = area.new_zeros(m * m).index_add_(0, fp["flat"][small_mask].flatten(), area.flatten())
        return rho.view(m, m) + (fp["lx"] * self.weight[big, None])[big_mask].T @ fp["ly"][big_mask]

    @torch.no_grad()
    def evaluate(self, centers: torch.Tensor, alpha: float | None = None):
        """Energy, its gradient w.r.t. the centers, the overflow and, if `alpha`
        is given, the local density terms (see `local_terms`)."""
        n, s, big = self.num_cells, self.small, self.large
        fp = self._footprints(centers, slopes=alpha is not None)
        rho_cells = self._map(fp, s < n, big < n)
        rho = rho_cells + self._map(fp, s >= n, big >= n) + self.fixed_rho

        d, sn = self.dct, self.dst
        coeffs = d @ rho @ d.T
        e_x = sn.T @ (coeffs * self.wu_w2) @ d
        e_y = d.T @ (coeffs * self.wv_w2) @ sn

        # Energy N = 1/2 sum_i q_i phi(x_i) ~ 1/2 * bin_area * sum_bins rho * phi
        # (charges in area units, as in ePlace); by Parseval it equals the sum
        # over spectral coefficients. Its gradient is minus the force q_i * E on
        # each object, with E averaged over the bins it overlaps. (Differentiating
        # the bin map instead would also be exact for the discretized energy, but
        # that gradient jumps whenever an object edge crosses a bin boundary,
        # which breaks Nesterov's Lipschitz step prediction.)
        energy = 0.5 * self.bin_area * (coeffs**2 * self.inv_w2).sum()
        grad = -self.bin_area * torch.stack([self._gather(fp, e_x), self._gather(fp, e_y)], 1)

        # Overflow tau: movable cell area in excess of each bin's available
        # (non-fixed) area at the target density, over total movable area.
        excess = rho_cells * self.bin_area - self.target_density * self.free_area
        overflow = excess.clamp(min=0).sum().item() / self.movable_area
        if alpha is None:
            return energy, grad, overflow
        phi = d.T @ (coeffs * self.inv_w2) @ d
        return energy, grad, overflow, self.local_terms(fp, alpha, excess, rho_cells, phi, e_x, e_y)

    def local_terms(self, fp, alpha, excess, rho_cells, phi, e_x, e_y):
        """RePlAce local density function (Section III, Eqs. 2, 3, 5, 8).

        Each bin gets a penalty factor nu_j = exp(alpha * overflow_j / bin area)
        (Eq. 2), large in overflowed bins. Returns
          * the gradient of D_local (Eq. 8, as in the original code): the
            electrostatic force reweighted per bin by nu_j, plus the change of
            nu_j as the object moves its area in or out of bin j;
          * per object, the overflow of the bins it touches over the total
            movable area, which accumulates into its coefficient Delta_i (Eq. 3).
        """
        nu = torch.exp((alpha * excess / self.bin_area).clamp(max=50.0))
        demand = rho_cells * self.bin_area
        # d nu_j / d x_i = nu_j * alpha / bin_area * d(area of i in j) / dx, weighted
        # by bin j's energy phi_j * demand_j.
        g = (alpha / self.bin_area) * nu * phi * demand
        grad = torch.stack([
            -self.bin_area * self._gather(fp, nu * e_x) + self.bin_area * self._gather(fp, g, x="dx"),
            -self.bin_area * self._gather(fp, nu * e_y) + self.bin_area * self._gather(fp, g, y="dy"),
        ], 1)
        touched = {k: (fp[k] > 0).to(e_x.dtype) for k in ("sx", "sy", "lx", "ly")}
        touched["flat"] = fp["flat"]
        bin_overflow = self._gather(touched, excess.clamp(min=0), weighted=False) / self.movable_area
        return grad, bin_overflow

    def __call__(self, centers: torch.Tensor, alpha: float | None = None):
        """(energy, overflow[, local terms]); the energy back-propagates the
        electrostatic gradient."""
        energy, grad, *rest = self.evaluate(centers.detach(), alpha)
        return _Energy.apply(centers, energy, grad), *rest


class _Energy(torch.autograd.Function):
    """Plugs a precomputed gradient into autograd."""

    @staticmethod
    def forward(ctx, centers, energy, grad):
        ctx.save_for_backward(grad)
        return energy.clone()

    @staticmethod
    def backward(ctx, out_grad):
        (grad,) = ctx.saved_tensors
        return out_grad * grad, None, None
