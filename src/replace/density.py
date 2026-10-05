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
        """Bin indices (k, WINDOW) and overlaps (k, WINDOW) of small objects
        along one axis, over the WINDOW bins starting at each object's first bin."""
        size = self.bin_size[axis]
        first = ((center - half - self.origin[axis]) / size).floor().long().clamp(0, self.bins - 1)
        idx = first[:, None] + torch.arange(WINDOW)
        lo = self.origin[axis] + idx * size
        # Bins past the die edge get zero overlap; clamp their index to stay valid.
        return idx.clamp(max=self.bins - 1), overlap(center, half, lo, lo + size)

    @torch.no_grad()
    def evaluate(self, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Energy, its gradient w.r.t. the centers, and the overflow."""
        m, n = self.bins, self.num_cells
        s, big = self.small, self.large

        # Small objects: overlap area with each bin of their WINDOW x WINDOW
        # window (area factorizes into x-overlap * y-overlap), scattered into
        # the flattened bin grid.
        ix, ox = self._window(centers[s, 0], self.half[s, 0], 0)
        iy, oy = self._window(centers[s, 1], self.half[s, 1], 1)
        flat = ix[:, :, None] * m + iy[:, None, :]
        area = (ox * self.weight[s, None])[:, :, None] * oy[:, None, :]

        # Large objects: overlaps with every bin; the map is one matmul.
        bx = overlap(centers[big, 0], self.half[big, 0], self.lo[0], self.hi[0]) * self.weight[big, None]
        by = overlap(centers[big, 1], self.half[big, 1], self.lo[1], self.hi[1])

        def density_map(small_mask: torch.Tensor, big_mask: torch.Tensor) -> torch.Tensor:
            rho = centers.new_zeros(m * m).index_add_(0, flat[small_mask].flatten(), area[small_mask].flatten())
            return rho.view(m, m) + bx[big_mask].T @ by[big_mask]

        rho_cells = density_map(s < n, big < n)
        rho = rho_cells + density_map(s >= n, big >= n) + self.fixed_rho

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
        grad = torch.empty_like(centers)
        for axis, e in enumerate((e_x, e_y)):
            grad[s, axis] = -self.bin_area * (area * e.flatten()[flat]).sum((1, 2))
            grad[big, axis] = -self.bin_area * ((bx @ e) * by).sum(1)

        # Overflow tau: movable cell area in excess of each bin's available
        # (non-fixed) area at the target density, over total movable area.
        excess = rho_cells * self.bin_area - self.target_density * self.free_area
        overflow = excess.clamp(min=0).sum().item() / self.movable_area
        return energy, grad, overflow

    def __call__(self, centers: torch.Tensor) -> tuple[torch.Tensor, float]:
        """(energy, overflow); the energy back-propagates the electrostatic gradient."""
        energy, grad, overflow = self.evaluate(centers.detach())
        return _Energy.apply(centers, energy, grad), overflow


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
