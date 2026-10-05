"""Wirelength-only initial placement with the bound-to-bound (B2B) net model.

As in RePlAce, movable objects start at the die center and a few rounds of
quadratic placement pull them toward a wirelength-minimal (heavily overlapped)
solution. The B2B model [Spindler et al., Kraftwerk2] connects each pin to its
net's two extreme (bound) pins with weight 2 / ((p - 1) * distance), which makes
the quadratic cost equal HPWL at the current positions. Each round rebuilds the
weights and solves the linear system  A x = b  per axis with conjugate gradient.
"""

import warnings
from dataclasses import replace

import torch

from .design import Design


def _b2b_system(design: Design, pos: torch.Tensor, axis: int, var: torch.Tensor, num_vars: int, min_dist: float):
    """Sparse matrix A and right-hand side b of the B2B quadratic for one axis."""
    x = pos[design.pin_obj, axis] + design.pin_offset[:, axis]
    net, n_nets = design.pin_net, design.num_nets
    pin = torch.arange(design.num_pins)
    degree = torch.bincount(net, minlength=n_nets)

    # The bound pins of each net. Ties pick the lowest index for min and the
    # highest for max, so the two bounds differ whenever a net has >= 2 pins.
    def bound(reduce: str, pick: str) -> torch.Tensor:
        extreme = x.new_zeros(n_nets).scatter_reduce(0, net, x, reduce, include_self=False)
        at = x == extreme[net]
        return pin.new_zeros(n_nets).scatter_reduce(0, net[at], pin[at], pick, include_self=False)

    lo, hi = bound("amin", "amin")[net], bound("amax", "amax")[net]
    # Edges: every pin to its min bound, every inner pin also to its max bound.
    first, second = pin != lo, (pin != lo) & (pin != hi)
    p = torch.cat([pin[first], pin[second]])
    q = torch.cat([lo[first], hi[second]])
    w = 2.0 / ((degree[net[p]] - 1) * (x[p] - x[q]).abs().clamp(min=min_dist))

    # Each edge adds w * (x_p - x_q)^2. Setting its derivative to zero, the row
    # of a movable endpoint gets +w on the diagonal and either -w toward a
    # movable partner or a w * (fixed position) term on the right-hand side.
    rows, cols, vals = [], [], []
    diag = x.new_zeros(num_vars)
    rhs = x.new_zeros(num_vars)
    off = design.pin_offset[:, axis]
    for a, b in ((p, q), (q, p)):
        va, vb = var[design.pin_obj[a]], var[design.pin_obj[b]]
        keep = (va >= 0) & (design.pin_obj[a] != design.pin_obj[b])
        va, vb, a, b, wk = va[keep], vb[keep], a[keep], b[keep], w[keep]
        diag.index_add_(0, va, wk)
        movable_b = vb >= 0
        rows.append(va[movable_b])
        cols.append(vb[movable_b])
        vals.append(-wk[movable_b])
        target = torch.where(movable_b, off[b], x[b])
        rhs.index_add_(0, va, wk * (target - off[a]))

    # Weak anchor to the current position keeps unconnected objects in place.
    eps = 1e-6 * diag.mean().clamp(min=1e-12)
    diag += eps
    rhs += eps * pos[var >= 0, axis]
    idx = torch.arange(num_vars)
    rows, cols, vals = torch.cat(rows + [idx]), torch.cat(cols + [idx]), torch.cat(vals + [diag])
    a_mat = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (num_vars, num_vars), check_invariants=False)
    with warnings.catch_warnings():  # CSR is "beta" in PyTorch, but ~100x faster than COO here
        warnings.simplefilter("ignore")
        a_mat = a_mat.coalesce().to_sparse_csr()
    return a_mat, rhs, diag


def _pcg(a: torch.Tensor, b: torch.Tensor, diag: torch.Tensor, x: torch.Tensor, iters: int, tol: float) -> torch.Tensor:
    """Jacobi-preconditioned conjugate gradient for symmetric positive definite A."""
    r = b - a @ x
    z = r / diag
    p = z.clone()
    rz = r @ z
    b_norm = b.norm().clamp(min=1e-30)
    for _ in range(iters):
        ap = a @ p
        alpha = rz / (p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        if r.norm() < tol * b_norm:
            break
        z = r / diag
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


@torch.no_grad()
def initial_place(design: Design, rounds: int = 20, cg_iters: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """(n, 2) positions with movable objects placed by B2B quadratic placement."""
    xl, yl, xh, yh = design.die
    pos = design.pos.double().clone()
    mov = design.movable
    num_vars = int(mov.sum())
    var = torch.full((design.num_objects,), -1, dtype=torch.long)
    var[mov] = torch.arange(num_vars)

    pos[mov] = torch.tensor([(xl + xh) / 2, (yl + yh) / 2], dtype=pos.dtype)
    half = design.size.double() / 2
    lo = torch.tensor([xl, yl], dtype=pos.dtype) + half
    hi = torch.tensor([xh, yh], dtype=pos.dtype) - half
    # Distances below this are clamped so B2B weights stay finite.
    min_dist = 0.01 * design.row_height

    d64 = replace(design, pin_offset=design.pin_offset.double())
    for _ in range(rounds):
        for axis in (0, 1):
            a, b, diag = _b2b_system(d64, pos, axis, var, num_vars, min_dist)
            pos[mov, axis] = _pcg(a, b, diag, pos[mov, axis], cg_iters, tol)
        pos[mov] = torch.minimum(torch.maximum(pos[mov], lo[mov]), hi[mov])
    return pos.to(design.pos.dtype)
