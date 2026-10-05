import math

import torch

from replace.density import Density, dct_matrix, dst_matrix, make_fillers
from replace.generate import generate


def make_density(num_cells=500, macros=3, bins=32):
    d = generate(num_cells, macros, seed=1)
    mov = d.movable.nonzero().squeeze(1)
    fillers = make_fillers(d, 1.0)
    sizes = torch.cat([d.size[mov], fillers])
    dens = Density(d, sizes, len(mov), 1.0, bins)
    return d, mov, fillers, sizes, dens


def test_transform_matrices_are_orthonormal():
    d = dct_matrix(16, torch.float64)
    assert torch.allclose(d @ d.T, torch.eye(16, dtype=torch.float64), atol=1e-12)
    s = dst_matrix(16, torch.float64)
    assert torch.allclose((s @ s.T)[1:, 1:], torch.eye(15, dtype=torch.float64), atol=1e-12)


def test_poisson_solution_of_a_single_mode():
    # rho = cos(w_u x) cos(w_v y)  =>  phi = rho / (w_u^2 + w_v^2)
    _, _, _, _, dens = make_density()
    m, (u, v) = dens.bins, (3, 5)
    coeffs = torch.zeros(m, m)
    coeffs[u, v] = 1.0
    rho = dens.dct.T @ coeffs @ dens.dct
    phi = dens.dct.T @ ((dens.dct @ rho @ dens.dct.T) * dens.inv_w2) @ dens.dct
    assert torch.allclose(phi, rho * dens.inv_w2[u, v], atol=1e-4 * phi.abs().max())


def test_density_map_conserves_charge():
    d, mov, fillers, sizes, dens = make_density()
    centers = torch.cat([d.pos[mov], torch.rand(len(fillers), 2) * torch.tensor([d.die_width, d.die_height])])
    # Keep stretched footprints inside the die (beyond it, charge is cut off).
    centers = torch.minimum(torch.maximum(centers, dens.half), torch.tensor([d.die_width, d.die_height]) - dens.half)
    energy, grad, overflow = dens.evaluate(centers)
    assert torch.isfinite(grad).all() and energy > 0 and 0 <= overflow <= 1
    # Charge of small (windowed) and large (dense) objects is fully accounted for.
    rho = dens._map(dens._footprints(centers), torch.ones_like(dens.small, dtype=torch.bool), torch.ones_like(dens.large, dtype=torch.bool))
    assert math.isclose(rho.sum().item() * dens.bin_area, sizes.prod(1).sum().item(), rel_tol=1e-4)


def test_density_force_spreads_a_clump():
    d, mov, fillers, sizes, dens = make_density(macros=0)
    torch.manual_seed(0)
    center = torch.tensor([d.die_width, d.die_height]) / 2
    centers = center + torch.randn(len(sizes), 2) * 10
    _, grad, _ = dens.evaluate(centers)
    # Moving against the gradient pushes objects away from the clump center.
    outward = centers - center
    assert torch.nn.functional.cosine_similarity(-grad, outward).mean() > 0.9


def test_local_density_terms():
    d, mov, fillers, sizes, dens = make_density()
    torch.manual_seed(0)
    center = torch.tensor([d.die_width, d.die_height]) / 2
    clumped = center + torch.randn(len(sizes), 2) * 30
    _, grad, _, (local_grad, bin_overflow) = dens.evaluate(clumped, alpha=1e-12)
    # With nu_j ~ 1 (alpha -> 0) the local gradient is the global one (Eq. 8).
    assert torch.allclose(local_grad, grad, rtol=1e-4, atol=1e-6 * grad.abs().max())
    # Objects touching overflowed bins accumulate overflow; the rest none.
    assert (bin_overflow >= 0).all() and bin_overflow.max() > 0
    # A larger alpha strengthens the push out of overflowed bins.
    _, _, _, (strong, _) = dens.evaluate(clumped, alpha=0.1)
    assert strong.norm() > local_grad.norm()
