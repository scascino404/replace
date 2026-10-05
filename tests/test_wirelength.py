from dataclasses import replace

import torch

from replace.generate import generate
from replace.wirelength import hpwl, wa_wirelength


def small_design(dtype=torch.float64):
    d = generate(300, 2, seed=3)
    return replace(d, pos=d.pos.to(dtype), size=d.size.to(dtype), pin_offset=d.pin_offset.to(dtype))


def test_wa_converges_to_hpwl():
    d = small_design()
    exact = hpwl(d, d.pos)
    # WA underestimates HPWL and approaches it as gamma -> 0.
    assert wa_wirelength(d, d.pos, 10.0).item() < exact
    assert abs(wa_wirelength(d, d.pos, 0.01).item() - exact) / exact < 1e-3


def test_wa_gradient_matches_finite_differences():
    d = small_design()
    pos = d.pos.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda p: wa_wirelength(d, p, 5.0), (pos,), eps=1e-6, atol=1e-5)
