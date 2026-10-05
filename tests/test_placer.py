import torch

from replace.generate import generate
from replace.initial import initial_place
from replace.placer import PlacerConfig, global_place
from replace.wirelength import hpwl


def inside_die(design, pos):
    lo = torch.tensor(design.die[:2]) + design.size / 2
    hi = torch.tensor(design.die[2:]) - design.size / 2
    return bool(((pos >= lo - 1e-3) & (pos <= hi + 1e-3)).all())


def test_generator_produces_valid_design():
    d = generate(1000, 4, seed=2)
    assert d.pin_obj.max() < d.num_objects and d.pin_net.max() < d.num_nets
    assert (torch.bincount(d.pin_net) >= 2).all()
    assert inside_die(d, d.pos)
    assert int(d.macro.sum()) == 4 and int(d.fixed.sum()) > 0  # pads are fixed


def test_initial_placement_beats_reference_wirelength():
    d = generate(1000, 0, seed=2)
    pos = initial_place(d)
    assert inside_die(d, pos)
    assert torch.equal(pos[d.fixed], d.pos[d.fixed])
    assert hpwl(d, pos) < hpwl(d, d.pos)  # overlapping, wirelength-only solution


def test_global_placement_converges():
    d = generate(1000, 3, seed=2)
    res = global_place(d, PlacerConfig(max_iters=1500))
    assert res.overflow <= 0.1
    assert inside_die(d, res.pos)
    assert torch.equal(res.pos[d.fixed], d.pos[d.fixed])
    assert res.hpwl < 1.5 * hpwl(d, d.pos)
