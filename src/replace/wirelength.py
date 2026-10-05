"""Wirelength: exact half-perimeter wirelength (HPWL) and its smooth WA model."""

import torch

from .design import Design


def _net_bounds(design: Design, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-net max and min of per-pin (p, 2) values -> two (num_nets, 2)."""
    deg = design.net_degree
    return torch.segment_reduce(xy, "max", lengths=deg), torch.segment_reduce(xy, "min", lengths=deg)


def _hpwl(x_max: torch.Tensor, x_min: torch.Tensor) -> float:
    return (x_max - x_min).double().sum().item()


@torch.no_grad()
def hpwl(design: Design, pos: torch.Tensor) -> float:
    """Exact HPWL: sum over nets of the bounding-box half perimeter."""
    return _hpwl(*_net_bounds(design, design.pin_pos(pos)))


def wa_wirelength(design: Design, pos: torch.Tensor, gamma: float) -> torch.Tensor:
    """Weighted-average (WA) wirelength model [ePlace / Hsu et al.].

    Per net and axis, the max pin coordinate is approximated by
        sum(x * exp(x / gamma)) / sum(exp(x / gamma))
    and the min by the same expression with -gamma. As gamma -> 0 the model
    converges to HPWL. Differentiable through autograd.
    """
    return wirelength(design, pos, gamma)[0]


def wirelength(design: Design, pos: torch.Tensor, gamma: float) -> tuple[torch.Tensor, float]:
    """WA wirelength (see `wa_wirelength`) and the exact HPWL at `pos`."""
    xy = design.pin_pos(pos)
    net = design.pin_net
    # Shift exponents by the per-net max/min so exp() never overflows; the
    # shift cancels between numerator and denominator.
    with torch.no_grad():
        x_max, x_min = _net_bounds(design, xy)
    e_max = torch.exp((xy - x_max[net]) / gamma)
    e_min = torch.exp((x_min[net] - xy) / gamma)
    # All four per-net sums in one pass.
    sums = xy.new_zeros(design.num_nets, 8).index_add(0, net, torch.cat([xy * e_max, e_max, xy * e_min, e_min], 1))
    wa_max = sums[:, 0:2] / sums[:, 2:4]
    wa_min = sums[:, 4:6] / sums[:, 6:8]
    return (wa_max - wa_min).sum(), _hpwl(x_max, x_min)
