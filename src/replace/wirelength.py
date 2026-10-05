"""Wirelength: exact half-perimeter wirelength (HPWL) and its smooth WA model."""

import torch

from .design import Design


def _net_reduce(design: Design, values: torch.Tensor, reduce: str) -> torch.Tensor:
    """Per-net reduction ('amax', 'amin') of per-pin (p, 2) values -> (num_nets, 2)."""
    index = design.pin_net[:, None].expand_as(values)
    out = values.new_zeros(design.num_nets, 2)
    return out.scatter_reduce(0, index, values, reduce, include_self=False)


@torch.no_grad()
def hpwl(design: Design, pos: torch.Tensor) -> float:
    """Exact HPWL: sum over nets of the bounding-box half perimeter."""
    xy = design.pin_pos(pos)
    span = _net_reduce(design, xy, "amax") - _net_reduce(design, xy, "amin")
    return span.double().sum().item()


def wa_wirelength(design: Design, pos: torch.Tensor, gamma: float) -> torch.Tensor:
    """Weighted-average (WA) wirelength model [ePlace / Hsu et al.].

    Per net and axis, the max pin coordinate is approximated by
        sum(x * exp(x / gamma)) / sum(exp(x / gamma))
    and the min by the same expression with -gamma. As gamma -> 0 the model
    converges to HPWL. Differentiable through autograd.
    """
    xy = design.pin_pos(pos)
    net = design.pin_net
    # Shift exponents by the per-net max/min so exp() never overflows; the
    # shift cancels between numerator and denominator.
    with torch.no_grad():
        x_max = _net_reduce(design, xy, "amax")[net]
        x_min = _net_reduce(design, xy, "amin")[net]
    e_max = torch.exp((xy - x_max) / gamma)
    e_min = torch.exp((x_min - xy) / gamma)

    def net_sum(v: torch.Tensor) -> torch.Tensor:
        return v.new_zeros(design.num_nets, 2).index_add(0, net, v)

    wa_max = net_sum(xy * e_max) / net_sum(e_max)
    wa_min = net_sum(xy * e_min) / net_sum(e_min)
    return (wa_max - wa_min).sum()
