"""Placement and convergence plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

from .design import Design  # noqa: E402


def _rects(pos: torch.Tensor, size: torch.Tensor):
    lo, hi = (pos - size / 2).numpy(), (pos + size / 2).numpy()
    return [[(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])] for a, b in zip(lo, hi)]


def plot_placement(design: Design, pos: torch.Tensor, path: str, filler_pos=None, filler_size=None, title: str = ""):
    """Draw the die, fillers (green), cells (red), movable macros (blue) and fixed macros (gray)."""
    fig, ax = plt.subplots(figsize=(8, 8 * design.die_height / design.die_width))
    xl, yl, xh, yh = design.die
    ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, fill=False, lw=1))
    pos, size = pos.detach().float(), design.size.float()
    groups = [
        (design.movable & ~design.macro, dict(facecolor="tab:red", edgecolor="none", alpha=0.6)),
        (design.movable & design.macro, dict(facecolor="tab:blue", edgecolor="navy", alpha=0.4)),
        (design.fixed & design.macro, dict(facecolor="gray", edgecolor="black", alpha=0.5)),
    ]
    if filler_pos is not None and len(filler_pos):
        ax.add_collection(PolyCollection(_rects(filler_pos.detach().float(), filler_size.float()), facecolor="tab:green", edgecolor="none", alpha=0.25))
    for mask, style in groups:
        if mask.any():
            ax.add_collection(PolyCollection(_rects(pos[mask], size[mask]), **style))
    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    ax.set_aspect("equal")
    ax.set_title(title or design.name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: list[dict], path: str, title: str = ""):
    """HPWL and overflow versus Nesterov iteration."""
    it = [h["iter"] for h in history]
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    a.plot(it, [h["hpwl"] for h in history])
    a.set_ylabel("HPWL")
    b.plot(it, [h["overflow"] for h in history], color="tab:orange")
    b.set_ylabel("overflow")
    b.set_xlabel("iteration")
    a.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
