"""Placement result figure: placement before and after global placement, with
the HPWL and overflow curves underneath."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .design import Design  # noqa: E402
from .placer import PlaceResult  # noqa: E402

DPI = 200


def _rects(pos: torch.Tensor, size: torch.Tensor):
    lo, hi = (pos - size / 2).numpy(), (pos + size / 2).numpy()
    return [[(a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1])] for a, b in zip(lo, hi)]


def draw_placement(ax, design: Design, pos: torch.Tensor, filler_pos: torch.Tensor, filler_size: torch.Tensor, title: str):
    """Die outline, fillers (green), cells (red), movable macros (blue), fixed macros (gray)."""
    xl, yl, xh, yh = design.die
    ax.add_patch(plt.Rectangle((xl, yl), xh - xl, yh - yl, fill=False, lw=0.8))
    pos, size = pos.detach().float(), design.size.float()
    if len(filler_pos):
        ax.add_collection(PolyCollection(_rects(filler_pos.detach().float(), filler_size.float()), facecolor="tab:green", edgecolor="none", alpha=0.25))
    groups = [
        ("cells", design.movable & ~design.macro, dict(facecolor="tab:red", edgecolor="none", alpha=0.6)),
        ("movable macros", design.movable & design.macro, dict(facecolor="tab:blue", edgecolor="navy", lw=0.5, alpha=0.4)),
        ("fixed macros", design.fixed & design.macro, dict(facecolor="gray", edgecolor="black", lw=0.5, alpha=0.5)),
    ]
    legend = []
    if len(filler_pos):
        legend.append(Patch(facecolor="tab:green", alpha=0.4, label="fillers"))
    for label, mask, style in groups:
        if mask.any():
            ax.add_collection(PolyCollection(_rects(pos[mask], size[mask]), **style))
            legend.append(Patch(**style, label=label))
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    ax.set_aspect("equal")
    ax.set_title(title)


def plot_result(design: Design, res: PlaceResult, path: str, title: str = ""):
    fig = plt.figure(figsize=(14, 13), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[3, 1, 1])
    draw_placement(fig.add_subplot(grid[0, 0]), design, res.initial_pos, res.initial_filler_pos, res.filler_size,
                   f"before: initial placement, HPWL {res.history[0]['hpwl']:.4g}, overflow {res.history[0]['overflow']:.3f}")
    draw_placement(fig.add_subplot(grid[0, 1]), design, res.pos, res.filler_pos, res.filler_size,
                   f"after: {res.iterations} iterations, HPWL {res.hpwl:.4g}, overflow {res.overflow:.3f}")

    ax_hpwl = fig.add_subplot(grid[1, :])
    ax_ovf = fig.add_subplot(grid[2, :], sharex=ax_hpwl)
    curves = [(res.history, "placement", {})]
    if res.trial_history:
        curves.insert(0, (res.trial_history, "trial placement (-ds)", dict(color="gray", alpha=0.7)))
    for hist, label, style in curves:
        it = [h["iter"] for h in hist]
        ax_hpwl.plot(it, [h["hpwl"] for h in hist], label=label, **style)
        ax_ovf.plot(it, [h["overflow"] for h in hist], label=label, **style)
    if res.transition_points:
        # Anchors alternate: start, TP1, TP2, TP1, TP2, TP1, end.
        for k, i in enumerate(res.transition_points.index[1:-1]):
            marker = ("s", "gold", "TP1") if k % 2 == 0 else ("*", "red", "TP2")
            ax_hpwl.plot(i, res.trial_history[i]["hpwl"], marker[0], color=marker[1], ms=9, mec="black",
                         label=marker[2] if k < 2 else None)
    ax_hpwl.set_ylabel("HPWL")
    ax_hpwl.legend(loc="lower right")
    ax_ovf.set_ylabel("overflow")
    ax_ovf.set_xlabel("iteration")
    for ax in (ax_hpwl, ax_ovf):
        ax.grid(alpha=0.3)
    fig.suptitle(title or design.summary())
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
