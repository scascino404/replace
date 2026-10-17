"""Placement result figure: placement before and after global placement, with
the HPWL and overflow curves underneath. Optionally, an animated GIF of the
placement from the snapshots taken during the run."""

import textwrap

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from .design import Design
from .placer import PlaceResult

DPI = 200


def _rects(pos: torch.Tensor, size: torch.Tensor):
    """(k, 4, 2) rectangle corners (an array builds a PolyCollection much faster than lists)."""
    lo, hi = pos - size / 2, pos + size / 2
    return torch.stack([lo, torch.stack([hi[:, 0], lo[:, 1]], 1), hi, torch.stack([lo[:, 0], hi[:, 1]], 1)], 1).numpy()


def draw_placement(ax, design: Design, pos: torch.Tensor, filler_pos: torch.Tensor, filler_size: torch.Tensor, title: str):
    """Die outline, fillers (green), cells (red), movable macros (blue), fixed macros (gray).
    Returns a function move(pos, filler_pos) that redraws the objects at new positions."""
    xl, yl, xh, yh = design.die
    ax.add_patch(Rectangle((xl, yl), xh - xl, yh - yl, fill=False, lw=0.8))
    pos, size, filler_size = pos.detach().float(), design.size.float(), filler_size.float()
    layers = []  # (collection, object mask; None for the fillers)
    # Collections skip autolim (slow for many polygons): the limits are the die.
    if len(filler_pos):
        fillers = PolyCollection(_rects(filler_pos.detach().float(), filler_size), facecolor="tab:green", edgecolor="none", alpha=0.25)
        ax.add_collection(fillers, autolim=False)
        layers.append((fillers, None))
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
            objects = PolyCollection(_rects(pos[mask], size[mask]), **style)
            ax.add_collection(objects, autolim=False)
            layers.append((objects, mask))
            legend.append(Patch(**style, label=label))
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlim(xl, xh)
    ax.set_ylim(yl, yh)
    ax.set_aspect("equal")
    ax.set_title(title)

    def move(pos: torch.Tensor, filler_pos: torch.Tensor) -> None:
        pos, filler_pos = pos.detach().float(), filler_pos.detach().float()
        for coll, mask in layers:
            coll.set_verts(_rects(filler_pos, filler_size) if mask is None else _rects(pos[mask], size[mask]))

    return move


def plot_result(design: Design, res: PlaceResult, path: str, title: str = ""):
    fig = Figure(figsize=(14, 13), layout="constrained")
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


def animate(design: Design, res: PlaceResult, path: str, title: str = "", frames: int = 80, fps: int = 12, hold: float = 1.5, dpi: int = 100):
    """Animated GIF of the actual placement, from res.snapshots (PlacerConfig.snapshot_every).

    The pace of the placement varies a lot (fast spreading right after the
    initial placement, a near standstill while lambda is small, steady spreading
    later), so frames are picked by movement rather than by iteration: objects
    and fillers move by about the same mean distance between consecutive frames.
    The last frame is held for `hold` seconds and the first for a third of
    that, so the loop restarts without a long pause. All frames share one
    undithered palette, which keeps the file ~30% smaller than per-frame
    palettes.
    """
    from PIL import Image  # a matplotlib dependency

    snaps = res.snapshots
    if not snaps:
        raise ValueError("no snapshots: run the placer with PlacerConfig.snapshot_every > 0")
    n = len(snaps[0][1]) + len(snaps[0][2])

    def dist(a, b):  # mean distance moved by objects and fillers from snapshot a to b
        (_, pa, fa), (_, pb, fb) = snaps[a], snaps[b]
        return ((pb - pa).norm(dim=1).sum() + (fb - fa).norm(dim=1).sum()).item() / n

    def pick(step):  # greedy: next frame once the mean distance from the last frame reaches step, plus the final one
        out = [0]
        for k in range(1, len(snaps)):
            if dist(out[-1], k) >= step:
                out.append(k)
        return sorted(set(out) | {len(snaps) - 1})

    # Bisect for the smallest step that gives at most `frames` frames.
    lo, hi = 0.0, max(dist(0, k) for k in range(len(snaps))) + 1e-6
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if len(pick(mid)) <= frames else (mid, hi)
    picks = pick(hi)

    fig = Figure(figsize=(6, 8.2), dpi=dpi, layout="constrained")
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 1, height_ratios=[6, 1.1, 1.1])
    ax = fig.add_subplot(grid[0])
    _, pos, filler_pos = snaps[0]
    move = draw_placement(ax, design, pos, filler_pos, res.filler_size, "")
    ax_hpwl = fig.add_subplot(grid[1])
    ax_ovf = fig.add_subplot(grid[2], sharex=ax_hpwl)
    it = [h["iter"] for h in res.history]
    cursors = []
    for a, key in ((ax_hpwl, "hpwl"), (ax_ovf, "overflow")):
        a.plot(it, [h[key] for h in res.history], lw=1)
        cursors.append(a.axvline(0, color="tab:red", lw=1.5))
        a.set_ylabel(key if key == "overflow" else "HPWL")
        a.grid(alpha=0.3)
    ax_hpwl.tick_params(labelbottom=False)
    ax_ovf.set_xlabel("iteration")
    ax.set_title("iteration 0")  # reserve the title's space before fixing the layout
    fig.suptitle(textwrap.fill(title or design.summary(), 80), fontsize=9)
    # Fix the layout so the panels don't shift as the title text changes.
    fig.draw_without_rendering()
    fig.set_layout_engine("none")

    images = []
    for k in picks:
        it, pos, filler_pos = snaps[k]
        h = res.history[it]
        move(pos, filler_pos)
        ax.set_title(f"iteration {it}: HPWL {h['hpwl']:.4g}, overflow {h['overflow']:.3f}")
        for c in cursors:
            c.set_xdata([it, it])
        canvas.draw()
        images.append(Image.fromarray(np.asarray(canvas.buffer_rgba())).convert("RGB"))
    # The palette comes from the first (clumped, overlapping) and last (spread) frames.
    first, last = images[0], images[-1]
    both = Image.new("RGB", (first.width * 2, first.height))
    both.paste(first, (0, 0))
    both.paste(last, (first.width, 0))
    palette = both.quantize(colors=64, dither=Image.Dither.NONE)
    images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    durations = [round(1000 / fps)] * len(images)
    durations[0] = round(1000 * max(hold / 3, 1 / fps))
    durations[-1] = round(1000 * max(hold, 1 / fps))
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
