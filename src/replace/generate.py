"""Random placement benchmarks with a known good reference placement.

Fully random netlists have no structure: every placement is about equally bad.
Instead we first draw a hidden reference placement (macros at random
non-overlapping spots, standard cells packed into rows around them) and then
connect objects that are close in it. The reference HPWL is therefore a
meaningful yardstick for the placer at every design size.

Units follow the ISPD contests loosely: row height 12, cell widths in sites of 1.
"""

import math

import numpy as np
import torch

from .design import Design

ROW_HEIGHT = 12.0


def _place_macros(rng, die_w, die_h, total_area, count):
    """Random non-overlapping macros (lower-left corners) snapped to rows."""
    if count == 0:
        return np.zeros((0, 4))
    areas = rng.uniform(0.5, 1.5, count)
    areas *= total_area / areas.sum()
    macros = []  # (x, y, w, h)
    for area in sorted(areas, reverse=True):
        aspect = rng.uniform(0.5, 2.0)
        h = max(2, round(math.sqrt(area * aspect) / ROW_HEIGHT)) * ROW_HEIGHT
        w = area / h
        for _ in range(10000):
            x = rng.uniform(0, die_w - w)
            y = rng.integers(0, int((die_h - h) / ROW_HEIGHT) + 1) * ROW_HEIGHT
            gap = ROW_HEIGHT  # keep some space between macros
            if all(x + w + gap <= mx or mx + mw + gap <= x or y + h + gap <= my or my + mh + gap <= y for mx, my, mw, mh in macros):
                macros.append((x, y, w, h))
                break
        else:
            raise RuntimeError("could not place macros; lower macro_area_frac or num_macros")
    return np.array(macros).reshape(-1, 4)


def _free_row_segments(die_w, num_rows, macros, min_len):
    """(row, x0, x1) row segments not covered by macros."""
    segments = []
    for r in range(num_rows):
        y0, y1 = r * ROW_HEIGHT, (r + 1) * ROW_HEIGHT
        blocked = sorted((mx, mx + mw) for mx, my, mw, mh in macros if my < y1 and my + mh > y0)
        x = 0.0
        for b0, b1 in blocked + [(die_w, die_w)]:
            if b0 - x >= min_len:
                segments.append((r, x, b0))
            x = max(x, b1)
    return np.array(segments)


def generate(
    num_cells: int,
    num_macros: int = 0,
    movable_macros: bool = True,
    macro_area_frac: float = 0.2,
    utilization: float = 0.7,
    nets_per_cell: float = 1.0,
    seed: int = 0,
) -> Design:
    """Random design. `design.pos` holds the hidden reference placement.

    utilization: standard cell area / area not covered by macros.
    macro_area_frac: macro area / die area (if num_macros > 0).
    """
    rng = np.random.default_rng(seed)
    if num_macros == 0:
        macro_area_frac = 0.0

    # Standard cells: widths roughly log-normal around 6 sites.
    widths = np.clip(np.round(rng.lognormal(math.log(6), 0.5, num_cells)), 2, 40)
    cell_area = widths.sum() * ROW_HEIGHT
    die_area = cell_area / (utilization * (1 - macro_area_frac))
    num_rows = max(1, round(math.sqrt(die_area) / ROW_HEIGHT))
    die_h = num_rows * ROW_HEIGHT
    die_w = die_area / die_h

    macros = _place_macros(rng, die_w, die_h, macro_area_frac * die_area, num_macros)

    # Reference cell placement: walk the free row segments left to right,
    # bottom to top, spreading cells evenly so whitespace is uniform. A cell
    # that would cross a segment end is pushed back inside it (it may then
    # slightly overlap its left neighbor: the reference is near-legal).
    segs = _free_row_segments(die_w, num_rows, macros, min_len=widths.max())
    seg_len = segs[:, 2] - segs[:, 1]
    seg_start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
    s = (np.cumsum(widths) - widths) * (seg_len.sum() / widths.sum())
    k = np.searchsorted(seg_start, s, side="right") - 1
    left = np.minimum(segs[k, 1] + s - seg_start[k], segs[k, 2] - widths)
    cell_xy = np.stack([left + widths / 2, segs[k, 0] * ROW_HEIGHT + ROW_HEIGHT / 2], 1)

    # IO pads: zero-size fixed terminals evenly spread on the die boundary.
    num_pads = max(4, int(2 * math.sqrt(num_cells)))
    # t is the distance along the boundary, counter-clockwise from (0, 0).
    t = (np.arange(num_pads) + rng.uniform(0, 1)) / num_pads * 2 * (die_w + die_h)
    pad_x = np.clip(t, 0, die_w) - np.clip(t - (die_w + die_h), 0, die_w)
    pad_y = np.clip(t - die_w, 0, die_h) - np.clip(t - (2 * die_w + die_h), 0, die_h)
    pad_xy = np.stack([pad_x, pad_y], 1)

    # Object table: cells, then macros, then pads.
    macro_xy = macros[:, :2] + macros[:, 2:] / 2
    size = np.concatenate([np.stack([widths, np.full(num_cells, ROW_HEIGHT)], 1), macros[:, 2:], np.zeros((num_pads, 2))])
    xy = np.concatenate([cell_xy, macro_xy, pad_xy])
    fixed = np.concatenate([np.zeros(num_cells, bool), np.full(num_macros, not movable_macros), np.ones(num_pads, bool)])
    pad0 = num_cells + num_macros

    # Nets: one seed object per net (a random cell, or each pad), plus d - 1
    # pins at random points around the seed. Degrees: 2 + geometric, with a
    # small fraction of large nets. Spread grows with degree and varies
    # log-normally so a few nets are long (global wires).
    seeds = np.concatenate([rng.integers(0, num_cells, round(nets_per_cell * num_cells)), np.arange(pad0, pad0 + num_pads)])
    num_nets = len(seeds)
    degree = 1 + rng.geometric(0.45, num_nets)
    big = rng.random(num_nets) < 0.02
    degree[big] = rng.integers(10, 60, big.sum())
    pitch = math.sqrt(die_area / num_cells)  # average distance between neighboring cells
    sigma = pitch * np.sqrt(degree) * rng.lognormal(0, 0.7, num_nets)
    t_net = np.repeat(np.arange(num_nets), degree - 1)
    target = xy[seeds[t_net]] + rng.normal(0, 1, (len(t_net), 2)) * sigma[t_net, None]
    target = np.clip(target, 0, [die_w, die_h])

    # Map each target point to the object there: a macro if inside one, else a
    # random cell whose center falls in the same bucket (bucket ~ 2 x 2 cells).
    t_obj = np.full(len(target), -1)
    for m, (mx, my, mw, mh) in enumerate(macros):
        inside = (target[:, 0] >= mx) & (target[:, 0] <= mx + mw) & (target[:, 1] >= my) & (target[:, 1] <= my + mh)
        t_obj[inside] = num_cells + m
    nbx, nby = max(1, int(die_w / (2 * pitch))), max(1, int(die_h / (2 * pitch)))

    def bucket(p):
        bx = np.minimum((p[:, 0] / die_w * nbx).astype(int), nbx - 1)
        by = np.minimum((p[:, 1] / die_h * nby).astype(int), nby - 1)
        return bx * nby + by

    cell_bucket = bucket(cell_xy)
    order = np.argsort(cell_bucket, kind="stable")
    count = np.bincount(cell_bucket, minlength=nbx * nby)
    start = np.cumsum(count) - count
    tb = bucket(target)
    free = (t_obj < 0) & (count[tb] > 0)
    pick = start[tb[free]] + (rng.random(free.sum()) * count[tb[free]]).astype(int)
    t_obj[free] = order[pick]

    # Pin offsets: cell pins uniformly inside the cell, macro pins exactly at the
    # target point (on the macro face nearest the net), pad pins at the pad.
    pin_obj = np.concatenate([seeds, t_obj])
    pin_net = np.concatenate([np.arange(num_nets), t_net])
    pin_off = (rng.random((len(pin_obj), 2)) - 0.5) * size[pin_obj] * 0.8
    is_macro_t = np.concatenate([np.zeros(num_nets, bool), (t_obj >= num_cells) & (t_obj < pad0)])
    pin_off[is_macro_t] = np.concatenate([np.zeros((num_nets, 2)), target])[is_macro_t] - xy[pin_obj[is_macro_t]]

    # Drop pins that hit empty space, then nets left with fewer than 2 pins.
    keep = pin_obj >= 0
    pin_obj, pin_net, pin_off = pin_obj[keep], pin_net[keep], pin_off[keep]
    alive = np.bincount(pin_net, minlength=num_nets) >= 2
    keep = alive[pin_net]
    new_id = np.cumsum(alive) - 1
    pin_obj, pin_net, pin_off = pin_obj[keep], new_id[pin_net[keep]], pin_off[keep]

    f32 = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return Design(
        name=f"rand{num_cells}_m{num_macros}{'' if movable_macros else 'f'}_s{seed}",
        die=(0.0, 0.0, float(die_w), float(die_h)),
        row_height=ROW_HEIGHT,
        size=f32(size),
        pos=f32(xy),
        fixed=torch.tensor(fixed),
        pin_obj=torch.tensor(pin_obj, dtype=torch.long),
        pin_net=torch.tensor(pin_net, dtype=torch.long),
        pin_offset=f32(pin_off),
        num_nets=int(alive.sum()),
    )
