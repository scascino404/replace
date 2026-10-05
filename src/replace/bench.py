"""Benchmark the placer on random designs of increasing size."""

import csv
import math
import os

from .generate import generate
from .placer import PlacerConfig, global_place
from .plot import plot_history, plot_placement
from .wirelength import hpwl

DEFAULT_SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]


def macros_for(num_cells: int) -> int:
    """Macro count growing with design size: 0 at 1k cells, ~26 at 100k."""
    return max(0, round(4 * math.log2(num_cells / 1000)))


def run(sizes=DEFAULT_SIZES, out_dir="benchmarks/results", movable_macros=True, seed=0, cfg=PlacerConfig(), plots=True) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for n in sizes:
        design = generate(n, macros_for(n), movable_macros=movable_macros, seed=seed)
        print(design.summary(), flush=True)
        ref = hpwl(design, design.pos)
        res = global_place(design, cfg)
        row = dict(
            design=design.name,
            cells=n,
            macros=int(design.macro.sum()),
            nets=design.num_nets,
            pins=design.num_pins,
            bins=res.bins,
            fillers=len(res.filler_size),
            ref_hpwl=ref,
            initial_hpwl=res.initial_hpwl,
            hpwl=res.hpwl,
            hpwl_vs_ref=res.hpwl / ref,
            overflow=res.overflow,
            iterations=res.iterations,
            time_initial_s=res.time_initial,
            time_global_s=res.time_global,
            ms_per_iter=1000 * res.time_global / max(res.iterations, 1),
        )
        rows.append(row)
        print(
            f"  hpwl {res.hpwl:.4e} ({row['hpwl_vs_ref']:.3f} x ref), overflow {res.overflow:.3f}, "
            f"{res.iterations} iters, {res.time_initial:.1f}s initial + {res.time_global:.1f}s global",
            flush=True,
        )
        if plots:
            plot_placement(design, res.pos, f"{out_dir}/{design.name}.png", res.filler_pos, res.filler_size)
            plot_history(res.history, f"{out_dir}/{design.name}_history.png", design.name)
        write_tables(rows, out_dir)
    return rows


def write_tables(rows: list[dict], out_dir: str) -> None:
    with open(f"{out_dir}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cols = ["design", "cells", "macros", "nets", "bins", "hpwl", "hpwl_vs_ref", "overflow", "iterations", "time_initial_s", "time_global_s", "ms_per_iter"]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    with open(f"{out_dir}/results.md", "w") as f:
        f.write("\n".join(lines) + "\n")
