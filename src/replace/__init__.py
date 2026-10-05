"""RePlAce global placement in PyTorch."""

import argparse

from .design import Design
from .generate import generate
from .placer import PlaceResult, PlacerConfig, global_place
from .wirelength import hpwl

__all__ = ["Design", "PlaceResult", "PlacerConfig", "generate", "global_place", "hpwl", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="replace", description="RePlAce global placement on random designs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def placer_args(p):
        p.add_argument("--target-density", type=float, default=1.0)
        p.add_argument("--max-iters", type=int, default=3000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--fixed-macros", action="store_true", help="make macros fixed instead of movable")

    p = sub.add_parser("place", help="place one random design")
    p.add_argument("--cells", type=int, default=10000)
    p.add_argument("--macros", type=int, default=0)
    p.add_argument("--plot", help="write placement and convergence plots with this path prefix")
    p.add_argument("--log-every", type=int, default=50)
    placer_args(p)

    b = sub.add_parser("bench", help="benchmark on random designs of increasing size")
    b.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], help="comma-separated cell counts")
    b.add_argument("--out", default="benchmarks/results")
    b.add_argument("--no-plots", action="store_true")
    placer_args(b)

    args = parser.parse_args()
    cfg = PlacerConfig(target_density=args.target_density, max_iters=args.max_iters, seed=args.seed)

    if args.cmd == "place":
        from .plot import plot_history, plot_placement

        design = generate(args.cells, args.macros, movable_macros=not args.fixed_macros, seed=args.seed)
        print(design.summary())
        ref = hpwl(design, design.pos)
        cfg.log_every = args.log_every
        res = global_place(design, cfg)
        print(
            f"HPWL {res.hpwl:.4e} ({res.hpwl / ref:.3f} x reference {ref:.4e}), overflow {res.overflow:.3f}, "
            f"{res.iterations} iterations, {res.time_initial:.1f}s initial + {res.time_global:.1f}s global placement"
        )
        if args.plot:
            plot_placement(design, res.pos, f"{args.plot}.png", res.filler_pos, res.filler_size)
            plot_history(res.history, f"{args.plot}_history.png", design.name)
    else:
        from . import bench

        bench.run(args.sizes or bench.DEFAULT_SIZES, args.out, movable_macros=not args.fixed_macros, seed=args.seed, cfg=cfg, plots=not args.no_plots)
