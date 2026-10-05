"""Constant step size scales vs dynamic step size adaptation (cf. Fig. 7 of the paper).

    uv run python benchmarks/step_sweep.py
"""

from replace.bench import macros_for
from replace.generate import generate
from replace.placer import PlacerConfig, global_place

for n in (10000, 50000):
    d = generate(n, macros_for(n), seed=0)
    base = None
    runs = [(f"[0.95, {c}]", PlacerConfig(cof_max=c, max_iters=20000)) for c in (1.10, 1.05, 1.02, 1.01, 1.005, 1.002)]
    runs.append(("-ds", PlacerConfig(dynamic_step=True, max_iters=20000)))
    for label, cfg in runs:
        r = global_place(d, cfg)
        base = base or r.hpwl
        print(
            f"| {d.name} | {label} | {r.hpwl:.4e} | {100 * (r.hpwl / base - 1):+.2f}% | "
            f"{r.trial_iterations + r.iterations} | {r.time_global:.1f} |",
            flush=True,
        )
