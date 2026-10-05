# RePlAce in PyTorch

A small, readable implementation of the RePlAce global placer
(Cheng, Kahng, Kang, Wang, *"RePlAce: Advancing Solution Quality and Routability
Validation in Global Placement"*, TCAD 2018; see `papers/replace.pdf`) in Python and PyTorch.

Scope: global placement only. The output is not legalized (cells may still
overlap slightly, macros may overlap) and there is no routability mode, since
both need tools outside RePlAce (a legalizer / detailed placer and a global router).

## Algorithm

    min_v  f(v) = W(v) + λ·D(v)

| Piece | Module | Source |
|---|---|---|
| Netlist as flat tensors (objects, pins, nets) | `design.py` | |
| Exact HPWL and weighted-average (WA) wirelength `W` | `wirelength.py` | ePlace |
| Electrostatic density `D`: bin grid, fillers, local smoothing, spectral (DCT) Poisson solve, overflow | `density.py` | ePlace / ePlace-MS |
| Bound-to-bound quadratic initial placement solved with conjugate gradient | `initial.py` | RePlAce |
| Nesterov's method with Lipschitz step prediction and backtracking | `nesterov.py` | ePlace |
| Main loop: preconditioner, γ schedule, λ update (Alg. 2), stopping rule | `placer.py` | RePlAce |
| Dynamic step size adaptation `-ds`: trial placement, transition points (Alg. 3), cof_max schedule (Eq. 10) | `dynamic_step.py` | RePlAce |
| Random benchmark generator | `generate.py` | |
| Plots, benchmark harness, CLI | `plot.py`, `bench.py`, `__init__.py` | |

Implementation notes:

- **Gradients:** the wirelength gradient comes from autograd. The density
  gradient is the ePlace electric force `q_i·E`, where the field `E` is computed
  spectrally per bin and averaged over the bins each object overlaps. It is
  plugged into autograd with a small custom `Function`, so the objective stays
  `W + λ·D`. Differentiating the bin density map directly would also be exact
  for the discretized energy, but that gradient jumps whenever an object edge
  crosses a bin boundary. That breaks Nesterov's Lipschitz step prediction, and
  placement then stalls at about 20% overflow.
- **Density map:** a rectangle's overlap with a bin factorizes into
  x-overlap × y-overlap. Standard cells and fillers (span ≤ 4 bins) use a 5×5
  window of bins; macros use dense overlaps with every bin (`ρ = Oxᵀ·Oy`).
- **DCT:** computed as products with precomputed M×M cosine/sine matrices.
- **Constants** not given in the paper are taken from ePlace and the open-source
  RePlAce and noted in comments: the γ schedule, λ₀ = 8e-5·|∇W|/|∇D|,
  cof ∈ [0.95, 1.05], the preconditioner `#pins + λ·area`, and filler sizing.
  ΔHPWL_ref (3.5e5 on ISPD designs) is rescaled to 0.075·#nets·bin width.
- **Additions not in RePlAce:** after initial placement every movable object
  gets a random kick of up to half a bin. Quadratic placement puts cells with
  identical connectivity at exactly the same point, where their own density
  peak exerts no separating force.

### Dynamic step size adaptation (`-ds`)

A trial placement runs with the default λ schedule until the overflow drops
below 1/2.5 of its initial value, recording the HPWL curve. Algorithm 3
splits the curve into three phases at two 2nd-order transition points (TP2)
and finds one 1st-order transition point (TP1) in each phase. The actual
placement then caps λ's per-iteration growth at cof_max (Eq. 10). The cap is
smallest at the TP2s (phase minima 1.0001 / 1.001 / 1.005) and largest at the
TP1s (phase maxima 1.001 / 1.025 / 1.05).

The paper leaves some details open; these follow the original RePlAce code
(`trial.cpp`, `opt.cpp`):
- The phase boundaries, i.e. the trial's start and end points plus the TP2s,
  serve as the TP2 in Eq. 10. Which one depends on whether the current point is
  before or after the phase's TP1.
- The ratio in Eq. 10 is capped at 1.
- Progress through the phases is tracked by the density potential, which keeps
  decreasing even while HPWL is flat.
- After the trial's end point, cof_max stays at 1.005.

Constant step size scales vs `-ds` (cf. the paper's Fig. 7), from
`benchmarks/step_sweep.py`. HPWL is relative to cof_max = 1.10; iterations
for `-ds` include the trial.

| design | λ step range | HPWL | vs [0.95, 1.10] | iterations | time (s) |
|---|---|---|---|---|---|
| rand10000_m13_s0 | [0.95, 1.10] | 6.0959e+05 | +0.00% | 229 | 1.5 |
| rand10000_m13_s0 | [0.95, 1.05] (default) | 6.0588e+05 | -0.61% | 412 | 2.5 |
| rand10000_m13_s0 | [0.95, 1.02] | 6.0417e+05 | -0.89% | 963 | 5.7 |
| rand10000_m13_s0 | [0.95, 1.01] | 6.0400e+05 | -0.92% | 1879 | 11.0 |
| rand10000_m13_s0 | [0.95, 1.005] | 6.0392e+05 | -0.93% | 3711 | 22.1 |
| rand10000_m13_s0 | [0.95, 1.002] | 6.0384e+05 | -0.94% | 9207 | 57.6 |
| rand10000_m13_s0 | -ds | 6.0374e+05 | -0.96% | 2013 | 13.2 |
| rand50000_m23_s0 | [0.95, 1.10] | 3.1611e+06 | +0.00% | 237 | 9.6 |
| rand50000_m23_s0 | [0.95, 1.05] (default) | 3.1231e+06 | -1.20% | 420 | 16.3 |
| rand50000_m23_s0 | [0.95, 1.02] | 3.1052e+06 | -1.77% | 983 | 38.6 |
| rand50000_m23_s0 | [0.95, 1.01] | 3.1020e+06 | -1.87% | 1923 | 74.6 |
| rand50000_m23_s0 | [0.95, 1.005] | 3.1015e+06 | -1.89% | 3802 | 147.6 |
| rand50000_m23_s0 | [0.95, 1.002] | 3.1008e+06 | -1.91% | 9438 | 367.8 |
| rand50000_m23_s0 | -ds | 3.0998e+06 | -1.94% | 2167 | 84.1 |

As in the paper, smaller constant step sizes give better HPWL with
diminishing returns, and `-ds` matches or beats the smallest constant step at
a fraction of its iterations. On these random designs the gain over the
default schedule is small (0.35–0.75%; the paper reports ~1.1% on ADAPTEC1 in Fig. 7).

The original code also contains extra heuristics not described in the paper
(trimming the trial curve, different per-phase constants, margins for TP1).
These are not implemented.

## Random benchmarks

`generate.py` first draws a hidden reference placement: macros at random
non-overlapping positions and standard cells packed into rows with uniform
whitespace. It then connects objects that are close in that placement. Net
degrees are mostly 2–3 pins with a heavy tail, and a few long nets act as
global wires. IO pads sit on the die boundary. A fully random netlist would have
no structure. With this one, the reference placement's HPWL is a consistent
yardstick at every size.

The reference is near-legal and spreads cells at the design utilization
(70%), while global placement packs cells at the target density (100%), so
global placement usually ends below the reference HPWL. Use `hpwl_vs_ref` to
compare runs, not as an optimality gap.

## Usage

    uv sync
    uv run replace place --cells 20000 --macros 8 --plot out/rand20k
    uv run replace place --cells 20000 --macros 8 --ds   # dynamic step size
    uv run replace bench                       # 1k ... 200k cells, default and -ds
    uv run replace bench --modes default       # default only
    uv run replace bench --sizes 1000,5000,20000 --out benchmarks/results
    uv run pytest

## Results

`uv run replace bench`: CPU only (8 threads), float32, target density 1.0,
stopped at overflow 0.1, macros movable. `hpwl_vs_first_mode` is the HPWL
relative to the default mode on the same design. Plots are in
`benchmarks/results/`.

`-ds` costs about 4–5× the iterations (17% of them in the trial, as in the
paper's Fig. 11). Its HPWL gain grows with design size, from noise level at
1k cells to −2.1% at 200k; the paper reports −2.0% on ISPD-2005/2006.

| design | mode | cells | macros | nets | bins | hpwl | hpwl_vs_ref | hpwl_vs_first_mode | overflow | trial_iterations | iterations | time_initial_s | time_global_s | ms_per_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rand1000_m0_s0 | default | 1000 | 0 | 1063 | 32 | 4.591e+04 | 0.598 | 1 | 0.09885 | 0 | 388 | 0.1974 | 0.6544 | 1.687 |
| rand1000_m0_s0 | ds | 1000 | 0 | 1063 | 32 | 4.593e+04 | 0.5982 | 1 | 0.09991 | 301 | 1476 | 0.2011 | 2.927 | 1.647 |
| rand2000_m4_s0 | default | 2000 | 4 | 2084 | 64 | 1.127e+05 | 0.6386 | 1 | 0.09991 | 0 | 392 | 0.2955 | 1.042 | 2.658 |
| rand2000_m4_s0 | ds | 2000 | 4 | 2084 | 64 | 1.125e+05 | 0.6377 | 0.9986 | 0.09991 | 297 | 1481 | 0.2934 | 4.44 | 2.497 |
| rand5000_m9_s0 | default | 5000 | 9 | 5130 | 64 | 2.952e+05 | 0.6391 | 1 | 0.0997 | 0 | 376 | 0.9303 | 1.607 | 4.274 |
| rand5000_m9_s0 | ds | 5000 | 9 | 5130 | 64 | 2.947e+05 | 0.6379 | 0.9982 | 0.09994 | 281 | 1511 | 0.8832 | 7.544 | 4.21 |
| rand10000_m13_s0 | default | 10000 | 13 | 10178 | 128 | 6.059e+05 | 0.6439 | 1 | 0.09952 | 0 | 412 | 1.122 | 3.047 | 7.395 |
| rand10000_m13_s0 | ds | 10000 | 13 | 10178 | 128 | 6.038e+05 | 0.6416 | 0.9965 | 0.09994 | 303 | 1710 | 1.098 | 14.29 | 7.099 |
| rand20000_m17_s0 | default | 20000 | 17 | 20265 | 128 | 1.213e+06 | 0.6401 | 1 | 0.09897 | 0 | 390 | 1.452 | 5.176 | 13.27 |
| rand20000_m17_s0 | ds | 20000 | 17 | 20265 | 128 | 1.207e+06 | 0.6368 | 0.9948 | 0.09989 | 281 | 1929 | 1.453 | 29.05 | 13.15 |
| rand50000_m23_s0 | default | 50000 | 23 | 50388 | 256 | 3.121e+06 | 0.647 | 1 | 0.09868 | 0 | 421 | 2.915 | 15.23 | 36.18 |
| rand50000_m23_s0 | ds | 50000 | 23 | 50388 | 256 | 3.1e+06 | 0.6427 | 0.9933 | 0.09989 | 299 | 1868 | 2.819 | 82.02 | 37.85 |
| rand100000_m27_s0 | default | 100000 | 27 | 100586 | 512 | 6.366e+06 | 0.6544 | 1 | 0.09888 | 0 | 468 | 6.874 | 35.63 | 76.14 |
| rand100000_m27_s0 | ds | 100000 | 27 | 100586 | 512 | 6.295e+06 | 0.6471 | 0.9888 | 0.09996 | 342 | 1944 | 6.707 | 164.4 | 71.93 |
| rand200000_m31_s0 | default | 200000 | 31 | 200800 | 512 | 1.279e+07 | 0.6564 | 1 | 0.09879 | 0 | 451 | 17.49 | 71.72 | 159 |
| rand200000_m31_s0 | ds | 200000 | 31 | 200800 | 512 | 1.252e+07 | 0.6428 | 0.9792 | 0.09992 | 306 | 1880 | 17.87 | 314.8 | 144 |
