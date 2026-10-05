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
    uv run replace bench                       # 1k ... 200k cells
    uv run replace bench --sizes 1000,5000,20000 --out benchmarks/results
    uv run pytest

## Results

Phase 1 (ePlace core, no `-ds` / `-ld` yet). Run with `uv run replace bench`:
CPU only (8 threads), float32, target density 1.0, stopped at overflow 0.1.
Macros are movable. Plots are in `benchmarks/results/`.

| design | cells | macros | nets | bins | hpwl | hpwl_vs_ref | overflow | iterations | time_initial_s | time_global_s | ms_per_iter |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rand1000_m0_s0 | 1000 | 0 | 1063 | 32 | 4.591e+04 | 0.598 | 0.09885 | 388 | 0.1675 | 0.5692 | 1.467 |
| rand2000_m4_s0 | 2000 | 4 | 2084 | 64 | 1.127e+05 | 0.6386 | 0.09991 | 392 | 0.2648 | 0.858 | 2.189 |
| rand5000_m9_s0 | 5000 | 9 | 5130 | 64 | 2.953e+05 | 0.6392 | 0.0996 | 376 | 0.7893 | 1.291 | 3.433 |
| rand10000_m13_s0 | 10000 | 13 | 10178 | 128 | 6.059e+05 | 0.6439 | 0.09929 | 412 | 0.9723 | 2.488 | 6.04 |
| rand20000_m17_s0 | 20000 | 17 | 20265 | 128 | 1.213e+06 | 0.6401 | 0.09912 | 390 | 1.278 | 5.461 | 14 |
| rand50000_m23_s0 | 50000 | 23 | 50388 | 256 | 3.122e+06 | 0.6471 | 0.09834 | 421 | 3.014 | 12.84 | 30.51 |
| rand100000_m27_s0 | 100000 | 27 | 100586 | 512 | 6.363e+06 | 0.6541 | 0.09991 | 467 | 6.658 | 32.76 | 70.15 |
| rand200000_m31_s0 | 200000 | 31 | 200800 | 512 | 1.279e+07 | 0.6564 | 0.09849 | 451 | 16.23 | 58.73 | 130.2 |
