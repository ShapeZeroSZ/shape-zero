# Shape Zero numerical tests

Scripts and raw results from the J-compatibility, readout, and q = 3 prediction
tests run against the Shape Zero archive's `model.py`. Every number below was
produced by the script named next to it, with the settings listed.

## Running

Everything runs from this repository. Run scripts from inside
`shape_zero_tests/` (several read their results files by relative path). The
pinned scripts use the hash-pinned copies in `model_versions/` below;
`q3_gate.py` uses the single working `model.py`, `04_scripts/session/model.py`,
which it finds relative to its own location. There is no `model.py` in this
folder.

    cd shape_zero_tests
    python3 <script> [args]

Python 3 with NumPy.

## Pinned model versions

Each script imports the exact `model.py` it was run with, copied unmodified from
the Shape Zero archive (`ark/04_scripts/session/model.py`):

| folder | archive | sha256 | used by |
|---|---|---|---|
| `model_versions/c49da46f/` | `c49da46f-shape_zero_ark.zip` | `f486a0083e5fe2956fcb41bc717c6bc6d07ee36676b6b5d8c12f4fd439adb99f` | `j_compat_test.py` (and `grid.py`, `checks.py`, `kscan.py`, `resid.py`, `openrows.py`), `gate7_readout.py` — sections 1–6 |
| `model_versions/948b09e8/` | `948b09e8-shape_zero_ark.zip` | `427e4c934f0cba3515961d5b9fdf88ebd480a920d1196117412e6736a29a8e9b` | `q3_readout.py`, `q3_combine.py`, `q3_kavg.py` — sections 7–8 |

The two versions differ only in a docstring note in `run_until_exit` and in
`model.py`'s own gate-7 pass criterion inside `main()`, which these scripts never
call. Setting `MODEL_DIR`
still overrides the pinned version. Verify the pins with:

    sha256sum model_versions/*/model.py

Verified from this folder alone (no `MODEL_DIR`): `q3_combine.py` regenerates the
section-7 table, `q3_kavg.py` reproduces `q3_kavg.json` exactly, and
`kscan.py check` reproduces its instrument-check values (π/2 ratio 0.0883). Common model settings throughout: c = 1, κ = 0.5, DT = 0.02,
packet amplitude 10⁻³, colour-0 packet, RK4 integrator from `model.py`.

## Two readouts

- **Old readout** (`j_compat_test.exit_time`): read out when the packet centre is
  3 initial widths past the segment end. **Known to be premature** — up to 3% of
  the packet is still inside the segment at q = 1 (98% at q = 3 under the
  `run_until_exit` centroid certificate). Results in sections 1–3 use it.
- **Clearing readout** (`resid.clear_time`, `gate7_readout.py`, `q3_readout.py`):
  read out only once every segment window `[start − 10, start + ramp + 10)` holds
  less than 10⁻⁶ of the packet weight. Results in sections 4–8 use it.

## Shared definitions

- **Wc, Wx** (`j_compat_test.split_W`): one random symmetric 4×4 W (NumPy seed 1
  unless stated), split into Wc = (W − JWJ)/2 and Wx = (W + JWJ)/2 with
  J = `rho(i·I)`, each rescaled to Frobenius norm 2.
- **Effect of a segment**: Bures angle (degrees) between the lattice-summed,
  block-diagonal chirality density matrix after the run and a no-segment reference.
- **Ratio**: effect(Wx) / effect(Wc).
- **Residual** (sections 4–5): intercept 2·ratio(g) − ratio(2g), g = 0.0025.

## Script → result map

### 1. J-compatibility vs stiffness, k₀ = π/2 (old readout)
`grid.py` → `grid.json`, `grid.log`

n = 2, q = 1, N = 200, packet n0 = 20, width 8, one ramped segment starting at
site 60 (`RAMP`, length 12), seed 1.
K ∈ {√5/4, √5, 4√5, 16√5}, g ∈ {0.0025, 0.005, 0.01, 0.02, 0.04}.
Reported: ratio table; model row (K = √5) levels at ~0.076; stiff rows ∝ g.

### 2. Robustness and threshold checks (old readout)
`checks.py` → **printed only, no results file**

Same settings as section 1. Seeds 2 and 3 at K ∈ {√5, 4√5}, g ∈ {0.0025, 0.01};
threshold runs K ∈ {2.8, 3.2}, seed 1. Rerun the script to regenerate.

### 3. J-compatibility vs wavenumber (old readout)
`kscan.py scan` → `kscan.jsonl`; `kscan.py check` → printed only

K = √5, k₀ ∈ {π/4, 3π/4}, g ∈ {0.0025 … 0.04}, seeds 1 (all g) and 2, 3
(g = 0.0025, 0.01). Geometry as section 1. Check mode reproduces the π/2,
g = 0.01 ratio (0.0883) and the Wc rotation at g = 0.03, 0.06.
Channel threshold k_c ≈ 1.4536 rad (0.4627π) was computed inline from
cos k′ = cos k₀ + κω/c.

### 4. π/4 residual: packet width and ramp length (clearing readout)
`resid.py scan` → `resid_scan.jsonl`; `resid.py checknew` → printed only

k₀ = π/4, K = √5, g ∈ {0.0025, 0.005}, seed 1. N = 1200, packet n0 = 160,
segment start 300. Ramp of length L: rising quarter, flat half, falling quarter
(L = 12 reproduces `RAMP`). Width scan W ∈ {8, 16, 32} at L = 12; ramp scan
L ∈ {12, 24, 48} at W = 8. Clearing threshold 10⁻⁶ with a 10% time margin.
Reported: residual ≈ 1×10⁻⁵ in every configuration.
`checknew` compares old vs clearing readout at W = 8, L = 12 (residual 0.0105 → 0).
Note: an earlier 720-site check used a previous version of this file; the
current file is the 1200-site version that produced the reported numbers.

### 5. Open-channel rows re-measured (clearing readout)
`openrows.py pi2` → `open_pi2.jsonl`; `openrows.py 3pi4` → `open_3pi4.jsonl`

Uses `resid.py`'s geometry and clearing readout: K = √5, W = 8, L = 12,
g ∈ {0.0025 … 0.04} seed 1, seeds 2, 3 at g ∈ {0.0025, 0.01}.
Reported: ratios unchanged from the old readout (π/2 ≈ 0.076, 3π/4 ≈ 0.070 at
g = 0.0025).

### 6. Gate 7 at q = 1 (model.py's ordering test)
`gate7_readout.py` → `gate7.jsonl`

q = 1, packet n0 = 20, width 8, segments at 60 and 80, u(2) g = 0.12/0.08,
u(3) g = 0.15/0.10. Readouts: model.py as shipped (N = 200, T = 180);
N = 1200 at T = 180; N = 1200 clearing.
Reported: 16–18% of weight in the second segment window at T = 180; split and
sim-vs-product unchanged (≤ 0.1°); Abelian floor 0.016° → 0.000°.

### 7. Gates 7 and 8 at q = 3 (three readouts)
`q3_readout.py <n> <job>` → `q3/<n>_<job>.json` (progress in `q3/*.log`);
jobs listed in `q3/jobs.txt`. **Headline table:** `python3 q3_combine.py 2 3`
reads the `q3/*.json` files (no simulation) and writes `q3_old_vs_new.json`;
its printed table is saved as `q3_old_vs_new.txt`.

Lattice 320 × 12 × 12 (long axis = propagation), full transverse slab, isotropic
3-D Gaussian packet width 3 centred at x = 30, k₀ = π/2. Segments at 50 and 70.
u(2): g = 0.12/0.08; u(3): g = 0.15/0.15; Abelian floors use unequal strengths
(u(2) 0.12/0.08, u(3) 0.15/0.10). Instrument check: `2 single`, one segment,
axis 0, g = 0.12. Readouts per run: fixed T = 180; centroid 2 sites past the last
segment (MODEL_SPEC §4d / `run_until_exit`); clearing (< 10⁻⁶).
Run with `OMP_NUM_THREADS=1`, four jobs in parallel (~20–60 min each).
Reported: Abelian floor ~1° → 0.000° (u(2), u(3)); split error 5.09° (u(2)) and
4.43° (u(3)) under clearing; single segment 2.70°.

### 8. Spectrum-averaged prediction at q = 3 (no new dynamics)
`q3_kavg.py` → `q3_kavg.json` (needs `q3/*.json` from section 7)

Regenerates the section-7 packet, Fourier-transforms it on the 320 × 12 × 12
lattice (17,112 components kept, all but 5×10⁻¹¹ of the weight), and propagates
each component through the segments with its own ω(k) and transverse term.
Reported: errors 2.70° → 0.11° (single), 5.09° → 0.04° (u(2) split),
4.43° → 0.17° (u(3) split).

### 9. q = 3 ordering gate (permanent check)
`q3_gate.py` → `q3_gate_runs_<L0>x<S>.json` (runs) and
`q3_gate_result_<L0>x<S>_<predictor>.json` (verdict)

Runs on the **working** `model.py`, `04_scripts/session/model.py` — the single
working copy, loaded by a path relative to `q3_gate.py` (`../04_scripts/session/`),
so the script runs from any directory. (It formerly ran on a duplicate kept in
this folder — archive 948b09e8 plus a note in `run_until_exit` that at q = 3 its
certificate fires with ~98% of the packet still in the windows; that note now
lives in the working copy, and the duplicate is removed. The pinned copies are
untouched.) Re-scoring the saved runs on the working copy reproduces both result
files exactly, for both lattices and both predictors. Eight evolutions
(u(2), u(3): AB, BA, two Abelian-floor orders), same packet, segments and
strengths as section 7, clearing readout, spectrum-averaged prediction.

**PASS** iff every per-order error < 1°, both split errors < 1°, and both
Abelian floors < 0.5°. The gate gives no verdict (exit 2) if a window fails to
clear to 10⁻⁶ or if the lattice is too short to rule out a wave re-entering a
window before readout (both fronts: centre + 4 packet widths at the band's
maximum group velocity, 0.497).

    python3 q3_gate.py                                  # 260 x 8 x 8, averaged
    python3 q3_gate.py --from-saved q3_gate_runs_260x8.json --predictor carrier

| lattice | clears | averaged prediction | single-wavenumber prediction | runtime (4 workers) |
|---|---|---|---|---|
| 320 × 12 × 12 (section 7 runs) | yes, t = 335–346 | **PASS** — 0.04–0.17° | **FAIL** — 3.0–5.2° | not timed precisely (roughly 2–3 h wall for the section-7 jobs) |
| **260 × 8 × 8 (default)** | yes, t = 335–346 | **PASS** — 0.10–0.18° | **FAIL** — 2.4–5.1° | **11.6 min wall, 46 min CPU** |
| 240 × 8 × 8 | **no** — backward stray re-enters window 2 at t ≈ 350 (bottomed at 1.7×10⁻⁶) | — | — | — |

Smallest lattice: 260 is the shortest length the no-wrap check accepts (240 is
rejected at −6 sites and failed to clear in a direct run). The slab is kept at
8 × 8: its transverse profile still varies 35× across the slab (12 × 12: ~3000×)
with transverse term Qt = 0.074 (12 × 12: 0.102); at 6 × 6 the contrast falls to
7×, approaching the transverse-uniform trap in which q = 3 reduces to q = 1.

### 10. κ(w = 2, side) at q = 3 (self-contained; does not import `model.py`)
`kappa_readout_test.py` → `kappa_readout_test_output.txt`;
`kappa_readout_test.py --swapped-seed` → `kappa_readout_test_swapped.txt`

β = 0.05, A = 0.30 (linear check A = 0.02), transverse width 2, periodic BC,
T = 300, DOP853 rtol 10⁻⁹, sides 8–32, three readouts, plane-wave control.
Default seed: Fourier space, each wavevector at its own branch frequency.
`--swapped-seed` reproduces the original run (+k at the lower root, one carrier;
RETRACTED values). Each run ~32 min wall on 4 cores. Reported: plane-wave
κ = −0.01869 (swapped: +0.07994); localised κ −0.00447 → −0.00047 over side
8–32, no side-independent limit (PROVENANCE §6o).

### 11. κ(w = 2) against box size, GPU (Colab, Tesla T4, PyTorch float64)
`kappa_side_gpu.py` → `kappa_side_gpu_colab_output.txt` (static vs dynamic origin);
`kappa_boxscan_gpu.py` → `kappa_boxscan_gpu_colab_output.txt` (L = 8–48);
`kappa_extended_gpu.py` → `kappa_extended_gpu_colab_output.txt` (L = 8–80, error bars).

Self-contained; PyTorch on a GPU if present, otherwise NumPy (slow). Physics,
seed and readout as section 10; RK4 dt = 0.01. Raw output in `*_colab_raw.txt`,
annotated reading in `*_colab_output.txt`. Re-checked on
CPU with `kappa_extended_gpu.py` at L = 20–32: identical. Result: κ(w = 2, side)
CLOSED (PROVENANCE §6o).

## Helpers

- `j_compat_test.py` — shared machinery: `KLattice` (stiffness K as a parameter),
  segment builder, Wc/Wx split, internal-state readout, Bures angle, old readout.
- `fmt.py`, `fmt2.py` — format `.jsonl` output as tables (`python3 fmt.py < file`).
