# P-1 investigation — which file produced which finding

Scripts and outputs behind the P-1 annotation in
`01_source/shape_zero_predictions_v1.md` and `00_START_HERE/PROVENANCE.md` §6o
(2026-09-24). Run each from this directory: `python3 <file>`. numpy + scipy.
Scripts that need the platform or session modules add `../../04_scripts/...` to
the path themselves.

**Shared code**

| file | what it is |
|---|---|
| `hill.py` | exact travelling wave by harmonic balance (`wave`) and Hill/Floquet linear stability (`growth`, max Im Ω over Bloch channel j, q = 2πj/64; j = 0 is the (0, π) pair). Its `__main__` prints the calibration (self-shift −0.0974 A², asymmetry +0.0350 βA², matching `phi_gauge_delta.py`'s PT) and an all-channel scan. |
| `sim.py` | batched RK4 of the platform lattice (same force and DT as `phi_gauge_delta.py`), with the script's plain-cosine start or the exact harmonic-balance wave; records retention \|u[M]\|, \|u[0]\| and \|u[N/2]\| over time. |

**Findings**

| finding | file(s) |
|---|---|
| The map reproduces as documented: prediction 0.0619 → 0.0470; measured retention 0.612 at (A = 0.4, β = 0.07); all A ≤ 0.25 cells 1.000 ± 0.005 | `t0.txt` — output of `04_scripts/platform/phi_gauge_decaymap.py`, run with its `np.save` path (`/home/claude/...`) redirected to a writable directory; the script as committed stops at that line |
| 1. The prediction script's resonance condition is incomplete: `W_SUM` is the fixed linear w(0) + w(π) and the curve is exactly −0.0991 A² (pump self-shift / dω/dβ) | `t1.py` |
| 1. The model's own (0, π) band moves up: [0.0630, 0.0643] at A = 0.10 to [0.0653, 0.0868] at A = 0.40, midpoint ≈ 0.063 + 0.08 A² | `t2c.py` (fine β scan, H = 6 and 8); `t2f_convergence.py` (H = 8/10/12, ±0.001 at A = 0.4). `t2b.py` was the first attempt on a coarse grid; it stops at A = 0.10, where the band is narrower than the grid step, and is superseded by `t2c.py` |
| §5 sign check: the PT asymmetry term (κ ≈ −0.0175) is opposite in sign to the retracted +0.0799 but moves β_res(0.4) by only 0.0008 (0.0470 → 0.0462) | `t3a_asym_sign.py` |
| κ with each direction seeded at its own root in the reference lattice: −0.0179 (A = 0.2), −0.0187 (A = 0.3), β = 0.05 | `t3b.py` → `t3b.txt` |
| 2. The measured dips come from the plain-cosine start: retention is identical to four digits with noise off, reseeded, or 1000× larger, and 0.9998 from the exact wave; the start seeds q = 0 and q = π at ≈ 0.027 and 0.017 | `t4a.py` |
| 2. Fixed-time readout: on a 0.0025 β grid the dip moves from β = 0.075 (T = 200–300) to 0.0725 (T = 400) to 0.0675 (T ≥ 500); min over time 0.26 | `t4b.py` → `t4b.txt` |
| 3. A clean start shows broad instability, not a window: other pair channels grow as fast as or faster than (0, π) at every β from 0.05 to 0.10 (A = 0.3, 0.4); clean-start runs to T = 2500 lose pump energy across the range | `t2d_channels.py`; `t4d.py` → `t4d.txt` |
| 4. Reverse-direction protection holds: the −k pump is stable on every channel at A = 0.3, 0.4 and β = 0.05, 0.07, 0.08 | `t2e_reverse.py` |

`t2d_channels.py`, `t2e_reverse.py`, `t2f_convergence.py` and `t3a_asym_sign.py`
are the commands that were run inline during the investigation, saved as files
verbatim apart from the path setup and a header comment.

`t1`, `t3a` and `t2e` finish in seconds; `t4d` (three amplitudes × 17 β, to
T = 2500) is the longest.
