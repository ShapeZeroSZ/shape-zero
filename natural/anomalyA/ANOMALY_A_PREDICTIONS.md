# Anomalies A and E — derivation and predictions (committed before any gate run)

**What the labels refer to.**
- **A** is the open misses recorded with C43, the adoption of A′ (PREMISE_LEDGER C43; MODEL_SPEC
  §1a; `certify_gates_output_nodewell.txt`):
  - split slopes only 1.5–6× smaller than under form A;
  - u(3)'s split slope changing sign;
  - q = 3 u(3) per-order 0.024;
  - q3_gate deviations of 0.011–0.020°.
- **E** is `tower_populated`'s isolated reference losing chirality purity, 0.9872 → 0.9494
  (0.038), under the single-carrier readout (PROVENANCE, P4 miss).

**The A-predictions are not blind.** The A′ certification slopes were measured and recorded before
this derivation. The derivation is fixed in advance by the stated law, with **no parameter**; the
recorded values played no part in forming it. **The smooth-force control (A2) and E are blind.**

## A1 — the derived slopes under A′ (`anomA_predict.py` → `anomA_predict_output.txt`, `anomA_predicted.json`)

**Mechanism.**
- Under A′ the stiffness is √5 + |ψ|, common to all components. A packet therefore carries a
  frequency shift δω = A_eff/(2ω + κ), with A_eff = ΣF³/ΣF² its density-weighted |ψ|.
- The frequency is conserved into the links. At fixed frequency, each link site's Peierls angle per
  eigen-direction j is θ_j = arctan(g·w_s·ω·h_j) (`main`'s link-sector derivation, 93debc6), so
  **δθ_j = g·w_s·h_j·δω / (1 + (g·w_s·ω·h_j)²)** (w_s is the RAMP weight).
- δω is taken at each segment's centre-crossing time, from the exact linear free envelope:

| | A_eff/A at the crossings | δω |
|---|---|---|
| q = 1 | 0.8113, 0.8059 | 1.918×10⁻⁴, 1.906×10⁻⁴ |
| q = 3 | 0.4721, 0.4437 | — |

- **The V2 refinement** — the model's own k_branch at ω + δω, with Q fixed — agrees with V1 to five
  digits.

**Predicted slopes (deg per 10⁻³ of amplitude):**

| | split slope | per-order AB | per-order BA | floor |
|---|---|---|---|---|
| q = 1, u(2) (0.12 / 0.08) | **−0.01435** | 0.0273 | 0.0222 | +4.9×10⁻⁵ |
| q = 1, u(3) (0.15 / 0.10) | **+0.02792** | 0.0253 | 0.0291 | +5.2×10⁻⁵ |
| q = 3, u(2) (0.12 / 0.08) | **−0.00640** | 0.0154 | 0.0126 | +2.6×10⁻⁴ |
| q = 3, u(3) (0.15 / 0.15; floor 0.15 / 0.10) | **−0.01457** | 0.0148 | 0.0148 | +2.7×10⁻⁴ |

**u(3)'s sign:** positive at q = 1, **negative at q = 3.**

**The floors.** They are nonzero only because the two segments sit at different dispersion states
(δω differs by ~1%, and more at q = 3), so a commuting pair with unequal strengths picks up slightly
different total phases in the two orders.

**q3_gate deviations at 10⁻³** (from the averaged linear prediction): each per-order deviation lies
in [slope − |intercept|, slope + |intercept|], using the recorded small-amplitude intercepts (0.0008
for u(2), 0.0037 for u(3)):
- u(2): AB 0.0146–0.0162, BA 0.0118–0.0134;
- u(3): 0.0111–0.0185.

**Criteria, per quantity:**
- **explained** if the measured slope has the predicted sign and lies within 25% of the predicted
  magnitude;
- **partly explained** if it has the right sign but is off by more than 25%;
- **not explained** if the sign is opposite.
- Floors are judged within ±50% or 5×10⁻⁵°, whichever is larger.

**Rerun of the A′ certification in a clean worktree of `main` (ed16311).** Predicted to reproduce
the recorded values: CERTIFIED at q = 1 and q = 3, with every slope and intercept within 10⁻⁴° of
`certify_gates_output_nodewell.txt`.

## A2 — the smooth-force control (diagnostic only; not a change of premise)

- **The change:** replace A′'s on-site nonlinearity |ψ|ψ by the analytic |ψ|²ψ, with energy |ψ|⁴/4.
  Everything else is `main`'s, and it is applied only inside the diagnostic driver.
- **Then** δω = ⟨|ψ|²⟩/(2ω + κ), with ⟨|ψ|²⟩ = ΣF⁴/ΣF² ∝ A². **So every deviation from the A → 0
  value scales as A², not A.**
- **Amplitudes 0.04, 0.02, 0.01, 0.005.** At 10⁻³ the effect would be ~10³× below the numerical
  floor.

**Predicted split-error changes from the A → 0 value** (q = 1, gate 7 protocol):

| A | u(2) | u(3) |
|---|---|---|
| 0.04 | −0.0197 | +0.0383 |
| 0.02 | −0.0049 | +0.0096 |
| 0.01 | −0.0012 | +0.0024 |
| 0.005 | −0.0003 | +0.0006 |

Per-order changes: AB 0.0375 / 0.0348 at A = 0.04.

**Criteria:**
- the fitted exponent p in dev(A) = a + c·A^p (4 amplitudes) is **2.0 ± 0.25**;
- the ratio [dev(0.04) − dev(0.01)] / [dev(0.02) − dev(0.01)] is **5 ± 1** (A² gives 5; A gives 3);
- the sign is as predicted, and the magnitude at 0.04 is within a factor of 2 of the prediction.

**At q = 3** (q3_gate, amplitudes 0.04, 0.02, 0.01): the same ratio test, **5 ± 1** for every split
and per-order deviation.

## E — the isolated packet's purity (blind)

- **Set-up:** `tower_populated`'s isolated reference, recomputed with `main`'s own code: D8,
  A′, ring N = 128, κ\*, amplitude 0.05, width 8, k₀ = π/2, per-mode launch, T = 2000.
- **Single-carrier readout (χ = ψ + iψ̇/ω):** reproduces 0.9872 → 0.9494 within 0.001.
- **Per-mode readout (`readout_modes`):**
  - purity at t = 0 **≥ 0.9999** — the per-mode launch has no b-branch content;
  - at T **≥ 0.999**;
  - change **< 10⁻³**.
- **So the 0.038 loss was the single-carrier readout, not physics.** The single carrier assigns
  modes with ω(k) ≠ ω(k₀) partly to the wrong branch, and dispersion changes their phases in time.
- **Would count against:** a per-mode purity loss ≥ 0.01, which would make E physical — for
  example, nonlinear conversion into the b-branch.

## Bearing on P9 (stated in advance)

- **If A1 is explained,** the residual slopes under A′ are the ordinary O(A·g) consequence of P9's
  radial φ-well (a common frequency shift) feeding the link sector's frequency-dependent Peierls
  angle — P9 working as intended. **Nothing new is required of P9.**
- **If A2 shows A² scaling,** the linear-in-A slopes are confirmed to come from the non-analytic
  |ψ|, i.e. from P9's radial form.
- **Parts not explained** stay open, and would point to effects this mechanism omits:
  reflections, the wavenumber spread, and the O(A²) terms.
