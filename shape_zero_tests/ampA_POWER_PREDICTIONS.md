# Anomaly A — stationary wave and the well-power test: predictions

*Committed 2026-10-01, before any run.*

Script: `ampA_power.py`. `predict` has been run; its output is `ampA_power_predictions.txt` and `.json`.
Design choices are ours.

## Disclosures
- Everything measured so far is known:
  - A′ single-segment factor 0.666 / 0.666 / 0.667 at widths 8 / 16 / 32 (ae17fc2);
  - smooth-well certification residue ∝ A² (99c8a0e).
- I saw the model outputs before committing.
- Tolerances were set now.
- The candidate in (2) is yours.

## (1) Stationary nonlinear wave through one segment

**Setup.** A flat-top wave: plateau 300 sites, tanh edges of width 15, per-mode launch at k₀ = π/2.
It passes one segment (u(2), axis 0, g = 0.12). The internal state is read locally, in a 100-site
window of the steady transmitted plateau (the back edge is still upstream of the segment). The two
half-windows check that the plateau is steady.

**Exact fact used.** A circular wave of constant |u| makes the A′ lattice exactly the linear lattice
with stiffness √5 + V and frequency Ω = ω + V/(2ω+κ), for a channel mixture of any spatial phase. So
the stationary transfer matrix (K3b) is the stationary answer.

**Predicted rotation change:**

| well | amplitude | V | predicted |
|---|---|---|---|
| A′ | 10⁻³ | 10⁻³ | 0.028406° |
| smooth | 10⁻² | 10⁻⁴ | 0.002841° |

**Factor (measured / transfer matrix) = 1.00 ± 0.05 for both wells.** The two half-windows agree
within 5%.

**How to read the outcome:**
- **Factor 1:** the 2/3 belongs to how a launched packet converts its self-phase into carried
  frequency. The stationary check is complete.
- **Factor 2/3 for A′ (or 1/2 for smooth):** the stationary transfer-matrix check is missing a term.
  That would contradict the exact reduction above, which would then itself be the thing to examine.

## (2) The candidate: energy per action versus frequency shift

**Derivation.**
- Take a radial potential ∝ r^p on a circular orbit, with action I = (2Ω+κ)r²/2.
- The frequency shift is ∂⟨δH⟩/∂I = r^{p−2}/D.
- The energy per action is ⟨δH⟩/I = (2/p)·r^{p−2}/D.
- If the eikonal conversion uses the frequency shift where the dynamics follows the energy per action,
  the factor is **2/p**: 2/3 for A′ (p = 3), **1/2 for the smooth well** (p = 4).

**Test.** The smooth well in the clean single-segment geometry (widths 8, 16, 32; ampA_regions width
scan), at A = 10⁻² and 5×10⁻³. The linear runs are reused. Eikonal model: δω = |ψ|²/D with weight
⟨F⁴⟩/⟨F²⟩ at each site crossing.

**Predicted rotation changes** (degrees; eikonal × factor):

| width | A | eikonal (factor 1) | candidate (× 1/2) | universal 2/3 |
|---|---|---|---|---|
| 8 | 10⁻² | 0.001760 | **0.000880** | 0.001174 |
| 16 | 10⁻² | 0.001988 | **0.000994** | 0.001326 |
| 32 | 10⁻² | 0.002007 | **0.001003** | 0.001338 |

At 5×10⁻³ every value is ×¼.

**Criteria:**
- **Candidate holds:** factor 0.50 ± 0.05 at every width.
- **Universal 2/3:** 0.667 ± 0.05.
- **Eikonal:** 1.00 ± 0.05.
- **A² order:** the 10⁻²/5×10⁻³ ratio of the rotation changes is 4.00 ± 0.08, and the direction
  cosine to the eikonal vector is ≥ 0.99.
- **The well-power dependence holds as 2/p** if A′ gives 0.667 (already measured) and smooth gives
  0.50.

**Stationary prediction for the candidate.** For a stationary circular wave the frequency is fixed
exactly, so the candidate predicts factor 1 in (1) for both wells. It can only act through the
packet's transient.

## Correction pending for the record (unchanged)

When anomaly A is next recorded, the fccf291 reading "K4 flips the q = 3 u(3) split sign" is marked
**corrected**: the positive split comes mainly from the in-segment (narrow-width) term.
