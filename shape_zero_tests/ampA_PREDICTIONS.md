# Anomaly A: predictions (committed 2026-09-29, before any run)

Scripts: `ampA_link_predict.py` (derivation: see its docstring) → `ampA_link_predictions.txt/.json`;
`anomalyE_purity.py` (anomaly E, expectation in its docstring). Model change for the diagnostic only:
`SZ_J_WELL=smooth` (force −(√5 + |ψ|²)ψ). The default (A′) is unchanged and this is not a change of premise.

**Disclosure.** The recorded A′ certification slopes (MODEL_SPEC §1a, §9) were known before this
derivation. No parameter was fitted. The construction choices (ours) were fixed before computing:
A_eff weighting, per-ramp-site crossing times, the single-carrier product at q = 1 and the
spectrum-averaged predictor at q = 3. The comparison tolerance below was set **after** seeing the
predicted numbers.

## (1) A′: link-sector angle law, δθ_j = g h_j δω/(1+(gωh_j)²), δω = ⟨|ψ|⟩/(2ω+κ)

The slopes are in degrees per 10⁻³ of amplitude, from the certification's own fits applied to the
predicted Bloch vectors.

| quantity | q = 1 u(2) | q = 1 u(3) | q = 3 u(2) | q = 3 u(3) |
|---|---|---|---|---|
| split-error slope | **−0.0144** | **+0.0279** | **−0.0058** | **−0.0120** |
| per-order AB slope | 0.0273 | 0.0253 | 0.0158 | 0.0136 |
| per-order BA slope | 0.0223 | 0.0291 | 0.0125 | 0.0136 |
| floor slope | 0.00005 | 0.00005 | 0.0005 | 0.0005 |

- u(3)'s predicted sign is **+ at q = 1** and **− at q = 3**.
- The q3_gate per-order deviations predicted at A = 10⁻³ equal the per-order slopes above (0.0125–0.0158°).
- Secondary variant (the full branch equation at ω + δω with Q fixed): identical to the primary to all printed digits. At k₀ = π/2 the hopping-renormalisation term vanishes (1 − Q/2c = 0), so the angle law is the whole first-order link effect there. This is a consequence, not a separate prediction.

**Comparison criterion (ours, set after seeing the predictions).** A slope is **explained** when its sign
matches and |pred − meas| ≤ 0.3|meas| + 0.002. The A′ rerun is expected to reproduce the recorded
runs (the dynamics are deterministic).

## (2) Smooth-force diagnostic, −(√5 + |ψ|²)ψ

Every deviation from the linear dynamics is second order in A, so each residual scales as A², not A.

- **Angle law with δω = ⟨|ψ|²⟩/(2ω+κ).**
  - Predicted deviations at A = 10⁻³ are ≤ 2.5×10⁻⁵°: about 1/1000 of A′.
  - This is at or below the numerical floor (≈ 4×10⁻⁵°), so A² scaling can be resolved only if the measured deviations rise above that floor.
- **Committed test.**
  - Every slope (floor, split error, per-order), at q = 1 and q = 3, satisfies |b| < 0.001 per 10⁻³, which is ≥ 10× below A′.
  - Where the deviations exceed 10⁻⁴°, dev(A)/A² is constant within 20%.
- **Intercepts.**
  - q = 1: the single-carrier product's own error, as with the elementwise well. Split +0.26° (u(2)) and −0.02° (u(3)); per-order 0.12/0.27° and 0.40/0.26°.
  - q = 3: zero.
- Any mechanism beyond the angle law that is first order in the force is also suppressed by the factor A (|ψ|²ψ versus |ψ|ψ). So a slope collapse tests P9's non-analyticity as the source of every linear-in-A effect, not only the angle law.

## (4) Anomaly E

Expectation (in `anomalyE_purity.py`): if the 0.038 purity loss is the single-carrier readout, the
per-mode purity loses < 0.005 over T = 2000.
