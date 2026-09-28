# Circuit build — pre-registration

*Committed 2026-09-28 on the branch `realisation-gyroscopic`, **before anything is built, simulated in SPICE,
or measured**. Every design choice is ours. The predictions follow from the model's equations (the lattice of
`LAB_NOTE.md` §1 and §6) with the component values below; the numbers were computed with
`shape_zero_tests/circuit_error_budget.py` at β̂ = 0.106.*

**β̂ = 0.106 was chosen post hoc**, after the error budget showed that the earlier design point β̂ = 0.10 puts the
k = π/2 probe mode 8.4×10⁻⁴ √K from the ω(−13π/16) mode — an accidental near-degeneracy the size of the P2 signal
(`LAB_NOTE.md` §6.4). At 0.106 the nearest other mode is 8.8×10⁻³ √K away at N = 32 and 64. The choice of design
point is post hoc; **the predictions below are committed before any measurement.**

## 1. Component values (32 nodes on a ring)

| element | value | tolerance assumed | role |
|---|---|---|---|
| node capacitor C | 10.00 nF, C0G | 0.1% (sorted) | mass |
| node inductor L_g,phys | 1.176 mH, shielded ferrite, Q ≈ 100 at 60 kHz | trimmed with the GIC so that each node's total K_n = 10¹¹ s⁻² to 0.1% | on-site stiffness |
| node GIC (baseline) | Antoniou grounded synthetic inductor, 6.667 mH (C4 = 10 nF, R1 = R2 = R3 = 10.0 kΩ, R5 = 66.7 kΩ programmable) | set by measurement | trim and bumps |
| total on-site inductance | 1.176 mH ∥ 6.667 mH = 1.000 mH | 0.1% | K = 1/(L_g C) = 1.000×10¹¹ s⁻² |
| bond inductor L_c | 4.000 mH, shielded ferrite, Q ≈ 100 | 0.1% (sorted/trimmed) | spring; ĉ = L_g/L_c = 0.250 |
| gyrator, per bond | two VCCS, each an OPA4197 follower plus a basic Howland pump; R_set = 2.983 kΩ (G = 3.352×10⁻⁴ S); Howland R2 = 2.983 kΩ, R3 = R4 = 10.0 kΩ | R_set 0.1%; Howland ratio R4/R3 = R2/R1 to 0.1% | b = G/C = 3.352×10⁴ s⁻¹; **β̂ = G√(L_g/C) = 0.106** |
| op-amps | OPA4197 (quad, 10 MHz GBW) for the gyrators, OPA2197 for the GICs; supplies ±12 V | — | — |
| parasitics assumed | ≤ 20 pF across each L_c; ≤ 5 pF per op-amp input | — | — |

**Stability window assumed:** inductor Q between 80 and 250 (the budget's instability threshold with a ~5 MHz
VCCS pole is Q ≈ 300–500; below ~80 the lines broaden past the P2 resolution).

## 2. Calibrations and preconditions (a failed precondition voids the run; it is not a model failure)

1. Gyrators unpowered: every ±k doublet degenerate to < 0.1% of Δω(π/2).
2. Gyrators unpowered: the band ω(k)² = K + 2c(1 − cos k) fits every m = 0 … 16 to 0.2% (K and c from the fit;
   ĉ must be 0.250 ± 0.002).
3. Gyrators powered: b = Δω(π/2)/2; β̂ = b/√K must be 0.106 ± 0.001.
4. Gyrators powered: the product rule ω(k)·ω(−k) = Q(k) (band from step 2) within 10⁻³ at every m — the check that
   the powered gyrators are velocity-linear.
5. Stability: no self-oscillation; every measured line has a finite width.

**Estimator (fixed now):** drive a current into node 0; measure every node's complex voltage; for each m form
P_±(f) = Σ_n V_n e^{∓2πimn/32}; fit a rational function of degree 3/3 to P_± over ±3 linewidths around the line
(Sanathanan–Koerner iteration) and take the pole nearest the design frequency; ω is its real part.

## 3. Predictions

**P1 — the sin k shape.** Δω(m)/Δω(8) = sin(2πm/32). Design values (ideal components):

| m | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 … 15 |
|---|---|---|---|---|---|---|---|---|---|
| f₊ (kHz) | 51.622 | 53.360 | 55.454 | 57.793 | 60.253 | 62.714 | 65.064 | 67.206 | (below) |
| f₋ (kHz) | 49.540 | 49.277 | 49.527 | 50.248 | 51.382 | 52.857 | 54.599 | 56.536 | |
| Δf (Hz) | 2081.6 | 4083.2 | 5927.8 | 7544.7 | 8871.6 | 9857.6 | 10464.8 | 10669.8 | mirror of m′ = 16 − m |
| sin k | 0.19509 | 0.38268 | 0.55557 | 0.70711 | 0.83147 | 0.92388 | 0.98079 | 1 | |

(m = 9 … 15: f₊ = 69.060, 70.568, 71.688, 72.397, 72.688, 72.567, 72.054 kHz; f₋ = 58.596, 60.710, 62.816, 64.852,
66.760, 68.484, 69.972 kHz; Δf and sin k as m′ = 16 − m.)
*Tolerance:* the budget's realistic circuit (0.1% parts, parasitics, VCCS pole) gives residuals ≤ 1.1×10⁻³.
*Failure:* any measured m with |Δω(m)/Δω(8) − sin k_m| > 5×10⁻³ + 3σ_m.

**P2 — the coefficient ¼, with the finite-ring correction.** Bump: δK_n = K S η_n, η a mean-zero, unit-rms
Gaussian of width σ = 4 sites centred on node 16, S = 0.05, applied through the GICs; reversed for −S. Even part
δ(Δω) = [Δω(+S) + Δω(−S)]/2 − Δω(0) at m = 8. The ¼ law gives −¼ b⟨δK²⟩/c² = −53.35 Hz (in f); **the
32-node ring's exact second-order value is 1.035× that: δ(Δf) = −55.2 Hz.** Reported as
R = δ(Δω)/(−¼ b_cal⟨δK²⟩_meas/c_cal²), with ⟨δK²⟩ from single-node spectroscopy of the applied bump.
*Predicted:* R = 1.035 (ideal components); 1.026–1.036 with the budget's realistic non-idealities (parasitics
−0.4%; scatter 0.2% at 0.1% parts). *Failure:* |R − 1.031| > 0.015 + 3σ_R. (For reference: a coefficient ⅛ would
give R ≈ 0.52 and ½ would give R ≈ 2.07.)

**P3 — shape independence.** At S = 0.05, R for four localised shapes and one control, finite-ring values:
Gaussian σ = 3: 1.058; Gaussian σ = 5: 1.026; sech², width 4: 1.058; two Gaussians (σ = 2.5, 16 sites apart):
1.077; linear ramp (discontinuous on the ring; a different class): 0.894. The ¼ law's shape independence holds in
the wide-bump, large-ring limit; on 32 nodes the localised shapes spread by 0.051 in R.
*Failure:* any localised shape with |R − R_pred| > 0.02 + 3σ, or the ramp with |R − 0.894| > 0.03 + 3σ.

## 4. What this does and does not test

These are consequences of the model's equations applied to a circuit built to realise them; a pass says the
circuit realises the lattice to the stated tolerance and that the three relations hold in hardware. The model is
not a theory of nature (`OVERVIEW.md`, "Scope"; `LAB_NOTE.md` §5).
