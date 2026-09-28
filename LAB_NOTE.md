# Three parameter-free predictions for a gyroscopically coupled pendulum ring

*Technical note, 2026-09-28, branch `realisation-gyroscopic` only — not merged into main. Sources:
`UNIVERSAL_RELATIONS.md` on main ("The count — corrected accounting"); `REALISATION.md` (this branch),
§4 (bond rotors) and §5–§6. Simulations: `shape_zero_tests/platform_error_budget.py` (expectations committed
first, 1c63b6d) and the post-hoc diagnostics `platform_error_budget_checks.py`. Every design choice below is
ours, not taken from a source.*

## 1. Platform

**System.** A ring of N planar pendula (one angle θ_n each) with the motors off: no on-site spin, so the
model's gyroscopic ratio κ = 0 and each node is scalar. Neighbouring pendula are joined by coil springs
(nearest-neighbour stiffness). Each bond carries one **bond rotor** (`REALISATION.md` §4): a spinning disc on
a two-axis gimbal whose tilt axes are slaved 1:1 to the axles of pendula n and n+1 by timing belts. The
gyroscopic torque of a rotor with spin angular momentum H couples the two rates:
F_n ⊃ −H θ̇_{n+1}, F_{n+1} ⊃ +H θ̇_n, which is passive (it does no work). With all rotors spinning in the
same sense, per unit effective inertia I_eff,

  θ̈_n = −K_n sin θ_n − γ θ̇_n + c_n(θ_{n+1} − θ_n) + c_{n−1}(θ_{n−1} − θ_n) − b_n θ̇_{n+1} + b_{n−1} θ̇_{n−1},

K = τ_g/I_eff, c = k_t/I_eff, b = βc = H/I_eff. For uniform parameters a plane wave θ ∝ e^{i(kn − ωt)}
obeys ω² − 2b sin k·ω − Q(k) = 0 with Q(k) = K + 2c(1 − cos k), so **Δω(k) ≡ ω(k) − ω(−k) = 2b sin k**
exactly. The model's dimensionless parameters are ĉ = c/K, β̂ = b/√K and κ̂ = 0.

**Design (all choices ours).**

| item | value |
|---|---|
| ring | N = 32 pendula, radius 1.02 m, spacing 0.20 m |
| pendulum | steel rod 0.30 m (0.05 kg), brass bob 0.50 kg at ℓ = 0.30 m; I_eff = 0.0471 kg m² (incl. gimbals and pulleys) |
| gravity stiffness | τ_g = 1.545 N m/rad → K = 32.8 s⁻², ω₀ = 5.73 rad/s (0.91 Hz) |
| springs | k_s = 6.2 N/m attached 0.25 m below the pivot → k_t = 0.386 N m/rad, c = 8.20 s⁻², **ĉ = 0.25** |
| bond rotor | aluminium disc, radius 40 mm, 8 mm thick (0.109 kg, I_r = 8.7×10⁻⁵ kg m²), 2965 rpm under PLL speed control → H = 0.0270 kg m²/s, b = 0.573 s⁻¹, **β̂ = 0.10** |
| per-site trim | electromagnet under each (steel-inserted) bob: sets K_n and the pinning bumps; peak 1.7 N/m at the bob for S = 0.05 |
| readout | 20-bit axle encoders, sampled at ≥ 20 Hz |
| excitation | phased coil impulses launching θ_n = A cos kn, θ̇_n = ±Aω sin kn (A = 0.02 rad) |
| damping | Q ≈ 500 assumed (amplitude decay time 143 s at 7 rad/s) |

Band: ω(k) from 5.73 (k = 0) to 8.10 rad/s (k = π) with the rotors stopped; at k = π/2 the rotors split the
doublet into ω₊ = 7.611 and ω₋ = 6.465 rad/s, **Δω(π/2) = 1.1455 rad/s** (16% of ω).

## 2. Calibrations, in order

1. **κ̂ — the zero check (rotors stopped).** With H = 0 every ±k doublet must be degenerate: κ̂ = 0 by
   construction (planar pendula have no on-site gyroscopic term). A residual splitting measures stray
   nonreciprocity and is subtracted as an offset.
2. **ĉ — the band (rotors stopped).** Measure ω(k) at m = 0 … 16 (k = 2πm/32) and fit
   ω² = K + 2c(1 − cos k): gives K and c, hence ĉ. Measure each site's K_n by single-site spectroscopy
   (neighbours clamped) at every coil setting used; this fixes ⟨δK²⟩ for the bumps.
3. **β̂ — one asymmetry (rotors at speed).** Measure Δω at k = π/2: b = Δω(π/2)/2, β̂ = b/√K.

**Diagnostic (not a calibration).** With the rotors on, ω(k)·ω(−k) must equal the rotors-off Q(k) at every
k (census row 14: Vieta's formula for a coupling linear in velocity). A change shows that the rotor coupling
is not purely velocity-linear, and the predictions below do not apply.

## 3. The three predictions

Each follows from the equation of motion with no further input; within the model they are exact, and they
can fail only in a realisation (`UNIVERSAL_RELATIONS.md`, "The count — corrected accounting").

**P1 — the sin k shape.** With b fixed at k = π/2, **Δω(k) = Δω(π/2)·sin k** at every other k.

| m (k = 2πm/32) | 1 | 2 | 4 | 6 | 8 | 12 | 14 |
|---|---|---|---|---|---|---|---|
| Δω predicted (rad/s) | 0.2235 | 0.4384 | 0.8100 | 1.0583 | 1.1455 (cal.) | 0.8100 | 0.4384 |

*Measurement:* launch ±k running waves at each m; Δω from the phase slopes of Σ_n θ_n e^{∓ikn} over a
300 s ring-down. *Failure:* any m with |Δω(m)/Δω(π/2) − sin k| > 0.01 (1.4× the worst disordered-ring
residual simulated, §4). The alternatives it excludes: linear Doppler (Δω ∝ k gives 0.573 rad/s at m = 4,
against 0.810), and a next-nearest-neighbour velocity coupling (a sin 2k component).

**P2 — the coefficient ¼.** A mean-zero stiffness bump δK_n shifts the k = π/2 asymmetry by

  **δ(Δω) = −¼ · b · ⟨δK²⟩ / c²**, i.e. δ(Δω)/Δω(π/2) = −S²/(8ĉ²),

where ⟨δK²⟩ is the ring average and S the ring-rms of δK/K. At S = 0.05 (Gaussian bump, σ = 4 sites):
**δ(Δω) = −5.73×10⁻³ rad/s (−5.0×10⁻³ of Δω).** On the finite ring the exact second-order value is 3.4%
larger (−5.92×10⁻³; 2.0% at S = 0.02; the ¼ is the wide-bump, large-ring limit). *Measurement:* the even
part C = [Δω(+S) + Δω(−S) − 2Δω(0)]/(2S²), with the bump reversed by the coil currents, compared with
−¼ b⟨δK²⟩/c² from the calibrations. *Failure:* the measured ratio to the ¼ law differs from the finite-ring
value by more than three times its uncertainty; at the precision of §4 this distinguishes ¼ from ⅛ or ½.

**P3 — shape independence.** At equal S, any localised mean-zero bump gives the same δ(Δω). Ideal N = 32
ring, S = 0.05 (ratio to the ¼ law): Gaussian σ = 3: 1.057; Gaussian σ = 5: 1.025; sech², width 4: 1.058;
two Gaussians (σ = 2.5, 16 sites apart): 1.074. A linear ramp (discontinuous on the ring; a different class):
1.744. *Failure:* the four localised shapes differ by more than their finite-ring spread plus measurement
uncertainty (§4), or the ramp agrees with them.

## 4. Error budget

Expectations for each non-ideality were committed before the simulation (1c63b6d); the simulated platform uses
the full nonlinear pendulum, damping, site-to-site disorder, the finite ring and a finite ring-down with the
phase-slope estimator. **Post-hoc** diagnostics, written after the budget output was seen, are marked.

| non-ideality | expected | simulated | verdict |
|---|---|---|---|
| finite ring, N = 32 (σ = 4) | C within 5% of ¼ | +3.4% (S = 0.05), +2.0% (S = 0.02) | hit; computable correction |
| larger rings at S = 0.05 | N = 64 within 2%, N = 128 within 1% | **breakdown**: ratio −18 (N = 64), 1.28 (N = 128); at S = 0.02: +0.7%, +0.3% | **miss** — at fixed S the response turns non-perturbative on large rings (as recorded in MODEL_SPEC §5b.6a); use S = 0.02 there |
| damping, Q = 500 and 200 | Δω < 10⁻⁴, C < 10⁻³ | 10⁻¹⁴ and ≤ 1.2×10⁻⁶ | hit |
| disorder: K 0.3%, springs 1%, rotors 0.5% (20 realisations) | P1 ≤ 5×10⁻³; P2 sd ≤ 3%; P3 shapes within 5% | P1 max 6.9×10⁻³ (sd 1.5×10⁻³); **P2 sd 19%** (range 0.66–1.37); **P3 spread 15%** (ramp still 1.5× off) | **miss** on all three |
| — which component (post hoc) | — | P2 sd: springs 1% alone 17.5%; K 0.3% alone 3.3%; rotors 0.5% alone 0.3% | springs dominate |
| — spring disorder needed (post hoc; K 0.03%, rotors 0.1%) | — | P2 sd 2.2% at 0.3%, 0.26% at 0.1%, 0.07% at 0.03% (N = 32, S = 0.05) | roughly ∝ (spring spread)² |
| — the ±S protocol | removes a linear cross term; one-sided sd > 10% | one-sided sd 19%, the same as ±S | **wrong reason**: the spread is not the linear cross term |
| pendulum anharmonicity (sin θ) | Δω shifts ~2×10⁻⁵ (A = 0.02) and ~4×10⁻⁴ (A = 0.1) | 10⁻⁸ (post hoc, ideal ring) | **miss — our derivation error**: for a running wave the anharmonicity is a uniform stiffness change, and Δω = 2b sin k is stiffness-independent; it cancels exactly |
| finite ring-down (T = 300 s, 20 Hz) | Δω within 10⁻⁴ of the eigenvalue; C within 1% | Δω +2.1×10⁻⁴; C +0.7% (A = 0.02), +1.5% (A = 0.1) | Δω miss (estimator/launch on a disordered ring); C hit |
| P3 at reduced disorder (post hoc) | — | shape spread (S = 0.02): 4.4% at N = 32, 1.1% at N = 64, unchanged at K 0.03%, springs 0.1% | the spread is finite size, not disorder |

**What survives at realistic precision.**
- **P1 survives** on the nominal design: disorder limits it to < 1%, against a 29% difference from linear
  Doppler at m = 4. It is the first experiment to run.
- **P2 does not survive with 1% springs** (19% scatter). With springs matched to 0.3% (K 0.03%, rotors
  0.1%) the scatter is 2.2%; with springs at 0.1%, 0.3%. K disorder at 0.3% alone gives 3.3%, so K must be
  trimmed well below that (≲ 0.1%, assuming the same quadratic scaling — not simulated). The finite-ring correction (+2 to +3.4% at N = 32) is
  computable from the calibrations and must be applied. Matched springs are the one component that needs
  better apparatus.
- **P3 survives only at the few-% level on N = 32**: the four localised shapes differ intrinsically by 4.4%
  (finite size), so the ring distinguishes the localised class from a ramp (×1.5–1.7) but cannot test shape
  independence at 1%. That needs N = 64 (1.1% intrinsic spread, S = 0.02) with springs at ≤ 0.1%; the
  model's own 0.4% check needs N ≳ 128.

## 5. Scope

This is a theory of a class of realisable lattices: linear, passive, gyroscopically coupled oscillator
chains described by three dimensionless parameters, ĉ = c/K, κ̂ = κ/√K and β̂ = βc/√K. Three measurements
fix them; the three predictions above are further conditions on the same chain that no calibration uses, and
they can fail. The note makes no claims about fundamental physics.
