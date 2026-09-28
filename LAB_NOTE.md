# Three parameter-free predictions for a gyroscopically coupled pendulum ring

*Technical note, 2026-09-28, branch `realisation-gyroscopic` only — not merged into main. Sources:
`UNIVERSAL_RELATIONS.md` on main ("The count — corrected accounting"); `REALISATION.md` (this branch),
§4 (bond rotors) and §5–§6. Simulations: `shape_zero_tests/platform_error_budget.py` (expectations committed
first, 1c63b6d) and the post-hoc diagnostics `platform_error_budget_checks.py`. Every design choice below is
ours, not taken from a source.*

*Second platform (added 2026-09-28): an LC ring with gyrator bonds, §6. §6.4 records a design flaw found
there that also affects this section's §4 (post hoc).*

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

**[Post hoc, 2026-09-28 — see §6.4.]** The design point ĉ = 0.25, β̂ = 0.10 places the k = π/2 probe mode
8.4×10⁻⁴ √K from the ω(−13π/16) mode — an accidental near-degeneracy about the size of the P2 signal, which
disorder couples. Much of the P2 scatter above comes from it. Rerun at β̂ = 0.106 (rotor speed ×1.06; nearest
other mode 8.8×10⁻³ √K away): with the nominal parts P2 scatters by **2.9%** (not 19%); with springs at 0.3%,
0.96%; with springs and K at 0.1%, 0.20%; P1 max 5.9×10⁻³, 1.5×10⁻³, 3.2×10⁻⁴. These numbers are post hoc,
not pre-registered; the design should use β̂ = 0.106.

## 5. Scope

This is a theory of a class of realisable lattices: linear, passive, gyroscopically coupled oscillator
chains described by three dimensionless parameters, ĉ = c/K, κ̂ = κ/√K and β̂ = βc/√K. Three measurements
fix them; the three predictions above are further conditions on the same chain that no calibration uses, and
they can fail. The note makes no claims about fundamental physics.

## 6. Second platform — an LC ring with gyrator bonds

*Added 2026-09-28. Every design choice here is ours. Expectations committed before the simulation
(`shape_zero_tests/circuit_error_budget_predictions.txt`, 65ee4eb); simulation `circuit_error_budget.py`;
post-hoc diagnostics `circuit_error_budget_checks.py` and `detuned_design_checks.py`, labelled as such.*

### 6.1 Circuit and mapping

Each node n has a capacitor C to ground and an inductor L_g to ground; each bond has an inductor L_c and a
gyrator made of two voltage-controlled current sources (VCCS) with buffered, high-impedance inputs: one
injects −G V_{n+1} into node n, the other +G V_n into node n+1. With the node flux φ_n (V_n = φ̇_n),
Kirchhoff's current law gives

  C φ̈_n = −φ_n/L_g + (φ_{n+1} + φ_{n−1} − 2φ_n)/L_c − G φ̇_{n+1} + G φ̇_{n−1},

the pendulum ring's equation (§1) term by term:

| pendulum | circuit | model |
|---|---|---|
| θ_n | φ_n (node flux) | u_n |
| I_eff | C | 1 |
| τ_g | 1/L_g | K = 1/(L_g C) |
| k_t | 1/L_c | c = 1/(L_c C); ĉ = L_g/L_c |
| H (bond rotor) | G (gyrator) | b = βc = G/C; β̂ = G√(L_g/C) |
| no spin | no on-site gyrator | κ = 0 |

**Passivity and velocity-linearity.** An ideal gyrator delivers P = V_n(−G V_{n+1}) + V_{n+1}(G V_n) = 0: it is
lossless, and its currents are linear in the node voltages, i.e. in φ̇ — the velocity-linear bond. Real VCCS
break this in four ways:
- **gain mismatch** G_a ≠ G_b adds a symmetric conductance coupling (P = −δG V_nV_{n+1}, not sign-definite);
- **finite bandwidth** G(ω) = G/(1 + iωτ) adds a current in phase with V: a direction-dependent conductance
  ±2Gωτ sin k — loss for one propagation direction and **gain** for the other;
- **finite output impedance** of a Howland pump with resistor mismatch ε: a shunt conductance ~ ±εG per output;
- **offsets** (input offset, bias currents) set a DC operating point only; they cannot change a linear
  circuit's frequencies.
A Howland pump with unbuffered inputs would also load each sensed node with ~G/2 (Q ≈ 12 here), hence the
buffered inputs. The circuit has no anharmonicity to the precision that matters here.

### 6.2 Design (ours)

| item | value |
|---|---|
| node | C = 10 nF (C0G), L_g = 1 mH (shielded ferrite), series R for Q = 100 |
| bond | L_c = 4 mH (shielded ferrite); ĉ = 0.25 |
| gyrator | two buffered improved-Howland VCCS per bond (one quad op-amp, ~10 MHz GBW, VCCS pole ~5 MHz), R_set = 2.98 kΩ → G = 3.35×10⁻⁴ S, **β̂ = 0.106** (not 0.10; §6.4) |
| bumps | a programmable grounded synthetic inductor (GIC) in parallel with L_g at each node, setting δK_n |
| frequencies | K = 10¹¹ s⁻² (50.3 kHz); band 50.3–71.2 kHz; at k = π/2, ω₊ = 67.2 kHz, ω₋ = 56.5 kHz, **Δω(π/2) = 6.70×10⁴ rad/s (10.7 kHz)**; P2 signal at S = 0.05: −335 rad/s (−53 Hz); linewidth at Q = 100: ~670 Hz |
| readout | current drive into one node through a resistor from a function generator; lock-in (or synchronous DAQ) reading of every node voltage via an analog multiplexer; spatial Fourier transform to ±k spectra; resonance fits |

**Calibrations** as in §2: gyrators unpowered — ±k doublets degenerate (κ̂ = 0) and the band gives K, c;
gyrators on — Δω(π/2) gives b. The product rule ω(k)ω(−k) = Q(k) is the diagnostic that the powered gyrators
are velocity-linear. A uniform temperature drift of the ferrite inductors changes K and c together and is
recalibrated; non-uniform drift acts as tolerance.

### 6.3 Error budget

N = 32, protocol and predictions as §3–§4, at the pre-registered design point β̂ = 0.10 unless marked. The
first run's mode selector admitted the fast, non-oscillating VCCS-pole modes and returned wrong modes wherever
the VCCS pole was included, and in some τ = 0 realisations (the 0.1% row read sd 12.8%). It was restricted to
oscillatory modes in the band, marked in the script, with the expectations unchanged; that output is kept as
`circuit_error_budget_output_modeid.txt`.

| non-ideality | expected | simulated | verdict |
|---|---|---|---|
| ideal circuit | identical to the ideal pendulum ring | P1 4×10⁻¹⁴; C ratio 1.03404 | hit — same equations |
| tolerance 1% (C, L_g, L_c, G_a, G_b, shunts) | P1 ≤ 10⁻²; P2 sd 15–25%; P3 fails | P1 max 2.1×10⁻²; **P2 sd 53%**; P3 fails | miss on P1 and P2 |
| tolerance 0.1% | P1 ≤ 1.5×10⁻³; P2 sd ≤ 1%; P3 finite-size-limited | P1 4.7×10⁻⁴; P2 sd 1.3% (60 realisations, post hoc: 2.0%, 28% of them > 2% off); P3 4.7% | P2 miss |
| inductor Q = 50, 100, 300 | Δω ≤ 10⁻⁴; C ≤ 1%; stable | Δω ~10⁻¹⁴; C ≤ 2.6×10⁻⁴; stable | hit |
| VCCS pole 5 MHz / 50 MHz | Δω ~1.5×10⁻⁴; C ≤ 10⁻³; gain ~4×10² s⁻¹; unstable above Q ≈ 500 / 5000 | Δω −1.6×10⁻⁴ / −1.6×10⁻⁶; C −3×10⁻⁵; gain 459 / 46 s⁻¹ (lossless inductors); unstable between Q = 300 and 500 / 3000 and 10000 | hit |
| VCCS mismatch and shunts, 1% / 0.1% | Δω ≤ 10⁻⁴ | Δω up to 2.7×10⁻³ / 2.3×10⁻⁴; gain up to 165 / 16 s⁻¹ | **miss** — post hoc: the change tracks the mean-G change and is absorbed by the β̂ calibration |
| parasitics: 10 pF across L_c, 3 pF per VCCS input | P1 ≤ 3×10⁻³; C ≤ 0.5% | P1 1.0×10⁻³ (systematic, from the k-dependent capacitance); C −0.42% | hit |
| offsets: 1 mV, 10 nA per op-amp | frequencies unchanged; V_dc ≤ 10 µV, i_dc ≤ 1 µA | V_dc 2 µV; i_dc 0.5 µA | hit |
| combined realistic (0.1%, Q 100, 5 MHz, parasitics, mismatch, shunts) | P1 ≤ 5×10⁻³; P2 sd ≤ 1%; P3 (N = 64, S = 0.02) ≤ 2%; stable | P1 1.1×10⁻³; P2 sd 0.22% (0.29% at S = 0.02; 0.49% at N = 64); P3 1.1% mean, **2.3% max**; stable (−1.3×10³ s⁻¹) | hit, except P3 max |
| lock-in estimator (single-node drive) | Δω ≤ 10⁻⁴ of eigen; δ(Δω) within 2% | Δω −3.7×10⁻⁴; δ(Δω) +3.3% | **miss** — overlapping, asymmetric lines under a one-node drive |
| which part drives the 1% P2 spread (post hoc) | — | C alone 50%; L_g 26%; L_c 15%; gyrators 2.8% | mass (C) disorder dominates |
| added non-idealities at 0.1% (post hoc) | — | + Q 100: sd 1.3%; + parasitics: 13.5%; + VCCS pole: 0.39% — erratic | led to §6.4 |

### 6.4 An accidental degeneracy in the design point (post hoc; both platforms)

At ĉ = 0.25, β̂ = 0.10 the probe mode ω(+π/2) = 1.32882 √K lies 8.4×10⁻⁴ √K from ω(−13π/16) (m = −13 at
N = 32, −26 at N = 64). That is the size of the P2 signal (10⁻³ √K), and disorder couples the two modes, so the
P2 shift is ill-conditioned: small changes (a parasitic, a loss) move the detuning and the scatter erratically.
At **β̂ = 0.106** the nearest other mode is 8.8×10⁻³ √K away at both N = 32 and 64. Rerun there (30
realisations, post hoc):

| circuit, β̂ = 0.106 | P1 max | P2 median, sd (max dev) | P3 spread mean (max) |
|---|---|---|---|
| N = 32, S = 0.05, ideal | — | 1.0351 | ratio range 0.051 (finite size) |
| — 1% tolerance | 2.7×10⁻² | 1.018, 16% (0.43) | 14% (40%) |
| — 0.1% tolerance | 4.9×10⁻⁴ | 1.0352, **0.19%** (0.006) | 4.8% (5.3%) |
| — realistic | 1.1×10⁻³ | 1.0307, **0.17%** (0.006) | 4.8% (5.3%) |
| N = 64, S = 0.02, ideal | — | 1.0069 | ratio range 0.011 |
| — 0.1% tolerance | 7.2×10⁻⁴ | 1.010, 2.5% (0.060) | 1.9% (5.9%) |
| — realistic | 1.1×10⁻³ | 1.0034, **0.78%** (0.021) | **1.2% (2.8%)** |

At 1% tolerance (N = 32) the P2 scatter by part is C 8.7%, L_g 2.5%, L_c 1.9%, gyrators 0.4%. The pendulum
ring at β̂ = 0.106 is in §4's post-hoc note. The design value β̂ = 0.106 above follows from this check.

### 6.5 What a 32- or 64-node circuit could test

- **P1 (sin k):** testable at 0.1% parts to ~0.1% (the residual is the parasitic-capacitance systematic,
  ~10⁻³, correctable from the band). With 1% parts the residual reaches 2.7%: parts at ≲ 0.3% are needed for a
  1% test.
- **P2 (¼):** testable on 32 nodes with 0.1% parts to ~0.2% scatter, against a computable finite-ring
  correction of +3.5% (S = 0.05); the estimator must fit the spatially resolved lines properly (the naive
  one-node lock-in peak read was 3.3% off).
- **P3 (shape independence):** 32 nodes give a 5% intrinsic (finite-size) spread — only a class test (ramp vs
  localised). 64 nodes with 0.1% parts reach ~1.2% mean spread (2.8% worst realisation): a 2–3% test.
- **Parts and cost (rough, 2026 hobby prices).** Per node: C0G capacitor (sorted to 0.1%: ~$1), two shielded
  ferrite inductors (5–10% as bought, sorted or trimmed to 0.1%: ~$4–8), one quad op-amp plus ~10 0.1%
  resistors for the bond's two VCCS (~$8), a GIC bump stage with digitally switched resistors (~$5), PCB
  share (~$3). About $20–25 per node: **~$700–800 for 32 nodes, ~$1500 for 64**, plus instruments — a precision
  LCR meter at 50–70 kHz (0.05% class; the costliest item unless borrowed), a function generator and a lock-in
  or USB scope/AWG with a 32-to-1 analog multiplexer (~$300–500).
- **Could a skilled hobbyist build it?** The 32-node board, powered gyrators and P1 — yes: standard
  through-hole or SMD analog work at 50–70 kHz. P2 and P3 need 0.1% matching of ~100 inductors and capacitors
  at the operating frequency, a stable temperature (non-uniform ferrite drift at ~10⁻⁴/K acts as tolerance),
  and careful line fitting: feasible for a patient hobbyist with access to a precision LCR meter, and the
  sorting is the main labour. Stability needs inductor Q below ~300–400 with ~10 MHz op-amps (or faster op-amps).
