# Realisation — gyroscopic metamaterials (scoping)

*Recorded 2026-09-27 on the branch `realisation-gyroscopic`, not on main. A scoping report:
nothing here is adopted into the model, and the model is not presented as a theory of nature
(`OVERVIEW.md`, "Scope"). The census it is scored against is `UNIVERSAL_RELATIONS.md`.*

**Sources.**

- [N] L. M. Nash, D. Kleckner, A. Read, V. Vitelli, A. M. Turner, W. T. M. Irvine,
  "Topological mechanics of gyroscopic metamaterials", *PNAS* 112(47), 14495 (2015); arXiv:1504.03362.
- [M] N. P. Mitchell, L. M. Nash, W. T. M. Irvine, "Realization of a topological phase transition
  in a gyroscopic lattice", arXiv:1711.02433.

Both were read from their arXiv LaTeX sources; the PDF text could not be extracted here.

**What is whose.** The papers supply every equation and number attributed to them below.
Four items are **ours, not the papers'**, and are marked where they appear:

1. the second-order heavy-top equation;
2. the point-dipole idealisation of the magnets;
3. the bond-rotor design for β;
4. the two-branch sum rule.

One item is **from theory, not the papers**: the Haldane valley asymmetry.

---

## 1. The platform, from the papers

**Nash et al.** The fast-spin response of a gyroscope is dr̂/dt = (r/Iω) × F ([N], eq. "eqn-gyro").
With ψ = r_x + i r_y the tip displacement, the linearised lattice equation is ([N], eq. "lattice-eom")

  i ψ̇_p = Ω_g ψ_p + (Ω_k/2) Σ_q [ (ψ_p − ψ_q) + e^{2iθ_pq} (ψ*_p − ψ*_q) ],

with:

- Ω_g = mgℓ/(Iω), the precession frequency;
- Ω_k = kℓ²/(Iω), the spring frequency;
- θ_pq, the global angle of the bond from p to q.

Experimental details from [N]:

- gyroscopes spun by DC motors at about 300 Hz;
- each suspended from a top plate by a weak spring, giving Ω_g of about 1 Hz;
- coupled by vertically aligned neodymium magnets;
- non-idealities: about 10% spread in motor speed, position error, next-nearest-neighbour coupling, and
  "only partially first-order dynamics (due to finite spinning speed)".

The footnote to "lattice-eom" says that for repulsive magnetic interactions the ψ and ψ* terms carry
unequal real coefficients.

**Mitchell et al.** The general form ([M], eq. "eom") is

  i ψ̇_p = Ω_p ψ_p + ½ Σ_q [ (Ω⁺_pp ψ_p + Ω⁺_pq ψ_q) + e^{2iθ_pq} (Ω⁻_pp ψ*_p + Ω⁻_pq ψ*_q) ],

where:

- Ω^±_pq = −(ℓ²/Iω)(∂F_p∥/∂x_q∥ ± ∂F_p⊥/∂x_q⊥);
- Ω_p = (mg + F^susp + F^coil_z)ℓ/(Iω), which a coil under each site shifts to (1 ± Δ_AB)Ω_p⁰;
- the interaction scale is Ω_k = ℓ²k_m/(Iω), with Ω_k/Ω_p⁰ = 0.67 in the experiments;
- the motors are synchronised by pulse-width modulation.

[M] also states that magnets have "an anti-restoring response to perpendicular displacements"
(Ω⁺ ≠ Ω⁻).

## 2. Parameter mapping and the large-κ̂ limit

**The heavy-top equation (ours, not the papers').** Neither paper writes the second-order equation
that includes nutation. The standard linearised heavy symmetric top gives

  I_p ψ̈ = −(mgℓ + suspension + coil) ψ + i I₃ω_s ψ̇ + ℓ² F,

where:

- I_p is the moment of inertia about the pivot;
- I₃ω_s is the spin angular momentum;
- F is the bond force.

In the model's form, ü = −K u + c ∇²u + κ𝕁u̇, this gives:

| model | platform |
|---|---|
| K | (mgℓ + suspension + coil)/I_p |
| κ | I₃ω_s / I_p (the nutation rate) |
| c | J-compatible part of the bond stiffness, (k∥ + k⊥)ℓ²/(2I_p) |
| — | J-breaking part, (k∥ − k⊥)ℓ²/(2I_p): no counterpart in the model (§3) |
| ĉ = c/K | Ω_k/(2Ω_g) for springs: 0.5 in [N] Fig. 1 (Ω_g = Ω_k); about 0.3 at [M]'s Ω_k/Ω_p⁰ = 0.67 (Ω_k is only the scale of Ω^±, so this is an order of magnitude). The model has ĉ = 1/√5 = 0.447. |
| κ̂ = κ/√K | √(ω_nut/Ω_g), since Ω_g = K/κ. The papers do not give I₃/I_p; for I₃/I_p between 0.05 and 0.5, ω_s ≈ 300 Hz and Ω_g ≈ 1 Hz give **κ̂ ≈ 4–12** (our estimate). The model's floor value is κ̂\* = 0.6498. |
| β̂ = βc/√K | 0: nothing corresponds (§4) |

**The platform is the model's large-κ̂ limit, for its J-compatible part.**

- The model's a-branch is ω = ½(−κ + √(κ² + 4Q)), with Q(k) = K + 2c(1 − cos k).
- As κ → ∞ with K/κ and c/κ fixed, this tends to Q/κ.
- That is exactly [N]'s equation without the ψ* term, with Ω_g = K/κ and Ω_k/2 = c/κ.
- The correction is −Q²/κ³, a relative O(1/κ̂²) — a few percent at κ̂ ≈ 4–12. This is consistent with
  [N]'s remark about partially first-order dynamics.

**Reaching κ̂\* ≈ 0.65 means slowing the spin.** On our estimate the spin must drop by a factor of about
40–350, to roughly 1–8 Hz.

- A hanging gyroscope, restored by gravity, stays stable at any spin.
- In that regime the first-order equation fails, and the full second-order equation — the model's form —
  is needed.
- Spin disorder then enters only κ. K and c do not depend on spin.

The full platform is **not** a limit of the model: the model sets the J-breaking term to zero (P8), and
the platform does not (§3). The two are different slices of the general linear gyroscopic lattice.

**Node size.** A gyroscope is one n = 1 node, ψ ∈ ℂ. Nodes with n ≥ 2 and the u(n) velocity links W have
no counterpart on the platform.

## 3. The J-breaking bond term

A bond with longitudinal stiffness k∥ and transverse stiffness k⊥ acts on δ = ψ_p − ψ_q as

  ½(k∥ + k⊥) δ + ½(k∥ − k⊥) e^{2iθ} δ*.

- The first part is complex-linear and **commutes** with J (multiplication by i).
- The second is conjugate-linear, couples ψ to ψ*, and **anticommutes** with J.

It is the J-breaking part.

**Its size, by coupling type:**

| coupling | k⊥ / k∥ | J-breaking / J-compatible | source |
|---|---|---|---|
| unstretched spring | 0 | **1** (equal halves) | [N], eq. "lattice-eom" |
| spring of rest length a₀ at spacing a | 1 − a₀/a | a₀/(2a − a₀) | ours |
| zero-rest-length spring | 1 | **0** | ours |
| repulsive point dipoles, moments held vertical, V ∝ 1/r³ | −1/4 | **5/3** | **our idealisation**: k∥ = V″ = 12A/a⁵, k⊥ = V′/a = −3A/a⁵; [M] reports only the sign ("anti-restoring") |

**It is not small at long wavelength.**

- The on-site sum Σ_q e^{2iθ_pq} cancels in the bulk of honeycomb, square and triangular lattices.
- The hopping part survives at O(k²), the same order as the dispersion.
- In a 1-D chain it is ∝ 2(1 − cos k).

This term is what gives [N] and [M] their gap, Chern bands and chiral edge modes. A J-compatible
realisation removes that physics: with Ω⁻ = 0, [N]'s equation is a scalar tight-binding model with real
hopping, symmetric under ψ → ψ*, t → −t.

**Feasible isotropic modifications (ours):**

- zero-rest-length (pre-stretched) springs, k⊥ = k∥;
- tensioned springs combined with magnets, tuned to k∥ = k⊥ (tension adds T/a to k⊥; the magnets
  subtract).

**The check.** With the motors off (κ = 0), the x- and y-polarised pendulum bands give k∥ and k⊥
directly. They coincide only for a J-compatible coupling.

**The on-site terms are J-compatible.** Gravity, the suspension spring and the coil force (along z) are
isotropic.

## 4. β — no counterpart

- **Spin breaks time reversal only on-site;** that is κ.
- **A 1-D chain of spinning gyroscopes is inversion symmetric,** so ω(k) = ω(−k).
- **The chiral edge modes of [N] and [M] are one-way,** but that is a topological edge property, not a
  bulk sin k asymmetry.

**The nearest native effect (from theory, not the papers).**

- [N] states that at weak coupling (Ω_k ≪ Ω_g) the spring-coupled lattice maps onto the Haldane model.
- In the Haldane spectrum, a sublattice mass M (here [M]'s Δ_AB) together with the complex
  next-nearest-neighbour hopping gives valley-contrasting gaps. The bulk ω(k) ≠ ω(−k).
- This is **not β**. It is second order in the coupling (∝ Ω_k²/Ω_g), depends on Ω_k, Ω_g, Δ_AB and the
  bond geometry, and vanishes when the coupling is J-compatible.
- Census rows 1–2 would distinguish it from β.

**The bond-rotor design (ours, not the papers').**

- The model's term F_n = −βc(u̇_{n+1} − u̇_{n−1}) is a skew velocity coupling between neighbours. It
  comes from the Lagrangian term L ⊃ ½βc Σ_n (u_n·u̇_{n+1} − u_{n+1}·u̇_n).
- A spinning rotor on each bond gives exactly that term, with its two gimbal (tilt) angles driven by
  the displacements of sites n and n+1: the gyroscopic torque couples the rate of one angle to the other.
  - With spin angular momentum H, the forces are F_n ⊃ −H u̇_{n+1} and F_{n+1} ⊃ +H u̇_n.
  - Summed over bonds, that is the model's term, with βc = H in the linkage's units.
  - Every rotor spins in the same sense along the chain. That is the direction asymmetry.
- The model's β acts identically on both components, which needs one rotor per component per bond.
- Alternatively, planar pendula (one degree of freedom each) make the node truly scalar, as the model's
  β sector is.
- The equivalent element in electrical circuits is the gyrator.

**β on spinning nodes: the two-branch sum rule (ours; found, then verified, not predicted).**

The model's β sector is scalar and contains no κ (`04_scripts/session/pinned_asymmetry_reference.py`).
There Δω = 2βc sin k exactly. Put the same term on gyroscopic nodes (`shape_zero_tests/gyro_beta.py`):

- **The precession branch alone** (isotropic coupling, circular modes, iκJ → −κ) satisfies
  ω² + (κ − b)ω − Q = 0, with b = 2βc sin k. So

  Δω_a = b + ½[√((κ − b)² + 4Q) − √((κ + b)² + 4Q)] = b (1 − κ/√(κ² + 4Q)) + O(b³).

  This depends on κ̂ and ĉ, falls ∝ 1/κ̂², and through Q depends on the transverse wavenumbers. Census
  rows 1–2 would **fail** on this branch alone.

- **The two positive branches together** satisfy

  **[ω_a(k) − ω_a(−k)] + [ω_b(k) − ω_b(−k)] = 2 · (2βc sin k),**

  for any κ, K, c, anisotropy k∥ ≠ k⊥, bond angle, and transverse wavenumber.

*Derivation (trace).*

1. Take a plane wave u = a e^{i(k·n − ωt)}, so u̇ = −iωu. The equation becomes
   −ω² a = −D a + κJ(−iω a) − βc(−iω)(2i sin k₀) a, that is,
   ω² a = D a + ω (iκJ + b) a, with b = 2βc sin k₀.
2. With z = (a, ωa), this is ω z = A z, where A = [[0, I], [D, iκJ + bI]]. The four roots at k therefore
   sum to tr A = tr(iκJ) + 2b = 2b: J is traceless, and D does not enter.
3. At −k, b → −b, so the roots sum to −2b.
4. The field is real, so (ω, a) at k gives (−ω\*, a\*) at −k. The roots are real (a gyroscopic system with
   D > 0 is stable). So the roots at −k are the negatives of those at k, and the positive roots at −k are
   minus the negative roots at k.
5. Hence Σ_pos ω(k) − Σ_pos ω(−k) = Σ_all ω(k) = 2b.

Only the velocity matrix enters the trace. With a uniform isotropic inertia m, b → b/m.

*Verification* (`shape_zero_tests/gyro_sumrule_verify.py`, output in `gyro_sumrule_verify_output.txt`):

- **Sum rule:** 5000 random draws on a 2-D lattice, with oblique bonds, k⊥/k∥ from −0.4 to 1.5, κ from 0
  to 30, and any k₀ and k₁. Worst deviation **1.2 × 10⁻¹³**.
- **Slow-branch exact form:** worst deviation 5 × 10⁻¹⁵. The first-order form is off by at most
  9 × 10⁻⁶ at β = 0.05.
- **κ = 0:** each branch alone gives 2βc sin k to 7 × 10⁻¹⁵, for any anisotropy.

**Scope.** Uniform, linear lattices with identical isotropic inertia.

**Status.** Noticed in the output of `gyro_beta.py`, then derived and verified. It was not predicted.

## 5. The census, row by row

| # | relation | test on this platform or a modification | feasibility |
|---|---|---|---|
| 1 | Δω = 2βc sin k | Motors off (κ = 0): pendula in a ring with bond rotors. Measure the ±k standing-wave splitting at several k. At κ = 0 this is exact for any anisotropy (the stiffness independence is tested by varying K, k∥, k⊥). With spin on, only the two-branch sum tests it. Δω/ω is a few percent at β̂ = 0.05, so the resonances need Q ≳ 100. | feasible, with a new coupler |
| 2 | transverse independence | A 2-D square array with rotors on the x-bonds only; measure Δω(k_x, k_y) at several k_y. Holds at κ = 0 even with anisotropic coupling (square-lattice bonds along the axes do not mix x and y). With spin, only the sum rule holds. | feasible, but many couplers |
| 3 | q = 1 pin shift −¼βs²S², shape-independent | Row 1's chain plus a pinning bump K(x) = K₀(1 + Sη(x)), set by per-site coil currents (native to [M]). The signal is small: at S = 0.02 and k = π/2, δ(Δω)/Δω ≈ 2.5 × 10⁻⁴ (model units, β = 0.05). Site-to-site stiffness disorder is itself a pinning perturbation, so each site's K must be measured and included. | marginal: needs high Q and site-by-site calibration |
| 4 | κ̂ ≥ 2ĉ/√(1 + 2ĉ) | The inequality can be measured from single-site and band spectroscopy. It holds trivially (κ̂ ≈ 4–12 against a floor of about 0.5 at ĉ ≈ 0.33). What the floor protects is J-compatibility of the u(n) gauge transport with n ≥ 2 nodes (C13, C14), which the platform lacks. Lowering the spin could cross the floor, but at n = 1 without links nothing is predicted to fail. | a number only; no consequence to observe |
| 5 | Larmor splitting = κ | Single-gyroscope tap spectrum (§6). | a calibration |
| 6 | unit equivalence | not a measurement | — |
| 7 | D8 ratio | not a measurement; the lattice does not realise the D8 flow | — |
| 8 | P-3 (excluded) | Excluded from the count. Separately, the A′ well's energy \|ψ\|³/3 is non-analytic at ψ = 0; a smooth pendulum's leading nonlinearity is \|ψ\|²ψ (the gravity term 1 − cos θ), not \|ψ\|ψ. | — |
| 9 | ν → 0 at k = π/2, q ≥ 2 | A 2-D lattice with rotors, a coil bump, and scattering rates Γ(±k) for a sequence of increasingly wide bumps. Needs lattices of hundreds of sites per side, and damping well below Γ. | not feasible on a tabletop |
| 10 | structural (missions 1, 2, 4a, 5, 6) | not measurements. Mission 1's n = 1 case says the allowed J-commuting coupling is the isotropic part — the k∥ = k⊥ condition of §3. | — |
| 11 | small-amplitude reduction | an internal check | — |

## 6. Calibrations

- **K and κ: single-site spectroscopy.** Excite one gyroscope and read its precession frequency ω_a and
  nutation frequency ω_b (counter-circulating). For the isolated node, ω² ± κω − K = 0 exactly, so
  ω_b − ω_a = κ, ω_a ω_b = K, and κ̂ = (ω_b − ω_a)/√(ω_a ω_b). This needs an isotropic pivot (I_x = I_y).
  In the fast-spin regime nutation is fast and weak and needs high-frame-rate tracking; at κ̂ ~ 1 the two
  frequencies are comparable.
- **c: the band across the zone.** ω_a(k) ω_b(k) = Q(k) = K + 2c(1 − cos k) in 1-D, or the normal-mode
  splitting of two coupled sites. With the motors off, the x- and y-polarised bands give k∥ and k⊥
  separately, which measures the J-breaking term (§3).
- **β: Δω at one k.** At κ = 0 this gives 2βc sin k directly. With spin, use the mean of the two
  branches' Δω (§4). β̂ = βc/√K, with K from single-site spectroscopy.
- **Tuning.** Spin (PWM) sets κ in situ. Coil current sets K, uniformly or per site. Spacing and spring
  pre-stretch set c and the anisotropy.

## 7. Bottom line

- As built, the platform is the model's **large-κ̂ limit plus a J-breaking bond term** of the same size
  as c (springs) or larger (magnets, 5/3 in our idealisation). It has no β.
- A realisation faithful to the model needs three changes:
  - isotropic couplers, for the J-sector;
  - a new bond-rotor element, for β;
  - much slower spin, for κ̂ ≈ 0.65.
- The most direct tests of the model's genuine predictions are census rows 1–2 on the motors-off
  pendulum version with bond rotors. Row 3 is marginal and row 9 is out of reach.
