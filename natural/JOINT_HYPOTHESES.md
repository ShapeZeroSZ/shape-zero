# Target 3 — electromagnetism and gravity together: hypotheses

*Committed before any check for target 3 was written or run. Marked HELD, FAILED or OPEN in
`JOINT_SCOPE.md`. Numbers here are predictions.*

## (a) Shared selections and the joint parameter count

**H1 — the targets share a wave speed and the base, and gauge invariance removes one gravity
selection.**
- **Shared:**
  - one universal wave speed for photons, gravity and matter (c_γ = c_g = √c), one selection
    where the separate scopings counted two;
  - the q = 3 base, which both need for an inverse-square force.
- **Removed:** the gravity scoping's equivalence-principle selection, "couple to rotating-frame
  energy". With a dynamical u(1), κ is a uniform electrostatic potential eA₀ = κ/2 (EM_SCOPE §4).
  The only gauge-invariant energy density is then ½|D_t ψ|² + …, which **is** the
  rotating-frame energy. Gravity must couple to a gauge-invariant energy, so the choice is
  forced, not selected.
- **Joint numeric parameters:** baseline 3 − β̂ (to a state) + e + G + one shared speed =
  **5**, against the separate sum 3 + 2 (EM) + 1 (gravity) = **6**.
- **Joint discrete selections:**
  - Wilson form; q = 3 (shared); massless Φ; T⁰⁰ rather than the trace; one shift for all
    sectors; one lapse for all sectors, matter and field.
  - Frame dragging's Lorentz-type tie optional.
  - **6 (+1)**, against the separate 2 + 5 (+1) = **7 (+1)**.

## (b) Field energy as a source, and equal fall of matter and field

**H2 — electromagnetic field energy sources gravity only by selection, and light falls with
matter only if the wave speed is shared.**
- If the lapse multiplies every sector's energy density, including the link sector's electric and
  plaquette energy, field energy gravitates. Energy conservation alone does not force this: a
  lapse on matter only is also Noether-conserving. It is part of the one "all energy" selection.
- **Free fall, in the eikonal:**
  - a matter packet (ω² = K + ck²) accelerates across a lapse gradient at −c·g, at rest or
    moving;
  - a massless photon packet (ω = c_γ|k|) is deflected at −c_γ²·g, independent of its speed
    along the gradient's normal.
  - So **light and matter fall alike iff c_γ² = c.** The equivalence principle and the shared
    wave speed are the same selection.
- **Predictions (J1)** — transverse acceleration a/(−cg):

  | case | predicted |
  |---|---|
  | massive at rest | 1.000 |
  | massive moving (kₓ = 0.6) | 1.000 |
  | massless, c_γ² = c, kₓ = 0.6 | 1.000 |
  | massless, c_γ² = 1.5c | **1.500** |

- **Predictions (J2)** — with the lapse coupled to the gauge-invariant (rotating-frame) energy,
  H = Σ(1 + Φ)ε_inv − μΣQ (the charge term, A₀ times the Gauss constraint, is not
  lapse-weighted, as in ADM with Maxwell): the κ\* a- and b-branches both fall at **1.000**.
  Target 2 alone gave 0.691 / 1.309.

## (c) Self-gravitating charged lumps

**H3 — charged lumps exist iff gravity beats the Coulomb repulsion per quantum, at no extra
parameter cost.**
- A lump's energy and charge densities have the same profile (both ∝ |ψ|² at leading order), and
  both long-range forces use the same Green's function. So the 1/R terms combine into
  (e²/4π − Gω²)·N²·I/R, with the same shape factor I, independent of the ansatz.
- **Bound lumps exist iff λ = e²/(4πGω²) < 1.**
  - For λ > 1 every term in the energy (kinetic, the hardening A′ term, net electrostatic) is
    non-negative, so no lump exists.
  - Neutral lumps (a/b-balanced) exist for any G > 0.
- **Parameter cost: none** beyond e and G. It is a condition on their ratio.
- **But nature's charged constituents have λ ~ 10⁴⁰**, so in the natural regime charged lumps do
  not exist and only neutral ones do. EM's static-charge blocker is lifted only for an unnatural
  ratio.
- **Prediction (J3):** a Gaussian ansatz is bound for λ = 0.5 and 0.9 and unbound for λ = 1.1
  and 2.

## (d) Projected joint count

**H4.**
- **Joint:** 5 numeric parameters + 6 (+1) discrete selections; about 9 conditions.
- **Strict predictions**: baseline 4 + the lattice photon dispersion shape + the a/b equal fall
  (if gauge invariance forces it, J2) + frame dragging (only with its selection) = **6 (7)**.
- **Against the baseline (3 parameters, 4 strict):** the joint count beats the sum of the
  separate counts, but it **does not beat the baseline** once the discrete selections are counted,
  as the charter requires. On numeric parameters alone the strict ratio is 6/5 against 4/3.

---

## LATE HYPOTHESIS L1 (added after J1–J3 ran; committed before check J4 was written or run)

*The file cannot contain its own commit, so this block's commit hash and time are recorded in
`JOINT_SCOPE.md`. The original hypotheses above were committed at d054844
(2026-09-27T18:53:50Z), and J1–J3 ran and were reported at ca6a63c (2026-09-27T19:02:00Z).*

**Statement (as requested).** In the joint theory, gauge invariance requires gravity to couple to
the gauge-invariant energy, not the lab-frame energy. So the rotating-frame coupling that removes
the gyroscopic equivalence-principle violation is **forced, not selected**.
- This was already stated as H1 ("forced") and tested as J2, with predictions committed at
  d054844 and results at ca6a63c: 0.9959 / 0.9959. **J2 shows the outcome, not the reason.**

**What is new here — a test of the reason (J4).** In the joint theory a uniform κ is a uniform
potential eA₀ = κ/2. Moving to a frame rotating at rate ν is then a gauge transformation: it moves
ν of the frame rotation into A₀. The gyroscopic term becomes κ_ν = κ − 2ν and the stiffness
K_ν = K + νκ − ν², with K_ν + κ_ν²/4 = K + κ²/4 unchanged.
- **A coupling to "lab energy"** means the energy in whatever frame the lapse is attached to. In
  frame ν it gives

  a/(−cg) = (ω_rot ∓ (κ/2 − ν))/ω_rot,  ω_rot = √(K + κ²/4) = 1.5723,

  **which depends on ν, i.e. on the gauge.** A physical prediction that depends on the gauge is
  inconsistent, so gauge invariance rules that coupling out.
- **The gauge-invariant coupling** gives the same result, 1.000, in every gauge.

**Predictions (J4)** — κ = κ\*, K = √5, a/(−cg):

| ν | lab-type coupling, a-branch / b-branch | gauge-invariant coupling, both |
|---|---|---|
| −κ/2 | 0.382 / 1.618 | 1.000 |
| 0 | 0.691 / 1.309 | 1.000 |
| κ/4 | 0.846 / 1.155 | 1.000 |
| κ/2 | 1.000 / 1.000 | 1.000 |

All within the scalar sector's accuracy at this setting (0.996).

**Count prediction:** the joint count has one fewer discrete selection than the separate counts
(already H1: 6 (+1) against 7 (+1)). It is justified only if J4 holds — only if the coupling is
forced rather than chosen.
