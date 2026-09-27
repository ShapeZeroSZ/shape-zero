# Target 2 — gravity through energy-responsive, energy-conserving couplings: hypotheses

*Committed before any check for target 2 was written or run. Each is marked HELD, FAILED or OPEN
in `GRAVITY_SCOPE.md`. Numbers stated here are predictions.*

## (1) β as a shift vector

**H1 — β is exactly a uniform shift coupled to lattice momentum.** The β term derives from
L_β = −βc·P, where P = −Σ u̇_n (u_{n+1} − u_{n−1})/2 is the Noether momentum of lattice
translations in the scalar sector. So the coupling is N·P with shift N = βc — the ADM form
H = H₀ + NⁱPᵢ — exactly on the lattice, not only at long wavelength.
- The Peierls angle arctan(βω), growing with frequency, is how a momentum coupling looks at fixed
  frequency.
- **Discriminating test (G1).** At fixed k, β's frequency shift is cβ sin k + O(β²), independent
  of the stiffness K. A u(1) phase θ shifts a mode by v_g(K)·θ, which depends on K.
  - **Prediction:** at K = √5, 4√5 and 16√5, β's first-order shift at k = π/2 is 0.0500 (β = 0.05)
    in all three.
  - The u(1) shift changes by the ratio of group velocities, about 0.49 : 0.29 : 0.15 (to be
    computed).
- **But it is not yet gravitomagnetism.** A uniform shift is a coordinate choice with zero curl;
  a gravitomagnetic field needs a shift with curl, which requires spatially varying β at q ≥ 2.
  And β acts on the scalar sector only, so it is not universal.

## (2) Couplings that respond to local energy while conserving it

**H2 — the ingredient.** A dynamical lapse field Φ (N = 1 + Φ) multiplies every sector's
Hamiltonian density, H = Σ (1 + Φ_n) h_n, with its own action (1/8πG)[Φ̇²/c_g² − (∇Φ)²].
- Total energy is conserved (Noether), and the coupling responds to local energy density at first
  order (−Φ·ε).
- **Passivity does not constrain it.** It is a potential-type coupling, not a velocity link.
- **J-compatibility is automatic.** The energy density of the 𝕁 sector is phase-invariant.
- **Principles select neither** the massless Φ nor the coupling to T⁰⁰ rather than the trace.
  Both are selections, counted.
- **The shift, made dynamical**, is the existing velocity link made site-dependent. In the
  literal form with scalar W it is passive (mission 2). It must be one field on every sector's
  links to be universal.

**H3 — the Newtonian limit.**
- Static Φ obeys ∇²Φ = 4πGε, so between sources Φ = −GE/r. The interaction energy −GE₁E₂/r is
  attractive: a positive-energy scalar exchanged between same-sign sources attracts.
- **The inverse-square force holds at q = 3 only.** It is the same lattice Laplacian as the EM
  scoping's check C4.
- **Universal free fall (G2).**
  - **Scalar sector:** a packet at rest in Φ = g·x accelerates at −c·g, independent of K. The
    eikonal gives dk/dt = −gω, v = ck/ω.
  - **Prediction:** K = √5, 4√5 and 16√5 fall identically, at a/(−cg) = 1.000.
  - **Gyroscopic (κ) sector:** on branch s the group velocity is ck/(ω + sκ/2), so a/(−cg) =
    ω/(ω + sκ/2).
  - **Prediction at K = √5, κ = κ\* = 0.9717:** a-branch (ω = 1.0864) **0.691**, b-branch
    (ω = 2.0582) **1.309**. Their mean is exactly 1, a sum rule.
- **So coupling to lab energy violates the equivalence principle** in proportion to the Larmor
  charge. The cause is that κ is a rotating frame (MODEL_SPEC §1c), so lab energy differs from
  rotating-frame energy by ±μQ.
  - Restoring the equivalence principle would require coupling to rotating-frame energy — a
    selection by the fact of the equivalence principle, counted, and not a prediction.

**H4 — β̂ becomes a state, not a parameter.**
- A dynamical shift is sourced by momentum density: in the linearised constraint, ∇×∇×N ∝ G·(momentum
  current).
- A uniform N with no net momentum of matter is then a boundary condition — the lattice drifting
  relative to matter.
- So **β̂ leaves the list of coupling constants**, but its value, calibrated by one Δω exactly as
  now, becomes a state variable. **No net reduction in what must be supplied.**
- **New genuine condition:** nonreciprocity induced by moving mass (frame dragging) —
  Δω = 2N sin k with N fixed by G and the momentum current. It is a genuine condition only if
  the shift's coupling is tied to G (by a Lorentz-type symmetry of the gravitational sector,
  a counted selection); otherwise it adds a parameter.

## (3) The blocker — localised lumps

**H5 — what admits lumps, and whether gravity needs them.**
- **Coleman's criterion** (min_r 2V/r² < K) holds for any softening leading nonlinearity. Two
  examples:
  - the pendulum form V = K(1 − cos r), bounded;
  - V = Kr²/2 − λr⁴/4 + σr⁶/6 with σ > 0.
- **A smooth isotropic restoring force** gives |ψ|²ψ at that order (the P9 note). With the
  softening sign, as for a pendulum, it admits Q-balls.
- **Conflicts with `main`:**
  - it replaces P9's quadratic radial force, a premise change;
  - persistence as bounded motion needs stabilisation (bounded V or a sextic);
  - phase conservation, J-compatibility, the D2 rung, κ\* and P-3's zero precession (whole-node
    isotropy) are unaffected.
- **Gravity does not need it.** With node form A′ unchanged, a self-gravitating lump — the
  Schrödinger–Newton analogue — exists for every G > 0. In a Gaussian ansatz
  E(R) = aN/R² + bN^{3/2}R^{−3/2} − GN²/R, and the last term dominates at large R, so E has a
  minimum.
- **Gravity also supplies EM's static charges** (charged self-gravitating lumps) where
  gravitational binding beats charge repulsion.

## (4) Projected count

Parameters: +G (free), +c_g (selected, = √c by Lorentz invariance), β̂ moved from coupling to state.
Conditions: frame dragging (+1, only with the selection above). The equivalence principle, the
inverse-square law and the universal redshift are selecting facts, so +0.

**Projection:**

| | before | after |
|---|---|---|
| parameters | 3 | 4, plus one state and ≥ 3 discrete selections |
| conditions | ~6 | ~7 |
| strict predictions | 4 | 5 |

That is **no real improvement on the baseline.** The gyroscopic-sector violation of the equivalence
principle is a prediction that nature falsifies, unless it is removed by a selection.
