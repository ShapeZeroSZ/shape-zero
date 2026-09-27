# Target 2 — gravity through energy-responsive, energy-conserving couplings: scoping report

*Scoping only; nothing is built and nothing is adopted. Hypotheses and numerical predictions were
committed first: `GRAVITY_HYPOTHESES.md` (e3365d4). Checks: `gravity_scope_checks.py` →
`gravity_scope_checks_output.txt` (G1a, G1b, G2, G4). One diagnostic was added to G2 after the
first run and is marked as such.*

## Summary

- **β is exactly a uniform shift vector** coupled to the scalar sector's lattice momentum. That
  is the lattice form of the ADM shift coupling NⁱPᵢ, and it is what EM_SCOPE suggested.
  - A uniform shift carries **no gravitomagnetic field** (zero curl).
  - β acts on one sector only, so **it is not universal**.
- **Energy-responsive, energy-conserving couplings.** A dynamical **lapse** field Φ multiplying
  every sector's energy density, plus a dynamical shift, which is the existing velocity link made
  site-dependent.
  - Passivity does not constrain the lapse; J-compatibility is automatic.
  - The Newtonian limit gives universal attraction and an inverse-square force at q = 3.
- **The equivalence principle fails in the gyroscopic (κ) sector.** Measured before/after: the two
  chirality branches fall at **0.690 and 1.300** times the universal rate (predicted 0.691 and
  1.309). The cause is that κ is a rotating frame: lab energy and inertia disagree.
  - Restoring the equivalence principle means coupling gravity to rotating-frame energy — a
    selection by that fact.
- **β̂ does not disappear.** A dynamical shift is sourced by momentum, so a uniform β becomes a
  state (the lattice's drift relative to matter), calibrated by the same Δω. No input is saved.
- **The lump blocker.**
  - Any **softening** leading nonlinearity admits Q-balls. That is a premise change against P9.
  - Gravity **does not need it**: self-gravitating lumps exist under the adopted node form A′ for
    every G > 0.
- **Projected count: no improvement on the baseline.**

## 1. Is β a gravitomagnetic shift vector? (H1 — HELD in form, FAILED in one claim)

- **The coupling is exactly a shift coupled to lattice momentum.** The β force equals the
  Euler–Lagrange force of L_β = −βc·P, with P = −Σ u̇_n(u_{n+1} − u_{n−1})/2. The difference is 0
  to machine precision (G1b), on the lattice and not only at long wavelength. In Hamiltonian form
  this is H = H₀ + N·P with N = βc — the ADM structure. The continuum Lagrangian
  ½u̇² + Vu̇u_x − ½cu_x² (MODEL_SPEC §1c) is a scalar in the metric with lapse 1, shift V and
  γ^{xx} = c + V².
- **It couples to momentum, not charge (G1a — prediction HELD).** At k = π/2 the frequency shift
  is the same at every stiffness, while a u(1) phase's scales with the group velocity:

  | K | shift from β = 0.05 | shift from u(1) θ = 0.05 | θ·v_g |
  |---|---|---|---|
  | √5 | +0.0506 | +0.0241 | 0.0243 |
  | 4√5 | +0.0504 | +0.0151 | 0.0151 |
  | 16√5 | +0.0502 | +0.0081 | 0.0081 |

  The first-order β shift is cβ sin k = 0.0500 exactly; the remainder is O(β²). The Peierls angle
  arctan(βω), growing with frequency, is how a momentum coupling looks at fixed frequency. The
  stiffness independence of the pinned asymmetry (mission 3) is the same fact.
- **FAILED:** H1 called P "the Noether momentum of lattice translations". The lattice has only
  discrete translations, so P is a **pseudomomentum**, conserved in the linear lattice (drift
  1.4×10⁻⁸ over T = 200) but not with the φ-well (drift 1.0×10⁻²; G1b). The coupling form is
  exact; the conservation is linear-only.
- **Not yet gravitomagnetism.**
  - A uniform shift has zero curl and is a coordinate choice (MODEL_SPEC §1c: "a removable
    Galilean boost"). A gravitomagnetic field needs a site-dependent shift with curl, at q ≥ 2.
  - It is **not universal**: β acts on the scalar sector only. On spinning (κ) nodes the same link
    gives a κ-dependent Δω, and only a two-branch sum rule survives (`UNIVERSAL_RELATIONS.md`
    row 1 note). That has the same root as the equivalence-principle failure in §2.

## 2. Couplings that respond to local energy while conserving it (H2, H3 — HELD; one prediction of failure confirmed)

**The ingredient.**
- **A lapse field** Φ, with N = 1 + Φ, multiplies every sector's Hamiltonian density,
  H = Σ (1 + Φ_n) h_n. It has its own action (1/8πG)[Φ̇²/c_g² − (∇Φ)²]. At first order it
  couples to local energy density (−Φε), and total energy is conserved: Noether, and measured
  with Φ static (G2 drift ≤ 2×10⁻⁴).
- **The shift, made dynamical**, is the existing velocity link with a site-dependent value. With a
  scalar W it is passive in the literal form (mission 2). It must be **one** field on every
  sector's links.

**What the principles say.**
- **Passivity does not constrain the lapse**: it is a potential-type coupling, not a link.
- **J-compatibility is automatic**: the 𝕁 sector's energy density is phase-invariant.
- **Selections, counted:** a massless Φ (needed for a long-range force), coupling to T⁰⁰ rather
  than the trace, c_g = √c (one light cone), and one shift field for all sectors.

**The Newtonian limit.**
- A static Φ obeys ∇²Φ = 4πGε, so Φ = −GE/r and the interaction energy is −GE₁E₂/r:
  **attractive**, since a positive-energy scalar exchanged between same-sign sources attracts.
- **The inverse-square force holds at q = 3 only.** It is the same lattice Laplacian as EM_SCOPE's
  C4: 0.3% at r = 16; q = 2 gives 1/r, q = 1 a constant.

**Universal free fall (G2).** A packet at rest in Φ = g·x; acceleration in units of −c·g:

| sector | ω | measured | predicted |
|---|---|---|---|
| scalar, K = √5 | 1.495 | 0.9960 | 1.000 |
| scalar, K = 4√5 | 2.991 | 0.9926 | 1.000 |
| scalar, K = 16√5 | 5.981 | 0.9789 | 1.000 |
| gyroscopic a-branch, κ\* | 1.086 | **0.6895** | **0.691** |
| gyroscopic b-branch, κ\* | 2.058 | **1.3005** | **1.309** |

- **Scalar sector:** universal — packets of every stiffness fall alike.
  - The heavy packet's 2% shortfall scales as g², not with packet width (diagnostic, added after
    the first run: 0.9945 at half the gradient; 0.9791 at twice the width). The packet reaches
    k = gωT ≈ 0.36, where v = c·sin k/ω departs from ck/ω.
  - So universality is the small-velocity (Newtonian) statement, as it should be.
- **Gyroscopic sector: the equivalence principle is violated**, as predicted. The a-branch falls
  at 0.69 and the b-branch at 1.30 times the universal rate. Their mean is 1: a sum rule, since
  ω_a + κ/2 = ω_b − κ/2.
  - **Cause:** the lapse couples to lab energy ω, but a branch's inertia is set by
    dω/dk = ck/(ω ± κ/2). κ is a rotating frame (MODEL_SPEC §1c), and lab energy differs from
    rotating-frame energy by ±μ per unit phase charge.
  - **So a lapse coupling to lab energy is a fifth force proportional to the Larmor charge.**
    Nature forbids it (Eötvös-type tests). Coupling to rotating-frame energy instead restores
    universality, but that is a selection by the equivalence principle — counted, and it removes
    the "prediction".

## 3. Does β̂ become determined? (H4 — HELD)

- A dynamical shift is sourced by momentum density. In the linearised constraint, ∇×∇×N ∝ G·(momentum
  current).
- For matter with no net momentum the sourced shift vanishes, so a **uniform** β is not a
  coupling constant any more: it is a **boundary condition or state**, the lattice's drift
  relative to matter.
- **Its value is still calibrated by one Δω, exactly as now, so nothing supplied is saved.** β̂
  moves from the parameter list to the state list; the count does not fall.
- **New, genuine condition:** moving mass induces nonreciprocity (frame dragging), Δω = 2N sin k
  with N fixed by G and the momentum current. It counts only if the shift's coupling is tied to G
  by a Lorentz-type symmetry of the gravitational sector, which is a counted selection. Otherwise
  it adds a parameter.

## 4. The lump blocker (H5 — HELD)

- **Coleman's criterion** (G4, min_r 2V/r² − K):

  | node form | min 2V/r² − K | lumps |
  |---|---|---|
  | A′ (`main`): Kr²/2 + r³/3 | +0.0000 | none |
  | smooth, hardening: Kr²/2 + r⁴/4 | +0.0000 | none |
  | smooth, softening + sextic: Kr²/2 − r⁴/4 + r⁶/6 | −0.19 | **admitted** |
  | pendulum: K(1 − cos r) | −2.24 | **admitted** |

  **The change needed is a softening leading nonlinearity.**
- **The P9 note** (a smooth isotropic restoring force gives |ψ|²ψ at small amplitude) **does not
  by itself admit lumps.** Smoothness fixes the power, not the sign. A pendulum-like, softening
  sign admits Q-balls; a hardening one does not.
- **Conflicts with `main`:**
  - **P9 premise:** it replaces "a quadratic on-site force, applied radially" with a cubic
    (analytic) one — a premise change.
  - **Persistence as bounded motion:** a softening quartic alone is unbounded below and needs a
    stabilising sextic (bounded motion then holds). The pendulum's potential is bounded, but its
    radial coordinate runs away above V = 2K.
  - **Parameters:** it adds at least one (λ; σ as well if not fixed by units).
  - **Not in conflict:** phase conservation, J-compatibility, the D2 rung, κ\* (linear only) and
    P-3's zero self-precession (whole-node isotropy is kept).
- **Gravity does not need it.** Under A′ unchanged, a self-gravitating Gaussian lump has a bound
  minimum for every G tested (1 to 10⁻⁶), with R ∝ G⁻² (R ≈ 4.4, 1.5×10³, 8.4×10⁶, 8.4×10¹⁰).
  Weak gravity makes such lumps astronomically large on the lattice.
- **Gravity would also supply EM's static charges** where gravitational binding beats charge
  repulsion; not checked.

## 5. Projected count

| after | parameters | conditions | strict predictions | note |
|---|---|---|---|---|
| baseline (`main` 646a238) | 3 | ~6 | 4 (3) | inherited |
| target 2 as scoped, if built | 4 (G free; c_g selected; β̂ → a state) + ≥ 4 discrete selections | ~7 | 5 (frame dragging, only with the Lorentz-type selection) | **not built, not adopted** |

- **No improvement.** β̂ leaves the parameter list but returns as a calibrated state; G enters.
- The one new genuine condition, frame dragging, needs a selection to be a prediction at all, and
  its effect is suppressed by G.
- The equivalence principle, the inverse-square law and the universal redshift select the
  adaptation, so they count for nothing.
- The adaptation's one surprise — the equivalence-principle violation proportional to the Larmor
  charge — is a prediction that **nature falsifies**, unless it is removed by a selection.

## Hypotheses

| | statement (short) | status |
|---|---|---|
| H1 | β is exactly a uniform shift coupled to lattice momentum; momentum not charge; not yet gravitomagnetism | **HELD** (G1a, G1b exact coupling); **FAILED** in part: P is a pseudomomentum, conserved only in the linear lattice |
| H2 | lapse × energy density; passivity silent; J-compatibility automatic; shift = the link made site-dependent | **HELD** |
| H3 | Newtonian attraction, inverse-square at q = 3, universal free fall in the scalar sector; the equivalence principle violated in the κ sector at 0.691 / 1.309 | **HELD** — 0.9960 / 0.9926 / 0.9789 (velocity effect, diagnosed); 0.6895 / 1.3005 |
| H4 | β̂ becomes a state; no net reduction; frame dragging a condition only with a selection | **HELD** (analytic) |
| H5 | softening admits lumps; P9 conflict; gravity does not need it | **HELD** (G4) |
