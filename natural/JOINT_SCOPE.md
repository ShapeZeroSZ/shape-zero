# Target 3 — electromagnetism and gravity together: scoping report

*Scoping only; nothing is built and nothing is adopted. Hypotheses and numerical predictions were
committed first: `JOINT_HYPOTHESES.md` (d054844). Checks: `joint_scope_checks.py` →
`joint_scope_checks_output.txt` (J1–J3). One diagnostic was added to J1 after the first run and is
marked as such. The joint count is now the branch's primary measure (charter, target 3).*

## Summary

- **Together, the two targets cost less than apart, and the combination repairs target 2.**
  - They share one wave speed and the q = 3 base.
  - Gauge invariance **forces** gravity to couple to the rotating-frame (gauge-invariant) energy.
    That removes target 2's equivalence-principle selection, and **the κ-sector violation
    disappears**: both chirality branches fall at 0.9959, against 0.6895 / 1.3005 when coupled to
    lab energy.
- **Light and matter fall alike iff the photon speed equals the matter speed.** The equivalence
  principle and the shared wave speed are one selection.
- **Charged self-gravitating lumps exist iff λ = e²/(4πGω²) < 1**, at no parameter cost. Nature's
  charged constituents have λ ~ 10⁴⁰, so the static-charge blocker is lifted only in an unnatural
  regime. Neutral lumps always exist.
- **The joint count beats the sum of the separate counts, but not the baseline** once the discrete
  selections are counted, as the charter requires.

## (a) Shared selections, and the joint parameter count (H1 — HELD)

- **Shared.**
  - One wave speed c_γ = c_g = √c — one selection where targets 1 and 2 counted two. (b) shows it
    is also exactly the equivalence principle between light and matter.
  - The q = 3 base, which both need for an inverse-square force.
- **Removed by the combination:** target 2's selection "couple gravity to rotating-frame energy".
  - With a dynamical u(1), a uniform κ is a uniform potential eA₀ = κ/2 (EM_SCOPE §4). The
    gauge-invariant energy density is ½|D_tψ|² + ½(K + κ²/4)|ψ|² + …, which is the rotating-frame
    energy.
  - Gravity must couple to a gauge-invariant energy, and the charge term A₀·ρ multiplies the Gauss
    constraint, not the lapse (as in ADM with Maxwell). So the choice is **forced**.
  - J2 confirms the consequence (b).

| | numeric parameters | discrete selections |
|---|---|---|
| target 1 alone | 3 + c_γ (selected) + e = 5 | Wilson form, q = 3 |
| target 2 alone | 3 − β̂ (to a state) + G + c_g (selected) = 4 | massless Φ, T⁰⁰ not trace, one shift for all sectors, one lapse for all sectors, rotating-frame energy (+ frame-dragging tie) |
| **sum of separate** | **6** | **7 (+1)** |
| **joint** | ĉ, κ̂, e, G, one shared speed = **5** | Wilson, q = 3 (shared), massless Φ, T⁰⁰, one shift, one lapse — the rotating-frame choice is forced — **6 (+1)** |

## (b) Field energy as a source; do matter and field fall alike? (H2 — HELD)

- **Field energy sources gravity only by selection.** If the lapse multiplies the link sector's
  electric and plaquette energy as well as matter's, field energy gravitates. Energy conservation
  alone does not force this: a lapse on matter only is also Noether-conserving. It is the one
  "all energy" selection.
- **Light and matter fall alike iff the wave speed is shared** (J1). Transverse acceleration in a
  lapse gradient on a 2-D lattice, in units of −c·g:

  | case | measured | predicted |
  |---|---|---|
  | matter at rest (K = √5) | 0.9915 | 1.000 |
  | matter moving (kₓ = 0.6) | 0.9918 | 1.000 |
  | light, one lattice-Maxwell polarisation, c_γ² = c | 0.9761 | 1.000 |
  | light, c_γ² = 1.5c | 1.4621 | 1.500 |

  - The two light cases stand in ratio 1.498, against the predicted 1.5: light falls at c_γ²·g,
    matter at c·g.
  - Light's ~2% shortfall at the shared speed is a finite-width effect of the massless packet
    (diagnostic, added after the first run). Widening the packet from σ = 12 to 20 moves light
    from 0.976 to 0.990; halving g barely moves it (0.978). At σ = 20, light/matter = 0.994–0.995.
- **So light falls with matter only if the photon speed equals the matter speed.** The equivalence
  principle between light and matter and the single wave speed are the same selection. It is
  counted once, and neither is a prediction.
- **The κ sector with the gauge-invariant lapse** (J2): the a-branch falls at **0.9959** and the
  b-branch at **0.9959**, matching the scalar sector's 0.996 at K = √5, against 0.6895 / 1.3005
  with the lapse on lab energy. Energy drift ≤ 3×10⁻⁵.
  - **The equivalence-principle failure found in target 2 is an artefact of scoping gravity without
    electromagnetism.**
  - Because gauge invariance, not the equivalence principle, forces the coupling, the equal fall of
    the two chirality branches is a **consequence**. Under charter rule 3 it counts as a condition
    satisfied — though what it reproduces is a known fact.

## (c) Self-gravitating charged lumps (H3 — HELD)

- **The criterion.** The energy and charge densities of a lump have the same profile, and both
  long-range forces use the same Green's function, so their 1/R terms combine into
  (e²/4π − Gω²)·N²·I/R.
- **Bound lumps exist iff λ = e²/(4πGω²) < 1.** A Gaussian ansatz under node form A′, at Gω² = 0.01
  (J3):

  | λ | E_min | lump |
  |---|---|---|
  | 0.5 | −1.5×10⁻⁷ at R = 4.7×10³ | bound |
  | 0.9 | −1.5×10⁻⁹ at R = 9.1×10⁴ | bound |
  | 1.1 | > 0 | none |
  | 2.0 | > 0 | none |

  For λ > 1 every term in the energy is non-negative (kinetic, the hardening A′ term, net
  electrostatic), so no lump exists in any ansatz.
- **Where Coleman's criterion fails without gravity, gravity supplies lumps:** neutral ones
  (a/b-balanced) for every G > 0 (GRAVITY_SCOPE §4), and charged ones for λ < 1.
- **Cost in parameters: none** beyond e and G. It is a condition on their ratio.
- **But nature's charged constituents have λ ~ 10⁴⁰**, so natural charged lumps do not exist, and
  EM's static-charge blocker is lifted only for an unnatural ratio. Static **gravitational** sources
  are always available.

## (d) Projected joint count (H4 — HELD)

| after | numeric parameters | discrete selections | conditions | strict predictions | note |
|---|---|---|---|---|---|
| baseline (`main` 646a238) | 3 | — | ~6 | 4 (3) | inherited |
| target 1 alone | 5 | 2 | ~7 (8) | 5 (6) | projected |
| target 2 alone | 4 + a state | ≥ 5 | ~7 | 5 | projected |
| **joint (targets 1 + 2)** | **5** + a state | **6 (+1)** | **~9 (10)** | **6 (7)** | projected, **primary measure** |

- **Strict predictions, joint:** the baseline's 4, plus the lattice photon dispersion shape, plus
  the equal fall of the chirality branches (forced by gauge invariance, J2). Frame dragging adds
  one more only with its Lorentz-type selection.
- **Not counted:** light-matter equal fall (it selected the shared speed), the inverse-square law
  (it selected q = 3), and the charged-lump criterion (an inequality with no natural instance).
- **Against the sum of the separate targets:** one fewer numeric parameter, one fewer discrete
  selection, one more strict prediction. **The joint count beats the separate counts.**
- **Against the baseline:**
  - On numeric parameters alone the strict ratio is 6/5 = 1.2 (7/5 with frame dragging), against
    the baseline's 4/3.
  - With the discrete selections counted, as charter rule 2 requires, the joint model has 11 (12)
    inputs against ~9 conditions: **fitting level, below the baseline.**
- The combination is internally better, the equivalence-principle repair being the main gain, but
  **it does not yet raise the evidence above what `main` has without it.**

## Hypotheses

| | statement (short) | status |
|---|---|---|
| H1 | shared speed and q = 3; gauge invariance forces the rotating-frame coupling; joint 5 numeric / 6 (+1) discrete against the separate 6 / 7 (+1) | **HELD** |
| H2 | field energy gravitates by selection; light falls at c_γ²g, so equal fall iff c_γ² = c; the gauge-invariant lapse gives the κ\* branches 1.000 / 1.000 | **HELD** — 0.9915 / 0.9918 / 0.9761 / 1.4621 (light's shortfall a finite-width effect, diagnosed); 0.9959 / 0.9959 |
| H3 | charged lumps iff λ < 1, at no parameter cost; natural λ ~ 10⁴⁰ | **HELD** — bound at 0.5, 0.9; none at 1.1, 2 |
| H4 | the joint count beats the separate sum but not the baseline with selections counted | **HELD** |
