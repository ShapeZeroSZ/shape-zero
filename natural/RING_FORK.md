# The ring-node fork — the one inside route the pilot leaves (FORK, not adopted)

*Scoping only; nothing built. Hypotheses R1–R4 were committed first, in `CPG_PILOT_PREDICTIONS.md`
(45c4c8f). Statuses here are analytic, with `main`'s own recorded values cited where they exist; none
is newly measured.*

## Why this is the one route left

- **The pilot** (`cpg_pilot.py` → `cpg_pilot_output.txt`) found **no gapless mode in `main`**. The
  smallest |ω| over every sector is 1.0864: the J-sector a-branch at k = 0, identical for n = 1, 2, 3
  and at q = 3. The scalar β sector bottoms at 1.4935 and the κ = 0 nodes at 1.4953.
- **Main's static operator is exactly Q** (to 2×10⁻⁹). A static source's influence decays as
  e^{−r/ξ}/r with ξ = 0.71 sites (predicted 0.723, −2%); by r = 10 it has fallen to 4×10⁻⁷.
- **So IO3 holds:** within `main`, no local mechanism carries a long-range force, and CP-G cannot pass
  the two-body test.
- **The escape:** a massless mode must come from somewhere. By Goldstone's theorem, a ground state
  that breaks a continuous symmetry has one. Among the node forms `main` considered, only the ring,
  form (B), breaks one.

## R1 — the massless modes a ring node would carry

- **n = 1 (the dimer ring).** One Goldstone, the phase around the ring. It is acoustic (type A),
  with **v² = c√5/(√5 + κ²) = 0.703 at κ\*** — `main`'s own value (MODEL_SPEC §1a, the (B) row).
  The amplitude mode across the ring is gapped.
- **Whole-node ring, n ≥ 2** (the well on the node's radius, as in A′):
  - U(n) → U(n − 1) breaks 2n − 1 generators.
  - The gyroscopic term gives the static ground state a **non-zero canonical charge density**
    (p = u̇ + (κ/2)𝕁u ≠ 0 at u̇ = 0), so broken charges fail to commute in expectation.
  - Watanabe–Murayama counting then gives **one linear (type A) mode plus n − 1 quadratic (type B)
    modes**, ω ∝ k² — as in a ferromagnet.
  - *Analytic; not checked here.*
- **Per-dimer ring** (the well on each dimer's own radius, as `main` defined (B)): U(1)ⁿ broken, n
  type-A phase modes.

## R2 — the range and kind of force they could mediate

**Scalar (spin 0), never tensor.** Goldstones have spin 0; Weinberg–Witten forbids a composite
massless spin-2 in the Lorentz-invariant infrared in any case.

**Coupling is derivative (shift symmetry, Adler zero).** So a static energy source does not source a
Goldstone linearly. What a static mass can do:
- act through the **gapped amplitude mode**: Yukawa, short range;
- act at **second order**, through two-Goldstone exchange: a power law ~r⁻⁷ at q = 3 (the
  Casimir–Polder form) — **long range but far weaker than 1/r, and not inverse-square.**

**Long range at first order — only for sources that couple to the phase itself:**
- phase windings (vortex lines): a logarithmic interaction per unit length, like parallel line
  currents — a gauge-like force between defects;
- moving currents: dipolar, ~r⁻³.

**Kinematics only: an acoustic metric.** Goldstone quanta propagate on an effective metric set by the
background density and flow (Unruh's analogue gravity).
- But the background density responds to energy only through the gapped amplitude mode, so a mass's
  disturbance of that metric decays within the amplitude mode's range.
- **There is no long-range metric sourced by energy.**

**Verdict: even the ring gives no inverse-square attraction between masses.** It gives a massless
scalar with gauge-like defect forces and analogue kinematics.

## R3 — exactly which principles and results it conflicts with

**§1a node-form criteria** (`main`'s own table):

| criterion | ring (B) |
|---|---|
| phase conservation at all orders | **yes** (U(1)-symmetric) |
| persistence as bounded motion (Formal Proofs) | **yes** (max \|u\| = 4.84) |
| the D2 rung's isotropic, origin-centred node | **no** — "the origin is not an equilibrium (unit outward force), so L = 0 motion cannot reach it" |
| keeps κ\* | **no** — "the phase mode is massless …; no chirality branches" |

**Adopted results it reverses:**
- **C34**: "only the radial node form satisfies the four criteria";
- **C43**: node form A′ adopted.

**Built on the origin vacuum, so it would lapse or have to be redone:**
- **The chirality branches** — the a and b branches do not exist around a ring. Therefore:
  - census **row 5** (the Larmor splitting ω_b − ω_a = κ);
  - **row 4** (the κ̂ floor κ̂\*, and with it C14);
  - **row 12** (the κ-gradient force, merged into `main` today);
  - **P-3** (no self-precession, row 8).
- **P8, J-compatibility required at every wavelength.** Its derived consequence, κ ≥ κ\*, is a
  channel analysis between opposite-chirality branches. Around a ring there is no such channel, so
  the principle would need restating.
- **The gates:** 3 (chirality purity), 7 and 8 (the ordering splittings and Abelian floors, computed
  on chirality-branch dispersion) and 9 (the cone state, whose radial/angular split assumes the
  origin as apex, MODEL_SPEC §0a).
- **Joint #5's J-sector parts.**

**P9, the φ-well.** (B) is P9 "read literally" on |ψ|, so it is consistent with the φ-well as a force
law, but not with the radial application `main` adopted.

**Unaffected:** the scalar β sector — census rows 1–3, the pinned asymmetry and joint #3 — which keeps
its per-component well; missions 1–6 (structural); P0.

## R4 — accounting, and the status of the fork

- **Adopting (B) would be a premise change** (reversing C34 and C43), selected by the physical fact
  "gravity needs a massless mode" — a counted selection.
- **By R2 it still would not deliver an inverse-square law between masses**, so it buys the
  selection without the result CP-G needs.
- **Recorded as a FORK, not an adoption.** No branch is built on it.
- **What would reopen it:** a mechanism by which the ring's Goldstone couples non-derivatively to
  energy density. That is forbidden by the shift symmetry that makes it massless, so no such
  mechanism is expected.

## Consequence for CP-G

**CP-G fails its own test.**
- Inside `main` there is no massless mode (pilot).
- The one inside route that supplies one (the ring) gives a scalar, derivatively coupled Goldstone
  with no inverse-square force between masses, and costs `main` its chirality branches, κ\*, and the
  results built on them.
- **"Gravity is a consequence, not an ingredient" is not supported on this lattice.** Gravity would
  have to be an ingredient (targets 2 and 3), and the inside-out GPU program would be a formality.
