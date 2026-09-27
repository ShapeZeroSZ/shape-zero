# CP-G pilot — predictions, and hypotheses for the ring-node fork

*Committed before `cpg_pilot.py` was written or run, and before `RING_FORK.md`. Pilot = hypothesis IO3
of `INSIDE_OUT_HYPOTHESES.md`: every field of `main` is gapped, so no local mechanism can carry a
long-range force.*

## Pilot predictions

**P1 — mode census from `main`'s own force.** Linearise `model.py`'s force about its vacuum (the
origin), numerically, on small periodic lattices. Then report the smallest |ω| over the Brillouin
zone per sector:

| sector (`model.py` configuration) | predicted smallest \|ω\| | where |
|---|---|---|
| J sector, n = 1, κ = κ\*, q = 1 | **1.0864** | a-branch, k = 0: ω_a = (−κ + √(κ² + 4√5))/2 |
| J sector, n = 2 and n = 3, κ\*, q = 1 | **1.0864** | same, degenerate over the n components |
| J sector, n = 1, κ\*, q = 3 (side 6) | **1.0864** | k = 0 |
| scalar β sector, β = 0.05, q = 1 | **1.4935** | near k = 0 on the slow branch, √(√5) lowered by the drift |
| κ = 0 nodes (the tower's n = 8 configuration), q = 1 | **1.4953** | √(√5) |

- **No gapless mode anywhere. The global gap is 1.0864.**
- **Static links** (the u(n) segments) are parameters, not degrees of freedom, so they carry no
  modes. The residual sector is off (C_r = 0 adopted).

**P2 — the range of any force they can carry.** A static source couples through the static response
1/Q(k), with Q = √5 + 2cΣ(1 − cos kᵢ). At ω = 0 the gyroscopic term drops out, so this holds for
every sector.
- **q = 3 Green's function along an axis:** G(r) ∝ e^{−r/ξ}/r, with
  **ξ = 1/arccosh(1 + √5/2) = 0.7232 sites**. A fit of ln(r·G) on r = 3…8 should give ξ within 3%.
- **Single-field exchange** is therefore Yukawa with range 0.72 sites. **Two-field
  (fluctuation-induced) exchange** is shorter: range ≤ ξ/2 = 0.36 sites.
- **Hence IO3 holds:** at r = 10 sites a static influence is below 10⁻⁵ of its value at r = 1.

## Ring-node fork — hypotheses (for `RING_FORK.md`; scoping only, not an adoption)

**R1 — the massless modes.** Form (B), the ring well with minimum on |ψ| = φ, breaks the node's
phase symmetry spontaneously.
- **n = 1:** one Goldstone, the phase mode, acoustic (type A), with v² = c√5/(√5 + κ²) = 0.703 at
  κ\* — `main`'s own §1a value.
- **Whole-node ring, n ≥ 2:** U(n) → U(n − 1) breaks 2n − 1 generators. The gyroscopic term gives
  the static ground state a non-zero canonical charge density (p = u̇ + (κ/2)𝕁u ≠ 0 at u̇ = 0).
  Watanabe–Murayama counting then gives **one linear (type A) mode plus n − 1 quadratic (type B)
  pairs**.
- **Per-dimer ring:** U(1)ⁿ broken, n type-A phase modes.
- **The amplitude (radial) mode is gapped.**

**R2 — what force they can carry.**
- **Goldstones are spin-0 with a shift symmetry, so they couple derivatively.** A static energy
  source does not source them linearly (Adler zero). It couples through the gapped radial mode
  (Yukawa), or at second order through two-Goldstone exchange, a power law ~r⁻⁷ at q = 3 (the
  Casimir–Polder form).
- **Long range only between phase-winding sources** — vortex lines — or moving currents
  (dipolar). That is a gauge-like force between defects, not an attraction between masses.
- **Kind: scalar, never tensor** (Weinberg–Witten and spin 0).
- Goldstone quanta also see an acoustic metric set by the background density (analogue gravity,
  kinematics only). But the density responds to energy only through the gapped radial mode, so a
  mass's metric disturbance decays within the radial mode's range.
- **So even the ring gives no inverse-square law between masses.**

**R3 — conflicts with `main`.**
- **§1a criteria:** fails "the D2 rung's isotropic, origin-centred node" (the origin is not an
  equilibrium) and "keeps κ\*" (no chirality branches); satisfies phase conservation and
  persistence.
- **Adopted results it reverses:** C34 ("only the radial form satisfies all four") and C43 (A′
  adopted).
- **Results built on the origin vacuum that would have to be redone or would lapse:**
  - the chirality branches and census row 5 (Larmor splitting), row 4 (the κ̂ floor) and row 12
    (the κ-gradient force);
  - P8 / J-compatibility's channel analysis — no opposite-chirality channel exists around a ring;
  - P-3;
  - gates 3, 7, 8 and 9;
  - joint #5's J-sector parts.
- **Unaffected:** the scalar β sector (rows 1–3), which keeps its per-component well.

**R4 — accounting.** Switching to (B) would be a premise change, selected by the fact "gravity needs
a massless mode", and so counted. By R2 it still does not deliver an inverse-square law. **It is
recorded as a fork, not an adoption: no route to CP-G.**
