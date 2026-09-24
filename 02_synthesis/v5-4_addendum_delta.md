# v5.4 Addendum — Delta for Consolidation

**To the author of `shape_zero_v5-4_addendum.md`.** Four insert-ready items
plus a retraction block, keyed to your existing section numbers. Nothing here
replaces existing text except where marked REPLACE. Every number regenerates
from a named script in the accompanying `shape_zero_c1s_package.zip`.

Provenance note: this material was produced in one long session working from
the C1 package (54-file version, with `z1_*` rung scripts, `shape_zero_open_threads.md`,
`shape_zero_zero_ladder.md`). The session's correction record is item 5 and
should be read before weighting anything in items 1–4.

---

## ITEM 1 — INSERT into §6.2 (Metric structure: what is banked and what is not)

Insert after the paragraph beginning "Metric compatibility of the connection
is already established":

> **Two metric scales move from CHOSEN to FORCED.** Schur's lemma settles
> *which* invariant structure a rung carries; it is silent on size. The
> embedding settles size, and both algebras are normalised by 1·1 = 1 with
> the composition law, which forces |1| = 1.
>
> At D4, the orbit of orthogonal complex structures under SO(4) has tangent
> {[X, J₀] : X ∈ u(2)^⊥}, and
>
>     ‖[X, J₀]‖ / ‖X‖ = 2.000000     (std 1.2×10⁻¹⁶ over 400 directions)
>
> with the stabiliser giving 1.8×10⁻¹⁵ and the complement 2-dimensional as S²
> requires. Since the structure sphere is Kähler — ω(u,v) = g(Ju,v), with
> J² = −I leaving no scale freedom — fixing the metric fixes the symplectic
> form.
>
> At D8, the orbit of the structure constants under Spin(7) has tangent
> {X·c : X ∈ g₂^⊥}, and
>
>     ‖X·c‖ / ‖X‖ = 6.000000         (std 5.8×10⁻¹⁶ over 400 directions)
>
> with derivations giving 1.3×10⁻¹⁴, confirming the split. The constant is
> not arbitrary: ‖c‖² = 42 = 7 lines × 6 orientations, and 42/7 = 6 is the
> per-line multiplicity. The scale counts Fano structure.
>
> *Scripts:* `z1_d4_structure_sphere.py`, and the orbit computation in the
> session log.

**Also insert, as a bounded negative result** (this closes the ℏ half of B-4
without deriving ℏ):

> **What geometric quantisation gives, and what it does not.** The structure
> sphere is a closed 2-cycle, so prequantisation's integrality condition
> applies. The isotropy action leaves exactly one invariant antisymmetric form
> (ε, invariant to 2.2×10⁻¹⁶) and exactly one invariant symmetric form, with
> the traceless symmetric form failing invariance at 2.0 as a control.
> Integrality then **discretises** the single remaining scale — flux must be an
> integer multiple of 2πℏ. It does **not** produce a value for ℏ: ℏ is the
> unit, not the output. The honest entry is that the route removes the
> *continuum* of symplectic normalisations, leaving a discrete tower.

---

## ITEM 2 — NEW SUBSECTION, suggested §6.3 (The signature bound)

> ### 6.3 The signature bound: the ladder is Euclidean by necessity
>
> Hurwitz's theorem classifies two families, not one. Both satisfy the
> composition law |xy| = |x||y| — verified to 2.8×10⁻¹⁴ in each — and both
> occur in dimensions 1, 2, 4, 8:
>
> | dim | division (definite) | split (indefinite) |
> |---|---|---|
> | 2 | ℂ | split-complex, j² = +1 |
> | 4 | ℍ | split-quaternions, sig (2,2) |
> | 8 | 𝕆 | split-octonions, sig (4,4) |
>
> **Persistence makes the choice, and it is not a preference.** The conserved
> norm's level set is the orbit's confinement surface. A definite norm gives a
> sphere — compact, bounded motion, and Lemma-2.1-type periodicity applies. An
> indefinite norm gives a hyperboloid — non-compact, with orbits escaping along
> null directions. Measured on split-quaternions: **431 null vectors in 20,000
> samples** against **0** for quaternions, and explicit zero divisors,
> (1 + j)(1 − j) = 0 exactly with both factors nonzero.
>
> **Wick rotation exits the ladder.** Rotating two imaginary units of ℍ sends
> e² = −1 to f² = +1, yielding signature (2,2), 206 null vectors in 20,000
> samples, and a hyperboloid level set — precisely the branch persistence
> excluded. So the ladder is not an incomplete Lorentzian theory awaiting
> continuation; there is no Lorentzian theory to continue *to*. Euclidean
> signature, boundedness, compactness, the integer lattice and persistence are
> one package: remove any and the rest go.
>
> **Scope consequence, stated plainly.** The ladder cannot describe
> propagation. Null geodesics, light cones, causal structure and gravitational
> lensing all require indefinite signature and lie outside it. What it can
> describe is bounded, recurrent, structural content — spectra, algebras,
> orbits, invariants — which is what the verified results are.
>
> *Related:* the Euclidean/thermal correspondence is exact at the level of the
> apparatus (Matsubara mode sum equals the closed thermal correlator to
> 5.7×10⁻¹², KMS holds at 1.7×10⁻¹⁶, the β→∞ limit reproduces the vacuum
> correlator). But **ω is not a temperature**: a single D1 orbit is
> microcanonical, its position density the arcsine law peaked at the turning
> points (centre/edge = 0.29), while the canonical distribution is peaked at
> the centre (1.12) at every temperature — qualitatively opposite, best fit
> running to the scan boundary with an 84% residual. What survives is the
> stronger statement: **D1 derives the periodicity the thermal formulation
> imposes.**

---

## ITEM 3 — REPLACE the closing paragraph of §3.1 (Ladder verdict)

Your text currently reads: *"Remaining CHOSEN after the ladder: the values of
the constants (δ, c, β, κ), the lattice topology, and the overdamped sector's
explicit void."*

That list is correct and should stand. Add this refinement after it:

> **Refinement on κ (and its D8 analogue).** Plurality licenses the
> *existence* of the gyroscopic coupling without fixing its value. Measured:
> at κ = 0 the node has exactly **one** frequency; at κ > 0 exactly **two**,
> split by exactly κ — 0.0524, 0.1518, 0.3979, 1.0001 against κ = 0.05, 0.15,
> 0.40, 1.00, which independently reconfirms the exact-Larmor result. Since
> plurality states there is more than one dynamical degree of freedom, κ = 0
> is excluded by principle rather than by choice, and minimality's exemption
> clause then admits a nonzero κ. **Its value remains CHOSEN**; only its
> non-vanishing is forced. The same structure holds at D8 for the relative
> orientation of two transport generators.

---

## ITEM 4 — INSERT into the B-2 entry (open threads: metric parameter derivation)

> **B-2 advanced, not closed.** The "phase-boundary self-consistency
> condition" resolves to a boundary the spec already names. Shape space
> P(Ω) under the Hellinger embedding μ ↦ √(dμ/dx) lands on the **unit sphere**
> (verified to 6.7×10⁻¹⁶ across n = 2, 3, 7, 50), and measures with disjoint
> support are orthogonal, so its diameter is exactly **π/2** — which is the
> **π/2 horizon** identified in the spec as the transport–reaction crossover
> and the building/falling channel boundary. The phase boundary and the cone's
> angular extent are the same object.
>
> Taking the cone's angular extent to be the shape-space extent gives the
> candidate
>
>     β = (π/2) / (2π) = 1/4
>
> **The gap, stated as the closing condition.** This identification requires
> the cone's angular coordinate to be a *closed circle* of circumference π/2.
> P(Ω) is a simplex, and its Hellinger image is a spherical simplex **with
> boundary**; a cone over a manifold-with-boundary is a wedge, not a deficit
> cone, and the deflection law π(1/β − 1) assumes a closed angle. B-2 closes
> when that identification is either justified or replaced. `g3_cone_lensing.py`
> currently uses β = 0.95 as a display value demonstrating impact-independence,
> not as a derived quantity.

---

## ITEM 5 — RETRACTION BLOCK (suggested for §7, Revisions to v5.3)

> **Retracted from the accompanying session material.**
>
> 1. **The sigma-model term census is not about this ladder.** A body of work
>    computing invariant terms for a field ψ(x) on a base manifold — a
>    22-dimensional space of cubic four-derivative invariants, 8 divergences,
>    14 entering the field equations, a 6-versus-14 sector split under Fano
>    confinement — is mathematically sound but presupposes a **base space no
>    rung has**. `z1_d4_rung.py` is single-state dynamics in ℝ⁴ with no lattice
>    index; `z1_d8_attempt.py` is algebra only. The 2-dimensional base
>    originally used came from `phi_gauge_wilson.py`, which is the *platform*, a
>    separate construction that places these fibres on a lattice. The ladder
>    studies the fibre; the platform adds the base. Neither has the base the
>    census assumed. **Do not fold the census into the ladder sections.**
>
> 2. **A-2 leg (ii) is vacuous, not merely definitional.** Leg (ii) reads:
>    transport that fails to preserve the multiplication does not preserve the
>    role assignment. Its contrapositive requires role-preserving transport to
>    preserve the multiplication. Measured: so(8) = g₂ ⊕ span{L, R} exactly,
>    14 + 14 = 28, with **every** multiplication operator lying entirely
>    outside the derivations (fraction 1.0000000000). So no transport built
>    from left and right multiplication *ever* preserves the multiplication;
>    the hypothesis is never satisfied and the leg does no work. Related: the
>    octonion structure is therefore **dynamical**, not background.
>
> 3. **B-7's boundedness is counter-indicated, not merely unearned.** The
>    E_G-shaped envelope follows from boundedness of the bond coupling alone.
>    But a ceiling is an escape energy: a saturating bond confines only below
>    s², and above it the configuration separates without bound (excursion
>    ratio → 4.000 at 4× the integration time, versus 1.000 below the
>    ceiling), while an unbounded confining bond confines at every energy
>    tested including E = 500. **Persistence favours the unbounded coupling.**
>    The Penrose E_G correspondence is not refuted but demoted: it holds for
>    structures with bounded couplings, and the ladder has no reason to be one.
>
> 4. **Session correction record**, recorded because the rate bears on
>    weighting: roughly fifteen corrections, of which two were structural — a
>    five-session block computed on a mistaken base reading, and a false
>    retraction in which the rung scripts were declared non-existent because
>    an older 31-file package had been re-uploaded in place of the 54-file one.
>    The recurring failure mode was **a structured sample read as general**
>    (unions of Fano lines skipping the 4-point case; coordinate triples read
>    as generic 3-planes; one archive read as the archive). A secondary
>    recurring fault was a **relative-only numerical rank threshold**, which
>    reports spurious rank on matrices that are zero up to floating-point
>    noise; fixed with an absolute floor, with all headline counts unchanged.

---

## ITEM 6 — NOTES NOT FOR INSERTION (for the author's judgement)

- **Clifford was under-used.** `z1_d8_attempt.py` P1 already states
  {L_a, L_b} = −2δ_ab, and P4 already states that iterated commutators span
  21 = spin(7). Foregrounding this explains the 7-versus-8 structure directly
  — Im(𝕆) is the 7-dimensional **vector** space, 𝕆 the 8-dimensional
  **spinor** — and would have prevented the base error entirely. The full
  grading is 7 (vectors) + 21 (bivectors) = 28 = so(8); "non-associativity
  generates the automorphism group" is a description of the arithmetic, not a
  mechanism. Recommend the Clifford grading be stated wherever the 21 appears.

- **Notation collision.** β denotes two independent quantities: the cone
  deficit (D2/G3) and the gyroscopic coupling ratio (Section 7 of the spec).
  Worth separating before external circulation.

- **The platform sections are the live opportunity.** Sections 6–9 of the spec
  — passivity forcing u(2), synthetic U(1), the measurable dispersion
  asymmetry, closed-form nonlinear predictions with coefficients fixed by
  linear spectroscopy of the same platform — are lattice results and therefore
  **Euclidean-compatible**. The signature bound of Item 2 constrains the
  cosmological readings and does not touch the lab work. Of everything
  reviewed, this is the part with a testable prediction and no signature
  obstruction.
