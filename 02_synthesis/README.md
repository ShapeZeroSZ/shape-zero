# Shape Zero — C1 Supplemental (C1S)

> **REVISION 2 — READ `REVISION_2_BASE.md` FIRST.**
> A block of this package was computed on a 2-dimensional base taken from
> `phi_gauge_wilson.py`, which is a D2-level script, not the D8 arena. The C1
> rung scripts show the arena advancing 1 → 2 → 4 → 8, so the D8 base is
> 𝕆 = ℝ⁸ with Im(𝕆) = ℝ⁷, and G₂ acts on the base as well as the fibre.
> The revision note classifies every result as standing, superseded, or
> needing recheck. Nothing has been deleted.
>
> **`C1S_SYNTHESIS.md` is the standalone account of this work** —
> Sections 1-11 cover the octonionic rung; 12-15 cover the ladder-wide
> results: the parameter census, the scale results, the signature finding, and
> the Euclidean/thermal reading. Read that rather than reconstructing it from
> the scripts.

Supplement to `shape_zero_c1_package`. Resolves open-threads item **A-2**,
closes **A-3**, opens **B-7**, and carries the A-2 resolution forward to a
fully specified sigma model.

Every quantitative claim is backed by a self-contained script in `scripts/`.
Python 3 + NumPy only, no other dependencies. Predictions are stated in each
script header **before** the run; misses are left in the header and in the
output rather than edited out.

Read `C1_ERRATA.md` first — it lists what C1 got wrong, and what this
supplement got wrong along the way.

## What closed

| item | status | script |
|---|---|---|
| A-3 uniqueness beyond one instance | closed as a census, all 16 algebras | `a2_invariance_hinge.py` |
| A-2 leg (i) | verified; closes by **citation**, not proof | `a2_invariance_hinge.py` |
| A-2 leg (ii) | **withdrawn** — false or definitional | `a2_leg_ii.py` |
| A-2 leg (ii) replacement | intrinsic torsion, four classes | `a2_intrinsic_torsion.py` |
| A-2 unification | legs (i) and (ii) are one decomposition | `a2_unification_audit.py` |

## Headline results

- The G₂ **First Fundamental Theorem** was proven in the 1980s; the generating
  invariants of the 7-dimensional representation have degrees 2, 3, 4. The
  proof A-2 leg (i) owed already exists. Verified computationally at ranks
  2, 3, 4 across all 16 valid algebras.
- The four constructible rank-4 couplings **are** the four G₂-irrep projectors
  on V⊗V (span gap ~10⁻¹⁵). Coupling classification and transport
  classification are the same decomposition, not two arguments.
- `z1_hinge.py`'s "all 7 pulses fail table-preservation" quantified: defect
  6.4×10⁻¹⁶ on the 14 derivations, ≥ 0.926 on the 7 complement. Those 7
  directions **do not break the algebra** — they move it along a compact
  7-parameter orbit of equally valid octonion structures.
- The orbit is **S⁷ = Spin(7)/G₂**. Verified: orbit of 1 ∈ 𝕆 has unit norm,
  spans all 8 coordinates, stabiliser dimension 14.
- Target metric **forced up to one scale** — the isotropy representation is
  irreducible, so Schur leaves no freedom. No squashing parameter.
- Moduli directions **canonically labelled** by imaginary octonions via
  `v ↦ L_v`, `(L_v)_ab = c_vab`. Equivariance holds to 10⁻¹⁵; the opposite
  sign convention misses by 2.575, so it is an intertwiner and not a fit.
- **No Wess–Zumino term.** dΦ = −6·Ψ with relative residual 0.000e+00 — an
  identity, not a fit. This is the nearly-parallel condition, torsion class
  W₁ alone, agreeing with an unrelated computation that found φ-as-torsion
  pure W₁ at fraction 1.0000000000.
- **Quartic is the lowest order at which β can couple**, and two terms exist
  there — but both octonion-built candidates collapse into them. The
  composition identity |u×v|² = |u|²|v|² − (u·v)² cancels the Skyrme term
  exactly. The octonions contribute nothing to the only available coupling.
- **The cone does couple — at the apex.** The non-minimal term R|dφ|² sees
  the one point where the cone is not flat. ∫R√g = 4π(1−β) exactly,
  independent of how the tip is regularised, and the support collapses onto
  the apex (r₉₅/ε constant to 0.002%). The β-dependence is forced; only ξ is
  new. This order also admits an octonionic term, unlike the quartic sector.
- **The octonionic second-derivative term survives variation.** B·□φ is not
  a total derivative — nonzero on a closed surface against a machine-zero
  floor, with nonvanishing functional derivative. First place in the whole
  construction where the octonions contribute dynamics the target metric
  alone does not. Consequence: orientation becomes physical, tied to base
  parity, so the 16 valid algebras stop being interchangeable.
- **The octonionic term survives the covariant treatment.** Redone with the
  Maurer-Cartan form on Spin(7)/G₂ and the tension field in place of the
  coordinate Laplacian: still not a total derivative, still enters the field
  equations. Amplitude scaling is cubic-plus-quartic with B/A = −3.66, the
  quartic being the connection piece of τ.
- **Orientation is physical, paired with base parity.** Verified, not asserted:
  the octonionic term is odd under base reflection, odd under orientation
  reversal, even under both. Reversal is an involution on the 16 valid
  orientations (8 conjugate pairs), so the reversed table is always a genuine
  octonion algebra. Only the *relative* sign of orientation and base handedness
  is meaningful. The term is also the unique invariant at its order — forced
  by A-3's one-dimensional invariant cubic space.
- **The octonionic term measures the normal component of the tension field.**
  It has two independent kernels: collinear base gradients (killing the entire
  radial sector, seven free profiles) and a target image confined to a 2-plane.
  Geometrically both are one fact — c(∂₁φ,∂₂φ,τ) is τ projected on the
  octonionic normal to the swept surface. It therefore vanishes on harmonic
  maps (τ = 0) and is a genuine perturbation, and on the cone it can only
  couple to angular structure — which is what the deficit is.
- **π₁(S⁷) = π₂(S⁷) = 0.** No solitons, no winding sectors. Combined with 2D
  conformal invariance and a cone being flat away from its apex, the deficit
  angle β does not couple to the field on present structure.

## Review note on the Formal Proofs document

`d8_exact_verification.py` replaces Theorem 5.1's sampled criterion with the
exact integer one the document already mentions: the Clifford relations
{Lₐ,L_b} = −2δ_ab I. Exactly 16 of 128, no floating point, no tolerance, and
the set is identical to the sampled one — the sampled answer was correct, just
weaker than a proof. It also confirms the partition claim literally (7 blocks
of dimension 3, total rank 21, cross-block overlap 0.000e+00), which follows in
one line from the Steiner property rather than from enumeration.

## Revision 2 headline

- **The arena advances with the ladder.** D1 scalar, D2 ℝ², D4 ℝ⁴, D8 ℝ⁸ —
  every C1 rung script, no exceptions. The D8 base is Im(𝕆) = ℝ⁷.
- **At d = 7 the kinetic term is not marginal.** It scales as μ⁵ and bounds
  the action alone. The instability, the coupling bound λ < 2√(μc₄), and the
  "no window" result were all consequences of a marginal kinetic term, which
  is a d = 2 property.
- **A term forbidden in 2D exists in 7D.** φ^{μνρ}c_{abc}∂ψ∂ψ∂ψ measures
  exactly 0.000e+00 on any 2-dimensional base (Λ³ of ℝ² vanishes) and equals
  exactly 42 on the identity map in 7D. Purely first-derivative, at (3, 3).
- **The (3,3) term is a null Lagrangian.** It exists in 7D, is a genuinely new
  invariant, and contributes nothing to the field equations — the same total
  antisymmetry that permits it makes it inert. Fifth independent mechanism by
  which the octonionic sector declines to supply dynamics.
- **The Wick coefficient 28 is withdrawn.** A control test with ε_{ijk} on 3
  components reproduces the same discrepancy, so the fault is general
  machinery, not octonionic.

## Net effect on the ledger

The coupling class SELECTED tag rested on two legs. One is now verified and
citable; the other is withdrawn. It rests on **one leg**, and the correct
framing is one structural fact with two consequences — both covered by
standard literature (G₂ FFT on the coupling side, Fernández–Gray on the
transport side).

The sigma model that results carries **one constant**: the overall scale of
the forced target metric. No WZ term, no potential (leg (i) plus transitivity),
no topological sectors.

## Running

```
cd scripts
python3 a2_invariance_hinge.py     # ~1 min
python3 a2_leg_ii.py               # seconds
python3 a2_intrinsic_torsion.py    # seconds
python3 a2_unification_audit.py    # seconds
python3 g2_base_question.py        # seconds
python3 z1_holonomy_orbit.py       # seconds
python3 coset_audit.py             # seconds
python3 target_wz_check.py         # ~30 s
python3 sigma_topology.py          # seconds
python3 cone_coupling_search.py     # seconds
python3 cone_coupling_d2.py         # seconds
python3 octonionic_term_variation.py # seconds
python3 octonionic_term_covariant.py # ~1 min
python3 d8_exact_verification.py    # seconds
python3 d8_base_corrected.py        # ~1 min   [Revision 2]
python3 d8_cubic_term.py            # seconds  [Revision 2]
python3 cone_derrick.py             # ~1 min   [superseded: 2D base]
python3 cone_threshold.py           # ~1 min   [superseded: 2D base]
python3 cone_competitors.py         # ~2 min   [superseded: 2D base]
python3 cone_infimum.py             # ~3 min   [superseded: 2D base]
python3 cone_ground_state.py        # ~5 min   [superseded: 2D base]
python3 cone_vertex.py              # ~3 min   [superseded: 2D base]
python3 vertex_renormalisation.py   # ~2 min   [coefficient withdrawn]
python3 octonionic_term_parity.py   # seconds
python3 octonionic_term_kernel.py   # seconds
python3 b7_dislocation.py          # ~1 min
python3 b7_boundedness.py          # ~2 min
```

`z1_holonomy_orbit.py`, `coset_audit.py`, `target_wz_check.py` and
`sigma_topology.py` reproduce `z1_hinge.py`'s König assignment directly, so
counts are against the actual C1 table rather than a re-derived orientation.
