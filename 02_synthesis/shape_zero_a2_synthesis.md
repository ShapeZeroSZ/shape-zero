# A-2 resolved, and what it leaves

## The restatement

A-2 asked whether selection principles may be applied to *structure* rather
than to states within structure, and argued the coupling class SELECTED on two
legs. The correct statement is not two legs. It is **one decomposition with
two consequences**:

    V ⊗ V  =  1 + 7 + 14 + 27      (G₂ irreducibles)

- **Coupling side.** The four constructible rank-4 couplings — the three metric
  pairings and Σ_m c_ijm c_klm — span exactly the same space as the four
  irrep projectors. Span gap ~10⁻¹⁵, all 16 algebras. Completeness of that
  list is the G₂ FFT.
- **Transport side.** V ⊗ g₂^⊥ ≅ V ⊗ V is the intrinsic torsion space,
  decomposing into the same four pieces — the Fernández–Gray classes. Casimir
  eigenvalues agree exactly on both sides (0, 2, 4, 14/3), so the
  identification is canonical rather than dimensional.

Presenting these as independent arguments invites the objection that the
second is circular. It is. Presenting them as one fact does not, and both
halves cite standard literature.

## What the transport side says about the arena

The intrinsic-torsion classification requires the base to *be* the 7 that G₂
acts on. Shape Zero's 7 is Im(𝕆), reached by a ladder over division-algebra
dimensions; the arena carried up that ladder is a 2-dimensional cone. Same
number, different object.

`g2_base_question.py` makes the difference sharp. With G₂ acting only on the
fibre, the defect space is *d* copies of the **7** for any base dimension *d*
— **including d = 7**, where it is 7 × (the 7) and not 1+7+14+27. Both are
49-dimensional. Dimension matching is not structure matching.

**The identification "the ladder's 7 is also the spatial base" is a CHOSEN
that has not been spent.** On present structure the bundle reading holds, and
W₁, W₂, W₃ do not exist in the arena.

## What the arena does have

`z1_hinge.py` already measured it: holonomy 21, derivations 14 inside it, 7
directions failing table-preservation. Those 7 do not damage the algebra —
they move it along a compact orbit of equally valid octonion structures. The
orbit is **S⁷ = Spin(7)/G₂** (`sigma_topology.py`: unit norm, spans all 8
coordinates of 𝕆, stabiliser dimension 14).

So the object is a sigma model:

| ingredient | status |
|---|---|
| base | the D2 cone |
| target | S⁷ = Spin(7)/G₂ |
| target metric | **forced** up to one scale (isotropy irreducible → Schur) |
| tangent labelling | **canonical**, `v ↦ L_v`, `(L_v)_ab = c_vab` |
| Wess–Zumino term | **none** — dΦ = −6Ψ, residual 0.000e+00 |
| potential | **none** — leg (i) plus transitivity of the action |
| topological sectors | **none** — π₁ = π₂ = 0 |

**One constant**: the overall scale of the target metric.

## Two structural constraints worth knowing before spending sessions

**Not integrable by either standard route.** Harmonic maps from a surface into
a *symmetric* space are integrable. SO(7)/G₂ is not symmetric — ‖[m,m]_m‖ =
6.48. The 3-symmetric route needs m to split into conjugate ω/ω² eigenspaces
of an order-3 automorphism, which requires dim m even; dim m = 7. Parity
alone closes it. Neither usual reason to expect a Lax pair applies.

**The deficit angle decouples.** Harmonic maps in 2D are conformally
invariant and a cone is flat away from its apex, so ζ cannot enter the local
field equations. It could only enter through topological selection, and
π₁(S⁷) = 0 leaves nothing to select among. (cone deficit renamed β → ζ on 2026-09-25; β now denotes only the lattice gyroscopic coupling)

## The open question, stated precisely

Everything answerable from structure constants alone is answered. What remains
is not algebra:

1. **Is the ladder's 7 the spatial base?** If yes, the four torsion classes
   become available and the whole Fernández–Gray apparatus applies. If no, the
   bundle reading stands and the model is as tabulated above. This single
   CHOSEN decides which of two quite different theories the program is
   describing.

2. **Does the cone couple at all?** On present structure it does not — ζ drops
   out both locally (conformal invariance) and topologically (π₁ = 0). Either
   the coupling enters through something not yet in the model, or the D2 and
   D8 rungs are less connected than the ladder's continuity suggests.

Item 2 is the sharper of the two, because it is a *negative* result about
structure the program already has, rather than a question about structure it
might acquire.
