# Errata

**See also:** `01_source/proofs/ERRATUM_Theorem_6.1.md` — later corrections to
C1 Formal Proofs §6 (Theorem 6.1: passivity forces **symmetric**, not skew,
couplings; Theorem 2.10) and §3 (Theorem 3.3(a) and Theorem 3.6 fail at the
edges), found while formalizing the results in Lean on Prove2Me.

Two lists. The first corrects C1. The second corrects work done inside this
supplement — kept because the predictions-first discipline caught them, and
removing them would hide how.

---

## 1. Corrections to C1

### A-2 leg (ii) is withdrawn

C1 argues the coupling class is SELECTED on two legs, the second being
"transport that fails to preserve the multiplication does not preserve the
role assignment." Tested in `a2_leg_ii.py`, this is **reading-dependent**:

- **Bare-colouring reading** — the role assignment is the colouring of the
  incidence structure. Sign flips act on the algebra but not on the
  combinatorics, so they preserve it trivially. |A| = 896, of which ≥ 120
  elements lie outside the multiplication-preserving group. **Leg (ii) fails.**
- **Colouring-plus-orientation reading** — A = B = 1344 exactly, so leg (ii)
  holds. But the colouring was *constructed* to determine the multiplication,
  so this restates the construction. **Definitional, not argumentative.**

Neither version is independent support. The SELECTED tag rests on one leg.

### A-2 leg (i) needs a citation, not a proof

Leg (i) asserts constructible ⟺ invariant under the automorphism group. The
reverse direction is a First Fundamental Theorem, which holds per group and
per representation and is **false as a general principle**. It cannot be
justified as a scope extension — only restricted to where it is used.

Where the program uses it, the group is G₂ on ℝ⁷, and the FFT for G₂ was
proven in the 1980s with generating invariants of degrees 2, 3, 4. Verified
at those ranks in `a2_invariance_hinge.py`. The "proof owed" is discharged
by reference.

### A-2 does not transfer to the gravity items

C1's A-2 says the hinge "reappears as diffeomorphism covariance in the gravity
items." It does not. Diff(M) is infinite-dimensional and admits no FFT of this
kind — which is why general covariance is a *principle* in GR rather than a
theorem, and why Lovelock uniqueness needs extra inputs (dimension,
second-order field equations) that covariance alone does not supply.

The hinge **splits three ways**: closed for the octonion items; a different
and harder problem for the gravity items; and a different *kind* of question
for the quantum items, which ask about **selection** when invariance
underdetermines, whereas A-2 asks about **completeness** of generators.

### A-3 closes as a census

C1 records the invariant-cubic uniqueness as verified on one algebra instance.
`a2_invariance_hinge.py` extends it to all 16 valid orientations: dim Der = 14
and a 1-dimensional invariant cubic space every time, overlap with φ =
1.0000000000. Also reproduces the 16/128 count by a **different** criterion
than `z1_d8_census.py` used — norm multiplicativity rather than orientation
validity. Still a census, not a theorem; the theorem is the FFT above.

### "All 7 pulses fail table-preservation" is not a defect

`z1_hinge.py` records this as a failure. It is not. The 7 directions move the
octonion structure along a compact 7-parameter orbit of **equally valid**
composition algebras (`z1_holonomy_orbit.py`, validity holds at every t
sampled). The correct reading is that the octonion structure has exactly 7
degrees of freedom — a field, not a background. That answers leg (ii)'s
replacement question in the arena that exists, by the program's own
measurement.

---

## 2. Errors made inside this supplement

### Intrinsic torsion is not Einstein–Cartan torsion

Claimed mid-work that the intrinsic torsion is "the same object B-7 reaches
for from the defect side." It is not. A dislocation density is a full torsion
tensor (147 components); the intrinsic torsion is a 49-dimensional quotient,
with 98 absorbable by re-choosing the connection. They meet through a
**projection**, not an identity. For the totally antisymmetric part
specifically the split is 35 intrinsic / 0 gauge — see
`a2_unification_audit.py`.

### "One gate on four threads" was wrong

Claimed A-2 was a single blocker holding up four threads. The computation
showed it splits three ways (above). Two threads came unblocked cheaply; two
were re-scoped as genuinely open.

### The nearly-parallel lead was half right

Flagged that the sigma model target is the same homogeneous space that carries
nearly-parallel G₂ structures, and gestured toward W₁ torsion and a positive
cosmological constant. The coset identity is real and **canonical**
(`coset_audit.py`). The gestured consequence does not follow: nearly-parallel
geometry uses that coset as a **base manifold**, here it is a **target**. Same
space, opposite job. Its actual consequence is negative — it is what obstructs
closure of the WZ candidate.

### `b7_dislocation.py` P3 missed by 199%, cause identified, factor undiagnosed

Predicted continuum slope b²/4π = 0.0796; measured 0.2383, converging to
≈ 3.0× the continuum value. Diagnosed by the dipole test (P3b), which
recovers the continuum coefficient to +1.2% in a boundary-independent
geometry. The cause is a charged defect measured against a pinned boundary.
**The factor of ~3.0 itself is not derived** and remains a loose end.

Standing rule from that miss: a topologically charged defect cannot be
measured against a pinned boundary; only neutral configurations recover the
continuum coefficient.

### `b7_boundedness.py` Q2 threshold was set too tight

Predicted ≤ 1% spread across bounded families at small b; measured 1.8% at
b = 0.2. The prediction holds at genuinely small b and the threshold was the
error, not the result. Left as stated.

### Saturation is a CHOSEN introduced here, not earned

The bounded bond potential in `b7_dislocation.py` / `b7_boundedness.py` is
supplied by this supplement, not derived from the ladder. `b7_boundedness.py`
shows the E_G-shaped envelope follows from **boundedness alone** — the
plateau is N_cut × ceiling for three unrelated bounded families, and only the
crossover profile carries the functional form. So exactly one proposition
needs earning: whether conservativity, minimality and persistence force a
bounded coupling. Minimality argues against it (boundedness carries a scale);
the counter-argument is a tension with D1, which selected bounded motion.
Unresolved.
