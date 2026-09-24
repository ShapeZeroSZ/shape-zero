# The Three Gauge Factors — Where Each Enters, and by What Mechanism

*Reference note. Written to locate the third factor, which enters by a
different route from the first two, and to state honestly what is derived,
what is a consistency check with known results, and what is assumed.*

**Scripts:** `z1_d8_dynamics.py` (dim Der(𝕆) = 14), `phi_gauge_chiral.py`
(tests the u(2) class under passivity plus J-compatibility; passivity alone
forces only symmetric couplings — see `01_source/proofs/ERRATUM_Theorem_6.1.md`).

---

## 1. Summary table

| factor | rung | mechanism | status |
|---|---|---|---|
| **U(1)** | D2 | isotropy ⇒ conserved angular momentum ⇒ complex structure; in the lattice sector, the synthetic gauge field from antisymmetric velocity coupling | **derived** by a selection principle owned by the programme |
| **SU(2)** | D4 | the structure sphere; the zero-power condition forces the coupling Hermitian, giving u(2) = u(1) ⊕ su(2) | **derived** by the same principle (passivity) |
| **SU(3)** | D8 | the subgroup of G₂ = Aut(𝕆) that fixes one imaginary unit | **not derived** — it follows from a *choice*, and the choice is not made by any principle in the ladder |

**That asymmetry is the answer to "where is the third?"** The first two
arrive by *selection*: a conservation requirement excludes the
alternatives. The third arrives by *breaking*: a preferred direction is
picked, and SU(3) is what survives the picking. It is structurally a
different kind of object, which is why it does not sit alongside the other
two in the derivation chain.

---

## 2. The third factor, precisely

G₂ is the automorphism group of the octonions, of dimension 14. It acts
transitively on the six-sphere of unit imaginary octonions, and the
stabiliser of a single imaginary unit is **SU(3)** — dimension count
14 − 6 = 8, as required.

Fixing that unit makes the octonions a complex vector space, and they
decompose under the stabiliser as

  𝕆 = ℂ ⊕ ℂ³ , i.e. **1 ⊕ 3 ⊕ 3̄ ⊕ 1**

with Im(𝕆) = ℝ⁷ giving **1 ⊕ 3 ⊕ 3̄**. Günaydin and Gürsey identified the
**3** and **3̄** as a triplet of quarks and antiquarks under the colour
group SU(3).

**Reference:** M. Günaydin and F. Gürsey, *Quark structure and octonions*,
J. Math. Phys. **14**, 1651–1667 (1973), doi:10.1063/1.1666240. Their paper
performs the reduction of G₂ through both of its physically relevant
subgroups — SU(3) *and* SU(2) ⊗ SU(2) — which matters for §3 below. See
also G. Dixon, *Division Algebras* (Springer, 2013), and C. Furey,
arXiv:1806.00612, for the modern development.

**Status: known since 1973 and not novel here.** The colour decomposition
appearing in the programme's D8 material is a reproduction of the standard
result, correctly obtained. It is a consistency check with known physics,
not a prediction — which is what the programme's own input ledger already
says.

---

## 3. Why the three do not combine as a product

This is the structural point behind the "contact, not product" framing, and
it is worth stating sharply because it constrains what the octonion route
can deliver.

Inside G₂, the colour SU(3) and the SU(2) sector are **two different
maximal subgroups of the same group** — Günaydin and Gürsey reduce through
SU(3) and through SU(2) ⊗ SU(2) as alternatives. They are not independent
factors sitting side by side; they intersect, and their union is not a
subgroup. So

  SU(3) × SU(2) × U(1)  **does not embed in G₂ as a product.**

That is why the established literature in this area does not try to obtain
the Standard Model group from a single automorphism group. It uses tensor
products of algebras instead — ℂ ⊗ ℍ for the weak sector, ℂ ⊗ 𝕆 for the
colour sector, assembled as ℝ ⊗ ℂ ⊗ ℍ ⊗ 𝕆 — precisely so that the factors
are independent by construction rather than competing subgroups of one
group.

**Consequence for the ladder as built.** The rung correspondence
ℂ → U(1), ℍ → SU(2), 𝕆 → SU(3) — rungs 2, 4, 8 — is real as an organising
pattern and is the standard idea of that literature. But it is a
correspondence between *algebras and factors*, not a chain of subgroups.
The ladder inherits the pattern; it does not by itself assemble the product
group, and no assembly appears in the current material.

---

## 4. What the programme adds, and what it does not

**Adds:** a selection principle for the first two factors. Passivity
forcing the Hermitian class — hence u(2), hence U(1) and SU(2) — is the
programme's own theorem and, at the lattice level, is the one chain that
runs from a stated principle to a measurable number. That is the content of
the bench test.

**Does not add:** anything at the third factor. SU(3) is obtained the
standard way, by fixing an imaginary unit, and the fixing is not performed
by any ladder principle. Nor does the ladder supply the assembly of §3.

**The well-posed open question, which would change this:** *is there a
principle that selects the preferred imaginary unit?* If persistence,
minimality, conservativity or plurality forced that choice, SU(3) would
join the other two as derived rather than assumed, and the asymmetry noted
in §1 would close. Nothing in the current material does this. It is a sharp
question with a definite answer either way, and it is the correct thing to
ask before the D8 material is described as containing three gauge factors.

**How to describe it in the meantime.** Two factors derived from a stated
principle; the third reproduced from a 1973 result by fixing a direction
the framework does not fix; no assembly of the product group. Stated that
way the claim is accurate and survives contact with anyone who knows the
literature.
