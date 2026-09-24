# Errata — C1 Formal Proofs (Sections 3 and 6)

*Section 6 first; Section 3 at the end.*

**See also:** `02_synthesis/C1_ERRATA.md` — the earlier errata list for C1.

# Section 6, Theorem 6.1

**Document:** *Shape Zero — Formal Proofs of the C1 Verification Package*
(August 2026), Section 6, "Passivity Forces u(2)."

**Status:** the theorem's conclusion is right in substance, but its proof
contains a sign-class error, and two of its claims go further than what is
proved. The corrected version below is consistent with three machine-verified
results (Lean 4, published on Prove2Me, 2026-09-23).

---

## What was wrong

**1. Symmetric, not skew-symmetric.** The proof writes the neighbour coupling's
power as Σᵢ⟨vᵢ, W(vᵢ₊₁ − vᵢ₋₁)⟩ and states that it vanishes for all velocity
fields only if W is **skew**-symmetric. For that expression the correct
condition is **symmetric**. Shifting the index in the second sum gives

    P = Σᵢ ⟨vᵢ, (W − Wᵀ) vᵢ₊₁⟩,

which vanishes for all velocities exactly when W − Wᵀ = 0. Numerically:
symmetric W gives |P| ≤ 6×10⁻¹⁵, skew-symmetric W gives |P| ≈ 62.

"Skew-symmetric" is the correct condition for a different term — the **on-site**
gyroscopic coupling K·vᵢ, whose power Σᵢ⟨vᵢ, K vᵢ⟩ vanishes exactly when K is
skew. The two conditions were interchanged.

**2. Hermitian is not u(2).** Real symmetric matrices commuting with the complex
structure J correspond to **Hermitian** complex matrices, not to u(2). The Lie
algebra u(2) consists of **anti**-Hermitian matrices; the two are related by
multiplication by i. They have the same dimension (n² for u(n)), so the counting
is right, but the admissible couplings are not themselves the Lie algebra.

**3. Commuting with J is an extra requirement.** The original proof says the
intra-node complex structure "converts" the admissible couplings into Hermitian
matrices, suggesting this follows from passivity. It does not. Passivity forces
symmetry; commuting with J is an additional condition — that the coupling
respect each node's complex structure — which the model imposes.

**4. The ring needs at least three sites.** On a ring of one site the power is
zero for every W. On a ring of two sites, each site's two neighbours coincide,
the power reduces to ⟨v₀, (D − Dᵀ)v₁⟩ with D = W₀ − W₁, and two links carrying
the same non-symmetric matrix do exactly zero work. The theorem fails in both
cases.

**5. The last sentence of the proof.** "Any non-Hermitian component produces a
non-zero symmetric part that pumps or drains energy" should read *antisymmetric*
part, following correction 1.

---

## Corrected Theorem 6.1

**Theorem 6.1 (Passivity forces symmetric coupling; with J-compatibility, the
coupling space has the dimension of u(n)).** Consider a ring of N ≥ 3 nodes,
each carrying an internal real space ℝ²ⁿ with a complex structure J
(J² = −1). Couple neighbouring nodes through the velocity-dependent force

    Fᵢ = Wᵢ vᵢ₊₁ − Wᵢ₋₁ vᵢ₋₁,

with one real 2n × 2n matrix Wᵢ per link.

(a) The coupling does no net work for every motion **if and only if** every Wᵢ
is symmetric.

(b) The real symmetric 2n × 2n matrices that commute with J form a real vector
space of dimension **n²**, which is the dimension of u(n). Under the
identification ℝ²ⁿ ≅ ℂⁿ induced by J, they are exactly the n × n Hermitian
matrices; multiplying by i gives u(n).

**Proof.** (a) Shifting the index in the incoming term, the total power is
P = Σᵢ ⟨vᵢ, (Wᵢ − Wᵢᵀ) vᵢ₊₁⟩. If every Wᵢ is symmetric this is zero. Conversely,
for N ≥ 3 the sites i−1 and i+2 are distinct from i and i+1, so a velocity field
supported only on sites i and i+1 reduces P to ⟨vᵢ, (Wᵢ − Wᵢᵀ) vᵢ₊₁⟩; its
vanishing for all vᵢ, vᵢ₊₁ forces Wᵢ = Wᵢᵀ.

(b) A matrix commutes with J = [[0, −I], [I, 0]] exactly when it has the block
form [[A, −B], [B, A]], and such a matrix is symmetric exactly when A is
symmetric and B is antisymmetric. Every real n × n matrix X decomposes uniquely
as S + K with S symmetric and K antisymmetric, and X ↦ [[S, −K], [K, S]] is a
linear bijection onto the admissible space. Its dimension is therefore that of
all n × n real matrices, n². ∎

**Scope.** The commuting-with-J condition in (b) is a modelling premise, not a
consequence of passivity. The n = 1, 2, 3 cases give u(1), u(2) and u(3) in
dimension.

**Machine verification.** Part (a) is Prove2Me mission 2, "Zero net power forces
every link coupling to be symmetric (rings of ≥ 3 sites)", including a proof of
the two-site counterexample. Part (b) is Prove2Me mission 1, "The
passivity-admissible couplings have dimension n² = dim u(n)". Both proved in
Lean 4 with only the standard axioms.

---

## Theorem 2.10 — also affected

Theorem 2.10 states that the zero-power couplings, closed under the bracket, are
precisely u(2). By the correction above, zero power alone gives all symmetric
matrices (dimension n(2n+1); 10 at n = 2), and symmetric matrices are not closed
under the commutator — the bracket of two symmetric matrices is antisymmetric.
The u(2) count requires the separate J-compatibility condition, which is a
chosen principle (conservation of total phase charge), not a consequence of
passivity.

## Corollary 6.2 — no change needed

Corollary 6.2 (ordering and amplitude-Zeeman) does not depend on the corrected
sign condition and stands as written.

---

## Related: the pinned asymmetry (Section 7)

Section 7's dispersion asymmetry Δω = −2Cβ·sin k is now also machine-verified
in its exact linear form (Prove2Me mission 3): ω(q) − ω(−q) = 2βc·sin q, with
the on-site stiffness cancelling identically. The sign difference from
Section 7's statement is a convention for which direction is subtracted from
which. The protection holds at linear order on a uniform lattice only; the
nonlinear coefficient and non-uniform stiffness both break it, as measured.


---

# Section 3 — two edge-case errors

The core argument of §3 is correct: a Steiner triple system with **at least one
point** that admits a role colouring has exactly 7 points. Two statements are
false at the edges. (Found while drafting Prove2Me mission 5; verified by
enumeration.)

**Theorem 3.3, condition (a).** It states that "any two lines intersect" forces
n = 7, and that (a) and (b) are equivalent. **Both are false for small systems.**
A single triple (n = 3, r = 1) and a single point (n = 1, r = 0) satisfy (a)
vacuously — with at most one line there is no pair of lines to fail — but have
n ≠ 7. The proof begins "fix a line L and a point p ∉ L," which assumes both
exist. **Fix:** require at least two lines (equivalently n > 3). Condition (b)
is unaffected.

**Theorem 3.6.** "Any Steiner triple system admitting a role colouring is
necessarily the Fano plane" is **false for the empty system**, which satisfies
triadic closure and both role postulates vacuously and has 0 points. **Fix:**
require a nonempty point set. (n = 1 and n = 3 are already excluded by role
completeness, since a point on fewer than three lines cannot take all three
roles.)

**Minor.** The existence argument cites "the classical 1-factorisation of K₃,₃
components"; the Fano incidence graph is the Heawood graph, which is not a union
of K₃,₃ components. König's theorem, also cited, is the correct justification: a
3-regular bipartite graph has a proper 3-edge-colouring, which is exactly a role
colouring. The count of **48** role colourings (§5) is confirmed by enumeration.

**Note — not a correction.** The proof of Theorem 3.3 ends "Uniqueness of STS(7)
is classical." — asserted without reference or argument. The claim is true, and
is now machine-verified: Prove2Me mission 6 (2026-09-24, in review) proves that
every seven-point Steiner triple system is the Fano plane. With mission 5, which
proves that the role postulates force 7 points, Theorem 3.6 ("Roles Force Fano")
is machine-verified in full.
