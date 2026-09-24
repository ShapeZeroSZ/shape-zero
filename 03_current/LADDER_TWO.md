# Ladder Two — The Quasi-Periodic Branch

Standalone. **Not an addendum to C1/C1S/C1S2.** Those document Ladder One: what
follows from demanding closed orbits. This documents a different construction
that begins where that demand is relaxed, and it stands or falls on its own
terms.

Read §1 before anything else. The relationship between the two ladders is the
whole content, and getting it wrong — repeatedly, in the session that produced
this — is what made the material look like a failed appendix rather than a
separate object.

---

## 1. The two ladders are one principle at two strengths

Lemma 2.1 does not conclude *bounded*. It concludes **periodic**. That distinction
is the hinge, and it is measurable.

For the natural flow ψ̇ = aψ with a a unit imaginary element, the generator is
L_a. Two conditions can be asked of it:

- **skew** (L_a + L_aᵀ = 0) — the flow preserves the norm, motion is *bounded*
- **L_a² = −I** — all eigenvalues are ±i, one frequency, every orbit closed with
  a common period, motion is *periodic*

| dim | max \|L_a² + I\| | distinct \|eigenvalue\| | distinct frequencies |
|---|---|---|---|
| 2 | 0.000e+00 | 1 | — |
| 4 | 4.4e−16 | 1 | — |
| 8 | 3.3e−16 | 1 | **1.0** |
| **16** | **7.6e−01** | **121** | **0.285, 1.0, 1.385** |
| 32 | 7.2e−01 | 413 | — |

**Skewness holds at every dimension** — verified, |L_a + L_aᵀ| = 0.000e+00 at
both 8 and 16, with net work 9.7×10⁻¹⁷. So conservativity never fails, and
neither does boundedness: S¹⁵ is compact. **What fails at 16 is periodicity.**

L_a² = −I is the **Clifford relation**, and it is equivalent to composition
|xy| = |x||y|, equivalent to L_a being orthogonal, equivalent to the **Koopman**
operator U_Φf = f∘Φ being unitary. Four statements of one condition:

| | Ladder One | Ladder Two |
|---|---|---|
| persistence read as | **periodic** | **bounded** |
| Clifford L_a² = −I | holds | fails |
| composition | holds | fails |
| Koopman unitary | yes | no |
| dimensions | 1, 2, 4, 8 | 16, 32, … |
| spectra | quadratic towers k(k+n−1) | irregular |

**So Ladder Two is not a mirror algebra, not a second origin, and not the split
branch.** The split algebras are *indefinite and still compose*; the sedenion
norm is positive definite. Ladder Two is the tower past the point where orbits
stop closing, reached by reading persistence at its weaker strength.

**Why it was built.** Ladder One's spectra are all k(k+n−1) — quadratic towers —
because composition makes L_a an isometry, so the algebra's metric *is* the round
metric and representation theory fixes everything in advance. Measured spectra do
not live in quadratic towers: charged-lepton m² ratios are 1, 4.3×10⁴, 1.2×10⁷,
and hadron Regge trajectories grow *linearly* in n. Reaching numbers physics does
not supply requires a non-round metric, and a non-round metric requires
composition to fail. That is the entire motivation, and it is a requirement, not a
preference.

---

## 2. What survives the relaxation

Losing periodicity does not lose everything. Verified at D16:

- **Der saturates at 14** — the same g₂ as D8, at dimensions 8, 16 and 32 alike.
  The automorphism structure does not grow.
- **Power-associativity and flexibility hold at every dimension**, to 10⁻¹⁴ at 16
  and 32. Composition and alternativity are what break.
- **Zero divisors appear**, but confined: rank of L_a drops from 16 to **12,
  exactly, every time**, on precisely one locus.

**The tower is not chaos.** Each doubling switches on one more failure, and the
surviving structure is inherited intact.

---

## 3. D16 mapped

**Non-homogeneous, and for a countable reason.** Der stays 14 while the unit
sphere grows: S⁷, S¹⁵, S³¹. Once 14 < dim of the sphere, the group cannot act
transitively.

| dim | dim Der | sphere | max orbit | codimension |
|---|---|---|---|---|
| 8 | 14 | S⁷ | 6 | 1 (the real axis) |
| 16 | 14 | S¹⁵ | **11** | **4** |

**Four invariants**, all constant to 10⁻¹⁶ under random Der elements against a
control drifting at 7.8×10⁻²:

**Re(a)**, **Re(q)**, **|p|²**, **p·q**

Every one is the doubling's own bookkeeping — two real parts, one norm, one
overlap. The non-homogeneity is *structured*: orbits are labelled by how the two
octonion halves sit relative to each other, and by nothing else. Jacobian rank 4
everywhere, generic and singular alike, so the quotient is a genuine 4-manifold.

**The singular locus is a single homogeneous orbit.** Zero divisors occur exactly
at Re(p) = Re(q) = 0, |p| = |q|, p·q = 0 — the point (0, 0, ½, 0) — with kernel
always 4-dimensional. Dimension count p ∈ S⁶ then q ∈ S⁵ gives 11; measured orbit
dimension 11; stabiliser 3, bracket-closed at 1.4×10⁻¹⁵. **The locus is
G₂/SU(2).**

**So the wildness is the most ordered part.** What breaks composition and division
lives on one homogeneous space with constant-rank degeneracy; the *generic*
sedenions are the non-homogeneous ones.

**The stabiliser identified.** A single generator has distinct |Im eigenvalue|
ratio **[1.0]** with 8 zeros, across all three generators — no 3:1, so no j = 3/2.
Eight non-zero eigenvalues at one magnitude gives four j = ½ doublets, so the 16
branches as **8·(j=0) ⊕ 4·(j=½)**, and each **7 → 3·(j=0) ⊕ 2·(j=½)**. Not the
principal su(2); the one inside the quaternion-subalgebra stabiliser — which is
structurally right, since the zero-divisor condition is quaternion-like.

*Method note:* two attempts at this branching failed, both on normalisation. The
route that worked drops normalisation entirely and uses eigenvalue **ratios**,
which are scale-free.

---

## 4. The non-round metric

> **SUPERSEDED IN PART — read this first.** This section was written before the
> result now in `INPUT_LEDGER.md` §5: **L_aᵀL_a is exactly the identity on the
> invariant gradient span** (4.4×10⁻¹⁶, every sampled point), while ranging
> 0.086–1.914 on the eleven orbit directions. The distortion is therefore
> non-round **only along the orbits** and **exactly round transverse to them** —
> and the transverse directions are the quotient. So the algebra's own metric
> reduces to the round metric on the four invariants, and the spectrum under it
> is unchanged.
>
> **Consequence for §6.** The irregular spectrum reported there comes entirely
> from the *chosen* conformal factor Ω = |det L_a|^{1/16}, not from composition
> failure. Combined with Colbois–Dryden–El Soufi — the invariant eigenvalues are
> unbounded over any conformal class in dimension ≥ 3 — an arbitrary Ω yields an
> arbitrary spectrum, so §6 carries no information about the algebra.
>
> **What survives:** composition failure does produce a genuinely non-round
> structure (σ_min 0.102–0.853 at D16 against 1.0000 exactly at D8). What fails
> is the inference that this reaches the quotient. Four attempts to force Ω all
> closed; the last closed by the theorem above.

### 4a. Original text (retained for the record)

**The ambient sphere is round** — |a|² = Σaᵢ², SO(16) transitive, spectrum
k(k+14). Restricting to invariant functions gives only a sub-tower of that. This
was briefly taken as fatal; it is not, because **the ambient metric is not the
algebra's metric once composition fails**.

| | σ_min | σ_max | \|det L_a\| |
|---|---|---|---|
| **D8** | 1.0000–1.0000 | 1.0000–1.0000 | 1.0000–1.0000 |
| **D16** | **0.102–0.853** | **1.128–1.411** | **0.0004–0.858** |

At D8, composition makes L_a an isometry and the two metrics coincide — which is
*why* every Ladder One spectrum is known group theory. At D16 left multiplication
distorts by a factor varying eightfold. **The failure of composition is itself the
source of a non-round geometry**, appearing at exactly the dimension where
homogeneity ends.

**And it descends to the quotient.** L_{g(a)} = g L_a g⁻¹ verified to 2×10⁻¹⁶, so
det and every singular value are Der-invariant, drifting at 10⁻¹⁶ or exactly zero.

---

## 5. The reduced problem, exactly

For an invariant function f(u₁,u₂,u₃,u₄): Δf = Σ g^{ij}∂_i∂_j f + Σ (Δu_i)∂_i f,
and both coefficient sets close in the invariants.

**Metric** — max error 1.1×10⁻¹⁵ over 300 points:

| | | | |
|---|---|---|---|
| g¹¹ = 1 − u₁² | g¹² = −u₁u₂ | g¹³ = 2u₁(1−u₃) | g¹⁴ = u₂ − 2u₁u₄ |
| | g²² = 1 − u₂² | g²³ = −2u₂u₃ | g²⁴ = u₁ − 2u₂u₄ |
| | | g³³ = 4u₃(1−u₃) | g³⁴ = 2u₄(1−2u₃) |
| | | | g⁴⁴ = 1 − 4u₄² |

**Drift** — from homogeneity, matched to numerics:
**(−15u₁, −15u₂, 16−32u₃, −32u₄)**

**Conformal weight** — det(L_a) is a **degree-6 polynomial in the invariants with
33 nonzero integer coefficients** (64, −64, 32, −40, all multiples of 8),
reproducing the determinant to **1.1×10⁻¹⁴**. A weighted-homogeneous ansatz
failed at 0.73: on the unit sphere |p|²+|q|² = 1 mixes degrees.

**Domain** — writing p = (u₁, p′), q = (u₂, q′) with p′,q′ ∈ ℝ⁷:

    u₃ − u₁² ≥ 0
    (1 − u₃) − u₂² ≥ 0
    (u₄ − u₁u₂)² ≤ (u₃ − u₁²)((1 − u₃) − u₂²)

**Zero violations in 40,000 samples.** The third face needs p′ ∥ q′ in ℝ⁷ —
codimension 6, so never sampled (0 hits in 200,000) and to be imposed
analytically. **g is positive definite throughout the interior**, smallest
eigenvalue never below 3.1×10⁻².

**The zero-divisor point is the geometric centre in three independent senses.** At
(0, 0, ½, 0): metric eigenvalues exactly **(1,1,1,1)**, drift exactly
**(0,0,0,0)**, Cauchy–Schwarz slack **0.5000**. The most singular point of the
algebra is the most regular point of the geometry.

---

## 6. The spectrum — and why it carries no information

**Method.** Rayleigh–Ritz on polynomials in the four invariants:
A_mn = ⟨g^{ij}∂_iφ_m ∂_jφ_n⟩, B_mn = ⟨φ_mφ_n⟩, solve Ax = λBx. Symmetric by
construction, natural boundary conditions, no grid. The domain and the invariant
weight are both handled by sampling S¹⁵ uniformly and pushing forward — the
push-forward of the uniform measure *is* the orbit-volume-weighted measure, so
the weight is never constructed.

*Superseded:* a finite-difference build gave 1785% drift between grids and none of
its numbers meant anything. It had **no boundary condition** (a ragged staircase
changing shape with refinement), drift terms double-counted inside the
second-derivative loop, and no symmetrisation. Recorded because it produced
plausible output before the control caught it.

**Results** — 12,000 samples, degree-4 basis, symmetry 10⁻¹⁷, B positive
definite, lowest eigenvalue exactly 0:

| metric | eigenvalues | ratios |
|---|---|---|
| **round** (control) | 14.70, 15.13, 30.92, 31.31, 32.15, 32.63 | 1, 1.03, 2.10, 2.13, 2.19, 2.22 |
| **Ω-weighted** | 9.54, 9.85, 19.74, 20.86, 23.23, 24.84 | 1, 1.03, 2.07, 2.19, 2.44, 2.60 |

The round case reproduces **k(k+14) = 15, 32** — validating the reduction, the
metric, the sampling and the measure against a known answer. Stable to under 1%
from degree 3 to 4.

### 6a. The result does not stand — three findings against it

**First: the forced metric gives nothing.** L_aᵀL_a is **exactly the identity on
the invariant gradient span** — eigenvalues (1,1,1,1), deviation 4.4×10⁻¹⁶ at
every sampled point — while ranging 0.086–1.914 on the eleven orbit directions.
So the algebra's own metric is non-round **only along the orbits** and **exactly
round transverse to them**, and the transverse directions are the quotient. The
reduced metric under g̃ is identical to the round one, and the spectrum under it
is unchanged.

**Second: four attempts to force Ω all closed.**

| candidate | outcome |
|---|---|
| conformal ansatz Ω = \|det L\|^{1/16} | **chosen**, not forced |
| pullback of x ↦ L_x | tr(L_vᵀL_v) = 16.0000 exactly — constant, no geometry |
| pullback of x ↦ L_xᵀL_x | rank 11, kernel 4 — blind to the invariants |
| g̃_x(v,w) = ⟨L_xv, L_xw⟩ | canonical and non-round, but reduces to round on the quotient |

The obstruction is structural: **anything built equivariantly from L transforms by
conjugation, and conjugation cannot move between conjugacy classes** — which is
exactly what the invariants label.

**Third: an arbitrary Ω gives an arbitrary spectrum.** Colbois, Dryden and
El Soufi: in dimension ≥ 3, the invariant eigenvalue functional λ_k^G is
**unbounded** over any conformal class of G-invariant metrics of fixed volume.
The quotient is 4-dimensional. So a chosen conformal factor can produce
essentially any spectrum, and "the spectrum came out irregular" is not evidence
about the algebra.

**Verdict.** The Ω-weighted column above demonstrates only that *a* conformal
choice yields irregular spacing — which is nearly generic, since roundness is the
special case. It is **not** a demonstration that composition failure creates
non-round geometry on the quotient. The round control stands as a validation of
the reduction; the second column carries no information.

### 6b. What survives

Composition failure does produce a genuinely non-round structure: σ_min ranges
**0.102–0.853** at D16 against **1.0000 exactly** at D8, det L from 0.0004 to
0.858. That is real and it is the correct reading of §4. What fails is the
inference that this reaches the **quotient**, which is where an invariant
spectrum lives.

**So Ladder Two has a verified geometry and no verified spectrum**, and the gap is
not a matter of more computation — it is closed by the theorem in the first
finding above.

## 7. Status

**Established:** the persistence relaxation that defines this ladder, measured at
every dimension; Der saturation at 14; the four invariants; the zero-divisor locus
as G₂/SU(2) with its stabiliser branching; the non-round metric from composition
failure; the exact reduction; and the **round control** reproducing k(k+14),
which validates the reduction.

**Prior literature, not original here** (§§1–3 and 7): Der = g₂ at all levels is
**Schafer 1954**. Fourfold eigenspace multiplicity, and the [4,8,4] pattern
specifically, is **Biss–Christensen–Dugger–Isaksen 2009**. The zero-divisor
condition is **Moreno 1998**. The locus as V₂(ℝ⁷) is **Biss–Dugger–Isaksen**,
with G₂-invariant metrics on it by **Reggiani**. The det L_x closed form is
**Koebisu**, arXiv:2512.13002 — and the 33-term integer polynomial reported
earlier is that paper's D₂², where D₂ is a four-term quartic.

**Closed negatively:** the Ω question. Four candidates tried, all failed, the last
by theorem (§6a). Colbois–Dryden–El Soufi makes any chosen Ω uninformative.

**Open:** whether the reduced Laplacian coefficients and the Rayleigh–Ritz
construction are themselves new — searched and not located, but a negative search
is weak evidence and novelty assessment has failed three times in this work.
Whether D32 adds anything — it saturates the same 14 with thirteen more
invariants, so probably not.

**Not claimed:** any Standard Model number. The motivation was that Ladder One
cannot reach them and this branch can in principle; *in principle* is where it
stands.
