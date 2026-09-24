# Revision 2 — the base was wrong

**Read this before any other file in the package.** A substantial block of the
work here was computed on a 2-dimensional base. The D8 arena is not
2-dimensional. Results are classified below; nothing has been deleted, but
several conclusions do not mean what their scripts say they mean.

## The error

I took the base to be the 2D cone, from `phi_gauge_wilson.py` in the C1
package. That is a **D2-level** script — the synthetic U(1) and gauge-emergence
work, a 2D lattice carrying a 2-component internal space. It is not the D8
arena.

The C1 rung scripts show the arena advancing with the ladder at every rung
without exception:

| rung | arena in the C1 script |
|---|---|
| D1 | scalar |
| D2 | `z = np.array([rstar + 0.02, 0.0])` — ℝ² |
| D4 | 4×4 structure matrices, `u = [z1.real, z1.imag, z2.real, z2.imag]` — ℝ⁴ |
| D8 | `np.zeros(8)`, `rng.normal(size=8)` — ℝ⁸ |

So the D8 base is 𝕆 = ℝ⁸, and Im(𝕆) = ℝ⁷ after the deletion that Proposition
2.8 already uses at D3. **G₂ acts on the base as well as the fibre.** This was
not a judgement call that went the wrong way — it was reading a D2
implementation as the D8 arena.

## Why it changes conclusions rather than arithmetic

Two things depend on the base dimension, and both were load-bearing.

**Scaling.** A term with *n* derivatives on a *d*-dimensional base scales as
μ^(d−n) under dilation. At d = 2 the kinetic term (n = 2) is **marginal**,
which is why nothing could stabilise the collapse, why a coupling bound was
needed, and why the "no window" result followed. At d = 7 the kinetic term
scales as μ⁵ and bounds the action by itself. Verified in
`d8_base_corrected.py`: the exponent law reproduces at d = 2 (0, −2) and,
after a resolution study, at d = 3 (+0.998, −0.956 against +1, −1).

**An available term.** On a 2D base the cubic three-derivative octonionic term
φ^{μνρ} c_{abc} ∂_μψ^a ∂_νψ^b ∂_ρψ^c does not exist — Λ³ of a 2-dimensional
space is zero. Measured at exactly **0.000e+00** for every 2D configuration
sampled. That is precisely why the surviving 2D term had to carry a second
derivative and sit at (3, 4). With G₂ on the base the term exists at (3, 3),
is generically nonzero, and equals exactly **42** on the identity map.

## Follow-up: the (3,3) term is inert

`d8_cubic_term.py`. The newly available cubic term is a genuine new invariant —
rank 4 against rank 3 for the trace invariants alone — and its Euler–Lagrange
derivative vanishes identically (3.4×10⁻¹⁴). It is a **null Lagrangian**.

The mechanism: c contracted with a symmetric tensor is zero (8.9×10⁻¹⁶), and
both EL terms have exactly that form, since φ is antisymmetric in (σ,ν) and
(σ,ρ) while the second derivatives are symmetric there. **The antisymmetry that
lets the term exist in 7D is what makes it inert.** Control: tr(J)³ has EL = 290,
so nullity is specific to the octonionic contraction, not to cubics.

Still untested: (3,4) and higher octonionic terms in 7D. The 2D uniqueness
argument does not transfer — with G₂ acting on the base, base and target
indices can contract with each other, opening contraction classes that did not
exist before.

## The D8 rung: what replaces transport

`z1_d8_dynamics.py`. Theorem 4.1 forbids copying D4 — path-ordered transport is
associative, so it only ever realises the quaternion subalgebras. What is new
at D8 is measurable three ways:

- **Loop–group gap.** ℍ: unit loop 3, generates 3, gap 0. 𝕆: unit loop 7,
  generates **28 = so(8)**, gap 21. Under G₂ that decomposes as g₂ ⊕ 7 ⊕ 7
  (Casimir 4.0×14 and 2.0×14, the latter two isomorphic 7's), with the loop
  directions disjoint from g₂ at 3.8×10⁻¹⁶. Non-associativity manufactures the
  automorphism group out of the loop.
- **Associative sectors.** Artin: associator 9.1×10⁻¹³ inside any 2-generated
  subalgebra, 57.5 generic. The seven Fano lines are regions where the dynamics
  is D4-like; octonionic content lives only in transit between them.
- **Bracketing as a degree of freedom.** Product spread over parenthesisations
  is 2.2×10⁻¹⁶ within a Fano line and grows outside it: 0.77, 0.90, 1.11 for
  n = 3, 4, 5. Identically zero at D4 for every sequence.

Recorded miss: predicted the generated algebra at 21, measured 28. The 21 in
`z1_d8_attempt.py` is the commutator span alone.

## The pattern breaks at (3,4)

`d8_census_clean.py`. Cubic, four-derivative invariants in the corrected arena.
Index parity (7 = 3 + 2 + 2) means every one carries exactly one c, so there
are no non-octonionic terms at this order.

- (3,4) invariant space: **22**
- divergences, hence null: **8**
- **not divergences, hence contributing to the field equations: 14**

Stable at tolerances 1e-8 through 1e-12, confirmed by explicit projection
(rank 14 outside the divergence span, residual singular value at 86% of the
leading scale). **This is the first positive dynamical result for the
octonionic sector.** Six earlier mechanisms each denied it; at (3,4) fourteen
independent invariants survive variation. The difference from (3,3): with H
present as an explicit field, c can contract against its indices without being
forced to meet a symmetric second-derivative pair.

Recorded miss, and it was a logic error rather than a numerical one: the
coverage test was written as `rboth == r34`, which only checks containment.
Coverage requires `rdiv == r34`. The script printed the opposite conclusion
until corrected. Predicted dimension was also ≤ 8 against an actual 22.

**Scope:** existence of non-null terms is not selection. Whether any of the 14
is picked out by conservativity, minimality and persistence is untouched.

## Channels, and O vs S^7 compared on outcomes

`d8_channels.py`. Two follow-ups to the (3,4) census.

**O vs S^7 give identical counts** — 22 invariants, 8 divergences, 14 non-null,
both readings. The census is pointwise, and at a point the tangent space of S^7
is Im(O); the second fundamental form of S^7 in O is purely normal and the
normal direction was never counted. The choice does not bite here. Where it
would bite is global: the Moufang loop structure, Artin's associative sectors,
and anything topological — none tested.

**Channel content, partial by construction.** Restricting J to a single G₂
channel gives non-null counts 4, 0, 0, 0 for the 1, 7, 14, 27 channels. Those
sum to 4 against 14 in the full space, so **ten of the fourteen non-null
directions live in cross-channel mixing**, which a single-channel probe cannot
see. The surviving dynamics is therefore **not a clean irrep class** — it needs
the channels of J to talk to each other.

Two recorded misses. The earlier proposal to "decompose the 22 under G₂" was
ill-posed: those are G₂-invariant scalars, with nothing for the group to act
on. And a claim that 45 samples were ample to resolve rank 22 was wrong — the
rank cap applies to the sample matrix against 105 columns. It produced
impossible output (rank 32 under a restriction that can only lower rank from
22, divergence rank 45 against 8, a negative count), which is how it was
caught. Both the batched rewrite and `optimize=True` on the einsum paths were
needed to make the computation finish at all; the packaged script now carries
both and runs end to end.

**O vs S^7, decided on outcomes:** S^7. Every measured feature distinguishing
D8 from D4 lives on the loop — the 7 → 28 dimension gap, Artin's seven
associative sectors, bracketing as a degree of freedom. Flat O carries none of
them. The local term census is neutral between the two readings; the loop
structure is not.

## Sector selection: the octonionic content is the crossing

`d8_sector_selection.py`. The three selection principles are stated for
trajectories — no net work, bounded enduring motion — and the 14 surviving
terms are static on a 7D base with no time direction, so they cannot be applied
as written. What the rung does supply is Artin's sectors.

Restricting the field data to a Fano line's quaternion copy:

| | non-null |
|---|---|
| free to cross sectors | **14** |
| confined to one sector | **6** |
| irreducibly octonionic | **8** |

Identical for all seven lines. The 6 are inherited: within a line c degenerates
to the ε of that quaternion copy, so they are D4-order content. **The other 8
exist only when the field leaves a quaternion subalgebra** — the octonionic
dynamics lives in transit between sectors, not in any sector.

This explains the thread's pattern. Inside a sector everything associates and
the composition identity leaves the cross product carrying nothing beyond the
norm, which is why six earlier mechanisms found no octonionic dynamics. The 8
extra terms live exactly where associativity fails. It matches the loop
measurement directly: bracketing spread was 2.2e-16 inside a line and grew
outside it.

Recorded miss: predicted the confined count at ≤ 3, measured 6.

**Not** an application of conservativity, minimality or persistence. Those still
require trajectories, which D8 does not yet have.

## Crossing a Fano line: a step function

Measured by expanding the field's support across sectors:

| support | points | non-null |
|---|---|---|
| one Fano line | 3 | **6** |
| two lines | 5 | **14** |
| three lines | 6 or 7 | 14 |
| all of Im(O) | 7 | 14 |

The moment the support touches a second line, all eight cross-sector terms
switch on together. **No intermediate regime exists.** Any two Fano lines share
exactly one point, so two sectors intersect in span{1, e} — a copy of ℂ, the
D2 rung. The sectors are glued along D2.

**CORRECTED by `d8_crossing_trigger.py`.** The step-function claim above was an
artifact of scanning only unions of lines, which jump from 3 points to 5. A scan
over all supports finds an intermediate value:

| points | lines inside | non-null |
|---|---|---|
| 3 | 0 | **0** |
| 3 | 1 | 6 |
| 4 | 0 | **0** |
| 4 | 1 | **13** |
| 5+ | 2+ | 14 |

The real threshold is **containment of a complete Fano line**, not associativity.
Every (3,4) invariant carries exactly one φ, and φ is supported only on lines, so
a support missing every line gives exactly zero terms at any size. Above the
threshold the count grows with support and saturates at 14 by five points.

This also revises the interpretation: the octonionic content is not "what appears
in transit between sectors" but **what φ can see**, and φ exists only on lines.
The associator quantisation (0 or exactly 4) is separately verified and stands.

### Note on an external document

An independently produced write-up built a cosmological framework on this sector split. Its
one correct ingredient is the associator quantisation (0 for Fano triples, 4
otherwise) — independently confirmed here. Its central input is wrong: it uses
2 sector-local and 12 cross-sector terms; the measured values are **6 and 8**.
Dark matter, the Hubble ratio and the PMNS loop factor all carry N_cross = 12;
with 8 the dark-matter figure misses Planck by 33%. Its verification script also
contradicts its own text (computes λ = 0.227025 where the text claims 0.22429)
and its associator test assigns values from a lookup rather than computing them,
with a sorting bug that misclassifies 3 of 7 lines.

## A D8 flow written without reference to sectors

`z1_d8_flow.py`. State psi on S^7; conservativity forces psi-dot orthogonal to
psi; minimality gives the simplest algebraic evolution, psi-dot = psi*a for one
fixed imaginary octonion. No mention of Fano lines, Artin, or associativity.

| law | sweep dim | associator | octonionic terms |
|---|---|---|---|
| psi-dot = psi a | **2** | 4.9e-16 | **0** |
| psi-dot = psi a + b psi | **5** | 1.76 | **14** |

Norm conserved to 1e-12 / 1e-11 under integration. **Confinement emerges rather
than being imposed** — psi(t) = psi_0 exp(at) traces psi_0 times the COMPLEX
subalgebra generated by a, so the one-generator flow does not reach D4 at all;
it stays at D2. And **plurality is the switch**: with one degree of freedom the
octonionic sector is absent, not merely reduced. The program's single purchased
assumption is exactly what opens it.

### Generality, and what the principles select

Twelve random (psi_0, a, b): **12/12 identical**. One generator gives sweep 2,
zero terms, associator zero. Two generators give sweep exactly 5 — never 6 or
7 — with 22 / 8 / 14 and nonzero associator. The result is not an artifact of
one draw.

Applying the three principles to the flow:

| flow | sweep | non-null | associator |
|---|---|---|---|
| b = a (anticommutator) | 3 | **0** | ~0 |
| b = −a (**commutator**, psi-dot = [psi,a]) | 3 | **6** | ~0 |
| generic (15°–135°) | 5 | **14** | 1.5–1.8 |

**The commutator flow yields exactly the six sector-local terms with zero
associator** — the quaternionic content, and the commutator is the Lie-algebra
structure D4 runs on. The D4-preserving choice is precisely the commutator.

- *Conservativity* does not select: norm preservation is automatic for
  imaginary a, b.
- *Persistence* does not select: S^7 is compact, so all motion is bounded.
- *Minimality* half-selects: the a–b angle is a genuine free parameter
  (G_2 normalises a; the stabiliser SU(3) leaves the angle invariant), but it
  is generically unobservable — structure appears only at the two isolated
  points above.

**The principles do not select among the 14.** They select the flow, and the
flow delivers 0, 6, or 14 wholesale. The 14 arrive as a package because they
all live on the same 5-dimensional support generic plurality sweeps.

Scope: the 12/12 check covered generic flows only. The commutator and
anticommutator rows are single data points at special configurations and need
the same sweep before being relied on.

### The top rung cannot be reached parameter-free

`z1_d8_minimality.py`. Minimality removes the second generator (G2 normalises
a; an independent b carries a residual SU(3) orbit), leaving two parameter-free
laws. Closed forms verified to 4e-16:

    psi a + a psi = 2 psi_0 a - 2 <psi_vec, a>     lies in span{1, a}
    psi a - a psi = 2 psi_vec x a                  purely imaginary

| law | imaginary support of the motion | terms | associator |
|---|---|---|---|
| anticommutator | **2** | 0 | ~0 |
| commutator | **3** | 6 | ~0 |
| generic (needs b) | **5** | 14 | ≠ 0 |

The anticommutator rotates one complex line and freezes every perpendicular
component (drift 2.2e-15) — D2 motion. The commutator generates a quaternion
copy, associator 6.5e-16 — D4 motion. **Neither parameter-free law is a D8
flow.** The sign is a choice of which lower rung to re-run inside the top one.

**MINIMALITY AND D8 CONTENT ARE IN TENSION.** Eliminating the free parameter is
exactly what drops the flow to D2 or D4; the only law reaching octonionic
content carries a parameter nothing forces. This is a structural obstruction,
not an unspent choice.

Recorded miss: predicted the anticommutator's generated algebra at dimension 2;
it is 4. The prediction conflated the span of the MOTION with the algebra that
span closes into. The motion's imaginary support is 2, which is what gives zero
terms.

### Plurality selects the flow that minimality seemed to forbid

`z1_d8_plurality.py`. An audit of the C1 formal proofs document finds the
selection principle **minimality** stated in Section 1 and then **never cited in
any lemma, theorem or proof**. Every actual elimination is performed by
something else — conservativity (D1 first integral), Noether (D2 void constant),
Hurwitz/Frobenius (D3), passivity (D4 u(2), Section 6 explicitly). The word
recurs later only as "role minimality," a distinct postulate. So minimality has
never been load-bearing, and its bite at D8 is its **first application**.

The resolution is in the principle as written: minimality exempts what the
purchased plurality assumption forces. Counting independent frequencies at
T = 600:

| law | frequencies | over 6 trials |
|---|---|---|
| anticommutator | 1 | 6/6 |
| commutator | 1 | 6/6 |
| generic (two generators) | 2 | 6/6 |

Each parameter-free law is a single periodic angle — **one** dynamical degree of
freedom. Plurality states there is more than one, so it excludes both, and
minimality's own clause then admits the second generator. **The tension resolves
without a fourth principle**, by reading two existing ones jointly.

The frequency 0.31833 appears in every generic spectrum and equals the
single-generator frequency, so the generic flow is the one-generator motion plus
an independent second one.

Recorded miss: a coarse T = 60 run reported 5/6 for the generic case. The
outlier's two peaks are separated by 0.0133, below that run's 0.0167 resolution.
Reported before being checked; corrected to 6/6.

Residual: plurality forces *a* second generator, not *which*. The a–b angle
survives — permitted by exemption rather than forbidden, and shown earlier to
leave the outcome at 5 / 14 regardless of value.

### Third correction to the crossing threshold

A generic 3-plane gives 9 invariants — the same as a Fano line. So
"containment of a complete Fano line is the threshold" is a **basis-position
artifact**: coordinate triples off a line give 0 only because phi has no
component there. The invariant variable is the DIMENSION of the support:
2 -> 0, 3 -> 9, 5 -> 22. Three successive corrections to this one question,
each from a finer probe, each time a structured sample read as general.

Also fixed: a purely relative rank threshold on an identically-zero matrix
(sv[0] ~ 1e-14) reported rank 67, above the full-space 22. An absolute floor
was needed alongside it.

## Classification of every result

### Stands — base-independent

These concern the algebra, the target, or representation theory, and are
unaffected.

- `a2_invariance_hinge.py` — A-2 leg (i), A-3 census, the G₂ FFT citation
- `a2_leg_ii.py` — leg (ii) withdrawn as false-or-definitional
- `a2_intrinsic_torsion.py` — 147 = 98 + 49, classes 1+7+14+27
- `a2_unification_audit.py` — couplings **are** the irrep projectors; Λ³V
  spin-sourced torsion entirely intrinsic; φ pure W₁
- `d8_exact_verification.py` — 16 of 128 in exact integer arithmetic
- `z1_holonomy_orbit.py` — the 7 failing pulses move the algebra, don't break it
- `coset_audit.py` — the canonical map v ↦ L_v; target metric forced by Schur
- `target_wz_check.py` — dΦ = −6Ψ, no WZ term
- `sigma_topology.py` — target is S⁷; π₁ = π₂ = 0
- `b7_dislocation.py`, `b7_boundedness.py` — B-7, the E_G envelope from
  boundedness alone

### Superseded — computed on a 2D base

The computations are correct; their subject is a theory the program does not
have.

- `g2_base_question.py` — the bundle/manifold fork. Now resolved in favour of
  the manifold reading, so the four torsion classes **are** available.
- `cone_coupling_search.py`, `cone_coupling_d2.py` — coupling orders, the
  Skyrme collapse, the apex term
- `cone_derrick.py`, `cone_threshold.py`, `cone_competitors.py`,
  `cone_infimum.py`, `cone_ground_state.py` — the instability, the bound
  λ < 2√(μc₄), and the "no window" conclusion. **All three rest on the
  kinetic term being marginal, which is a d = 2 property.**
- `cone_vertex.py`, `vertex_renormalisation.py` — vacuum chirality and its
  divergence

### Needs recheck — meaning may survive, statement does not

- `octonionic_term_variation.py`, `octonionic_term_covariant.py` — the term is
  not a total derivative and enters the field equations. Likely still true, but
  established for the (3, 4) term, and the (3, 3) term now exists.
- `octonionic_term_kernel.py` — the two kernels (collinear gradients, planar
  target image) are geometric and probably survive; the "blind to the radial
  sector" reading was about the cone.
- `octonionic_term_parity.py` — orientation odd, base-reflection odd, even
  under both. The sign structure should persist, but "base parity" means
  reflection in ℝ⁷ now, and the (3, 3) term needs its own check.

## Second failure, unrelated to the base

`vertex_renormalisation.py` derived ⟨O²⟩ = 28 L⁴ Σ A′² G G G. A direct
Monte-Carlo check against the exact Wick sum gives ratios 0.52, 0.57, 0.59 at
N = 8, 12, 16 — not unity and **not constant**, so it is not a missing factor
I could identify. A control test substituting ε_{ijk} on 3 components for
c_{abc} on 7 reproduces the same ratios (0.524, 0.573, 0.564), which proves
**the fault is in the general Wick assembly and is not octonionic**.

What survives from that script: the collapse identity
A = 2(k₁×k₂)(k₁²+k₂²+k₃²), verified to 1.1×10⁻¹³, and the log² scaling with
its soft-region mechanism. The coefficient 28 is **withdrawn**.

## Standing tally of prediction failures

Recorded because the rate matters when weighting anything unverified here:
the aliasing amplitude, the pure-cubic scaling, log-vs-log², the apex-artefact
diagnosis, the linear quadrature, the V1 normalisation, the V1 *diagnosis*,
and the base itself. Most were caught by instrument checks rather than by
headline numbers; the base was caught only by reading the C1 rung scripts
directly.
