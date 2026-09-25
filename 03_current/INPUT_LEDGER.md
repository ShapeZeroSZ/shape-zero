# Shape Zero — Input Ledger and Validation Protocol

What the construction actually requires as input, where each enters, and what
measurement would fix or break it.

Written because "the CHOSEN list is shrinking" is not a testable statement, and
"here are five numbers, two of which a lab can measure this decade" is.

---

## 1. The one given: the ordering condition

Lemma 2.1 concludes that bounded conservative motion on a one-dimensional
configuration manifold is **periodic**, and Theorem 2.2 turns that into the
integer lattice. Written out, the lemma requires four things:

    q-dot                  a time parameter
    E = q-dot²/2 + V(q)    an energy
    q on a manifold        a configuration space
    boundedness            a bounding condition

None is derived. Earlier drafts recorded these as four unlisted inputs hidden in
the notation. **That was the wrong count.** They are one primitive — the minimal
condition for order to be ordered at all — arriving in a single step because no
element is coherent without the others. Time without energy is a parameter
nothing traverses; energy without space has nowhere to be; space without
boundedness is not locality.

**Tested by removal** (`d1_given_irreducible.py`). Each removal is asserted to
have taken effect before any conclusion is drawn from it, and periodicity is
tested directly — does the orbit return to its initial state — rather than
inferred from a spectrum.

| case | removal verified | periodic |
|---|---|---|
| control, all present | — | **yes** (harmonics 1, 1.98, 2.98) |
| remove boundedness | escaped: yes | **no** |
| remove conservation (γ = 0.02) | decayed: yes | **no** |
| remove space | no motion: yes | no trajectory |
| remove time | not statable | — |

Three verified removals, three failures of periodicity. Removing time cannot be
expressed at all: "q-dot" has no referent without a parameter to differentiate
against.

**This is internal consistency, not proof.** The test runs on the given it
examines — stepping requires a time parameter, state requires space, the
computation requires work, and finite arrays require boundedness. There is no
vantage point outside the given from which to check it. That is what makes it a
primitive rather than a hypothesis.

### 1a. The scope limit — the dimensional qualifier is load-bearing

Lemma 2.1 concludes periodicity for bounded conservative motion on a
**one-dimensional** configuration manifold. That qualifier is not decoration.
*"Bounded conservative motion is periodic"* is **false** as a general statement,
and the counterexample is standard.

Hénon–Heiles is conservative and bounded below E = 1/6 with a **two**-dimensional
configuration manifold (`d1_scope_limit.py`). Instrument: top-20 spectral power
concentration, Hanning window, 10% transient discard, energy drift reported with
every row.

| energy | section point | \|dH\|/H | concentration | peaks | verdict |
|---|---|---|---|---|---|
| 0.060 | four points | ~2×10⁻⁷ | 0.998–1.000 | 3–9 | line spectrum |
| 0.155 | (+0.15, 0.00) | 2.8×10⁻⁷ | 0.9975 | 7 | line |
| 0.155 | **(−0.10, 0.10)** | 2.7×10⁻⁷ | **0.4124** | **196** | **broadband — no clock** |
| 0.155 | (+0.30, 0.05) | 2.9×10⁻⁷ | 0.9984 | 6 | line |
| 0.155 | **(−0.25, 0.00)** | 1.8×10⁻⁷ | **0.4422** | **157** | **broadband — no clock** |

**Regular and chaotic regions coexist at the same energy.** Whether a clock
exists depends on *where in phase space you are*, not on the energy alone.

**Consequence.** D1's conclusion, and the integer lattice built on it, hold on
the one-dimensional rung and do **not** extend to higher-dimensional
configuration manifolds by default. Any use of "persistence gives periodicity"
above D1 needs its own argument.

*Method note:* a first version of this script used a single fixed initial
condition and reported line spectra at every energy — the opposite conclusion.
One trajectory is not the system. The finding was recovered only by sampling
several points on the surface of section.

**What the given does not supply:** the number of spatial dimensions. q = 3 is
fixed independently by three intersecting requirements (§2b), and is a constraint
the given must satisfy rather than a consequence of it.

---

## 2. Why the rest must be dimensionless

All five selection principles are **dimensionless**. Existence is a posit;
conservativity says a quantity vanishes; persistence is a topological condition
on an orbit; minimality and plurality are counts. Rescale every quantity and all
five read identically — so no derived quantity can carry a unit.

Every output of the construction is correspondingly a pure number: harmonic
ratios 1, 2, 3; orbit constants 2 and 6; ‖c‖² = 42; 16 of 128; dim Der = 14;
loop–group 7 → 28; associator 0 or 4; Casimirs 0, 1/3, 4/3, 5/6;
sin²θ_W = 3/8; multiplicities 1, 3, 3̄; block multiplicity 4 at every level
above 8.

*On sin²θ_W:* 3/8 is the **tree-level GUT value**, matching only at the
unification scale. The measured value at M_Z is **0.23122**, and closing the gap
requires renormalisation-group running with the full particle content and scale
hierarchy — neither of which this construction has. It is a group-theoretic
identity given the representations, not agreement with data.

**Scope.** This constrains the current principle set — it does **not** say that
adding a dimensionful input is illegitimate. Adding levels above D8 adds no such
input, which is why every route to a new number *within the existing set* closed.
Dimensions enter through the given of §1, or by a stated posit.

**A posited scale is normal physics.** What distinguishes derivation from fitting
is the count in §3: independent conditions against free parameters. The open
question is therefore **what minimum set of dimensionful inputs makes the physics
come out, and whether the output overdetermines them** — not which values are
forbidden.

## 2b. What the given must satisfy: q = 3

Three independent requirements intersect at one value.

| requirement | admits |
|---|---|
| a centrifugal term exists at all — so(q) non-trivial | q ≥ 2 |
| it beats the attraction — stable bound orbits | q ≤ 3 |
| gravity has local degrees of freedom — D(D−3)/2 > 0 | q ≥ 3 |
| **intersection** | **q = 3** |

At q = 1, so(1) = 0: there is no angular momentum, so the L²/2r² term does not
exist as a term. Nothing opposes r → 0. **This excludes the invariant torus** —
the flow's frequency count saturates at 2, giving a base of 1 time + 1 space,
which cannot hold locality.

So the base is not something the dynamics generates; it is a condition the
dynamics must sit inside. Persistence needs bounded orbits, bounded orbits need
a barrier, and a barrier needs q = 3.

## 2b2. The three gauge factors — one theorem, three node sizes

**All three sit on the same derivation**, not two derived and one imported.

The A1 passivity theorem generalises exactly: for a node of **n dimers**
(ℝ^{2n}), the admissible coupling class {W : W symmetric, [W,𝕁] = 0} has
dimension **n²= dim u(n)**. Verified n = 1…5.

| factor | node | mechanism | status |
|---|---|---|---|
| **U(1)** | n = 1 | antisymmetric velocity coupling | derived |
| **SU(2)** | n = 2 | passivity ⟹ u(2) = u(1) ⊕ su(2) | derived, **measured** |
| **SU(3)** | n = 3 | passivity ⟹ u(3) = u(1) ⊕ su(3) | derived, **measured** |

*Spec v5.3, "Gauge emergence mechanisms": SU(3) from triadic coupling derived
from Rank 3 — trimer sub-lattices, three internal modes per node. This is the
programme's own route. The G₂-stabiliser construction (Günaydin–Gürsey 1973) is
a separate, imported result and is not what derives the third factor here.*

**Why n = 3 — the role triad, and it is forced.** Blocks of a Steiner triple
system have cardinality three because there are three **roles**: observer,
observed, observation (Formal Proofs §3, Definition 3.1 and 3.5). Role
completeness and role minimality together give r = 3, hence n = 7 and the Fano
plane uniquely (Theorem 3.6); AG(2,3) is excluded by pigeonhole, four incidences
against three roles (Corollary 3.7). **The node carries one block, so it carries
three modes.**

**The algebra corroborates independently.** At n = 2 there is observer and
observed with no observation — the relation is unwitnessed — and that is exactly
ℍ, associative, **associator 1.1×10⁻¹⁶**: nothing is carried. On the full algebra
the associator is **0.81**. The associator *is* the observation term: two
elements associate freely, three is the first size at which the grouping matters.
So n = 2 is excluded twice — by pigeonhole and by the vanishing associator.

**Dynamical verification** (`04_scripts/platform/phi_gauge_u3_working.py`; the companion
`phi_gauge_u3.py` is the annotated handoff copy and raises NotImplementedError
by design). Gell-Mann
generators map to symmetric real 6×6 matrices commuting with 𝕁 (8/8, 8/8),
admissible class dimension **9 = dim u(3)**. Ordering measurement, with segments
at 20-site separation:

| | |
|---|---|
| Abelian control (commuting axes, both orders) | **0.0169°** — theory says 0 |
| sim vs independent prediction, both orders | 0.31°, 0.32° |
| non-commuting sim vs prediction | 0.61°, 0.57° |
| **ordering splitting** | **65.12° measured, 64.97° predicted** |
| energy drift | 10⁻⁷ throughout |

Same footing as the u(2) benchmark (59.86° vs 59.84°). [Both benchmarks are at
κ = 0.5 (platform scripts) — superseded by the change of operating point
(2026-09-25), not retracted, and not re-run at κ\*; `model.py`'s own gate 7 at κ\* is in
MODEL_SPEC §3, "ADOPTED".]

**Scope:** u(3) on a lattice fibre is a *synthetic* gauge structure, not colour
SU(3) acting on quark representations. "Consistency", not QCD.

---

## 2c. The count

**One given** — the ordering condition of §1, irreducible within the model.

**One purchase** — plurality, spent exactly once. Measured: at D8 the
parameter-free flows have one frequency and **zero** octonionic terms; spending
plurality gives two frequencies and all fourteen.

**Everything else** — forced, licensed by minimality's exemption clause, or
listed in §2d as remaining free.

## 2d. The remaining free parameters

| # | parameter | fixes | enters at | status | kind (2026-09-25) |
|---|---|---|---|---|---|
| 1 | **ω** | the unit of time | D1, the clock | it *is* the unit, not a prediction | **1 — unit convention.** Model time is measured in the unit the φ-well fixes; ω converts it to physical time, t_phys = t_model/ω. Every dimensionless prediction is a ratio computed in model units — Δω/W, κ, F, frequency ratios — and is unchanged under ω → λω, which multiplies every physical frequency by λ |
| 2 | **ζ**, cone deficit (formerly β) | the D2 arena | D2 | contingent on an unresolved embedding | **Unclassified.** Without the Lorentzian embedding ζ has no observable consequence (§3.3), so nothing depends on it; with it, the deflection π(1/ζ − 1) does, and it would be kind 3 unless B-2 closes. Settled by supplying or refuting the embedding (`02_synthesis/C1S_SYNTHESIS.md` §14) and by resolving the 2ζ inconsistency (§3.3) |
| 3 | **κ**, gyroscopic ratio | D4 coupling strength | D4 | **measurable in a lab now** | **3 — genuine.** The physics depends on it: the Larmor splitting equals κ exactly (§3.1), and closing the J-breaking channel needs κ from 0.02 to 0.97 with k (MODEL_SPEC §3). No principle fixes its value; plurality excludes only κ = 0 (`C1S_SYNTHESIS.md` §12). model.py's 0.5 is not shown to be irrelevant to any result. [**Candidate, not adopted, 2026-09-25:** requiring J-compatibility at every wavelength gives the floor κ ≥ 2c/√(K + 2c) = 0.971737 (q = 1; 2.091 at q = 3 if the coupling does not conserve transverse momentum — open); a lower bound only, no upper side from any principle (MODEL_SPEC §3)] [**ADOPTED 2026-09-25 — genuine parameter with a DERIVED FLOOR.** The principle "J-compatibility required at every wavelength" gives **κ ≥ κ\* = 2c/√(K + 2c) = 0.971737**, at q = 3 as at q = 1 (the model's slab segments conserve transverse momentum; a finite-width segment would need 2.091). The floor is derived; the exact value is not — it remains for experiment (the Larmor split, §3.1), and the floor is a test the model can fail (a measured split below 2c/√(K + 2c) contradicts the principle). model.py's operating value is now **κ = κ\*, CHOSEN**; results at 0.5 are superseded by the change of operating point, not retracted (MODEL_SPEC §3, "ADOPTED")] |
| 4 | **a–b angle** | D8 flow frequency ratio | D8 | measurable in the same setting | **3 — genuine.** The D8 frequency ratio sweeps monotonically over [1.04, 23.9] as the angle runs 5°–150° (§3.2); plurality forbids only b = ±a |
| 5 | **C_r** | residual coupling strength | the residual sector (MODEL_SPEC §4b.1) | dimensionless; nothing yet fixes it | **3 — genuine.** The residual oscillates at f_res ∝ C_r (MODEL_SPEC §9, closed items; §5b.7), so the physics depends on it; the form is fixed, the strength by nothing (MODEL_SPEC §4b.1) |
| 6 | **c**, elastic coupling | inter-node elastic coupling, F = c(x₊ + x₋ − 2x) | the lattice (MODEL_SPEC §3) | **CHOSEN** — c = 1 in every script; added 2026-09-25 (`03_current/SCALE_SCOPING.md` §1b). Not the speed of light — see MODEL_SPEC "Notation" | **3 — genuine, as the ratio c/√5.** *Marked CHOSEN.* It cannot be scaled away: the well's fixed coefficients (√5 and 1) fix both the time and the amplitude unit, and a lattice spacing of one site cannot be rescaled, so no rescaling absorbs c (`shape_zero_tests/param_classify.py`, docstring). The physics depends on it: derived κ₂ = −0.026389, −0.017480, −0.009316 at c = 0.5, 1, 2 (`param_classify_output.txt`). Only in the long-wavelength continuum limit would c become a length-unit convention |
| 7 | **β**, lattice gyroscopic coupling | the synthetic U(1), βc(ẋ₊ − ẋ₋) | the lattice (MODEL_SPEC §3, §5) | **CHOSEN** — **β = 0.05 has no derivation**: it was one point of the original sweep {0, 0.02, 0.05, 0.10} in `04_scripts/platform/phi_gauge_test.py`, and lies just below the (0, π) decay window near 0.06 (PROVENANCE §6o, P-1). Carries the dimension of time; dimensionless only in simulation units. Added 2026-09-25 | **2 for the pinning; 2 at leading order for κ; unclassified otherwise.** *Marked CHOSEN.* The normalised asymmetry Δω/(2cβ sin k) = 1 holds for every β (Prove2Me missions 3, 4b; MODEL_SPEC §3) — kind 2. κ: the reference β-sweep gives \|Δ/Δ₀\| = 0.998339, 0.998316, 0.998390, 0.998316 at β = 0.02, 0.05, 0.10, 0.20 (`PINNED_ASYMMETRY_TEST.md`, κ note; −0.0184 ± 0.00033), but the derived κ₂ drifts at O(β²) — −0.017448, −0.017480, −0.017595, −0.018071, +3.4% from 0.05 to 0.20 (`param_classify_output.txt`) — inside that sweep's spread, so κ is β-independent at leading order only. **Not shown β-independent:** every F and κ_box measurement, the launch pieces and the beam fourth-order tests (all at β = 0.05 only), and the (0, π) decay window, which is located at β ≈ 0.06 and so depends on β. Settled by β sweeps of F and of the launch pieces. The absolute asymmetry 2cβ sin k scales with β, so an experiment fixes β by one measurement of Δω, as it fixes κ by the Larmor splitting |


*Kinds (added 2026-09-25):* **1** unit convention — scaled away without changing any
dimensionless prediction; **2** test value — every result using it shown not to depend
on it; **3** genuine parameter — the physics depends on it and no principle fixes it.
Only rows 6 and 7 are marked CHOSEN; the others are classified too because the kind-3
set (`03_current/SCALE_SCOPING.md` §6) needs every row. Where the evidence is
incomplete the entry says unclassified and what would settle it.

**Premise, not a parameter: the φ-well force law.** The on-site force
F = −(x² − x − 1), with fixed points at the golden-ratio roots and linear stiffness
√5, is **stated** in MODEL_SPEC §1 and **not derived** there or in this ledger. It
is a premise of the lattice model. (Recorded 2026-09-25, from
`03_current/SCALE_SCOPING.md` §1b.)

**Principle, not a parameter: J-compatibility required at every wavelength.**
(Adopted 2026-09-25; it replaces the premise "J-compatibility chosen at short
wavelength".) The J-breaking part of any coupling must be suppressed by the
dynamics at every wavelength — the opposite-chirality channel closed at every
travelling wavenumber. Derived consequence: the floor κ ≥ 2c/√(K + 2c) = 0.971737
on row 3 (MODEL_SPEC §3, "ADOPTED"). It fixes a floor, not a value: no principle in
the repository supplies an upper side.

**ℏ is NOT derived — retracted.** See MODEL_SPEC §4c.2a-R. The step converting
tr(λ_aλ_b) = 2δ_ab into a geometric radius was a category error: a generator
normalisation is not a metric scale. Integrality gives Area = kπ ⟹ n = k/2,
integral for every even k, and **nothing selects one**. ℏ = μℓ_f²/(T·n) with n
unfixed. This restores C1S §9's original and correct assessment: *integrality
discretises; ℏ is the unit, not the output.*

**What survives:** the cycle is ℂP¹ ⊂ ℂP² fixed by node size; the lattice spacing
never enters the single-node moment map; integrality discretises the scale.

*Superseded text follows.* ~~ℏ is derived, and the cycle question is settled.~~ The structure manifold of an
n-dimer node is the Grassmannian Gr(k,n) = U(n)/(U(k)×U(n−k)). At n = 2 that is
Gr(1,2) = S², recovering the structure sphere as a control; **at n = 3 — the
model's node, by the role triad — it is Gr(1,3) = ℂP².** Its 2-cycle is ℂP¹ with
the su(3) normalisation tr(λ_aλ_b) = 2δ_ab (verified exactly), giving

**ℏ = (μℓ_f²/T)/4**

Topology did the eliminating: H₂(S³) = H₂(S⁷) = **0**, so neither can carry an
integrality condition at all.

**And the two gauge routes agree once the cycle is fixed** —
**Isom(ℂP²) = PSU(3), dimension 8**, exactly the su(3) part of passivity's u(3);
the leftover u(1) is the one the lattice supplies through the antisymmetric
velocity coupling.

**α is an input, not a prediction** (MODEL_SPEC §4c.4b). Five closure routes were
tested; four are blocked by the model's own established results and the fifth
fails on 137 being **prime** — the model's integers factor over {2,3,7} and no
product reaches it. The model is *consistent with* the measured α and constrains
the **form** of its contributions, not its **value**.

**ℓ_f/ℓ is irrelevant to ℏ**, not open: the fundamental cycle lives inside one
node's internal fibre, so the lattice spacing never enters the moment map
(MODEL_SPEC §4c.2a).

**Still owed:** the pure numbers in G (c₈) and Λ.

Five, of which one is a unit and two are ratios measurable on a bench. [Six since
2026-09-25: the elastic coupling c, row 6, is chosen rather than measured.] [Seven
with the lattice β, row 7, also chosen; the cone deficit, row 2, is now ζ.]

*On C_r:* the residual coupling **form** is determined — f = C_r·mul(g,x) with g
octonionic and imaginary — and every constraint the selection rule imposes is
satisfied by construction rather than by tuning: even in propagation direction
(no velocity dependence), no net work (skew generator, 2.7×10⁻¹⁵), norm-preserving
(3.3×10⁻¹⁶), and reducing to the existing D8 structure at B = 0. Only the
**strength** is free.

**Not on this list, because each was shown forced:** the gauge class u(2); both
orbit metric scales (2 and 6); the D8 law; the colour decomposition 1 ⊕ 3 ⊕ 3̄;
the isospin decomposition 1 ⊕ 2 ⊕ 4; the hypercharges, from anomaly cancellation
given the representations.

**Forced conditionally, on a base existing:** the signature (1,q), the dimension
q = 3, the field equations G_μν + Λg_μν, and Maxwell. §2b shows the ladder's own
principles admit no other value of q — which constrains the base without deriving
that there is one.

## 3. Validation, in order of reach

### 3.1 κ — the live one

Sections 6–9 of the spec: passivity forces u(2), synthetic U(1) emerges from
antisymmetric velocity coupling, and the dispersion asymmetry is measurable with
every coefficient fixed by **linear spectroscopy of the same platform**.

Physical setting: coupled-oscillator arrays, photonic lattices, metamaterials.
Lattice, Euclidean, no signature obstruction, no cosmology.

**Protocol.** Fix κ by one measurement — the Larmor splitting, verified here to
equal κ exactly (0.0524, 0.1518, 0.3979, 1.0001 against κ = 0.05, 0.15, 0.40,
1.00). Then Section 9's closed-form nonlinear predictions carry **zero free
parameters**. Measure them.

**Falsification test.** Count independent conditions against free parameters,
as anomaly cancellation does. Conditions > parameters and all satisfied is
evidence. Conditions ≤ parameters is fitting. The count is computable *before*
looking at the answer, which is the discipline that makes the test real.

### 3.2 The a–b angle

Plurality forces a second generator but not which one. The observable is the
frequency ratio: parameter-free flows give exactly **one** frequency (6/6
trials), the generic flow gives **two** (6/6 at adequate resolution), and the
ratio sweeps monotonically over [1.04, 23.9] as the angle runs 5° to 150°.

Measurable wherever the u(2) structure of §3.1 is realised, since the splitting
is the same physics one rung up.

**Note:** the golden ratio is attainable near 76°, and **so is every other value
in the range**. Attainability is not selection. Any claim that φ is picked out
here would be numerology.

### 3.3 ζ — testable but the target is unclear

(cone deficit renamed β → ζ on 2026-09-25; β now denotes only the lattice gyroscopic coupling)

The deflection law π(1/ζ − 1) is exact and verified: independent of impact
parameter to 7.6×10⁻¹⁵ across two decades, against a point mass varying 100×
over the same range.

**But the cone is the Hellinger–Kantorovich cone over measure space** — mass
radial, shape angular — not a spacetime cone. So this is geodesic deflection in
*measure space*, and calling it lensing requires the cone to be the transverse
slice of a Lorentzian spacetime. That embedding is not supplied by the ladder.

Without the embedding: ζ has no observable consequence and no bound applies.
With it: the CMB-lensing bound Gμ ≤ 4.3×10⁻⁵ gives 1 − ζ ≲ 1.7×10⁻⁴.

**B-2 route, partially closed.** The "phase-boundary self-consistency
condition" is the **π/2 horizon** the spec already names as the
transport–reaction crossover, and it is the Fisher–Rao diameter — Hellinger
embedding to the unit sphere verified to 6.7×10⁻¹⁶, disjoint supports
orthogonal. Candidate ζ = (π/2)/(2π) = **1/4**. *Closing condition:* shape space
is a simplex with boundary, and the deflection law needs a closed angle.

**OPEN INCONSISTENCY, recorded not resolved (2026-09-25).** `02_synthesis/C1S_SYNTHESIS.md`
§12 states that plurality forbids the rational values of 2ζ; no derivation of that
rule is in this repository. The D2 rung calls the cone flat
(`01_source/shape_zero_zero_ladder.md`, D2: "The cone itself is flat ℝ² in polar
coordinates [FORCED given the minimal metric]"), i.e. ζ = 1 — and 2ζ = 2 is
rational, so the rule would exclude it. The candidate ζ = 1/4 (2ζ = 1/2) would be
excluded too. One of the two statements needs its derivation or its scope.

### 3.4 The scale — not testable, definitional

Any dimensionful prediction requires one scale posited. This is what every
physical theory does and it is not a defect. It does mean claims of the form
"the theory derives G" are unavailable.

---

## 4. What would falsify the construction

Stated plainly, because a theory that cannot fail is not being tested.

1. **κ measured, and Section 9's zero-parameter predictions fail.** Direct.
2. **The dispersion asymmetry has the wrong sign or scaling.** Passivity forces
   W skew hence u(2); a measured non-Hermitian component at the linear order
   would break the derivation, not just the fit.
3. **A parameter-free flow found with two frequencies**, or a generic one with
   one. That would break the plurality argument that selects the D8 law.
4. **Stable bound orbits found in a 4-spatial-dimension analogue system**, which
   would break the persistence bound giving q = 3.

Items 1 and 2 are bench experiments. Items 3 and 4 are analogue-system tests.

---

## 5. What is already tested, and failed

Recorded because it is the most solid empirical content in the whole program.

**The forced spectra do not match measured spectra.** Compact homogeneous
spaces give quadratic towers m² ∝ k(k+n−1). Charged-lepton m² ratios are 1,
4.3×10⁴, 1.2×10⁷ — off by four to seven orders, unfixable by rescaling since
ratios are scale-free. Hadron Regge trajectories grow **linearly** in n; every
spectrum here grows **quadratically**. That is a mismatch of functional form,
not of number.

The one apparent match — S² giving 1, 3, 6, 10, matching molecular rotation —
is vacuous: a rigid rotor *is* the Laplacian on S².

**And the route around it closed.** Irregular spectra need a non-round metric;
a non-round metric needs composition to fail; composition failing is what
persistence excludes, since Lemma 2.1 concludes *periodic* and periodicity is
exactly L_a² = −I. Four attempts to force a metric above D8 all closed, the last
by theorem: L_aᵀL_a is exactly the identity on the invariant gradient span
(4.4×10⁻¹⁶), so the algebra's own metric reduces to the round one on the
quotient.

---

## 6. What the construction is

A ladder of **what can be tracked**, not of what can be. The 112 discarded
orientations are 8-dimensional unital algebras with **zero** zero divisors —
fully invertible, fully functional. They lose only |xy| = |x||y|, so the loss is
**measurability**, not structure: in the 16, measuring the parts tells you the
whole; in the 112 it does not.

Every rung inherits this bias. Persistence keeps what stays put long enough to
be measured; conservativity keeps what does not leak; minimality keeps what has
few enough parameters to pin down.

That is a defensible thing to be, and it sets the scope: the construction
derives structure that exists and no number that is missing. The five inputs
above are where measurement enters, and §3.1 is where it can enter first.
