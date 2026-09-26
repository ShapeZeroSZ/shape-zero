# Scale scoping — what the model can and cannot fix, before any derivation

Written 2026-09-25 as a scoping document, before any attempt at the scale
question. It changes no result and no other document. Every claim cites a file
and section; where the cited documents disagree or are silent, that is said.

**Sources.** The request named `PREMISE_LEDGER.md`; **no such file exists in this
repository.** [**Updated 2026-09-25:** it now exists — `03_current/PREMISE_LEDGER.md`,
written after this document, lists every premise with its status. This document was
written from the sources below and is not revised against it.] Its role is filled by
`03_current/INPUT_LEDGER.md` (the input ledger and validation protocol), used here with `00_START_HERE/MODEL_SPEC.md`
(§0b, §1–§3, §4b.1, §4c, §9) and `00_START_HERE/PROVENANCE.md` (§6c, §6d).

---

## 1. Every free parameter and every input

Status key: **FIXED** — fixed by a principle, with the proof or derivation cited;
**CHOSEN** — set by a choice the documents record as a choice; **OPEN** — not
fixed, and a fixing is sought or its absence recorded.

### 1a. Dimensionful inputs

| symbol | what it is | units | status | source |
|---|---|---|---|---|
| ω (or T = 1/ω) | the D1 clock frequency — *the* unit of time | [T]⁻¹ | **CHOSEN** — "it *is* the unit, not a prediction" | `INPUT_LEDGER.md` §2d #1; `MODEL_SPEC.md` §4c table |
| ℓ | base lattice spacing | [L] | **CHOSEN** (input) | `MODEL_SPEC.md` §4c table |
| μ | node mass | [M] | **CHOSEN** (input) | `MODEL_SPEC.md` §4c table |
| ℓ_f | fibre size | [L] | **OPEN** — "possibly = ℓ"; rigidity forbids it *varying* but does not identify it with ℓ | `MODEL_SPEC.md` §4c table and §4c.3(2) |
| fibre metric scale k | overall scale of the metric on the angular fibre ℂP² | pure number multiplying a length² (sets ℓ_f's meaning) | **OPEN** — the largest open item; ℏ, c₈ and Λ are all functions of it | `MODEL_SPEC.md` §9, first row |

The speed of light enters only as the convention c = 1 used for dimensional
arithmetic ([L] = [T]); it is not a model parameter (`MODEL_SPEC.md` §4c.0,
checked by `04_scripts/session/dimensions.py`). **Symbol collision:** the same
letter c is the elastic inter-node coupling in the force law (§1b below).

### 1b. Dimensionless couplings and ratios

| symbol | what it is | status | source |
|---|---|---|---|
| on-site force −(x² − x − 1), linear stiffness √5 | the φ-well; fixed points at the golden-ratio roots | **stated as the model's force law**; the sources used here state it and do not derive it [**2026-09-26:** contributes no dimensionless parameter — √5 and φ are coordinates; the premise is a quadratic on-site force applied radially (`MODEL_SPEC.md` §1b)] | `MODEL_SPEC.md` §1 |
| c (elastic) | inter-node elastic coupling, F = c(x₊ + x₋ − 2x) | **CHOSEN** — c = 1.0 in `model.py` and in every lattice script; **not listed** among the free parameters of `INPUT_LEDGER.md` §2d, so its status is unrecorded there | `MODEL_SPEC.md` §3; `04_scripts/session/model.py` (C = 1.0) |
| κ (gyroscopic ratio) | intra-node gyroscopic coupling, F = κ𝕁v; D4 coupling strength | **OPEN, measurable** — fixed by one bench measurement (the Larmor splitting equals κ). [Candidate floor, not adopted, 2026-09-25: κ ≥ 0.971737 if J-compatibility is required at every wavelength — a floor, not a value; MODEL_SPEC §3] [**ADOPTED 2026-09-25:** **DERIVED FLOOR** κ ≥ 2c/√(K + 2c) = 0.971737; value still OPEN, measurable; operating value κ = κ\* CHOSEN] [**Finding 2026-09-25, not adopted:** the strongest persistence the model admits confines κ to [4.9, 7.5] at q = 3, c = 1, excluding κ\* (MODEL_SPEC §3, "FINDING")] [2026-09-26: that window was the elementwise node form's; under the adopted radial form it is [κ\*, ∞) (MODEL_SPEC §1a, §3)] | `INPUT_LEDGER.md` §2d #3 and §3.1; `MODEL_SPEC.md` §2, §3; `model.py` (KAPPA = 0.5 until 2026-09-25, now κ\*) |
| β (lattice) | inter-node antisymmetric velocity coupling, βc(v₊ − v₋) — the synthetic U(1) | **CHOSEN** — β = 0.05 in the κ scripts; `MODEL_SPEC.md` §4c.4 calls it the U(1) "charge" and "already an input (§2d of the ledger)" | `MODEL_SPEC.md` §4b.1, §4c.4 |
| ζ (cone deficit; β before 2026-09-25) | the D2 arena's cone deficit | **OPEN** — contingent on an unresolved embedding; B-2 candidate ζ = 1/4 has an unmet closing condition | `INPUT_LEDGER.md` §2d #2, §3.3 |
| a–b angle | D8 flow frequency ratio (second generator's direction) | **OPEN, measurable** — plurality forces a second generator, not which; every ratio in [1.04, 23.9] [corrected 2026-09-26: 23.9 → 22.93, a resolution artifact; exactly 1/sin(θ/2), `INPUT_LEDGER.md` §3.2] is attainable | `INPUT_LEDGER.md` §2d #4, §3.2 |
| C_r | residual coupling strength, f = C_r·mul(g, v) | **OPEN** — the *form* is determined; the *strength* is free | `MODEL_SPEC.md` §4b.1; `INPUT_LEDGER.md` §2d #5; `MODEL_SPEC.md` §9 |
| J-compatibility ([W, 𝕁] = 0) | a condition on the coupling, not a number | **CHOSEN** at short wavelength, **derived at long** (emergent below k_c) [**SUPERSEDED 2026-09-25:** now a **PRINCIPLE — required at every wavelength**; delivered by the dynamics once κ ≥ κ\*, which is its derived consequence] [**CORRECTED 2026-09-25:** at linear order only — at κ\* the model's u∘u nonlinearity converts a-waves near k = 0.49, 1.25 into the opposite chirality (MODEL_SPEC §3, "CORRECTION")] | `MODEL_SPEC.md` §3 |
| n (flux integer in ℏ) | ℏ = μℓ_f²/(T·n) | **OPEN** — integrality permits the even family n ∈ {1, 2, 4, 8, …} and selects none | `MODEL_SPEC.md` §4c.2a-R |
| c₈ | pure number in G₄ = [c₈/(2π²)]·(ℓ⁵/ℓ_f⁴)/μ | **OPEN** — "not yet extracted" | `MODEL_SPEC.md` §4c.3 |
| Λ's number | Λ = (number)/ℓ² | **OPEN** — "not yet extracted" | `MODEL_SPEC.md` §4c table |
| α | e²/ℏc | **CLOSED as an input** — not derivable; five routes tested | `MODEL_SPEC.md` §4c.4b |

**Symbol collisions, flagged and not resolved here.** [(i) **resolved 2026-09-25:**
the cone deficit is renamed ζ; β is the lattice gyroscopic coupling, `MODEL_SPEC.md`
§4c.4 corrected.] (i) β names both the
lattice gyroscopic coupling and the cone deficit; `MODEL_SPEC.md` §4c.4 cites the
ledger's §2d for the lattice β, whose §2d entry is the cone deficit. Whether they
are one parameter is not established in the cited sections. (ii) κ names both the
intra-node gyroscopic ratio (`INPUT_LEDGER.md` §2d #3) and the A² coefficient of
the pinned asymmetry (`MODEL_SPEC.md` §5). (iii) c names both the elastic coupling
and the speed of light (§1a). A derivation that uses any of these must say which
it means.

### 1c. Fixed by a principle

| quantity | value | the principle | source |
|---|---|---|---|
| orbit metric scales | 2 at D4, 6 at D8 | \|1\| = 1, forced by 1·1 = 1 and the composition law | `MODEL_SPEC.md` §0b |
| node size | n = 3 | the role triad (Formal Proofs §3, Theorem 3.6); second route: SU(3) in dimension 6 | `INPUT_LEDGER.md` §2b2; `MODEL_SPEC.md` §4c.2b |
| spatial dimension | q = 3 | three intersecting requirements | `INPUT_LEDGER.md` §2b |
| total dimension | D = 8 | dim(base) + dim(ℂP²) = 4 + 4 | `MODEL_SPEC.md` §4c.2b |
| gauge class dimension | n² = dim u(n) | passivity plus J-compatibility (machine-verified, Prove2Me missions 1, 2) | `MODEL_SPEC.md` §3 |
| propagation asymmetry | 2βc sin k₀, independent of stiffness and transverse wavenumbers | proved (Prove2Me missions 3, 4b) | `MODEL_SPEC.md` §3 table |
| residual coupling form | C_r·mul(g, v) | conservativity, norm preservation, evenness | `MODEL_SPEC.md` §4b.1 |

---

## 2. Why the earlier ℏ derivation was retracted

The claim was ℏ = (μℓ_f²/T)/4, from a flux quantum on ℂP¹ ⊂ ℂP²
(`PROVENANCE.md` §6d, last table row; `MODEL_SPEC.md` §4c.2a, marked SUPERSEDED).

**The error** (`MODEL_SPEC.md` §4c.2a-R; `PROVENANCE.md` §6d): tr(λ_aλ_b) = 2δ_ab
normalises the **Lie-algebra generators**. Turning it into a definite
**geometric radius** on ℂP² is a category error — a generator normalisation
says nothing about the overall scale of the metric on the quotient.

**What integrality actually gives** (§4c.2a-R): Area(ℂP¹) = kπ ⟹ n = k/2,
integral for every even k, so n ∈ {1, 2, 4, 8, …} all pass and nothing selects
one. Three of the derivation's own statements implied k = 8, 2 and 1, the last
violating integrality (§4c.2a-R table). An earlier correction had already found a
π-versus-4π convention mismatch between the lemmas (`PROVENANCE.md` §6c).

**Corrected status:** ℏ = μℓ_f²/(T·n) with n unfixed — ℏ is the unit fixed by
integrality, not an output. This restores the original assessment of
`02_synthesis/C1S_SYNTHESIS.md` §9, *integrality discretises; ℏ is the unit, not
the output* (§4c.2a-R). **What survives:** the cycle is ℂP¹ ⊂ ℂP², the lattice
spacing never enters the single-node moment map, and integrality discretises the
scale (§4c.2a-R, "What survives").

*Note on the record:* `PROVENANCE.md` §6c, written before the retraction, still
reads "holds" for the ℏ lemma set. The retraction is recorded in §6d and in
`MODEL_SPEC.md` §4c.2a-R; §6c is not updated by this document.

**The lesson recorded** (`PROVENANCE.md` §6d): the ℏ claim reversed a prior
cautious assessment that was correct; "a reversal of a prior cautious assessment
deserves more scrutiny than a new claim, not less."

---

## 3. Units are conventions; only pure numbers are derivable

**No model derives the SI value of ℏ or G.** A value such as
ℏ = 1.054571817×10⁻³⁴ J·s encodes the choice of the joule and the second as much
as any physics: change the units and the number changes while nothing physical
does. The model's own statement of this is `MODEL_SPEC.md` §0b: all five
selection principles are dimensionless, so no quantity derived from them alone
carries a unit, and "a dimensionful prediction claimed as following from the
principles alone is an error." `INPUT_LEDGER.md` §3.4 states the same: "claims of
the form 'the theory derives G' are unavailable."

What a model *can* fix is a **dimensionless** number: a ratio of couplings, an
integer, or a quantity already expressed in the model's own units. For ℏ that is
the integer n in ℏ = μℓ_f²/(T·n); for G it is c₈ in G₄ = [c₈/(2π²)]·(ℓ⁵/ℓ_f⁴)/μ
(`MODEL_SPEC.md` §4c). A dimensionful prediction is legitimate as **principles
plus stated inputs**, judged by the count of independent conditions against free
parameters (`MODEL_SPEC.md` §0b; `INPUT_LEDGER.md` §3.1).

### 3a. Dimensionless quantities the model could in principle fix

| quantity | fixed now? | source |
|---|---|---|
| **sin²θ_W = 3/8** | **yes** — group-theoretic, from the representations (Tr T₃² = 2, Tr Q² = 16/3); tracked by `00_START_HERE/check_consistency.py` (TRACKED, "sin2 theta_W"). It is the **tree-level unification value**: matching the measured 0.23122 at M_Z needs renormalisation-group running the model does not have, so it is not agreement with data | `INPUT_LEDGER.md` §2; `MODEL_SPEC.md` §4c.4 |
| orbit constants 2, 6; ‖c‖² = 42; dim Der = 14; Casimirs 0, 1/3, 4/3, 5/6; 16 of 128; multiplicities 1, 3, 3̄ | **yes** | `MODEL_SPEC.md` §0b; `INPUT_LEDGER.md` §2 |
| hypercharges | **yes, given the representations** — anomaly cancellation, conditions > parameters | `MODEL_SPEC.md` §0b; `INPUT_LEDGER.md` §2d |
| n = 3, q = 3, D = 8 | **yes** | §1c above |
| the flux integer n in ℏ | **no** — an even family | `MODEL_SPEC.md` §4c.2a-R |
| c₈, Λ's number | **no** — not yet extracted | `MODEL_SPEC.md` §4c.3, §4c table |
| ℓ_f/ℓ | **no** — open | `MODEL_SPEC.md` §4c.3(2) |
| the fibre metric scale k | **no** — the §9 blocker | `MODEL_SPEC.md` §9 |
| coupling ratios among κ, β, C_r, and the a–b angle | **no** — each is free (§1b); no relation between them is derived | `INPUT_LEDGER.md` §2d; `MODEL_SPEC.md` §4c.4a |
| α_GUT and α | **no, and closed** — the model owns the 5/8 group factor, not the magnitude | `MODEL_SPEC.md` §4c.4, §4c.4b |

---

## 4. The two open items that block scale

### 4a. The fibre metric scale

**The problem** (`MODEL_SPEC.md` §9, first row). ℏ, c₈ and Λ all depend on the
overall scale k of the metric on ℂP² (κ_ℏ ∝ 1/k, c₈ ∝ k², Λ ∝ 1/k). Nothing in the
architecture supplies a length: structure constants are ±1, |1| = 1 is a norm,
and 2, 6, 42, 3/8, 2π² are ratios. The φ-well does supply a scale, √5
(`MODEL_SPEC.md` §1), but it lives on the **radial** coordinate of the cone, while
ℂP² is the **angular** one (`MODEL_SPEC.md` §0: node state (m, σ), mass radial,
shape angular). The Hellinger–Kantorovich cone relation ties them only as k = m,
which discretises the node mass without fixing its unit (§9).

**What a derivation would need.**
1. A relation, internal to the model, between the radial scale (the φ-well's √5,
   or the node mass's unit) and the metric scale on the angular fibre — for
   example from the cone metric itself, stated in the model's own units, and not
   from a normalisation convention (the §6c π-versus-4π mismatch and the §4c.2a-R
   generator-versus-metric error are the two ways this has already gone wrong).
2. The result expressed as a pure number: a value of k, or equivalently of the
   integer n in ℏ = μℓ_f²/(T·n).
3. The same k carried into all three quantities that depend on it (ℏ, c₈, Λ).

**Success.** A single k fixed by a stated rule, computed before any comparison,
with the rule stated so that a different k would have been reached had the
structure been different; and **over-determination** — the one k fixes ℏ, c₈ and
Λ together, so the prediction imposes more conditions than it spends parameters
(`MODEL_SPEC.md` §0b count).

**Failure.** Any of: the rule admits a family (as integrality admits
n ∈ {1, 2, 4, 8, …}); the value depends on a normalisation convention (Fubini–Study
versus round); the rule was selected after seeing a target value; or the k
required by one of ℏ, c₈, Λ contradicts the others. A result of "k is an input"
is a legitimate outcome and should be recorded as such, as α was
(`MODEL_SPEC.md` §4c.4b).

### 4b. The residual coupling strength C_r

**The problem.** The residual coupling's form, C_r·mul(g, v), is determined by
conservativity, norm preservation and evenness (`MODEL_SPEC.md` §4b.1); its
strength C_r is free (`INPUT_LEDGER.md` §2d #5; `MODEL_SPEC.md` §9, second row).
No higher-order passivity or consistency condition currently normalises it
against the leading gauge coupling (`MODEL_SPEC.md` §4c.4a, "F_residual — not
computable at present"). The earlier saturation anomaly (B ≈ 0.020 at both
C_r = 0.05 and 0.20, `MODEL_SPEC.md` §4b.1) is closed as an oscillation with
frequency ∝ C_r (`MODEL_SPEC.md` §9, "Closed this session"), so C_r does act as a
dial.

**What a derivation would need.** A condition the model already imposes — a
higher-order passivity, conservativity or composition requirement — that fails
for all but one value, or one ratio, of C_r relative to an existing coupling
(β or κ). It must be stated without reference to α or any other measured
constant; `MODEL_SPEC.md` §4c.4a records that α's modular decomposition has
"three unknowns, one equation" and holds no numerical claim.

**Success.** A value or ratio of C_r fixed by such a condition, together with an
observable consequence in the model's own dynamics (for example the B
oscillation frequency, which scales with C_r) that could have come out otherwise.

**Failure.** C_r remains a free dial whose value matters only through a fit; or
the proposed condition holds for every C_r; or its value is chosen to reproduce a
measured number.

---

## 5. Rule for any claimed match to a constant of nature

**Any claimed match between a model output and a measured constant of nature must
be predicted and committed to the repository before the comparison is made, with
no tunable choices.** Specifically:

1. The prediction is a number computed by a script, committed (with its commit
   hash recorded) before the measured value is consulted.
2. Every input and every convention it depends on — including metric and
   generator normalisations — is stated in that commit. No choice may remain to be
   made after the comparison.
3. The count of independent conditions against free parameters is stated with it
   (`MODEL_SPEC.md` §0b; `INPUT_LEDGER.md` §3.1). A match that spends as many
   parameters as it tests is a fit, not a prediction.
4. A number found while looking at the target is refused, however close
   (`PROVENANCE.md` §6d: exp(42), 128 against 137.036, and the retracted ℏ;
   `MODEL_SPEC.md` §4c.4, §4c.4a).

This extends the rule `PROVENANCE.md` §6d already states — "compute the factor
from a stated rule *before* comparing to the target" — from a principle to a
procedure, the same one used for the κ and F predictions in `MODEL_SPEC.md` §5
(predictions committed before comparison, with their commit hashes).

---

## 6. The genuine parameters — what a complete derivation must fix

From the kind column of `INPUT_LEDGER.md` §2d (added 2026-09-25). These are the
numbers the physics depends on and no principle fixes; a complete derivation must
fix each of them, or record it as an input.

**In invariant form (2026-09-26, `MODEL_SPEC.md` §1b)** — the minimal independent set,
after every rescaling of amplitude and time (length is fixed by the lattice):
**ĉ = c/K** (1/√5 now), **κ̂ = κ/√K** (≥ κ̂\* = 2ĉ/√(1 + 2ĉ) = 0.6498), **β̂ = βc/√K**
(0.0334 at β = 0.05; the unclassified entry below), **Ĉ_r = C_r/√K**, **θ_ab** (\|b\|/\|a\|
fixed at 1). The well (K = √5 and its unit nonlinear coefficient) contributes none; the
amplitude A/K, ĝ = gc/√K, k₀, packet width, geometry, T√K, n and q are protocol settings.
The list below is kept as written; items 1, 2 and 4 are ĉ, κ̂ and Ĉ_r.

**Kind 3 — genuine:**
1. **c/√5** — the elastic coupling relative to the well stiffness (§2d #6); the
   derived κ₂ moves from −0.026389 to −0.009316 between c = 0.5 and 2
   (`shape_zero_tests/param_classify_output.txt`). [Finding 2026-09-25, not adopted:
   under the strongest persistence the model admits, the q = 3 window in κ vanishes
   for c/√5 between 0.45 and 0.89 — a bound on c/√5 if that form were adopted
   (MODEL_SPEC §3, "FINDING").] [2026-09-26: not under the adopted radial node form, where
   the window is [κ\*, ∞) at every c tested.]
2. **κ**, the gyroscopic ratio (§2d #3) — measurable by the Larmor splitting.
   [2026-09-25: **genuine parameter with a derived floor**, κ ≥ 2c/√(K + 2c) = 0.971737,
   from the principle "J-compatibility required at every wavelength". A complete
   derivation still has to fix the value (or record it as an input); the floor is
   a test the model can fail. `model.py` operates at κ = κ\*, a chosen point.]
   [Finding 2026-09-25, not adopted: the strongest persistence the model admits — no
   decay, no change of branch — would confine κ to [4.9, 7.5] at q = 3, c = 1, an
   interval, not a value (MODEL_SPEC §3, "FINDING").]
   [2026-09-26: under the adopted radial node form the window is [κ\*, ∞) — the floor
   only (MODEL_SPEC §1a).]
3. **the a–b angle** (§2d #4) — measurable as the D8 frequency ratio.
   [2026-09-26: the ratio is exactly 1/sin(θ/2). Investigated what fixes θ_ab: **still a chosen
   genuine parameter (kind 3)**. Orthogonality is not required (every θ in (0°, 180°) gives a
   4-dimensional associative pair); the model contains **no nonlinear perturbation of the D8
   flow**, so endurance cannot select an angle, and none was added; the golden angle stays
   "attainable, not selected". Prediction misses: swept span 5, not 8; 𝕁 is neither a left nor a
   right octonionic multiplication (best-fit residual 0.82–1.00; that fit post hoc).
   `INPUT_LEDGER.md` §3.2; `shape_zero_tests/d8_theta.py`.]
4. **C_r**, the residual coupling strength (§2d #5).

**Unclassified — may join the list:**
- **β**, the lattice gyroscopic coupling (§2d #7): a test value for the pinning,
  and for κ at leading order, but not shown β-independent for F, κ_box, the
  launch pieces or the (0, π) window; the absolute asymmetry scales with it.
- **ζ**, the cone deficit (§2d #2): observable only through an unearned
  embedding, and subject to the open 2ζ inconsistency (`INPUT_LEDGER.md` §3.3).

**Kind 1 — conventions, not on the list:** ω, and the base units ℓ and μ
(`MODEL_SPEC.md` §4c). They are dimensionful, so no model can derive them (§3).

**Not in the §2d table but open** (§1a, §4a): the fibre metric scale k — equivalently
the integer n in ℏ — and the ratio ℓ_f/ℓ. c₈ and Λ's number are outputs of k, not
further parameters (`MODEL_SPEC.md` §9).
