# Shape Zero — Premise Ledger

Written 2026-09-25. A new document; it changes no other. It lists every premise the
model rests on, every established consequence with the premises it uses, and what
was hoped for but is not established. Every line cites a file and section. Where a
source states a philosophical motivation for a premise, only its location is given
(§4); it is not paraphrased or interpreted here.

**Sources.** `03_current/INPUT_LEDGER.md` (including the §2d kind column),
`03_current/SCALE_SCOPING.md`, `00_START_HERE/MODEL_SPEC.md`, with the documents
they cite: `01_source/shape_zero_zero_ladder.md`, `01_source/spec/shape_zero_v5-3.txt`,
`01_source/proofs/ERRATUM_Theorem_6.1.md`, `02_synthesis/C1S_SYNTHESIS.md`,
`01_source/shape_zero_predictions_v1.md`, `00_START_HERE/PROVENANCE.md`. The C1 Formal
Proofs (`01_source/proofs/ShapeZero_C1_Formal_Proofs.pdf`) are cited by theorem number
as the sources above cite them. File names below omit the folder where unambiguous:
MODEL_SPEC = `00_START_HERE/MODEL_SPEC.md`, INPUT_LEDGER and SCALE_SCOPING = `03_current/`.

**Status key.**
- **principle** — a stated starting point; not derived anywhere in the repository.
- **derived** — with the proof, Prove2Me mission or derivation cited. Prove2Me
  missions are machine-verified in Lean 4; their review state (published, or proved
  and in review) is given as MODEL_SPEC §3's mission table records it.
- **derived in range** — derived only in a stated range, which is given.
- **chosen** — with its kind, as INPUT_LEDGER §2d defines them: **1** unit convention
  (scaled away without changing any dimensionless prediction), **2** test value (every
  result using it shown not to depend on it), **3** genuine parameter (the physics
  depends on it and no principle fixes it). Where no source classifies it, that is said.
- **open** — not fixed; a fixing is sought or its absence is recorded.

---

## 1. Premises

### 1a. Principles and postulates

| # | premise | plain statement | enters the model | status | source |
|---|---|---|---|---|---|
| P1 | **Existence / the ordering condition** | There is something, with a time parameter, an energy, a configuration space and boundedness — recorded as one primitive with four inseparable elements | D0–D1; the base arrives with it (MODEL_SPEC §4) | **principle** — "the one given", tested by removal for internal consistency, "not proof" | INPUT_LEDGER §1; `shape_zero_zero_ladder.md`, D0; MODEL_SPEC §4 |
| P2 | **Conservativity** | Couplings do no net work | D1 first integral, D2 Noether reduction, D4 passivity (P6), the residual form (C21) | **principle** — called "earned" inside the verification programme | `shape_zero_zero_ladder.md`, "The question and the method"; `C1S_SYNTHESIS.md` §5 |
| P3 | **Persistence** | What endures is what exists to be studied | D1 periodicity; q ≤ 3 (C5); Euclidean signature (C6) | **principle** — used at D1 only in the formal proofs [Finding 2026-09-25, not adopted: *persistence at every wavelength* in the strict form is unsatisfiable for all parameters; the strongest form the model admits confines κ to windows (C32, C33)] | `shape_zero_zero_ladder.md`, "The question and the method"; `C1S_SYNTHESIS.md` §5, §14 |
| P4 | **Minimality** | No unforced parameters, "unless forced by a prior principle or by the purchased plurality assumption" (the exemption clause) | D2 isotropy (C4); the D8 law (C18) | **principle** — selects nothing until D8; its "earned" status derives from D8 | `shape_zero_zero_ladder.md`, "The question and the method"; `C1S_SYNTHESIS.md` §4, §5 |
| P5 | **Plurality** | There is more than one thing | spent once: at D8 it selects the generic flow; it excludes κ = 0 and b = ±a | **principle** — "purchased outright and spent exactly once" | `shape_zero_zero_ladder.md`, "The question and the method"; INPUT_LEDGER §2c; `C1S_SYNTHESIS.md` §4, §12 |
| P6 | **Passivity** | Each inter-node coupling does no net work (conservativity applied to the coupling) | the inter-node gauge term F = c(W v₊ − W v₋) | **principle** (an application of P2); its consequence, W symmetric, is **derived** (C9) | MODEL_SPEC §3; `ERRATUM_Theorem_6.1.md`, "Corrected Theorem 6.1"; `shape_zero_zero_ladder.md`, D4 |
| P7 | **The three roles** — role completeness, role minimality, triadic closure | Blocks of the incidence structure are triads because there are three roles (observer, observed, observation); every point takes every role; nothing more than needed | the node size n = 3 (C11) and the Fano plane (C10) | **principle** — two formal postulates plus triadic closure (Formal Proofs §3, Definitions 3.1, 3.5); the count and shape are derived, "the role semantics remain interpretive" | INPUT_LEDGER §2b2, "Why n = 3"; `shape_zero_zero_ladder.md`, "Rung 8" and "Postscript"; `01_source/spec/shape_zero_v5-3.txt`, line 162; `01_source/cover_notes/self_reference.md`, "Ask" |
| P8 | **J-compatibility, required at every wavelength** | The part of any coupling that does not respect each node's complex structure J must be suppressed by the dynamics at every wavelength — the opposite-chirality channel closed at every travelling wavenumber | the inter-node coupling class (C9, C13) and the floor on κ (C14) | **principle** — adopted 2026-09-25, replacing the premise "J-compatibility chosen at short wavelength" (which had replaced "chosen" outright: "the coupling conserves total phase charge") [**CORRECTED 2026-09-25:** at κ\* this holds for the **linear** coupling channel only; the model's own u∘u nonlinearity (P9) is J-breaking and converts a-waves into the opposite chirality (C31)] | MODEL_SPEC §3, "ADOPTED" and its SUPERSEDED markers; INPUT_LEDGER §2d, "Principle, not a parameter"; `ERRATUM_Theorem_6.1.md`, "What was wrong" item 3; PROVENANCE §6m, §6p |
| P9 | **The φ-well force law** | Each node sits in the on-site force F = −(x² − x − 1): fixed points at the golden-ratio roots, linear stiffness √5, quadratic nonlinearity | MODEL_SPEC §1, every force law in `04_scripts/session/model.py` | **principle** — "stated … and not derived"; the source spec's universality control reports every measured dynamical law reproducing on a generic-cubic lattice (K = 2, α = 0.7) [**2026-09-26 — node form adopted:** at a dimer node (the J sector) the well acts on the dimer radius, F = −(√5 + \|ψ\|)ψ (form (A)); the scalar β sector keeps the per-component form. Chosen by the principles alone (§1d). The φ-well itself **remains an underived premise**] | INPUT_LEDGER §2d, "Premise, not a parameter"; SCALE_SCOPING §1b; MODEL_SPEC §1; `shape_zero_v5-3.txt`, "Universality control and the status of the golden ratio (v5.3)" |
| P10 | **The residual as a state coordinate** | The node carries a level-≥16 element; its octonionic part is the shape the gauge acts on, and B measures how far the state is from octonionic | the residual sector, f = C_r·mul(g, v) | **principle** for the attachment ("an extension of the state"); the *form* of the term is **derived** (C21) | MODEL_SPEC §4b, §4b.1 |

### 1b. Structure

| # | premise | plain statement | enters the model | status | source |
|---|---|---|---|---|---|
| S1 | **The lattice (the base)** | Nodes on a periodic, nearest-neighbour lattice; `model.py` builds a ring (q = 1) and q = 2, 3 cubic lattices | MODEL_SPEC §4; `model.py` `Lattice` | the base's **existence** arrives with P1 (MODEL_SPEC §4); its **dimension** is derived conditionally (C5); **periodic nearest-neighbour topology: chosen**, kind not classified in the sources ("the lattice topology" is listed as CHOSEN) | MODEL_SPEC §4; `shape_zero_zero_ladder.md`, "The line" |
| S2 | **The 1-D chain as platform** | Most measurements run on a q = 1 chain | every q = 1 gate and test | **chosen** — "correct as a bench analogue … and outside what the construction permits for bound motion"; kind not classified | MODEL_SPEC §4 |
| S3 | **Node sizes** | A node carries n dimers, ℝ²ⁿ ≅ ℂⁿ; the model runs n = 1, 2, 3 | MODEL_SPEC §0 table; §3 table | **n = 3 derived** (C11); n = 1 and n = 2 are the D2 and D4 rungs, used as instances of one theorem (C9); for the physical node n = 2 is excluded twice | MODEL_SPEC §0, §3; INPUT_LEDGER §2b2 |
| S4 | **The cone state** | A node's state is a mass/radial coordinate and a shape/angular coordinate, (m, σ) | MODEL_SPEC §0 | **derived in range** — derived at D2 and inherited by the model at the platform amplitude (holds to 3×10⁻⁴ at 10⁻³; breaks at \|u\| ≳ 0.1) | MODEL_SPEC §0, §0a |
| S5 | **The complex structure J** | Each node's internal plane carries J, J² = −1 | the gyroscopic term κ𝕁v; the coupling class (C9) | **derived** — dynamically selected by the gyroscopic term (Theorem 2.9: compatible complex structures on ℝ⁴ form S²; measured in `phi_gauge_chiral.py`), given κ ≠ 0 | MODEL_SPEC §2; `shape_zero_zero_ladder.md`, D4 |
| S6 | **The gyroscopic term** | An intra-node velocity force F = κ𝕁v | MODEL_SPEC §2; `model.py` `force` | **licensed** — plurality excludes κ = 0; its value is κ (row K3) | `C1S_SYNTHESIS.md` §12; `02_synthesis/v5-4_addendum_delta.md`, "Refinement on κ" |
| S7 | **The antisymmetric velocity coupling** | βc(v₊ − v₋): the synthetic U(1) | MODEL_SPEC §3, §4b.1 | the U(1) at n = 1 is listed as **derived** by that mechanism; the coefficient β is chosen (row K7) | INPUT_LEDGER §2b2 table; MODEL_SPEC §4b.1 |

### 1c-0. The node form and the selection rule (2026-09-26)

**The node form — how the φ-well (P9) acts at a dimer node.** Adopted: (A). Chosen by the
principles alone — no physical criterion outside the model was used (MODEL_SPEC §1a).

| node form | phase conservation at all orders | persistence as bounded motion (Formal Proofs) | D2's isotropic, origin-centred node | keeps κ\* | source |
|---|---|---|---|---|---|
| elementwise, −(√5u + u∘u) per component (until 2026-09-26) | **no** — linear order only (C31) | **no** — escapes above energy 1.863 per component (C35) | **no** — anisotropic | yes | MODEL_SPEC §1a, §3 |
| **(A) radial, −(√5 + \|ψ\|)ψ — ADOPTED** | **yes** — U(1)-equivariant | **yes** — V ≥ 0, confining | **yes** | **yes** | MODEL_SPEC §1a |
| (B) ring, minimum on \|ψ\| = φ | yes | yes | **no** — origin not an equilibrium | **no** — massless phase mode, no chirality branches | MODEL_SPEC §1a; `shape_zero_tests/ring_spectrum.py` |

Recorded with it (MODEL_SPEC §1a): an isotropic cubic well cannot be analytic at the
origin (r³ is not a polynomial in the components); the D2 rung's own confinement is
quadratic (V = δ\|z\|²/2); the φ-well remains an underived premise.

| # | premise | plain statement | enters the model | status | source |
|---|---|---|---|---|---|
| M1 | **Selection rule (method)** | "The model's principles decide first. Where they leave a choice open or unclear, the option that leads to known physical structure is selected; where neither does, a chosen value from known physics is used. Any choice made this way is recorded as 'selected by physics' — an input, not a derivation — and any fact used in making it cannot afterwards be counted as a prediction or as evidence for the model. The physical criterion must be written down before options are compared." | every open modelling choice | **principle (method)**, adopted 2026-09-26; not used for the node form, which the principles decided | MODEL_SPEC §1a |

### 1c. Parameters and inputs

Kinds from INPUT_LEDGER §2d (added 2026-09-25) unless stated.

| # | parameter | plain statement | enters | status | source |
|---|---|---|---|---|---|
| K1 | **ω** | the D1 clock frequency — the unit of time | D1 | **chosen, kind 1** (unit convention) | INPUT_LEDGER §2d #1; MODEL_SPEC §4c |
| K2 | **ℓ, μ** | base lattice spacing; node mass | dimensional inputs | **chosen, kind 1** | MODEL_SPEC §4c table; SCALE_SCOPING §1a, §6 |
| K3 | **κ**, gyroscopic ratio | strength of F = κ𝕁v; the Larmor splitting equals κ exactly | D4; `model.py` KAPPA | **chosen, kind 3 (genuine), with a derived floor** κ ≥ κ\* = 2c/√(K + 2c) = 0.971737 (C14). The floor is derived, the value is not; the operating value `model.py` KAPPA = κ\* is **chosen**; experiment fixes κ by the Larmor splitting. Was 0.5 until 2026-09-25 | INPUT_LEDGER §2d #3, §3.1; MODEL_SPEC §3, "ADOPTED"; SCALE_SCOPING §6 |
| K4 | **the a–b angle** | direction of D8's second generator | D8 | **chosen, kind 3** — the D8 frequency ratio sweeps [1.04, 23.9]; plurality forbids only b = ±a | INPUT_LEDGER §2d #4, §3.2 |
| K5 | **C_r** | residual coupling strength | the residual sector | **chosen, kind 3** — the form is fixed, the strength by nothing | INPUT_LEDGER §2d #5; MODEL_SPEC §4b.1, §9 |
| K6 | **c**, elastic coupling | F = c(x₊ + x₋ − 2x); c = 1 in every script (not the speed of light) | every force law | **chosen; kind 3 as the ratio c/√5** — cannot be scaled away; the derived κ₂ depends on it | INPUT_LEDGER §2d #6; `shape_zero_tests/param_classify.py` |
| K7 | **β**, lattice gyroscopic coupling | coefficient of βc(v₊ − v₋); 0.05 in the κ scripts, no derivation; carries the dimension of time | the synthetic U(1) | **chosen; kind 2 for the pinning, kind 2 at leading order for the amplitude coefficient, unclassified otherwise** | INPUT_LEDGER §2d #7; MODEL_SPEC §4c.4 |
| K8 | **ζ**, cone deficit | the D2 arena's deficit (named β before 2026-09-25) | D2 | **open / unclassified** — observable only through an unsupplied Lorentzian embedding; an open 2ζ inconsistency | INPUT_LEDGER §2d #2, §3.3; `C1S_SYNTHESIS.md` §14 |
| K9 | **K = √5** | the well's linear stiffness | every dispersion relation | part of **P9** (principle); stiffness varied only in tests (MODEL_SPEC §3, J-compatibility vs stiffness) | INPUT_LEDGER §2d, "Premise, not a parameter"; MODEL_SPEC §1, §3 |
| K10 | **δ, and the D2 constants** | the D2 rung's stiffness, V = δ\|z\|²/2 | D2 | **chosen** ("the values of the constants (δ, c, β, κ)"); kind not classified | `shape_zero_zero_ladder.md`, "The line"; MODEL_SPEC §0a |
| K11 | **ℓ_f** | fibre size | G's reduction | **open** — "possibly = ℓ"; rigidity forbids it varying, does not identify it with ℓ | MODEL_SPEC §4c.3(2); SCALE_SCOPING §1a |
| K12 | **k**, fibre metric scale | overall scale of the metric on ℂP² | ℏ, c₈, Λ | **open** — the largest open item | MODEL_SPEC §9, first row; SCALE_SCOPING §1a, §4a |
| K13 | **n**, flux integer in ℏ | ℏ = μℓ_f²/(T·n) | ℏ | **open** — integrality permits the even family and selects none | MODEL_SPEC §4c.2a-R; SCALE_SCOPING §1b |
| K14 | **α** | e²/ℏc | electromagnetic coupling | **input** — closed as not derivable (five routes tested) | MODEL_SPEC §4c.4b; SCALE_SCOPING §1b |
| K15 | **Protocol settings** | packet wavenumber k₀ = π/2, amplitude 10⁻³, DT = 0.02, lattice sizes, segment strengths g, RAMP | every simulation | **chosen**; kind not classified in the sources. Known dependences: J-compatibility depends on k₀ (C13); the cone holds only at small amplitude (S4) | `shape_zero_tests/README.md`, "Pinned model versions" (common settings); MODEL_SPEC §0a, §3 |

---

## 2. Established consequences

"Premises" lists the rows of §1 each depends on. Review state for Prove2Me missions
as MODEL_SPEC §3's mission table records it.

### 2a. Proved (machine-verified or a cited proof)

| # | consequence | how established | premises | source |
|---|---|---|---|---|
| C1 | Bounded conservative motion on a **one-dimensional** configuration manifold is periodic, and its spectrum is the integer lattice | Lemma 2.1, Theorem 2.2 (Formal Proofs); FORCED at D1 | P1, P2, P3 | INPUT_LEDGER §1; `shape_zero_zero_ladder.md`, D1 |
| C2 | The periodicity of C1 does **not** extend to higher-dimensional configuration manifolds (Hénon–Heiles: regular and chaotic regions at one energy) — C1 is **derived in range** (one dimension) | measured, `d1_scope_limit.py` | P1–P3 | INPUT_LEDGER §1a |
| C9 | **Passivity ⟹ each link matrix W symmetric** (ring N ≥ 3; any number of axes, L ≥ 3); **symmetric + commuting with J ⟹ dimension n² = dim u(n)**, for every n | Prove2Me missions 2 and 4a (passivity), 1 (dimension) — proved, in review; corrected Theorem 6.1 | P6, S5, P8 (for the commuting condition) | MODEL_SPEC §3 mission table; `ERRATUM_Theorem_6.1.md`, "Corrected Theorem 6.1" |
| C10 | A Steiner triple system with a role colouring has exactly 7 points, and every STS(7) is the Fano plane — Theorem 3.6, "Roles Force Fano", in full | Prove2Me missions 5, 6 — **approved, published** 2026-09-24 | P7 | MODEL_SPEC §3 mission table; `ERRATUM_Theorem_6.1.md`, "Section 3" |
| C12 | On a uniform lattice ω(q) − ω(−q) = 2βc·sin q — **independent of the on-site stiffness** and (any number of axes) of every transverse wavenumber | Prove2Me mission 3 — **approved, published**; mission 4b — proved, in review | P6, S7, K6, K7 | MODEL_SPEC §3 mission table; `ERRATUM_Theorem_6.1.md`, "Related: the pinned asymmetry" |

### 2b. Derived

| # | consequence | how established | premises | source |
|---|---|---|---|---|
| C3 | The Larmor splitting equals κ exactly (0.0524, 0.1518, 0.3979, 1.0001 at κ = 0.05, 0.15, 0.40, 1.00) | exact algebra, verified | S5, S6, K3 | INPUT_LEDGER §3.1; `shape_zero_zero_ladder.md`, D4; `v5-4_addendum_delta.md`, "Refinement on κ" |
| C4 | D2: the centrifugal term is the void term, g = L²/2; the cone is flat ℝ² in polar coordinates; universal 2:1 radial:angular ratio | Noether reduction; FORCED given the minimal metric; measured | P2, P4 (isotropy SELECTED) | `shape_zero_zero_ladder.md`, D2; MODEL_SPEC §0, §0a |
| C5 | **q = 3** — the only spatial dimension the ladder's principles admit (so(q) non-trivial ⟹ q ≥ 2; stable bound orbits ⟹ q ≤ 3; gravity with local degrees of freedom ⟹ q ≥ 3) | three intersecting requirements; "forced conditionally, on a base existing" | P1, P3 (and the third requirement's gravitational premise) | INPUT_LEDGER §2b, §2d ("Forced conditionally"); MODEL_SPEC §4 |
| C6 | The arenas are **Euclidean** at every rung | persistence excludes the indefinite (split) branch of Hurwitz | P3 | `C1S_SYNTHESIS.md` §14 |
| C7 | D3 carries exactly one gauge-blind direction (odd dimension); handedness appears there | two-line theorem; bit-identical null measured | P2 | `shape_zero_zero_ladder.md`, D3 |
| C8 | D4: compatible complex structures on ℝ⁴ form a sphere (the Bloch sphere); dynamics touching two of its axes cannot commute | FORCED — so(4) = su(2)⊕su(2); quaternion algebra | P2, S5 | `shape_zero_zero_ladder.md`, D4; MODEL_SPEC §2 |
| C11 | **Node size n = 3** | role triad (C10); a second, independent route: the holonomy group preserving J and a 3-form is SU(3) in dimension 6 = ℂ³ | P7; P6, S5 | INPUT_LEDGER §2b2; MODEL_SPEC §4c.2b; SCALE_SCOPING §1c |
| C14 | **κ ≥ κ\* = 2c/√(K + 2c) = 0.971737** at c = 1, K = √5 (equal to 2φ^(−3/2) because 2 + √5 = φ³ — a consequence of K = √5, not a selection). A floor only: no principle supplies an upper side. At q = 3 the same floor applies because the model's slab segments conserve transverse momentum; a segment of finite transverse width would need 2qc/√(K + 2qc) = 2.091 [2026-09-25: a floor for the linear channel; at κ\* the nonlinear conversion C31 is open] | κ_req(k₀) = x/√(K + x), x = c(1 − cos k₀), maximal at the band edge; q = 3 from `make_links` and `jcompat_q3.py` (predictions committed first, 9bb9951) | P8, P9, K6, S1 | MODEL_SPEC §3, "CANDIDATE" and "ADOPTED"; INPUT_LEDGER §2d #3 |
| C15 | Orbit metric scales **2 at D4, 6 at D8** | \|1\| = 1, forced by 1·1 = 1 and the composition law | the composition law (D4, D8 algebras) | MODEL_SPEC §0b; SCALE_SCOPING §1c; `C1S_SYNTHESIS.md` §13 |
| C16 | **D = 8** = dim(base) + dim(ℂP²) = 4 + 4 | arithmetic from C5 and the fibre ℂP² = Gr(1,3) at n = 3 | C5, C11 | MODEL_SPEC §4c.2b, §4c.3 |
| C17 | Isom(ℂP²) = PSU(3), dimension 8 — the su(3) part of passivity's u(3); the leftover u(1) is the lattice's | computed | C9, C11 | MODEL_SPEC §4c.3 |
| C18 | The **D8 law** ψ̇ = ψa + bψ (generic flow: two frequencies, all fourteen terms); confinement to Fano lines emerged, not imposed | the parameter-free laws have one frequency; plurality excludes them | P2, P4, P5 | `C1S_SYNTHESIS.md` §4, §6; INPUT_LEDGER §2c |
| C19 | Pure numbers: harmonic ratios 1, 2, 3; ‖c‖² = 42; 16 valid orientations of 128; dim Der = 14; Casimirs 0, 1/3, 4/3, 5/6; multiplicities 1, 3, 3̄; block multiplicity 4 above 8 | listed as the construction's outputs | the ladder (P1–P5, P7) | MODEL_SPEC §0b; INPUT_LEDGER §2 |
| C20 | sin²θ_W = **3/8** (tree-level unification value) and the hypercharges from anomaly cancellation — **given the representations** | group theory; anomaly cancellation (conditions > parameters) | the representations 1 ⊕ 3 ⊕ 3̄, 1 ⊕ 2 ⊕ 4 (INPUT_LEDGER §2d, "Not on this list") | INPUT_LEDGER §2, §2d; MODEL_SPEC §0b, §4c.4; SCALE_SCOPING §3a |
| C21 | The residual coupling's **form**, f = C_r·mul(g, v): even in propagation direction, no net work, norm-preserving, reduces to D8 at B = 0; the exact potential V = −2 log(1 − 4B) | by construction, each property measured | P2, P10 | MODEL_SPEC §4b, §4b.1 |
| C22 | The first stable articulation of energetic distinction is the resonant triad — "the count and the shape are derived" | normal-form argument, measured to 10⁻¹³–10⁻¹⁴ | P2, P3, P4 | `shape_zero_zero_ladder.md`, "Postscript" |
| C23 | The fundamental 2-cycle is ℂP¹ ⊂ ℂP², fixed by node size; the lattice spacing never enters the single-node moment map; integrality **discretises** the scale | what survives the ℏ retraction | C11 | MODEL_SPEC §4c.2a-R, "What survives"; INPUT_LEDGER §2d |

### 2c. Measured (simulation, with checks that could have failed)

| # | consequence | how established | premises | source |
|---|---|---|---|---|
| C13 | **J-compatibility emerges from the dynamics**: the J-breaking part of a coupling is suppressed completely at first order in g when the opposite-chirality channel is closed (cos k′ = cos k₀ + κω/c > 1), and not when open. At κ = κ\* every travelling wavenumber is closed (ratio ∝ g at π/4, π/2, 3π/4). Scope: n = 2, q = 1 and full slabs at q = 3, amplitude 10⁻³; the second-order leftover in a closed channel is open [**CORRECTED 2026-09-25:** this is the linear channel; at κ\* the u∘u nonlinearity converts a-waves near k = 0.49, 1.25 into the opposite chirality (C31) — J-compatibility at κ\* holds at linear order only] | predicted from the dispersion relation, then run | P8, P9, S5, S6, K3, K6 | MODEL_SPEC §3 ("UPDATE", "ADOPTED"); PROVENANCE §6m, §6p |
| C24 | **Non-commuting gauge ordering matches its independent prediction**: q = 1 gate 7 to 0.2–0.45°; q = 3 `q3_gate.py` to 0.001–0.005° with the per-mode launch and one readout time per pair (floor 0.000° at κ\* and κ = 0.5) | `model.py` gates 7–8; `q3_gate.py` (shown to fail against the single-wavenumber prediction) | C9, C11, S1, S5, K3 | MODEL_SPEC §3 ("ADOPTED", after-fix table), §4d, §4d.1; `README.md`, "Measured" |
| C25 | u(3) dynamical verification on the platform: admissible class dimension 9; ordering 65.12° measured, 64.97° predicted (κ = 0.5; platform script) — "synthetic … not colour SU(3)" | `04_scripts/platform/phi_gauge_u3_working.py` | C9, C11 | INPUT_LEDGER §2b2; MODEL_SPEC §6b (u(3) row, label corrected) |
| C26 | The **pinned asymmetry's A² coefficient** (the second κ): plane-wave κ = −0.0175 at small amplitude, derived by perturbation theory (−0.01748) and measured (−0.0176 at A = 0.10); fourth order κ(A) = −0.017480 − 0.002947·A²; β-independent at leading order; depends on c/√5 | harmonic balance; reference scripts; `param_classify.py` | P9, K6, K7, C12 | MODEL_SPEC §5; `README.md`, "Measured"; INPUT_LEDGER §2d #6, #7 |
| C27 | **P-3, spinor self-precession**: Ω = C·A²·n_z with n_z conserved; at κ\* C = −0.0491 predicted, −0.0495 measured (κ = 0.5: −0.0896 / −0.0899) [**Superseded 2026-09-26 by the change of node form:** this is the elementwise form's law. Under the adopted radial form, **Ω = A(cos χ − sin χ)/(2ω + κ)** — linear in A — confirmed within 0.4–1.4% of the exact per-dimer frequency difference (`shape_zero_tests/radialA_tests.py`)] | closed form; `phi_gauge_precession.py`, `shape_zero_tests/p3_kstar.py` | P9, S5, S6, K3, K6 | `01_source/shape_zero_predictions_v1.md`, P-3; MODEL_SPEC §3, "ADOPTED" |
| C28 | Odd residual couplings break the pinned asymmetry at first order with no threshold; even ones leave it pinned | live-instrument re-measurement | C12, P10 | MODEL_SPEC §4b, §4b.2 |
| C29 | The cone state holds at the platform amplitude (L conserved to 3×10⁻⁴ at 10⁻³) | measured | S4, P9 | MODEL_SPEC §0a |
| C30 | **Negative results, established:** the forced spectra do not match measured spectra (charged-lepton m² ratios off by four to seven orders; Regge trajectories linear, not quadratic); α is not derivable (137 prime against integers over {2, 3, 7}); ω is not a temperature | measured / arithmetic | the ladder; C19 | INPUT_LEDGER §5; MODEL_SPEC §4c.4b; `C1S_SYNTHESIS.md` §15 |
| C31 | **At κ\* the model's nonlinearity converts a-waves into the opposite chirality**: −u∘u per dimer is −(1−i)/4ψ² − (1+i)/2\|ψ\|² − (1−i)/4ψ\*², J-breaking; the ψ\*² term drives the b-branch at 2k where 2ω_a(k) = ω_b(2k), at k = 0.492 and 1.255 (c = 1, q = 1). J-compatibility at κ\* holds at linear order only [2026-09-26: a property of the elementwise form; under the adopted radial form the conversion is forbidden by the U(1) symmetry (R1 rerun: 9.9×10⁻¹⁷ A) and phase charge is conserved at every order; only charge-neutral pair creation can populate the b-branch (MODEL_SPEC §3, update)] | derived; direct simulation R1, slope 0.991 × rA² (predictions committed first, a213069) | P9, S5, S6, K3, K6 | MODEL_SPEC §3, "CORRECTION — the scope of J-compatibility at κ\*"; `shape_zero_tests/persist_sim.py` |
| C32 | **Strict persistence at every wavelength is unsatisfiable, for all parameters**: same-branch four-wave quartets (a+a → a+a) are always open — in 1-D for pumps between the inflection point and π/2, in 3-D at every saddle of ω(k) | derived (channel ranges); run R5 grew the predicted quartet (×16 at p = 1.398 vs 1.341) | P2, P3 (as a requirement), P9, S1 | MODEL_SPEC §3, "FINDING"; `shape_zero_tests/persist_resonance.py`, `persist_sim.py` |
| C33 | **The strongest persistence the model admits** — no decay, no change of branch, same-branch scattering allowed; *identified after the strict form failed* — confines κ to windows; **q = 3, c = 1: κ ∈ [4.9, 7.5]** (upper side confirmed by S1, slope 0.9986 × rA²), **excluding κ\***; the window vanishes for c/√5 between 0.45 and 0.89; no bound on β beyond the single point 0.06285; inside the window the nonlinear chirality conversion (C31) is closed. A finding, **not adopted**. Caveats: numerical optimiser for q = 3, κ resolution 0.02–0.05 / 0.1, four-wave couplings not derived, R6's finite-amplitude modulational growth, a post-hoc replacement of the committed b-branch criterion [2026-09-26: under the adopted radial form the windows become **[κ\*, ∞)** at every c tested (q = 1, 3); the [4.9, 7.5] window was the elementwise form's, superseded by the change of node form (MODEL_SPEC §3, FINDING update)] | derived (channel ranges); simulations R1–R6, S1–S2 (predictions committed first, a213069, 7215165) | P3 (weakened, as a requirement), P8, P9, K3, K6, S1 | MODEL_SPEC §3, "FINDING"; `shape_zero_tests/persist_q3_scan.py`, `persist_sim3d.py`, `persist_sim_posthoc.py` |
| C34 | **Only the radial node form satisfies the four criteria** — phase conservation at all orders, persistence as bounded motion (Formal Proofs), the D2 rung's isotropic origin-centred node, keeping κ\* (table in §1d) | derived (symmetry, potential bounds, spectrum) and simulated | P2, P3, P9, C4, C14 | MODEL_SPEC §1a; `shape_zero_tests/persist_radial.py`, `ring_spectrum.py` |
| C35 | **The elementwise form violates persistence as bounded motion**: its per-component potential √5u²/2 + u³/3 is unbounded below past u = −√5, so a node escapes above energy 5√5/6 = 1.863 per component (a node with kinetic energy 5 escaped at t = 3.5) | derived; simulated | P3, P9 (elementwise) | MODEL_SPEC §1a; `shape_zero_tests/ring_spectrum.py` |

---

## 3. Hoped for, not established

| item | what is established instead | source |
|---|---|---|
| **The physical scale** — ℏ, G, Λ as outputs | No model derives SI values; the dimensionless parts are open: the flux integer n (even family, none selected), c₈, Λ's number, the fibre metric scale k, and ℓ_f/ℓ. The ℏ derivation ℏ = (μℓ_f²/T)/4 is **retracted** | MODEL_SPEC §4c, §4c.2a-R, §4c.3, §9; SCALE_SCOPING §2, §3, §4a; INPUT_LEDGER §2d, §3.4 |
| **α** | closed as an input; the model constrains the form of its contributions, not its value; the 128 and exp(42) proximities are recorded as numerology | MODEL_SPEC §4c.4, §4c.4a, §4c.4b |
| **sin²θ_W as measured** | 3/8 is the tree-level unification value; matching 0.23122 at M_Z needs RG running the model does not have | INPUT_LEDGER §2; SCALE_SCOPING §3a |
| **Hierarchies** | the model produces hierarchy structure, not magnitude; all three untested mechanisms fail to anchor a size | MODEL_SPEC §4c.4; `03_current/HIERARCHIES.md` §5 (as cited there) |
| **Particle spectra** | failed — see C30 | INPUT_LEDGER §5 |
| **Standard Model numbers** from the second ladder | "Not claimed"; "*in principle* is where it stands" | `03_current/LADDER_TWO.md`, closing "Not claimed" paragraph |
| **Colour SU(3), QCD** | u(3) here is a synthetic gauge structure on a lattice fibre — "Consistency, not QCD" | INPUT_LEDGER §2b2, "Scope" |
| **Lorentzian spacetime, gravity, lensing** | the arenas are Euclidean (C6); the field equations and Maxwell are "forced conditionally, on a base existing"; ζ's lensing reading needs an embedding the ladder does not supply and whose signature persistence forbids; the B-2 candidate ζ = 1/4 has an unmet closing condition; the 2ζ rule conflicts with the flat cone | `C1S_SYNTHESIS.md` §14; INPUT_LEDGER §2d, §3.3 |
| **The golden ratio as physics** | the source spec reports √5 "decorative in the dynamics layer" on its universality control; at D8 φ is attainable as a frequency ratio, and so is every value in [1.04, 23.9] — "attainability is not selection" | `shape_zero_v5-3.txt`, "Universality control and the status of the golden ratio (v5.3)"; INPUT_LEDGER §3.2 |
| **The value of κ, C_r, the a–b angle, c/√5** | genuine parameters (kind 3); κ has a derived floor only; a complete derivation must fix each or record it as an input | SCALE_SCOPING §6; INPUT_LEDGER §2d |
| **J-compatibility beyond first order** | the second-order leftover in a closed channel: open [2026-09-25: at κ\* J-compatibility already fails at nonlinear order — the u∘u term converts a-waves near k = 0.49, 1.25 into the opposite chirality (C31)] | MODEL_SPEC §3, "UPDATE", "CORRECTION" |
| **Persistence at every wavelength as a parameter principle** | the strict form is unsatisfiable (C32); the strongest admissible form gives a window, not a value, and excludes the current operating point (C33) — a finding, not adopted [2026-09-26: under the adopted radial form the window is [κ\*, ∞) (C33)] | MODEL_SPEC §3, "FINDING" |
| **β-independence** of F, κ_box, the launch pieces and the (0, π) window | not shown; settled by β sweeps | INPUT_LEDGER §2d #7 |
| **P-1, the nonreciprocal decay window** | not supported (PROVENANCE §6o) | PROVENANCE §6o; `shape_zero_predictions_v1.md`, P-1 |
| **Why the two roads to Fano meet** (roles; associative subalgebras of 𝕆) | "either a coincidence or a theorem, and we do not know which" | `shape_zero_zero_ladder.md`, "Rung 8" |
| **The classical/quantum interface term** | "Conceptually motivated, derivation incomplete" | `shape_zero_v5-3.txt`, line 198 |
| **The overdamped sector's explicit void** | "the flagged open seam" | `shape_zero_zero_ladder.md`, "The line" |
| **q = 3 as the measured platform** | the bench platform is a 1-D chain; q = 3 runs exist for the gauge sector (C24) | MODEL_SPEC §4, §4d |

---

## 4. Where sources state philosophical motivations

Locations only — not paraphrased or interpreted here.

| premise | file and location |
|---|---|
| the ranks 0–3, the void, the golden ratio (P7, P9) | `01_source/spec/shape_zero_v5-3.txt`, §1 "Core Ontology" (lines 16–19) |
| the golden ratio in the force law (P9) | `01_source/spec/shape_zero_v5-3.txt`, "Universality control and the status of the golden ratio (v5.3)", paragraph beginning "Where the golden ratio actually lives" (line 157) |
| the arena as a whole | `01_source/shape_zero_zero_ladder.md`, "The question and the method" (lines 10–13); D2 (lines 51–52) |
| the three roles (P7) | `03_current/INPUT_LEDGER.md` §2b2, "Why n = 3" and "The algebra corroborates independently" (lines 160–173); `01_source/shape_zero_zero_ladder.md`, "Postscript" (line 164); `01_source/cover_notes/self_reference.md` |
| the given (P1) | `03_current/INPUT_LEDGER.md` §1 (lines 22–27) |
| the principles' bias (P2–P4) | `03_current/INPUT_LEDGER.md` §6, "What the construction is" |
| the model's lack of scale | `00_START_HERE/MODEL_SPEC.md` §0b, closing paragraph ("Under inversion …") |
