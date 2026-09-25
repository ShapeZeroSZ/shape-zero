# MODEL SPEC — the machine, assembled

**What this is.** The buildable core, in build order, with every piece traced to
its source and its script. Not a research record — that is `PROVENANCE.md` and
the layered documents. This is what you type.

**Sorting rule used:** a piece is here if it **changes the code you write**.
Everything else is in the research record.

**Notation — overloaded symbols (added 2026-09-25; nothing renamed).** Three
letters carry two meanings each in this document. Read each section with the
meaning listed for it (`03_current/SCALE_SCOPING.md` §1b).

| symbol | meaning | sections using it |
|---|---|---|
| **κ** | intra-node **gyroscopic ratio**, F = κ𝕁v (`INPUT_LEDGER.md` §2d #3; `model.py` KAPPA) | §2, §3, §4b.1, §7 (build step 3) |
| **κ** | the **A² coefficient** of the pinned asymmetry, \|Δω\| = 2cβ sin k·[1 + κA²] | §4c.0, §4c.4b, §4d (gate notes), §4d.1, §5, §5b–§5b.7, §6b (gate 6), §7 (build step 5), §9 |
| κ_ℏ | the ℏ-scale constant in the §9 fibre-scale row | §9 only |
| **c** | inter-node **elastic coupling**, F = c(x₊ + x₋ − 2x), c = 1 in every script (`INPUT_LEDGER.md` §2d #6) | §1–§3, §4b, §5, §5b, §6b, §7 — every force law and dispersion relation |
| **c** | the **speed of light**, set to 1 for dimensional arithmetic | §4c (table, α = e²/ℏc), §4c.0, §4c.3, §4c.4 |
| ~~**β**~~ **ζ** | **cone deficit** of the D2 arena (`INPUT_LEDGER.md` §2d #2) — **renamed β → ζ, 2026-09-25**, so this collision is resolved | §0 |
| **β** | **lattice gyroscopic coupling**, βc(v₊ − v₋) — the synthetic U(1); β = 0.05 in the κ scripts, no derivation (`INPUT_LEDGER.md` §2d #7) | §3, §4b.1, §4b.2, §4c.4, §5, §5b, §6b, §7 |
| ~~**β**~~ | ~~**ambiguous**: §4c.4 calls the lattice coupling's "charge" β and cites the ledger's §2d, whose β is the cone deficit~~ **corrected 2026-09-25** — §4c.4 now says the lattice β is not the cone deficit | §4c.4 |

---

## 0. State

A node carries a point on the **cone**: a mass/radial coordinate and a shape/
angular coordinate.

    node state  =  (m, sigma)      m in R+ , sigma on the angular fibre

This resolves what looked like a conflict between the rung and platform state
spaces. The rung's flow preserves |psi| — that is the **shape** being unit. The
platform's node is an unbounded displacement in a phi-well — that is the
**radial** coordinate. Different coordinates on one object, not competing
descriptions.

| rung | angular fibre | node size n | real dim |
|---|---|---|---|
| D2 | S¹ | 1 | 2 |
| D4 | S² structure sphere → ℂ² | 2 | 4 |
| Rank 3 trimer | ℂ³ | **3** | 6 |
| D8 | S⁷ → ℂ⁴ | 4 | 8 |

**Source:** Formal Proofs §2.3 (the cone is forced at D2 — the centrifugal
barrier *is* the void repulsion, g = L²/2, eliminated in favour of a conserved
quantity of the state's own motion), §8 (HK convexity, the π/2 horizon).
`cone_vertex.py` closes the span: *"zeta is the cone's deficit and O is the
octonionic sector"* — a **D2-to-D8 observable**. (cone deficit renamed β → ζ on 2026-09-25; β now denotes only the lattice gyroscopic coupling).

---

### 0a. The cone — REPAIRED. It lives in the node, and it holds.

**An earlier version of this section flagged the cone as a foundations gap.**
That was wrong twice over, and the correction came from reading the ladder's
own history (the D2 rung) before testing anything at q = 3.

**What the cone is.** The D2 rung (`z1_d2_rung.py`) is a single degree of
freedom z ∈ ℝ² in its own plane — V = δ|z|²/2, L its angular momentum. The cone
is the statement that flat kinetic energy on ℝ², written in polar coordinates,
*is* the cone metric ds² = dr² + r²dθ², and the centrifugal term
V_eff = V + L²/2r² *is* the void repulsion with g = L²/2. **It is a statement
about one node's internal configuration space — not about the lattice.**

**The rung verifies it:** r\* = √L to four digits (1.00005 vs 1.00000), the
radial:angular ratio 2.04 (universal 2:1), and the turning point exact
(0.134464, 0.335186, 0.597908 at L = 0.2, 0.5, 0.9), with L = 0 reaching the
origin.

**The model node inherits it.** At n = 1 the node's internal space *is* ℝ².
In the linear limit the node reproduces D2's turning points to three digits
(0.133750, 0.334371, 0.601867) with L conserved to **6×10⁻⁹**.

**The φ-well's nonlinearity is anisotropic, and it does not matter where the
model runs.** C1's own `phi_gauge_chiral.py` uses −(√5·u + u∘u) with an
**elementwise** square — not rotationally invariant in the internal plane, so
Noether does not strictly apply. But the anisotropy scales with amplitude:

| amplitude r₀ | r_min/r₀ | L relative drift |
|---|---|---|
| 1 | 0.048 | 3.07 |
| 10⁻¹ | 0.496 | 3.2×10⁻² |
| 10⁻² | 0.4996 | 3.0×10⁻³ |
| **10⁻³ (platform)** | **0.5000** | **3.0×10⁻⁴** |

**At the platform amplitude the cone holds to 3×10⁻⁴.** It breaks only at
\|u\| ≳ 0.1.

**Why the earlier null was wrong.** It tested a packet circulating through
*space* on a 3-D lattice — the wrong object — at amplitude 1 — the wrong
regime. Neither is what the cone claims.

**Standing:** the cone state of §0 is the node's own configuration space,
derived at D2 and inherited by the model in its operating regime. Not an
imported particle-mechanical assumption.

## 0b. Normalisation — what fixes the constants, and what cannot be fixed

**The model has no scale of its own.** All five selection principles are
dimensionless: existence is a posit, conservativity says a quantity vanishes,
persistence is a topological condition on an orbit, minimality and plurality are
counts. Rescale every quantity and all five read identically, so no quantity
derived **from this principle set alone** carries a unit.

**SCOPE — read this before treating it as a wall.** That is a statement about
what the current principle set produces. It is **not** a statement that adding a
dimensionful input is illegitimate. Newton did not derive G; the Standard Model
takes nineteen parameters. A theory that posits a scale and then gets the physics
right is a derivation in the sense that matters.

**The test that separates derivation from fitting is countable**, and it is the
same one already used for anomaly cancellation in `INPUT_LEDGER.md` §3:

> Count the **independent conditions** the model imposes against the **free
> parameters** it takes. Conditions > parameters, and all satisfied, is evidence.
> Conditions ≤ parameters is fitting.

Anomaly cancellation passes it: five hypercharges, three independent linear
conditions plus one cubic, and the Standard Model assignment satisfies all four.

So the productive question is **not** "which values are forbidden." It is: **what
is the minimum set of dimensionful inputs, and does the model's output
overdetermine them?**

Every output is correspondingly a pure number: harmonic ratios 1, 2, 3; orbit
constants 2 and 6; ‖c‖² = 42; 16 of 128; dim Der = 14; Casimirs 0, 1/3, 4/3,
5/6; sin²θ_W = 3/8; multiplicities 1, 3, 3̄; block multiplicity 4 at every level
above 8.

**What fixes the constants that do appear.** The unit satisfies 1·1 = 1, which
with the composition law forces **|1| = 1**. That single normalisation fixes both
orbit metric scales — **2 at D4** and **6 at D8** — which were at one point
recorded as free and are not. G₂ is likewise *defined* as the stabiliser of the
unit. The unit is not a convention chosen for convenience; it is what the
construction's symmetry group is built around.

**Consequence for the build.** Every quantity the model produces is a **ratio**.
Nothing internal sets a metre, a second or a joule, so:

- the coupling constants 2 and 6 are **forced** and need no input
- ω, and any length or energy scale, must be **posited or matched to a
  measurement** — they do not follow from the five principles
  (`INPUT_LEDGER.md` §2d)
- **a dimensionful prediction claimed as following from the principles alone is
  an error**; a dimensionful prediction following from the principles *plus a
  stated input* is ordinary physics, and is judged by the conditions-versus-
  parameters count

**Under inversion x ↦ 1/x, 0 and ∞ exchange and 1 is the fixed point.** A theory
with no scale of its own is one written entirely at that fixed point, which is
why it produces pure numbers and nothing else. Note also that 0 × ∞ is
**indeterminate** — three sequences with the same limits give products 1, 0 and
∞ — so no limiting argument from either boundary determines a finite value.

---

## 1. On-site force

    F_onsite(x) = -(x^2 - x - 1)

Fixed points at the golden-ratio roots. **phi is in the force law itself**, not
decoration — the linearised stiffness is sqrt(5) and the packet oscillates about
PHI = 1.618034.

**Script:** `phi_gauge_nonlinear.py`. **Present in all 7 platform files.**

---

## 2. Intra-node: the complex structure is selected, not chosen

    F_gyro = kappa * JJ * v          JJ = block-diag complex structure

The gyroscopic term **dynamically selects** a complex structure; it is not
imposed. This is Theorem 2.9 (the manifold of orthogonal complex structures on
R^4 is S², the Bloch sphere) and it is measured in `phi_gauge_chiral.py`.

At D8 the same mechanism works because **L_a² = −I exactly** for any unit
imaginary octonion — the Clifford relation, residual **0.00e+00**. So a unit is
selected dynamically at D8 exactly as a complex structure is at D4.

---

## 3. Inter-node coupling, and the gauge class

    F_elastic  = c * (x_{n+1} + x_{n-1} - 2 x_n)
    F_gauge    = c * (W_n v_{n+1} - W_{n-1} v_{n-1})

**Passivity forces symmetry; J-compatibility is a separate, CHOSEN condition.**
Zero net power requires each W symmetric (machine-verified, Prove2Me mission 2,
rings of N ≥ 3). Passivity alone allows all real symmetric 2n×2n matrices —
dimension **n(2n+1)**. Requiring W also to commute with 𝕁 cuts this to the
Hermitian matrices on ℂⁿ, dimension **n² = dim u(n)** (machine-verified,
mission 1); u(n) itself is obtained by multiplying by i.

**Commuting with 𝕁 is NOT derived.** It is equivalent to the coupling conserving
the **total phase charge** — the Noether charge of one simultaneous 𝕁-rotation
of every node. No existing principle forces it: passivity allows the n(n+1)
extra directions; minimality favours the 1-dimensional coupling W = aI (keeping
u(1), losing su(2) and su(3)); the D2 isotropy is the wrong size at n ≥ 2
(O(2n), preserving it gives dimension 1) and acts per node, while a nonzero
coupling can only conserve a *global* phase. Status: **CHOSEN** — "the coupling
conserves total phase charge," extending conservativity from energy to phase.

**UPDATE — EMERGENT AT LONG WAVELENGTH (measured, model's own parameters).** The
J-breaking part of a coupling is suppressed by the dynamics **exactly when the
opposite chirality has no travelling wave at the packet's frequency**, i.e. when
cos k′ = cos k₀ + κω/c lies outside [−1, 1]. At K = √5, κ = 0.5 that holds for
**k₀ < k_c ≈ 1.45 rad (0.46π)**. Predicted from the dispersion relation, then
confirmed before/after:

| k₀ | channel | ratio (J-breaking / J-respecting) as g → 0 |
|---|---|---|
| π/4 | closed (cos k′ = 1.43) | falls ∝ g: 0.043 → 0.0033; leaked weight 9×10⁻⁹ |
| π/2 | open (0.91) — **the gates' wavenumber** | levels off ≈ 0.08 |
| 3π/4 | open (0.36) | levels off ≈ 0.07; leaked weight grows as g² |

Also varying stiffness at fixed k₀ = π/2: closed for K ≳ 3, open below; the
controlling variable is the channel, **not** g/ω (ratios at 4√5 and 16√5 agree
to 3% though ω differs by 2×). κ needed to close the channel at K = √5 runs from
0.02 (k = 0.25) to 0.97 (k → π); κ ≳ 1 would close it everywhere. [**Made exact
2026-09-25:** κ ≥ κ* = 0.971737 — see "CANDIDATE, NOT ADOPTED" below.]

**Status: DERIVED at long wavelength, CHOSEN at short.** Phase conservation is an
emergent low-energy symmetry of the model.

**The closed-channel suppression is complete at first order.** A residual ~10⁻³
reported at k₀ = π/4 was a **readout-timing artefact** — ~3% of the packet was
still inside the segment at readout. With readout after the packet has fully
cleared (weight in the segment window < 10⁻⁶, 1200-site lattice), the residual is
~10⁻⁵, **independent of packet width (8/16/32) and ramp length (12/24/48)** —
a numerical floor. The J-breaking/J-respecting ratio is exactly ∝ g. A static
segment cannot change a wave's frequency, so its edges cannot open a channel
closed at that frequency. **Open-channel values confirmed under clearing
readout**: at π/2 ratios 0.0762 → 0.1278 (g = 0.0025 → 0.04), at 3π/4
0.0699 → 0.1165, other W's unchanged — the ~3–12% first-order effect is real.
Open: whether the second-order leftover in the closed channel commutes with J.
Scope: n = 2, q = 1, amplitude 10⁻³, three random W.

**CANDIDATE, NOT ADOPTED (2026-09-25) — J-compatibility derived at every
wavelength, and what it requires of κ.** `model.py` keeps KAPPA = 0.5; nothing
below is adopted. If the channel is required closed at every wavenumber instead of
chosen at short wavelength:

- **Derivation.** The packet branch is ω² + κω = Q(k₀), Q(k) = K + 2c(1 − cos k);
  the channel is closed iff κω(k₀; κ) > c(1 − cos k₀). With x = c(1 − cos k₀) the
  boundary κω = x and ω² + κω = K + 2x give **κ_req(k₀) = x/√(K + x)**. It rises
  monotonically with x (d/dx = (K + x/2)/(K + x)^{3/2} > 0), so the band edge binds:
  **κ ≥ κ\* = 2c/√(K + 2c) = 0.971737** at c = 1, K = √5. At κ = κ\* every travelling
  wavenumber is strictly closed; only the standing mode k₀ = π is marginal (k′ = 0,
  v_g = 0). Values: κ_req(0.25) = 0.021, κ_req(π/2) = 0.556; at κ = 0.5 the channel is
  closed for k₀ < 1.4536 (the k_c above).
- **κ\* = 2φ^(−3/2)**, because 2 + √5 = φ³. This is a **consequence of K = √5**, not a
  selection: the bound is 2c/√(K + 2c) for any K and c.
- **Lower bound only.** No principle in the repository supplies an upper side:
  plurality excludes only κ = 0 (`C1S_SYNTHESIS.md` §12), and minimality ("no
  unforced parameters") admits κ by its exemption clause without choosing a value.
  **Adopting this fixes a floor, not a value.** κ stays fixed by the bench Larmor
  measurement (`INPUT_LEDGER.md` §3.1), which it would turn into a falsifiable
  inequality: the measured split must be ≥ 2c/√(K + 2c), with K and c from the
  same platform's linear spectroscopy. The floor scales with c (∝ √(2c) for c ≫ K).
- **Numerical confirmation** (`shape_zero_tests/jcompat_kappa.py`, `j_compat_test.py`
  machinery, width-16 packet, 400 sites). At κ = 1.15 κ_req the ratio falls ∝ g —
  0.255 (π/2), 0.247 (0.75π), 0.241 (0.9π) for g 0.04 → 0.01 — with leaked weight
  10⁻⁶–10⁻⁷. At κ = 0.85 κ_req the channel is open: at π/2 the ratio levels off (0.75)
  with leak 10⁻²; at 0.75π and 0.9π it stops falling at 0.0175 and 0.0216 by
  g = 0.0025. A κ sweep at g = 0.04 drops the leaked weight 3–4 orders at
  κ/κ_req = 1 (0.9π: 8.0×10⁻³ at 0.98 → 8.6×10⁻⁶ at 1.02; 0.75π edge smeared over
  1.02–1.05 by the packet's ~0.06 rad spread).
- **All eleven `model.py` gates pass at κ = 0.5, κ\* (0.9727) and 1.0**
  (`shape_zero_tests/jcompat_gates.py`). Gates 1, 2, 4, 5, 6, 10, 11 are identical at all
  three (gate 6's amplitude κ = −0.0184 is a different quantity — n = 1, no 𝕁 term).
  Values that move:

  | gate value | κ = 0.5 | κ\* | κ = 1.0 |
  |---|---|---|---|
  | 3 chirality purity | 0.9834 | 0.9818 | 0.9817 |
  | 7 u(2) split, measured / predicted | 101.12 / 100.80 | 98.76 / 98.28 | 99.61 / 99.12 |
  | 7 u(3) split, measured / predicted | 117.92 / 118.29 | 106.80 / 107.10 | 105.28 / 105.55 |
  | 8 Abelian floor | 0.016° | 0.049° | 0.052° |

- **Recorded κ = 0.5 results that would change on adoption:** this section's k_c ≈ 1.45
  (becomes the whole zone), its π/2 and 3π/4 "open" rows and the clearing-readout
  open-channel values (0.0762 → 0.1278, 0.0699 → 0.1165) — all would be closed-channel,
  ∝ g; the status line "CHOSEN at short" (becomes derived); `shape_zero_tests/README.md`'s
  kscan instrument check (π/2 ratio 0.0883); the gate tables' chirality purity 0.9834
  and ordering splittings (101.12/100.80, 59.86/59.84, 65.12/64.97, and the
  `INPUT_LEDGER.md` benchmark rows); the `gate7_readout.py` clearing values (105.89°, …)
  and `q3_gate.py` values (κ enters through ω and v_g; not re-run); the P-3
  simulation instance **C = −0.0896 → −0.0491** at κ\* (same formula,
  `shape_zero_tests/jcompat_effects.py`). Unaffected: the amplitude-κ results (β-collapse,
  fourth order), which have no gyroscopic term.
- **OPEN — which bound at q = 3.** κ\* = 0.972 holds if the coupling conserves
  transverse momentum (the opposite-chirality wave keeps k₀'s transverse part). If it
  does not, k′ can lie anywhere in the zone, the channel is closed iff
  2κω > Q(k₀) − K, and the zone corner binds: κ\*_q = 2qc/√(K + 2qc) = 1.602 (q = 2),
  **2.091 (q = 3)**.

| n | dim admissible W | group | measured |
|---|---|---|---|
| 1 | 1 | u(1) | Δω = −2cβ sin k |
| 2 | 4 | u(2) = u(1)⊕su(2) | ordering measured, matches independent prediction (§4d) |
| 3 | **9** | **u(3) = u(1)⊕su(3)** | ordering measured, matches independent prediction (§4d) |
| 4 | 16 | u(4) | — |
| 5 | 25 | u(5) | — |

**MACHINE-VERIFIED (Lean 4, Prove2Me, 2026-09-23).** Two results are now proved
for every size, relying only on Lean's three standard axioms:

| mission | statement | status |
|---|---|---|
| 1 | symmetric real matrices commuting with J on ℝ²ⁿ form a space of dimension **n² = dim u(n)**, for every n | proved, in review |
| 2 | on a ring of **N ≥ 3** sites, the per-link coupling Wᵢvᵢ₊₁ − Wᵢ₋₁vᵢ₋₁ does no net work for all motions **iff every Wᵢ is symmetric** | proved, in review |
| 3 | on a uniform ring, ω(q) − ω(−q) = **2βc·sin q** for the branch frequency ω, so the propagation asymmetry is **independent of the on-site stiffness K**; ω is shown to be a root of the dispersion relation | **proved, APPROVED — published** |
| 4a | on a periodic lattice with **any number of axes q** and **L ≥ 3** sites per axis, the per-link coupling (one matrix per site per axis) does no net work for all motions **iff every link matrix is symmetric**; the L = 2 counterexample is also proved | proved, in review |
| 4b | on a uniform lattice with **any number of axes**, reversing the wave along the propagation axis changes its frequency by exactly **2βc·sin k₀** — **independent of the stiffness and of every transverse wavenumber**; the formula is checked against the eigenvalues of a real q = 3 lattice to 1.8×10⁻¹⁴ | proved, in review |
| 5 | a Steiner triple system with at least one point admitting a **role colouring has exactly 7 points** (the count, not that it is the Fano plane) | **APPROVED — published** 2026-09-24; first solved by another solver, ours accepted as later solves (`PROVENANCE.md` §6l) |
| 6 | **every seven-point Steiner triple system is the Fano plane** — so the role postulates force the Fano plane; defines IsFano and imports mission 5's STS and fano | **APPROVED — published** 2026-09-24 (moderator Shuze Chen); all seven theorems accepted as first solves (`PROVENANCE.md` §6l) |

Missions 5 and 6 together machine-verify **C1 Theorem 3.6 ("Roles Force Fano")**
in full: the role postulates force 7 points (mission 5), and every STS(7) is the
Fano plane (mission 6).

Missions 4a and 4b reduce exactly to missions 2 and 3 at q = 1, so **the passivity
and asymmetry results hold on the three-dimensional base itself.**

Chain now machine-checked: **passive ⟹ symmetric**, and **symmetric +
J-compatible ⟹ dimension n²**. Still a premise: **why the coupling commutes with
J.** And the coupling space is the **Hermitian** matrices — u(n) in dimension,
with the Lie algebra itself appearing after multiplying by i.

N ≥ 3 is necessary, not convenient: at N = 2 two links carrying the same
non-symmetric matrix do exactly zero work (proved in Lean with [[0,1],[0,0]]).

**One theorem, three node sizes.** Verified n = 1…5.

**Why n = 3 — forced, not chosen.** Blocks of a Steiner triple system have
cardinality three because there are **three roles**: observer, observed,
observation (Formal Proofs §3.1, §3.5). Role completeness plus role minimality
give r = 3, hence n = 7 and the Fano plane uniquely (Theorem 3.6); AG(2,3) is
excluded by pigeonhole (Corollary 3.7). **The node carries one block.**

**The algebra corroborates.** At n = 2 there is observer and observed with no
observation — the relation is unwitnessed — and that is exactly ℍ, associative,
**associator 1.1×10⁻¹⁶**: nothing is carried. On the full algebra the associator
is **0.81**. *The associator is the observation term.*

**Second route, agreeing.** An octonionic node (ℝ⁸ = ℂ⁴) gives u(4) under
passivity **plus 𝕁-compatibility** (passivity alone gives all symmetric 8×8
matrices, dimension 36, not 16); selecting a unit splits 𝕆 = ℂ ⊕ ℂ³ and the class drops to
**u(1) ⊕ u(3)**, dimension 10 = 1 + 9, with the ℂ³ block exactly **9**.

**Scripts:** `phi_gauge_test.py`, `phi_gauge_chiral.py`,
`phi_gauge_u3_working.py`.

---

## 4. Lattice

    ring of N nodes, periodic, nearest-neighbour

**The base exists as part of the given** (`INPUT_LEDGER.md` §1). The ordering
condition is one irreducible primitive with four inseparable elements — a time
parameter, an energy, **a configuration space**, and boundedness — and removing
the space element leaves no motion and no trajectory (verified,
`d1_given_irreducible.py`). So the base is not an unsupplied antecedent; it
arrives with the given, in the same single step.

**Dimension.** A lattice *is* a base. **q = 3** is the only value the ladder's
own principles admit:

| requirement | admits |
|---|---|
| a centrifugal term exists — so(q) non-trivial | q ≥ 2 |
| it beats the attraction — stable bound orbits | q ≤ 3 |
| gravity has local dof — D(D−3)/2 > 0 | q ≥ 3 |

At q = 1, so(1) = 0: no angular momentum, no barrier, nothing opposes r → 0.

**The current platform is a 1D chain** — correct as a bench analogue, where the
falsifiable prediction lives, and outside what the construction permits for bound
motion. **State which you are building.** The *existence* of a base is settled;
what a 1D chain does not do is satisfy the q = 3 requirement.

---

## 4b. The residual — what the projection leaves behind

**Without this the model is a fibre with no surround.** The distillation (§7b)
projects every level ≥ 16 onto 𝕆. The **residual** is what does not survive the
projection, and it is not discarded structure — it is the condition under which
the projected space has the dimension it has.

**The residual scalar.** Write a point of the level above as two octonion halves
p, q with imaginary parts p′, q′. Then

    B  =  (u₃ − u₁²)((1 − u₃) − u₂²) − (u₄ − u₁u₂)²

the Cauchy–Schwarz expression in the four invariants Re(a), Re(q), |p|², p·q.

| property | value | status |
|---|---|---|
| **exact potential** | V = −2 log(1 − 4B) | matches −log\|det L\| to **8.9×10⁻¹⁵** |
| range | B ∈ [0, ¼] | measured 0.0026–0.2491 over S¹⁵ |
| **zero divisors** | B = **¼ exactly** | V diverges there |
| composition holds | **B = 0** | this is where L_a is an isometry |

**It opens the dimension.** The quotient metric has **rank 3 at B = 0** and
**rank 4 for B > 0**, and the null direction at B = 0 **is ∇B** — the direction
that would carry you off the boundary. Measured eigenvalues on B = 0:
(0, 0.80, 1.00, 1.20); in the interior: (0.85, 0.91, 1.03, 1.09).

**So the residual coordinate exists as a direction only when the residual is
non-zero.** It is not a source — it is the condition for its own dimension to be
present. That is why it has no stress-energy signature and why looking for one
found nothing.

**The selection rule, and it is a theorem for this operator.** Residual couplings
decompose into direction-**even** and direction-**odd** parts:

| sector | effect on the synthetic-U(1) identity | status |
|---|---|---|
| **odd** | violates at **first order, no threshold** — resolvable at the smallest coupling tested, r\* = **0** | **forbidden outright** |
| **even** | shift ~10⁻⁷ at ε = 0.001, sign-changing — integrator noise | **unconstrained at any magnitude** |

The odd prohibition is **stable under a 40× even background** (shift changes by
4%). So the even sector is the dynamical one and the odd sector is excluded by
the same identity that carries the bench prediction.

### 4b.2 A-2 CLOSED — the torsion class is identified

**The selection rule was re-measured against a live instrument** (the original
`residual_selection_rule.py` returned 0.000000 for every case *including the
baseline*, which should have read −0.100 — a dead instrument, not a null result).

| test | result |
|---|---|
| **baseline Δω** | **0.100020**, ratio 1.0002 — instrument confirmed live |
| **even** residual, f = ε·x³ | shift 10⁻⁶–10⁻⁵ — **Δω stays pinned** |
| **odd** residual, f = ε·(x₊ − x₋)·x² | shift **+0.61** at ε = 0.005 — **breaks the pin** |

**The torsion class is: field-cubic, nearest-neighbour difference, odd under
k → −k.** Not a spatial parity operator — a stand-in built on spatial sin/cos
parity tests a different operator class entirely and carries no information
either way. It has been retired.

**The observable is Δω, not energy drift.** The identity Δω = −2cβ sin(k) is
exact for any W²; the on-site nonlinearity shifts W² but is direction-blind, so
it cancels in the difference. A residual term either preserves that blindness or
breaks it.

**Result:** even (on-site cubic) leaves the pinned asymmetry intact; odd
(difference × x²) shifts it at small coupling. This is the r\* = 0, no-threshold
statement of §4b, now confirmed on a live instrument.

### 4b.1 The coupling form — determined, not chosen

**The residual is not a separate field. It is a coordinate of the same state,
one level up.** The node carries a level-≥16 element; its octonionic part is the
shape the gauge structure acts on; **B** measures how far the state is from being
octonionic. So the attachment is not a term to invent — it is an extension of the
state, and the term follows.

    f_residual  =  C_r * mul(g_n, v_n)        g_n octonionic, imaginary

**Every constraint the selection rule imposes is satisfied by construction:**

| requirement | why it holds automatically | measured |
|---|---|---|
| **even in propagation direction** | a **single-node** term with no v₊ − v₋ difference, so k → −k leaves it unchanged; the odd sector is **unreachable**, not merely avoided | — |
| **does no net work** (conservativity) | built from **velocity** with a skew generator, exactly as the D4 gyroscopic κ𝕁v term | F·v = **2.2×10⁻¹⁵** |
| **norm-preserving** | composition returns when one factor is octonionic | **3.3×10⁻¹⁶**, and exactly **0.00e+00** at D64, D256, D1024 |
| **acts non-trivially on B** | the action moves the residual | changed in **200/200**, mean 0.046, max 0.144 |
| **reduces correctly at B = 0** | there the state *is* octonionic, so the term becomes the existing D8 structure | — |

**Compare the gauge term.** βc(v₊ − v₋) is built from *velocity* and is the one
that is **odd** in k — which is exactly why it produces the dispersion asymmetry,
and exactly why an odd *residual* coupling would destroy the identity that
carries it (r\* = 0, no threshold).

**CORRECTION.** An earlier version of this section gave the form as
C_r·mul(g, x) — built from the **state**. That form **does work**: F·v measured at
**12.6**, violating conservativity. The conservative form is built from
**velocity**, F·v = 2.2×10⁻¹⁵, which is the same mechanism as the D4 gyroscopic
term one level up.

**Implemented and gated** (`04_scripts/session/model.py`):

| C_r | energy drift | B evolution |
|---|---|---|
| 0.00 | 1.4×10⁻⁷ | inert to **4.5×10⁻¹⁹** — reduces to the residual-free model |
| 0.05 | 1.4×10⁻⁷ | 0 → 0.0204 |
| 0.20 | 1.6×10⁻⁷ | 0 → 0.0200 |

**OPEN ANOMALY, recorded not buried:** B lands near 0.020 at both C_r = 0.05 and
C_r = 0.20 — a fourfold change in coupling producing nearly the same result. That
is not linear response. Either the residual saturates quickly or something is
clamping it, and **C_r should not be treated as a meaningful dial until this is
understood.**

**One free parameter appears: C_r**, the coupling strength. Dimensionless, and
recorded in `INPUT_LEDGER.md` §2d. Nothing yet fixes its value.

**Not a source of expansion.** ε_V ≥ 5.9 everywhere on the domain, diverging at
both ends, so the residual cannot slow-roll and cannot dominate. B = ¼ is a
potential **maximum** and a critical point of B (|∇B| = 0), so nothing drives a
field there.

---

## 4c. The dimensionful inputs, and what the model derives from them

**Three inputs, with a fourth pending a question.** ℓ (base lattice spacing) and
ℓ_f (fibre size) are written separately because nothing yet identifies them — see
§4c.3(2). If the cone relation identifies them, the count stays at three.

| input | fixes | anchor |
|---|---|---|
| ω | [T] | the D1 clock frequency |
| ℓ | [L] | lattice spacing |
| μ | [M] | node mass |
| ℓ_f | [L] | fibre size — **possibly = ℓ**, see §4c.3 |

| quantity | form | the number |
|---|---|---|
| **ℏ** | (μℓ_f²/T)/n | **NOT DERIVED** — n unfixed, §4c.2a-R |
| Newton **G** | [c₈/(2π²)]·(ℓ⁵/ℓ_f⁴)/μ | **c₈ not yet extracted** |
| **Λ** | (number)/ℓ² | not yet extracted |
| **α = e²/ℏc** | **dimensionless** — no input consumed | **NOT DERIVABLE**, §4c.4 |

**Three or four parameters depending on §4c.3(2) — the count is meaningful**, and it is
the same test used for anomaly cancellation (`INPUT_LEDGER.md` §3): conditions
exceeding parameters and all satisfied is evidence; conditions ≤ parameters is
fitting.

### 4c.1 ℏ — the flux quantum is forced

`04_scripts/session/flux_quantum.py`. Integrality on the structure sphere gives
(1/2πℏ)∮ω = n. The chain runs forward with no choice in it:

1. a complex structure on ℝ⁴ satisfies **J² = −I**, J skew
2. that forces **tr(JᵀJ) = 4** — measured across 400 sampled J, **zero spread**
3. so the sphere has radius **2** in the Frobenius metric — not a choice
4. flux = area = 4πr² = **16π**
5. **n = flux/2π = 8**

**The normalisation is not free.** Rescaling the ideal's basis by λ gives
|J² + I| = 0.75, 3.0, 48.0 — it **breaks** J² = −I. You cannot rescale to a
different radius; the defining condition of a complex structure fixes it.

**Controls.** ℝ² gives n = 4 against ℝ⁴'s n = 8, so the computation is
dimension-sensitive rather than measuring nothing. And the 2-cycle's symplectic
form is unique up to scale — the isotropy constraint matrix is identically zero
to **5.3×10⁻¹⁶**, so there is exactly one invariant antisymmetric form and one
invariant symmetric form.

*Correction banked:* `z1_d4_structure_sphere.py` reported **0** invariant
antisymmetric forms — a relative-only rank threshold on an identically-zero
matrix, the same fault caught four times elsewhere. Its stated conclusion was
right and its own check was not.

### 4c.4 α — searched hard, not derivable, and the reason is structural

**Decompose it.** α = α_GUT × (group factor) × (RG running):

| factor | model's reach |
|---|---|
| group-theoretic normalisation | **OWNED** — sin²θ_W = 3/8 exact (Tr T₃² = 2, Tr Q² = 16/3), so α = α_GUT × 5/8 at unification |
| **α_GUT** | **not derived** — a single dimensionless number |
| RG running to M_Z | needs particle content and scales — the model has neither |

**The model owns the ratio and not the magnitude.** α_GUT is a coupling *at a
scale*, and running needs a hierarchy whose **size** is fixed. §5 of
`HIERARCHIES.md` established the only mechanism the model has — Greene/Diophantine
— generates separations **without fixing their size**.

**A second route also fails.** The model's U(1) is *synthetic*: it comes from the
antisymmetric velocity coupling, so its "charge" is **β** — already dimensionless
and already an **input** (§2d of the ledger). Even the B-2 candidate β = 1/4
is off from α = 1/137.036 by a factor of 34, and nothing identifies a lattice
gyroscopic ratio with a quantum field theory coupling. [**CORRECTED 2026-09-25:**
the lattice U(1) coupling is **β**, the coefficient of βc(ẋ₊ − ẋ₋). It **carries the
dimension of time** — in the source spec's c_ij(x_j − x_i + β·ẋ_j)
(`01_source/spec/shape_zero_v5-3.txt`, "Coupled network") β·ẋ must have the units of
x — and is **dimensionless only in simulation units**. It is **not the cone deficit**:
the ledger's §2d #2 entry is the cone deficit, now named **ζ**, and the lattice β is
§2d #7 (chosen, 0.05, no derivation). The comparison of **1/4 with α concerns ζ**,
the B-2 cone candidate, not the lattice coupling — and neither is a QFT coupling.]

**All three untested hierarchy mechanisms were tried and none anchors a size:**

| mechanism | result |
|---|---|
| rarity / inverse-probability | 1/p = 8 from 16-of-128; reaching 10¹⁷ needs ~19 compounding events, and nothing selects 19 |
| thermodynamic penalty | exp(42) = 1.74×10¹⁸ against a 10¹⁷ target — **see the warning below** |
| attractor spectra | escape rates are exponential in *time*, so they need a time scale — same gap |

**⚠ THE exp(42) NEAR-MISS — recorded because it looked good.** 42 is ‖c‖², a
real quantity in the model, and exp(42) lands within a factor of 17 of the
electroweak–Planck hierarchy. **It is numerology.** The number was available, the
target was known, and they were put together *after seeing both*. No mechanism
says the free energy should be ‖c‖², and none says the hierarchy should be an
exponential. Found while *looking* for a hierarchy rather than while testing
whether hierarchies are anchorable, this would have been very easy to write up.

### 4c.4a The modular decomposition — computed, and still open

A modular form α⁻¹ = F_role × F_residual × F_dyn was proposed, with the first two
factors to be locked to verified architecture so only one number is at issue.
**Both computations were performed. Both factors remain free.**

**F_role — enumerated from the incidence data.** Base set: 3/8, 1, 2, 3, 4, 7, 8,
9, 14, 16, 21, 27, 28, 42, 49, 56, 128. All pairwise products and quotients plus
a sample of triples, filtered to (0.5, 2000). A large discrete set results, and
several land near 137 — **128, 147, 168, 189**. **None is singled out by any rule
already in the formal document** (role completeness, minimality, the passivity
dimension count, the block axiom).

**⚠ 128 is in the model** — it is the orientation count, of which 16 are valid —
and it sits 7% from 137.036. Like exp(42), it is *available* and *close*, and
selecting it would be proximity-after-seeing-the-target. **Forbidden.**

**F_residual — not computable at present.** C_r is a free parameter (§2d of the
ledger). No higher-order passivity or consistency condition currently normalises
it against the leading gauge coupling. A definite value needs a calculation that
does not yet exist.

**Verdict on the decomposition: three unknowns, one equation.** The modular form
is retained **only** as bookkeeping that keeps the required computations explicit
and makes post-hoc fitting visible. **No numerical claim about α is active.**

### 4c.4b CLOSED — α is an input, not a prediction

Five closure routes were proposed and tested against what the model has. **Four
are blocked by results the model itself established; the fifth fails an
arithmetic test.**

| route | verdict |
|---|---|
| geometric matching of node to continuum scale | **blocked** — needs the UV/IR ratio; HIERARCHIES §5 shows the only mechanism generates separations *without* fixing size |
| **topological integer from incidence data** | **blocked** — **137 is prime.** The model's integers factor over {2,3,7}, so **zero** products of up to three reach it. Sums reach it in several arbitrary ways (9+128, 2+7+128, 9+64+64) and therefore select nothing |
| infrared fixed point, C_r drops out | **blocked** — needs RG flow; the ladder is classical, no loops (HIERARCHIES §2) |
| an independent intermediate scale | **blocked** — ω, ℓ, μ, ℓ_f are all *inputs*; every derived quantity (κ = 0.0799 [retracted; corrected −0.0187, §5], 3/8, 2, 6, 42) is dimensionless. No derived scale exists |
| **withdraw the numerical claim** | **adopted** |

**The primality is a real obstruction, not a failed search.** A topological or
counting argument yields integers assembled from the model's structure — 7
points, 3 per line, 16 of 128, 42 constants — and every one factors over small
primes. **137 cannot be built from them multiplicatively.**

**CLOSURE: the model is consistent with the measured α and does not predict it.**
α normalises one free parameter, the same status ω, ℓ and μ hold. The
architectural statement survives — α receives contributions from the role
structure and the residual sector, and the model constrains their **form** while
the **value** is input.

**Do not re-attempt the four blocked routes.**

**Verdict.** Large numbers are *available*; **selection** is what is missing.
The model produces hierarchy **structure** and cannot produce hierarchy
**magnitude**. α needs magnitude. This is closed, not open.

### 4c.0 Dimensional arithmetic is done in code

`04_scripts/session/dimensions.py`. Every dimensional claim in this section is
checked by that module, not written by hand. **The same quantity was claimed
derived and then claimed dimensionally impossible, and both were wrong** —
hand-written exponents in prose, the same class of error as the factor-of-ten κ.
Dimensional analysis looks like reasoning rather than measurement, which is why
it escaped the Phase 0 rule. It is measurement.

Convention: c = 1, so [L] = [T] and the independent dimensions are L and M.
Anchor: r_s = 2GM in D = 4 gives [G₄] = L/M, and in general
**[G_D] = L^(D−3)/M**.

**A finding from the module:** ℏ and e² have the **same** dimensions, L·M. That
is why **α = e²/ℏc is dimensionless** — so the model does not need a scale for
the electromagnetic coupling. **α is a pure number the model either produces or
does not**, consuming no input. That is the same shape as the results that
worked, and a better target than e² itself.

### 4c.2b D = 8 — closed by arithmetic, not by holonomy

**D = dim(base) + dim(fibre) = 4 + dim(ℂP²) = 4 + 4 = 8.** Both are already
established: the base at q = 3 by three intersecting requirements (§2b of the
ledger), the fibre as ℂP² by the Grassmannian at n = 3 (§4c.2). No
special-holonomy argument is required.

**A proposed holonomy route does not hold, and its correction is worth keeping.**
The proposal was that a metric connection preserving both a complex structure and
a non-vanishing 3-form exists only in dimension 7 or 8. Against Berger's
classification:

| group | dim | preserves | note |
|---|---|---|---|
| G₂ | 7 | a **3-form** | **no complex structure** — 7 is odd |
| Spin(7) | 8 | a **4-form** (Cayley) | no complex structure |
| SU(4) | 8 | J and a **4-form** | not a 3-form |
| **SU(3)** | **6** | **J and a holomorphic 3-form Ω** | **both** |

**Neither 7 nor 8 has both.** The group preserving a complex structure *and* a
3-form is **SU(3) in dimension 6**.

**And 6 = ℝ⁶ = ℂ³ = the trimer node.** So the corrected argument is a **second,
independent route to n = 3** — passivity plus a role 3-form selects dimension 6,
which is the node the role triad already fixes. Two independent derivations of
the node size is stronger than one, and it is recorded here rather than as a
route to ambient D.

### 4c.3 G — dimensionally sound, one number owed

**The cycle question closed** (§4c.2): the structure manifold of an n-dimer node
is the Grassmannian Gr(k,n) = U(n)/(U(k)×U(n−k)). At n = 2 that is Gr(1,2) = S²,
recovering the structure sphere as a control. **At n = 3 — the model's node, by
the role triad — it is Gr(1,3) = ℂP².**

**And the two gauge routes then agree**, which is the strongest structural check
in this area: **Isom(ℂP²) = PSU(3), dimension 8**, exactly matching the su(3)
part of passivity's **u(3)**, dimension 9. The leftover u(1) is the one the
lattice supplies separately through the antisymmetric velocity coupling. The
mismatch flagged in §3 was a symptom of not having fixed the cycle.

**The reduction is dimensionally consistent.** With c = 1 the standard result is
[G_D] = L^(D−3)/M — checked against the Schwarzschild radius r_s = 2GM, giving
[G₄] = L/M. ℂP² is **four**-dimensional, so the total space is **D = 8**:

| fibre | f | D = 4+f | [G_D] | Vol | [G_D/Vol] |
|---|---|---|---|---|---|
| S¹ | 1 | 5 | L²/M | L¹ | **L/M** ✓ |
| S² | 2 | 6 | L³/M | L² | **L/M** ✓ |
| **ℂP²** | 4 | **8** | L⁵/M | L⁴ | **L/M** ✓ |

**G₄ = G₈/(2π²ℓ_f⁴)** is well-formed for any fibre dimension.

*Retraction:* an earlier version of this section claimed the reduction was short
by L³ and concluded G "does not follow". **That was an arithmetic error** — the
c ≠ 1 form of [G_D] combined with taking D = 5 for a four-dimensional fibre.
Recorded in `PROVENANCE.md`; the lesson is to do dimensional arithmetic **in
code** before drawing a structural conclusion from it.

**What remains open is narrower.** Writing G₈ = c₈ℓ⁵/μ:

    G₄ = [c₈/(2π²)] · (ℓ⁵/ℓ_f⁴) / μ

Two questions, both real and neither a dimensional inconsistency:

1. **the pure number c₈** — not yet extracted from the model
2. **whether ℓ_f = ℓ** — if so, G₄ = [c₈/(2π²)]·ℓ/μ and the input count stays
   at three

On (2): rigidity establishes the fibre has **no modulus** — deforming the
imaginary products destroys composition (1.4×10⁻¹ at 1% against 3.6×10⁻¹⁵ at
exact), and the 16 valid structures are discrete. That forbids the size
*varying*; it does not by itself *identify* ℓ_f with ℓ. The internal candidate is
the cone (§0): mass radial, shape angular, so the fibre's scale would be set by
the radial coordinate's own unit. Not established.

### 4c.2 Which cycle is physical — NOT SETTLED, and it is a factor of two

The n = 8 above was computed on **SO(4)/U(2)**, the **n = 2** structure sphere.
**The model's node is n = 3** by the role triad. So the calculation is internally
correct and may be about the wrong object.

**Topology eliminates two of four candidates outright:**

| cycle | H₂ | can carry integrality? |
|---|---|---|
| S² | ℤ | **yes** |
| **ℂP²** | ℤ | **yes** — the colour–isospin coset, the n = 3 structure |
| S³ | **0** | no 2-cycle |
| S⁷ | **0** | no 2-cycle |

So S³ and S⁷ cannot carry an integrality condition at all, and only two
candidates remain. **They disagree:**

| object | normalisation | radius | flux | **n** |
|---|---|---|---|---|
| structure sphere of ℝ⁴ | J² = −I forces tr(JᵀJ) = 4 | 2 | 16π | **8** |
| ℂP¹ inside ℂP² | su(3): tr(λ_aλ_b) = 2δ_ab, verified exactly | √2 | 8π | **4** |

Both are internally consistent. They are **different objects**, and nothing in
the model yet says which one physical quantisation sees.

### 4c.2a-R ℏ — RETRACTED. The derivation does not hold.

**The error.** `tr(λ_aλ_b) = 2δ_ab` normalises the **Lie-algebra generators**.
Converting that into a definite **geometric radius on ℂP²** is illegitimate —
they are different objects. A generator normalisation says nothing about the
overall scale of the metric on the quotient.

**What integrality actually gives.** Area(ℂP¹) = kπ ⟹ n = k/2, integral for every
**even** k. So n ∈ {1, 2, 4, 8, …} all satisfy integrality and **nothing selects
one member**.

**Three of my own statements were mutually inconsistent**, which is how the error
survived:

| statement | implied scale | n |
|---|---|---|
| "radius √2 → area 8π" | k = 8 | 4 |
| "lengths ×√2 → Vol = 2π²" | k = 2 | 1 |
| standard Fubini–Study, Vol = π²/2 | k = 1 | **½ — violates integrality** |

**Corrected status: ℏ = μℓ_f²/(T·n) with n an unfixed even-family integer.**
ℏ is the unit fixed by integrality, not an output.

**This returns the claim to C1S §9's original assessment** — *integrality
discretises, ℏ is the unit not the output.* That assessment was right and was
overturned here on an error.

**What survives, and it is real:**

| claim | status |
|---|---|
| the fundamental cycle is ℂP¹ ⊂ ℂP², fixed by node size | **stands** |
| the lattice spacing never enters the single-node moment map | **stands** |
| integrality **discretises** the scale rather than leaving it continuous | **stands** |
| a unique integer n is determined | **fails** |

*The section below is retained for the record and is superseded by this one.*

### 4c.2a [SUPERSEDED — full text in `PROVENANCE.md` §6c]

The three-lemma ℏ derivation, retained for the record. Its conclusion is
**retracted** by §4c.2a-R above; the corrections it required are in
`PROVENANCE.md` §6c. **Do not build from it.**

### 4c.2b D = 8 — closed by arithmetic, not by holonomy

**D = dim(base) + dim(fibre) = 4 + dim(ℂP²) = 4 + 4 = 8.** Both are already
established: the base at q = 3 by three intersecting requirements (§2b of the
ledger), the fibre as ℂP² by the Grassmannian at n = 3 (§4c.2). No
special-holonomy argument is required.

**A proposed holonomy route does not hold, and its correction is worth keeping.**
The proposal was that a metric connection preserving both a complex structure and
a non-vanishing 3-form exists only in dimension 7 or 8. Against Berger's
classification:

| group | dim | preserves | note |
|---|---|---|---|
| G₂ | 7 | a **3-form** | **no complex structure** — 7 is odd |
| Spin(7) | 8 | a **4-form** (Cayley) | no complex structure |
| SU(4) | 8 | J and a **4-form** | not a 3-form |
| **SU(3)** | **6** | **J and a holomorphic 3-form Ω** | **both** |

**Neither 7 nor 8 has both.** The group preserving a complex structure *and* a
3-form is **SU(3) in dimension 6**.

**And 6 = ℝ⁶ = ℂ³ = the trimer node.** So the corrected argument is a **second,
independent route to n = 3** — passivity plus a role 3-form selects dimension 6,
which is the node the role triad already fixes. Two independent derivations of
the node size is stronger than one, and it is recorded here rather than as a
route to ambient D.

### 4c.3 G — dimensionally sound, one number owed

**The cycle question closed** (§4c.2): the structure manifold of an n-dimer node
is the Grassmannian Gr(k,n) = U(n)/(U(k)×U(n−k)). At n = 2 that is Gr(1,2) = S²,
recovering the structure sphere as a control. **At n = 3 — the model's node, by
the role triad — it is Gr(1,3) = ℂP².**

**And the two gauge routes then agree**, which is the strongest structural check
in this area: **Isom(ℂP²) = PSU(3), dimension 8**, exactly matching the su(3)
part of passivity's **u(3)**, dimension 9. The leftover u(1) is the one the
lattice supplies separately through the antisymmetric velocity coupling. The
mismatch flagged in §3 was a symptom of not having fixed the cycle.

**The reduction is dimensionally consistent.** With c = 1 the standard result is
[G_D] = L^(D−3)/M — checked against the Schwarzschild radius r_s = 2GM, giving
[G₄] = L/M. ℂP² is **four**-dimensional, so the total space is **D = 8**:

| fibre | f | D = 4+f | [G_D] | Vol | [G_D/Vol] |
|---|---|---|---|---|---|
| S¹ | 1 | 5 | L²/M | L¹ | **L/M** ✓ |
| S² | 2 | 6 | L³/M | L² | **L/M** ✓ |
| **ℂP²** | 4 | **8** | L⁵/M | L⁴ | **L/M** ✓ |

**G₄ = G₈/(2π²ℓ_f⁴)** is well-formed for any fibre dimension.

*Retraction:* an earlier version of this section claimed the reduction was short
by L³ and concluded G "does not follow". **That was an arithmetic error** — the
c ≠ 1 form of [G_D] combined with taking D = 5 for a four-dimensional fibre.
Recorded in `PROVENANCE.md`; the lesson is to do dimensional arithmetic **in
code** before drawing a structural conclusion from it.

**What remains open is narrower.** Writing G₈ = c₈ℓ⁵/μ:

    G₄ = [c₈/(2π²)] · (ℓ⁵/ℓ_f⁴) / μ

Two questions, both real and neither a dimensional inconsistency:

1. **the pure number c₈** — not yet extracted from the model
2. **whether ℓ_f = ℓ** — if so, G₄ = [c₈/(2π²)]·ℓ/μ and the input count stays
   at three

On (2): rigidity establishes the fibre has **no modulus** — deforming the
imaginary products destroys composition (1.4×10⁻¹ at 1% against 3.6×10⁻¹⁵ at
exact), and the 16 valid structures are discrete. That forbids the size
*varying*; it does not by itself *identify* ℓ_f with ℓ. The internal candidate is
the cone (§0): mass radial, shape angular, so the fibre's scale would be set by
the radial coordinate's own unit. Not established.

### 4c.2 Which cycle is physical — NOT SETTLED, and it is a factor of two

The n = 8 above was computed on **SO(4)/U(2)**, the **n = 2** structure sphere.
**The model's node is n = 3** by the role triad. So the calculation is internally
correct and may be about the wrong object.

**Topology eliminates two of four candidates outright:**

| cycle | H₂ | can carry integrality? |
|---|---|---|
| S² | ℤ | **yes** |
| **ℂP²** | ℤ | **yes** — the colour–isospin coset, the n = 3 structure |
| S³ | **0** | no 2-cycle |
| S⁷ | **0** | no 2-cycle |

So S³ and S⁷ cannot carry an integrality condition at all, and only two
candidates remain. **They disagree:**

| object | normalisation | radius | flux | **n** |
|---|---|---|---|---|
| structure sphere of ℝ⁴ | J² = −I forces tr(JᵀJ) = 4 | 2 | 16π | **8** |
| ℂP¹ inside ℂP² | su(3): tr(λ_aλ_b) = 2δ_ab, verified exactly | √2 | 8π | **4** |

Both are internally consistent. They are **different objects**, and nothing in
the model yet says which one physical quantisation sees.

### 4c.2a-R ℏ — RETRACTED. The derivation does not hold.

**The error.** `tr(λ_aλ_b) = 2δ_ab` normalises the **Lie-algebra generators**.
Converting that into a definite **geometric radius on ℂP²** is illegitimate —
they are different objects. A generator normalisation says nothing about the
overall scale of the metric on the quotient.

**What integrality actually gives.** Area(ℂP¹) = kπ ⟹ n = k/2, integral for every
**even** k. So n ∈ {1, 2, 4, 8, …} all satisfy integrality and **nothing selects
one member**.

**Three of my own statements were mutually inconsistent**, which is how the error
survived:

| statement | implied scale | n |
|---|---|---|
| "radius √2 → area 8π" | k = 8 | 4 |
| "lengths ×√2 → Vol = 2π²" | k = 2 | 1 |
| standard Fubini–Study, Vol = π²/2 | k = 1 | **½ — violates integrality** |

**Corrected status: ℏ = μℓ_f²/(T·n) with n an unfixed even-family integer.**
ℏ is the unit fixed by integrality, not an output.

**This returns the claim to C1S §9's original assessment** — *integrality
discretises, ℏ is the unit not the output.* That assessment was right and was
overturned here on an error.

**What survives, and it is real:**

| claim | status |
|---|---|
| the fundamental cycle is ℂP¹ ⊂ ℂP², fixed by node size | **stands** |
| the lattice spacing never enters the single-node moment map | **stands** |
| integrality **discretises** the scale rather than leaving it continuous | **stands** |
| a unique integer n is determined | **fails** |

*The section below is retained for the record and is superseded by this one.*

### RESOLVED — the 1-D Bloch map DOES hold at q = 3

**Retraction.** An earlier version of this section reported a stable ~10–12×
discrepancy between q = 1 and q = 3 and concluded the 1-D segment map fails on a
3-D base. **That was wrong.** It was the product of three compounding protocol
faults, not physics.

**With axis-aware force, a FULL transverse slab, and post-exit readout certified
by cumulative displacement:**

| gauge region | cleared | B_y sim | B_y from U_1D | angle |
|---|---|---|---|---|
| **full slab** | yes | **+0.885** | **+0.884** | **1.7°** |
| tube r = 4 | yes | +0.582 | +0.884 | 21.5° |
| tube r = 2.5 | yes | +0.270 | +0.884 | 45.1° |

**q = 3 matches the 1-D prediction to 1.7° — the same quality as q = 1.**

**What the earlier discrepancy actually was**, in order of contribution:

1. **Tube clipping.** A gauge region narrower than the packet's transverse
   support reduces the effective coupling. The rotation falls monotonically with
   tube radius — 0.885 (slab) → 0.582 (r=4) → 0.270 (r=2.5) — which is exactly
   the "order of magnitude" that was misread as a q = 3 effect.
2. **Mid-segment readout.** Several runs measured while the packet was still
   inside the segment.
3. **A flat `np.roll` in `force`** in one working copy (the committed `model.py`
   has the axis-aware `_shift` and is correct).

### Gate 7 at q = 3 — provisional pass

Full slab, post-exit certified by cumulative displacement, axis-aware force:

| case | sim-vs-pred AB / BA | measured split | predicted | error |
|---|---|---|---|---|
| u(2), g = 0.12/0.08 | **3.1° / 3.2°** | 30.2° | 24.2° | 6.0° |
| u(3), g = 0.15/0.15 | **3.4° / 3.0°** | 61.4° | 55.7° | 5.7° |

Each individual ordering tracks the composed U_seg prediction to **~3°** — the
same quality as the single-segment test — with drift ~10⁻⁹ and
**non-commutativity clearly present** (splits of 30° and 61°, not zero).

**The ~6° split error has been decomposed**, and only part of it is removable:

| contribution | size | removable? |
|---|---|---|
| per-order tracking error (~3° each) | accumulates into the split | no — it is the map's own accuracy |
| ~~**kinematic**, from unequal \|g\| → differential group velocity~~ | ~~~1.4°~~ **0 — READOUT ARTEFACT** | see below |
| **order-dependent spatial profile**, non-commuting axes | **~4–5°** | **no** |
| **total** | **~6°** | — |

**⚠ CORRECTED (clearing-readout rerun).** The "~1.4° kinematic" row is wrong.
The q = 3 numbers in this section were read with the centroid certificate
(`run_until_exit`), which fires with **98% of the packet still inside the
segment windows** — it certifies exit 2 sites past the segment while the window
extends 10 further. Rerun on a 320×12×12 full slab, reading out only once every
window holds < 10⁻⁶ of the weight:

| | readout | weight in windows | split | predicted | split error | Abelian floor | per-order (AB/BA) |
|---|---|---|---|---|---|---|---|
| u(2) | centroid | 98% | 103.21° | 100.80° | 2.41° | 0.977° | 4.59° / 3.50° |
| u(2) | **clearing** | < 10⁻⁶ | 105.89° | 100.80° | **5.09°** | **0.000°** | **3.01° / 3.62°** |
| u(3) | centroid | 98% | 64.91° | 64.97° | 0.06° | 1.049° | 2.86° / 1.87° |
| u(3) | **clearing** | < 10⁻⁶ | 60.54° | 64.97° | **4.43°** | **0.000°** | **5.21° / 5.23°** |

**The Abelian floor is exactly 0 at q = 3**, as the algebra requires; ~1° was the
packet read mid-exit. Unequal strengths change *when* a packet arrives, not its
final internal state. **The centroid readout's good agreement (u(3): 0.06°) was a
coincidence of mid-exit timing.** A single segment misses its prediction by
**2.70°** after clearing. **The ~4–5° split error is real and intrinsic** to the
single-wavenumber product prediction; the candidate cause — the prediction uses
one wavenumber while a localised 3-D packet spans many — is **untested**.

**Provenance gap:** the original q = 3 configurations were run interactively and
never saved, so the numbers below and the ~30° u(2) split quoted elsewhere
cannot be reproduced. The rerun rebuilt the protocol from the stated
requirements; its u(2) split is ~106°, not ~30°, so it is **not** a
reproduction — old-vs-new comparisons are on identical runs.

**RESOLVED — the ~4–5° is the single-wavenumber approximation.** Averaging the
segment rotation over the packet's actual wavenumber content (17,112 Fourier
components of the regenerated initial field, each propagated with its own
frequency and transverse term, readout Σ P(k) U(k)ρ₀U(k)†) closes the gap. No
simulations rerun; the single-wavenumber column reproduces the old errors exactly:

| quantity | measured | single-k error | **spectrum-averaged error** |
|---|---|---|---|
| single segment, u(2) | — | 2.70° | **0.11°** |
| u(2) per-order AB / BA | — | 3.01° / 3.62° | **0.08° / 0.14°** |
| **u(2) split** | 105.89° | 5.09° (pred 100.80°) | **0.04°** (pred 105.85°) |
| u(3) per-order AB / BA | — | 5.21° / 5.23° | **0.10° / 0.10°** |
| **u(3) split** | 60.54° | 4.43° (pred 64.97°) | **0.17°** (pred 60.71°) |

Retained weight: all but 5×10⁻¹¹. **The product U_B·U_A works when taken mode by
mode.** It fails as a single-carrier product because a 3-D packet this localised
spans a wide range of wavenumbers, including transverse ones that shift the
frequency. At q = 1 the width-8 packet is narrow in k and has no transverse
modes, which is why the carrier approximation suffices there (0.2–0.45°).

**⚠ The claim below that the ~4–5° is "irreducible" and that "a pure product
carries no spatial information" is WRONG** — retained for the record.

**Every q = 3 discrepancy in this line is now explained:** the floor and gate 8's
0.39° were readout timing (exactly 0 under clearing); the split error was the
carrier approximation (0.04–0.17° when averaged). **The gauge sector is
quantitatively verified on the three-dimensional base.** Scope: one geometry,
width-3 packet; reflections and in-segment chirality mixing neglected — the
agreement suggests both are small.

*The original text follows for the record.*

**The ~4–5° survives equal strengths**, so it is not kinematic. Its mechanism:
after the first segment the state is a *different eigenstate mixture* depending
on which segment came first, so the **spatial** profile arriving at the second
segment is order-dependent even when \|g\| matches. **A pure product U_B·U_A
cannot represent this** — it composes internal-space operators and carries no
spatial information between them.

### Gate 7 reformulated — control-subtracted, and it passes

**The unitary product is the wrong pass criterion.** Applying U₂ to the
**measured mid-state** gives 3.7° / 5.9°, *worse* than the plain product's
3.1° / 3.2°. So the residual is not "bad mid-state, good second map" — global
composition of internal unitaries is the wrong object once the packet carries
spatial structure. That rules out the obvious fix.

**Measurement form, requiring no theory map:**

| quantity | u(2) | u(3) |
|---|---|---|
| non-commuting split | ~30° | ~60° |
| **Abelian split, unequal \|g\| — the instrument's zero** | **~1.4°** | **~1.4°** |
| **non-commutative excess** | **~29°** | **~59°** |

**Pass criterion:**
1. Abelian unequal-\|g\| split < 3° — the floor is under control
2. non-commuting split ≫ that floor — the signal is present
3. sim-vs-U₂U₁ reported, **not** used as a gate

**Both conditions met, with a 20× separation.** Gate 7 measures dynamical
non-commutativity at q = 3 directly, independent of the theory residual.

**Boundary, so the excess is not over-read:** the Abelian floor removes the
*kinematic* contribution but **not** the ~4–5° order-dependent profile effect,
which exists only for non-commuting axes (commuting axes give the same mid-state
mixture either way). So 29°/59° is a **detection** of non-commutativity, not a
precise measurement of its magnitude. It should not later be quoted as "the
splitting is 29.0°."

**The profile-weighted map was built and tested at q = 1** (segments 60/80):

| predictor | split | residual vs measured |
|---|---|---|
| **measured** | **101.123°** | — |
| plain product U_B·U_A | 100.799° | **0.323°** |
| **profile-weighted U₂** | 100.880° | **0.242°** |
| global U₂ on the mid-state | 122.615° | **21.5°** |

**Three findings:**

1. **Mid-state composition is definitively wrong** — 21.5°, an order of
   magnitude worse than the plain product. The obvious fix is not merely
   inadequate, it is actively harmful.
2. **Profile weighting is a marginal gain** — 0.24° against 0.32°. Directionally
   right, 0.08° of improvement, **not worth building** on the chain.
3. **The product residual at q = 1 is already sub-degree**, so the ~6° figure
   was a **q = 3 protocol** effect, not a missing bulk-arrival term. Two of the
   three items filed as "theory" were really protocol.

**Status: optional polish, not required for the gate.** A slice map would matter
more at q = 3, where the product residual is a few degrees — but that affects
*product agreement*, not the control-subtracted gate, which passes at ~6000×
separation.

**Consequence:** gap control and strength matching cannot remove this floor. A
full fix needs a spatial arrival model; otherwise the few-degree residual is
intrinsic to the short-gap protocol. This is the same *class* of limitation as
the ≤20-site gap requirement at q = 1 (§7 step 8), now quantified rather than
merely constrained.

**Provisional, not full, pass:** the physics is present and quantitative
agreement is a few degrees, but the split error sits just above a tight 5°
threshold and has a named unmodelled cause.

**Implemented in `model.py`:** `ordering_test(q=, side=, axes=)` with an
overlap assertion, `abelian_floor()` with a vacuous-equal-strength guard, gate 7
on the control-subtracted criterion with the product residual reported but not
gated, and **`run_until_exit()`** — which certifies clearance by **cumulative
displacement**, the check that modular position cannot perform.

**Requirements for any q = 3 gauge measurement**, learned here:

- **full slab**, or a tube large enough to contain the packet's transverse support
- **axis-aware** differences in *both* `force` and `make_links`
- **post-exit** readout certified by **cumulative displacement**, never modular
  position — a dispersed packet defeats `argmax` as well as wrap detection

**On gate 8's 0.39° — RESOLVED: readout timing, not group velocity.** Gate 8's
quantity is the u(2) Abelian floor, which is **exactly 0.000°** under clearing
readout at q = 3 (and at q = 1). The earlier attribution to arrival-profile
differences from unequal group velocities is **not supported**: it was the
packet read mid-exit.

**On gate 6:** κ is a property of the beam profile, not a universal constant.
[Superseded by the κ(w = 2, side) closure, §5: only the plane-wave κ = −0.0187 is
a real coefficient (the plain-cosine-launch value at A = 0.3; the exact wave gives
−0.01775 — §5, "κ to fourth order"); a localised beam's box-averaged κ depends on the box and goes
to zero as it grows, so it is not a property of the beam alone.]
The *pinning* survives every geometry; only the A² coefficient moves. See
`PINNED_ASYMMETRY_TEST.md`.

### 4d.1 Seven measurement traps, all encountered in this port

*(Six as first recorded; the seventh added 2026-09-25.)*

Recorded because each produced a plausible wrong number before being caught.

| trap | signature | cost |
|---|---|---|
| **flat indexing** | `np.roll(x, s, axis=0)` on a flat (N,D) array addresses the **last** spatial axis, not the first | gauge terms coupled along the wrong axis; drift 0.019 |
| **flat segment placement** | `W[start+j]` likewise placed a segment as a line along the last axis | same |
| **transverse-uniform seed** | q = 3 results **bit-identical** to q = 1 — zero transverse Laplacian, so 3D reduces exactly to 1D | a vacuous "verification" of κ |
| **overlapping segments** | segments 3 apart with RAMP length 12 overlap by 9 sites | a spurious 63° Abelian splitting |
| **wrap-around** | modular position **cannot** detect it — needs **cumulative displacement** | double traversal read as a single pass |
| readout before full clearing | a centroid past a segment does not mean the packet has left it: run_until_exit fires with 98% of the weight still in the segment windows at q = 3, and 16–18% remains at gate 7's fixed T = 180 at q = 1 | Abelian floors of ~1° (q = 3) and ~0.016° (q = 1) that are exactly 0 under a clearing readout; gate 8's 0.39° attributed to group velocity |
| **launching off the orbit** | a linearly seeded wave (plain cosine, velocity at the linear frequency, no static shift or harmonics) is **not the nonlinear wave**; the launch leaves free oscillations behind, and at fourth order the measured frequency depends on the launch | the plane-wave κ at A = 0.3 read as −0.01869 against the travelling wave's −0.01775 — about 80% of the apparent amplitude growth (§5, "κ to fourth order") |

**The pattern across the first five: identical or exactly-zero numbers from
configurations that should differ.** Equal strengths in an ordering test make the
two specs the *same array*; that control cannot fail and returned 0.000° three
times before being caught.

Trap 6 is a different pattern: a certificate that reports success without checking the thing it certifies. At q = 1, clearing leaves the gate-7 split unchanged to 0.01° and moves per-order residuals ≤ 0.09°, but takes the Abelian floor to exactly 0 — which made gate 7's old criterion, split > 10 × floor, impossible to fail. Gate 7 now tests the split against the independent prediction.

Trap 7 is a preparation error: the instrument was calibrated, the initial state was
not. It is invisible at second order — the linear launch gives the right A → 0
coefficient — and appears only when the amplitude dependence is read. A run launched
on the exact wave (`kappa_pw4_seed.py`) has a zero error bar; a launch-dependent one
shows the free oscillations as a weighted-versus-unweighted fit disagreement.

---

## 5. What the assembled model predicts

Every number below is emitted by a script in this archive.

**Dispersion asymmetry** (n = 1 sector):

    Δω(k, A) = −2 c β sin(k) · [1 + κ A²]        κ = 0.0799  (PLANE WAVE / 1-D)  ← RETRACTED

**⚠ κ = 0.0799 IS RETRACTED (2026-09-24).** The reference script seeded each
direction with the *other* direction's root: its gyro term has the opposite sign
to the platform scripts, so +k is the upper root there, and "+" was seeded with
the lower. The O(β) velocity mismatch biased the A² coefficient (the same
mechanism `phi_gauge_delta.py` Part 2 documents for v5.2). Seeded with each
direction's own root, the same integrator and estimator give

    κ = −0.0187  (β = 0.05, A = 0.30)      κ = −0.0184 ± 0.00033  (β-sweep, A = 0.30)

— the asymmetry **magnitude shrinks** with amplitude, |Δ/Δ₀| = 0.998339,
0.998316, 0.998390, 0.998316 at β = 0.02, 0.05, 0.10, 0.20. This agrees with
second-order PT (−0.0175, from the +0.0349 βA² term in `phi_gauge_delta.py`) and
was confirmed with separate code, by `joint3_kappa_stiffness.py`, and — a fourth
independent confirmation — by the plane-wave control of `kappa_readout_test.py`
(−0.01869; see κ(w = 2, side) below). **Scripts:** `pinned_asymmetry_reference.py`
(fixed), `model.py` gate 6 (now signed). **Unaffected:** the linear pinned
asymmetry Δω = −2cβ sin k (reproduced to 10⁻⁵ before and after the fix) and the
Lean missions (Prove2Me 2, 3, 4a, 4b), which are linear-order and contain no κ.

*Headline κ = −0.0187 is the value **at A = 0.3**. The small-amplitude coefficient is **−0.0175** —
derived by perturbation theory (−0.01748) and measured (−0.0176 at A = 0.10) — and
a fourth-order term adds about 7% by A = 0.3.* [**CORRECTED 2026-09-25:** the
true fourth-order term adds about **1.5%** at A = 0.3. −0.0187 is the
**plain-cosine-launch** value at A = 0.3; the exact travelling wave gives
**−0.01775** there. About 80% of the growth over −0.0175 is a launch effect —
"κ to fourth order, and the launch", below.]

Trail: `PROVENANCE.md` §6o. The original text follows, kept as the record.

**⚠ κ = 0.0799 IS THE PLANE-WAVE / 1-D VALUE ONLY.** [RETRACTED values — the
κ(w, side) figures in this paragraph were measured with the swapped seeding;
κ(w = 2, side) is re-measured below and is **CLOSED**: only the plane-wave κ is a
real coefficient; a localised beam's box-averaged κ has no box-independent value.] For a transversely
localised beam κ is smaller and depends on the transverse **domain** as well as
the beam width — measured κ(w=2) falls 0.0177 → 0.0019 from side 8 to 32 and
does **not** converge. Readout dilution, fill-fraction scaling and A²
normalisation were each tested and ruled out; the mechanism is unresolved.
**There is no publishable κ(w).** Quote 0.0799 for a plane-wave or 1-D beam;
measure per profile otherwise.

**κ(w = 2, side) re-measured (2026-09-24) — measured and open [now CLOSED, below].** `shape_zero_tests/kappa_readout_test.py` (β = 0.05, A = 0.30, q = 3, transverse
width w = 2, periodic BC, T = 300, uniform transverse readout). Its gyro term is
the reference convention, but it had seeded +k with the lower root — the §6o
swap — and one carrier frequency for every transverse component. It now builds
the initial velocity in Fourier space, each wavevector at its own branch
frequency (counter-propagating content 10⁻¹⁶; one carrier at the right root
still leaves 1.8–2.2%, the swap 4.0%). Outputs: `kappa_readout_test_output.txt`
(fixed) and `kappa_readout_test_swapped.txt` (`--swapped-seed`, the original run).

| side | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| transverse fill fraction | 0.194 | 0.087 | 0.049 | 0.022 | 0.012 |
| **κ, own-branch seed** | **−0.00447** | **−0.00289** | **−0.00204** | **−0.00115** | **−0.00047** |
| κ / κ(plane wave, −0.01869) | 0.239 | 0.155 | 0.109 | 0.062 | 0.025 |
| that ratio / fill fraction | 1.23 | 1.77 | 2.22 | 2.82 | 2.05 |
| κ, swapped seed, re-run — RETRACTED | +0.01772 | +0.01128 | +0.00801 | +0.00457 | +0.00185 |
| κ, swapped seed, as quoted in the script's docstring — RETRACTED | 0.0175 | 0.0113 | 0.0082 | **0.0027** (does not reproduce) | 0.0018 |

- **Plane-wave control: κ = −0.01869** (linear ratio 0.999993) at every side and
  readout — the **fourth independent confirmation** of the corrected value, after
  the fixed `pinned_asymmetry_reference.py`, the separate-code check, and
  `joint3_kappa_stiffness.py`. (Swapped seed: +0.07994, the retracted value.) It
  cannot test the readouts: a transverse-uniform wave is the 1D problem at every
  side.
- **Linear pinning holds** at 0.999997–1.000000 for every side and readout.
- **κ still has no side-independent limit.** It falls tenfold from side 8 to 32
  and steepens at the end (≈ side⁻³ from 24 to 32) [within error — below]. **No κ(w) is published.**
- **Mechanism unresolved** [since CLOSED — a fixed dilution with factor F = 2 − s, below]. It is **not seeding**: new/old is −0.25 at every side
  to 1%, so the fix changed κ's sign and scale but not its side-dependence. It
  is **not readout**: the uniform readout is clean (phase residual 0.003–0.011);
  the weighted and centre readouts are invalid as frequency readouts (residual
  0.5–3.5 rad) and show the same trend. It is **not proportional to fill
  fraction** — predicted before the run (transverse spreading dilutes the
  amplitude driving the shift) and **failed**: |κ|/(|κ_pw|·fill) runs 1.23 →
  2.82, not constant.
- **The original docstring's 0.0027 at side 24 does not reproduce**: the
  original script, re-run as supplied, gives +0.00457. The other four sides
  agree with it to ~2%.

Status: κ(w = 2, side) was **measured and open**; it is now **CLOSED** — below. The other transverse widths
(w = 3, the geometry table) were measured with `pinned_asymmetry_reference.py`'s
single-carrier seed and remain **unverified**. [Since SUPERSEDED by the width
scan — "κ across beam widths" below.]

**κ(w = 2, side) — CLOSED (2026-09-24).** Three GPU scans (Google Colab, Tesla
T4, PyTorch float64, fixed-step RK4 dt = 0.01; physics, Fourier-space seed and
uniform readout identical to `kappa_readout_test.py`), each with its output in
`shape_zero_tests/`: `kappa_side_gpu.py`, `kappa_boxscan_gpu.py`,
`kappa_extended_gpu.py` (`*_colab_output.txt`). All passed every validation
check; `kappa_extended_gpu.py` reproduces the table above exactly at every side
and extends it to L = 80. Re-measured here on CPU with the saved
`kappa_extended_gpu.py` (NumPy backend) at L = 20, 24, 28, 32: every κ, error bar,
[0, 150] value and energy drift identical to the Colab output.

| L | fill | κ ± err | F = κ/(κ_pw·fill) ± err | model 2 − s |
|---|---|---|---|---|
| 8 | 0.19387 | −0.00447 ± 0.00011 | 1.23 ± 0.03 | 1.35 |
| 12 | 0.08726 | −0.00289 ± 0.00014 | 1.77 ± 0.08 | 1.66 |
| 16 | 0.04909 | −0.00204 ± 0.00021 | 2.22 ± 0.23 | 1.80 |
| 20 | 0.03142 | −0.00147 ± 0.00030 | 2.50 ± 0.52 | 1.87 |
| 24 | 0.02182 | −0.00115 ± 0.00036 | 2.81 ± 0.89 | 1.91 |
| 28 | 0.01603 | −0.00075 ± 0.00034 | 2.49 ± 1.13 | 1.94 |
| 32 | 0.01227 | −0.00047 ± 0.00015 | 2.03 ± 0.63 | 1.95 |
| 36 | 0.00970 | −0.00040 ± 0.00016 | 2.22 ± 0.86 | 1.96 |
| 40 | 0.00785 | −0.00033 ± 0.00017 | 2.26 ± 1.16 | 1.97 |
| 48 | 0.00545 | −0.00025 ± 0.00019 | 2.43 ± 1.83 | 1.98 |
| 64 | 0.00307 | −0.00018 ± 0.00021 | 3.22 ± 3.64 | 1.99 |
| 80 | 0.00196 | −0.00017 ± 0.00022 | 4.59 ± 5.91 | 1.99 |

(κ over [0, 300]; the error bar is the larger of the weighted-vs-unweighted
phase-fit difference and the fit's standard error. s = share of the beam's power
in its box-wide transverse component.)

1. **The box-size dependence is static in origin.** It is present from t = 0 —
   κ(L = 16) over [0, 150] and over [0, 300] are both −0.00204 — and it is a clean
   A² coefficient (−0.00201 / −0.00202 / −0.00204 at A = 0.10 / 0.20 / 0.30, flat
   to 1%). **The prediction of a dynamic origin failed.** A slow secondary energy
   transfer into the measured wave is also present, riding on top and not the
   cause: its amplitude grows up to 11% (L = 16, T = 1200) and κ drifts ~10% over
   long runs (−0.00204 at T = 300 → −0.00184 at T = 1200), while the plane-wave
   control stays put (−0.01869 early, −0.01874 late). `kappa_side_gpu.py`.
2. **κ_box = κ_pw × fill × F**, with F rising from 1.23 ± 0.03 at L = 8 toward
   about 2. The cross-versus-self model **F = 2 − s** (the box-wide component is
   shifted by itself with weight 1 and by the beam's sideways components with
   weight 2; nothing fitted) is consistent within error — within two error bars,
   largest 1.8σ at L = 16 — at every L ≥ 12, though all eleven of those points
   lie above it. It is **9% high at L = 8** (1.35 against 1.23 ± 0.03). Deriving
   the exact cross factor for this lattice is the one refinement left. [Done:
   derived below — "The cross-modulation factor F, derived".]
   **What failed earlier still stands as failed:** dilution alone (F = 1, κ ∝
   fill — the measured F is 1.2–2.8) and dilution by transverse spreading during
   the run (the origin is static; and, as `kappa_side_gpu.py` notes, spreading
   alone cannot change the box average, since total energy is conserved). **What fits is a fixed dilution with factor
   F = 2 − s.**
3. **κ goes to zero as the box grows, keeping its sign.** Every measured value is
   negative. From L = 64 it cannot be distinguished from zero by this method
   (−0.00018 ± 0.00021 at 64, −0.00017 ± 0.00022 at 80); nothing suggests a sign
   change.
4. **No oscillation in F is resolved.** Every departure from its neighbours is
   within its error bar — e.g. +13% ± 32% at L = 24, −14% ± 31% at L = 32. The
   "oscillation" and "does not settle" readings of `kappa_boxscan_gpu.py`, which
   has no error bars, are withdrawn; so is the steepening noted above (≈ side⁻³
   from 24 to 32), which is inside the error bars.
5. **Only the plane-wave κ = −0.0187 is a real coefficient.** [−0.0187 is the
   plain-cosine-launch value at A = 0.3; the exact travelling wave gives −0.01775
   there, and −0.017480 at small amplitude — "κ to fourth order", below.] A localised beam's
   box-averaged κ has no box-independent value. **A localised-beam κ quoted
   without a box size is not a property of the beam.** The w = 3 value and the
   geometry table remain **unverified**; by this result, even re-measured they
   would describe a beam in a particular box. [Both since SUPERSEDED by the width
   scan below; the w = 3 item is CLOSED.]

*Provenance of the outputs:* the three `_colab_output.txt` files are annotated
transcripts, not raw output; the raw outputs are in the matching
`*_colab_raw.txt` files. Each opens with a note; the verdicts printed by the
first versions of `kappa_side_gpu.py` ("DYNAMIC") and `kappa_boxscan_gpu.py`
(Q1 "NO BUMP", Q2 "DOES NOT SETTLE") were wrong and were corrected afterwards,
and the scripts saved here carry the corrected reading code — re-running them
prints different verdict text over the same tables. The extended output's
validation and Q2 lines are condensed from the per-size lines the script
prints. The tables are the result.

**κ across beam widths — w = 3 CLOSED (2026-09-24).** `kappa_widthscan_gpu.py`
(Colab, Tesla T4; raw `kappa_widthscan_gpu_colab_raw.txt`, reading
`kappa_widthscan_gpu_colab_output.txt`) measures F in four groups of matched
proportions w/L, since a Gaussian beam's fill and s depend on w/L alone.
`kappa_resolution_test.py` (NumPy on CPU; `kappa_resolution_test_raw.txt`) holds
w/L = 1/4 and varies w from 1 to 6; it reproduced the Colab values for w = 2, 3
and 4 exactly. Physics, seed, readout and error bar as `kappa_extended_gpu.py`.

| w/L | w = 1 | w = 1.5 | w = 2 | w = 3 | w = 4 | w = 5 | w = 6 | 2 − s |
|---|---|---|---|---|---|---|---|---|
| 1/4 | 1.15 ± 0.01 | — | 1.23 ± 0.03 | 1.32 ± 0.04 | 1.33 ± 0.07 | 1.29 ± 0.10 | 1.34 ± 0.16 | 1.34–1.38 |
| 1/8 | — | 2.01 ± 0.12 | 2.22 ± 0.23 | 2.38 ± 0.65 | 1.61 ± 0.53 | — | — | 1.80 |
| 1/12 | — | — | 2.81 ± 0.89 | 1.99 ± 0.83 | 2.78 ± 1.66 | — | — | 1.91 |
| 1/16 | — | 2.86 ± 0.86 | 2.03 ± 0.63 | 2.60 ± 1.85 | 4.82 ± 3.09 | — | — | 1.95 |

(F = κ/(κ_pw·fill). The w = 1, 5, 6 entries are from the resolution test.)

1. **The box-size mechanism is general, not tuned to w = 2.** At matched w/L, F
   agrees across beam widths 1.5 to 4 within two combined error bars in every
   group (largest differences 1.8, 1.1, 0.7, 0.9). The test is sharp only at
   w/L = 1/4 (error bars ±0.03–0.07), partly sharp at 1/8, and **not a real test
   at 1/12 and 1/16**, where error bars reach ±3 — agreement there means only
   "not resolved".
2. **The 9% gap between F and 2 − s at w = 2, L = 8 is a narrow-beam lattice
   effect, established at w/L = 1/4:** the gap is −16.5% at w = 1 (18.6 of its
   error bars) and −8.8% at w = 2 (4.0), and within error for w = 3 to 6 (−2.2%,
   −1.4%, −3.7%, −0.1%). Holding w/L fixed holds the beam's geometry fixed, so the
   gap tracks the beam's width itself. The reading: a narrow beam carries sideways
   ripples on the scale of single lattice sites, where lattice waves differ from
   smooth space, and the model assumes smooth. It is **not** coarse sampling of
   the beam's shape — fill and s are computed on the actual grid. The prediction
   (the gap grows as the beam narrows and vanishes for wide beams) was stated
   before running and **held**. [**CORRECTED 2026-09-24:** at A = 0.30 the gap
   is **not a pure lattice effect**. It is partly the derived lattice kernel and
   partly fourth order in amplitude: from A = 0.10 to 0.30 the plane-wave κ grows
   6.5% while the narrow beam's barely changes, and at A = 0.10 the derived kernel
   accounts for the whole gap (w = 1: 1.221 ± 0.018 against 1.225; w = 2: 1.312 ±
   0.031 against 1.318). The prediction's result stands; its reading is corrected —
   see "The cross-modulation factor F, derived" below.] [**CORRECTED 2026-09-25:** the
   plane-wave growth that lowers F at A = 0.30 is **mostly a launch effect**, not
   fourth-order physics, acting through the plane-wave normalisation. With
   orbit-consistent (second-order) launches F at A = 0.30 is 1.201 (w = 1) and
   1.281 (w = 2), against 1.154 and 1.234 from plain cosines and 1.225 and 1.318
   derived at second order: the launch accounts for 66% and 56% of the gap, and the
   rest matches the plane wave's true fourth-order and velocity terms — "κ to fourth
   order, and the launch", below.]
3. **Unresolved:** whether F exceeds 2 − s at smaller w/L (10 of 14 width-scan
   points lie above it). The error bars there are too large to say. [**Since
   resolved:** it does, and the phase-coherent off-diagonal terms are why —
   derived below.]
4. **w = 3 κ, correctly seeded:** −0.00478 ± 0.00013 at L = 12, −0.00218 ± 0.00059
   at L = 24, −0.00081 ± 0.00034 at L = 36, −0.00060 ± 0.00043 at L = 48. These
   **supersede** the old w = 3 value (0.0311) and the old geometry table (1D 0.0799,
   w = 2 0.0168, w = 3 0.0311; swapped seeding, single carrier), which stay
   visible where they were quoted. By result 5 above, each is a beam in a
   particular box. **The w = 3 item is CLOSED.**

**The cross-modulation factor F, derived (2026-09-24).** `kappa_cross_pt.py`
extends the second-order perturbation theory behind the closure result
(δ = +0.0349β; the script reproduces the plane-wave κ as −0.01748 against
−0.01745) to the **cross kernel** R(q⊥) = K(q⊥)/K(0) on the lattice: the
direction-odd shift of the box-wide component caused by one sideways component
q⊥, per unit of that component's amplitude², in units of the self term. It keeps
the intermediate sum mode (2K, q⊥), the difference mode (0, −q⊥) and the DC mode,
each at its own lattice frequency. Nothing is fitted.

- **R → 2 as q⊥ → 0**, recovering the smooth model's cross-versus-self factor;
  on the lattice it falls to 1.39 at q⊥ = (π, 0) and 1.12 at (π, π).
- **The direction-odd part comes only from the sum mode.** The difference mode
  (kx = 0, frequency Ω₀ − Ω_q, the same in both directions), the DC mode and the
  frequency denominator are all even in direction; the odd part is carried by the
  sum mode at (π, q⊥), whose detuning grows with q⊥. That is why a narrow beam's
  sideways components cross-modulate less.
- **Checked directly** (`kappa_cross_kernel.py`, two-wave runs: a probe at (K, 0)
  plus one pump at (K, q⊥), 16 values of q⊥): at pump amplitude 0.10 the measured
  R matches the derived kernel within about one error bar at 15 of 16 points
  (1.389 ± 0.012 against 1.391 at (π, 0); 1.116 ± 0.011 against 1.118 at (π, π)),
  2.8σ at the smallest q⊥. At 0.30 it sits 3–4% below at large q⊥ — beyond second
  order.

Two predictions for F were recorded before comparison
(`kappa_cross_pt_output.txt`): **P3a** sums the kernel over each beam's grid
components (the diagonal terms only); **P3b** also keeps the **phase-coherent
off-diagonal terms** — three components such as (a, 0), (0, b) and (a, b) that
stay in step on the lattice, because the lattice dispersion is a sum over axes —
read out over the experiment's [0, 300] window, with linear detunings.
`kappa_cross_compare.py`, against every measured F (sharp = error bar ≤ 0.25,
match = within two error bars):

| | sharp points matched |
|---|---|
| 2 − s (smooth model) | 7 of 9 |
| P3a, diagonal kernel sum | 4 of 9 |
| **P3b, all triads** | **7 of 9** — every point except w = 1, L = 4 (1.225 against 1.15 ± 0.01) and w = 2, L = 8 (1.318 against 1.23 ± 0.03), both at A = 0.30 |

**The rise of F above 2 − s at small w/L comes from the phase-coherent
off-diagonal terms** (item 3 above): P3a stays below 2 − s there and misses the
sharp w/L = 1/8 and L = 12 points; P3b rises above it and lies within 1.1 error
bars of every non-sharp point.

**The two narrow-beam misses are fourth order in amplitude** (`kappa_cross_amplitude.py`).
Re-measured at A = 0.10 they give 1.221 ± 0.018 (w = 1, L = 4; derived 1.225)
and 1.312 ± 0.031 (w = 2, L = 8; derived 1.318). From A = 0.10 to 0.30 the
plane-wave κ grows 6.5% (−0.01755 → −0.01869) while the narrow beam's κ barely
changes (−0.004121 → −0.004147), so F, which divides by the plane-wave κ, falls.
[**CORRECTED 2026-09-25:** that plane-wave growth is mostly the plain-cosine launch,
not fourth-order physics — the exact travelling wave grows 1.5% from A → 0 to 0.30,
not 6.9% — see "κ to fourth order, and the launch", below.]

**Out-of-sample test** (`kappa_cross_oos.py`). Three never-measured beams, chosen
where P3a and P3b differ most in small boxes, measured at A = 0.10; both
predictions recorded and committed before the run
(`kappa_cross_oos_predictions.txt`, commit `3b19bf7` on branch
`claude/p1-resonance-window-disagreement-o3m1zm`). Validation passed, including
an 8 × 4 × 4 box reproducing the cubic L = 4 value (x length does not matter).

| beam | measured F | P3a | P3b |
|---|---|---|---|
| w = 0.75, 8 × 6 × 6 | 1.421 ± 0.111 | 1.536 (+1.0σ) | **1.425 (0.0σ)** |
| w = 1.25, 8 × 10 × 10 | 1.988 ± 0.123 | 1.676 (−2.6σ) | **1.994 (+0.1σ)** |
| w = 1.0, 12 × 12 × 12 | 2.127 ± 0.175 | 1.732 (−2.3σ) | **2.131 (0.0σ)** |

P3b lands on all three, including the one where it predicts **below** P3a. The
discrimination is weaker than planned: the error bars at A = 0.10 came out
0.11–0.18, not the 0.02–0.12 expected, so no configuration separates the two
predictions by the four error bars set in advance; P3a is excluded at 2.6σ and
2.3σ, and the first beam does not discriminate (1.0σ). The formal four-error-bar
criterion was **not met**. But P3b's three predictions land within **0.006** of
the measured F while the error bars are 0.11–0.18 — agreement that close would be
very unlikely by chance if those error bars reflected the true uncertainty. So at
A = 0.10 the error-bar estimator (the disagreement between weighted and unweighted
fits) is likely **conservative**.

*Caveat on order of work:* the measured F values were seen before the theory was
written. Nothing in the theory is adjustable, and neither prediction was changed
after comparison.

**Open:** the **fourth-order calculation** — the A⁴ terms that move the plane-wave
κ and the narrow-beam F at A = 0.30. P3b's detunings are also linear; the O(A²)
nonlinear shifts of the components are not in them. [**Done for the plane wave, 2026-09-25** — below;
the beam's fourth order remains open.]

**κ to fourth order, and the launch (2026-09-25).** `shape_zero_tests/kappa_pw4_pt.py`
derives the plane wave's A⁴ term by harmonic balance on the lattice (reference
convention; static shift and second harmonic to O(A⁴), third harmonic to O(A³)):

    κ(A) = −0.017480 − 0.002947·A²

It is confirmed by an exact 24-harmonic solution (the same κ₄ to every printed
digit) and by simulation launched on the exact travelling wave
(`kappa_pw4_seed.py`), which reproduces it to every printed digit with a zero
error bar. The physical value is **−0.01751 at A = 0.1**, **−0.01775 at A = 0.3**
and **−0.01797 at A = 0.4** (exact solution; the A² truncation gives −0.01795 at
0.4). The true fourth-order term adds about **1.5%** at A = 0.3.

**The measured amplitude sweep is mostly a launch effect.** The sweep −0.0176,
−0.0179, −0.0187, −0.0200 at A = 0.1–0.4 (`pinned_asymmetry_reference.py`; the same
on the side-8 cube and on a 1-D ring: −0.01755, −0.01792, −0.01869, −0.01996)
starts from a plain cosine, which omits the wave's static shift and second
harmonic and sets the velocity at the linear frequency. That adds **−0.00094 at
A = 0.3**, about **80%** of the apparent growth over −0.01748. Attribution at
A = 0.3 (`kappa_pw4_attrib.py`: the exact wave with pieces removed; identical at
T = 300 and T = 900, so a frequency shift, not a transient):

| omitted from the launch | κ increment | status |
|---|---|---|
| velocity at the linear frequency | −0.00016 | **derived** (forward/backward split of the fundamental; matches) |
| static shift | −0.00029 | **measured, not derived** |
| second harmonic | −0.00031 | **measured, not derived** |
| interaction of the two | about −0.00024 | **measured, not derived** |
| total, plain-cosine launch | −0.00094 | −0.01869 against the wave's −0.01775 |

**The static-shift and second-harmonic pieces are measured, not derived.** [**Since
derived, 2026-09-25** — "The launch pieces derived", below; the table's status column
is kept as first written.] The
leading cross-modulation formula for the free oscillations the launch leaves
behind does not capture them: for the uniform mode its direction-odd part
vanishes identically, and for the staggered (k = π) mode it gives a sixth to a
third of the measured shift (−2.5×10⁻⁶ against −1.55×10⁻⁵ and −9.9×10⁻⁶ against
−3.6×10⁻⁵ per direction), which grows roughly linearly with that mode's amplitude
while the formula grows quadratically (`kappa_pw4_attrib.py`, part 2). **Adding the second-order field
to the launch** (`kappa_seed2_test.py`) gives **−0.01787 ± 0.00006** at A = 0.3,
against the derived exact wave plus velocity term, −0.01791.

**Beam fourth-order tests** (`kappa4_predict.py`, predictions committed before
any comparison in commit `0e7f269`; `kappa4_compare.py`; `kappa4_measure.py`).
Every fourth-order term is quartic in the component amplitudes, so relative to
the second-order term it carries one more factor of the fill; the kernel is not
derived. Three hypotheses for the beam's own growth, as a fraction r of the
plane wave's: **H0** none; **S1** all quartic terms survive (incoherent local
sextic), r = fill·(6 − 3P₂ − 6s + 4s²)/F₂; **S2** only the box-wide component's
own term, r = fill·s²/F₂. At the nine sharp points (A = 0.30): H0 9/9, S1 8/9
(w = 1, L = 4 at +2.1σ), S2 9/9; the second-order F₂ alone 7/9. Two never-measured
large-fill beams at A = 0.10 and 0.30: the second-order F₂ holds there (1.0063 ±
0.014 against 1.0076; 1.0316 ± 0.014 against 1.0338), and **H0 is excluded at
large fill** — w = 3, L = 4: F(0.30)/F(0.10) at −2.2σ, F(0.30) at −3.4σ; w = 2,
L = 4: F(0.30) at −2.4σ; the box κ grows 4.4% and 3.0%. S1 and S2 both pass
(within 0.8σ), so **S1 against S2 is unresolved**; the narrow-beam evidence that
favours S2 was seen before the predictions were written. **These tests used
plain-cosine launches, so they mix physics with launch effects.** Redoing them
with orbit-consistent launches is open (§9). [**Done 2026-09-25** — below.]

**The launch pieces derived (2026-09-25)** (`kappa_launch_pt.py`; measured by
`kappa_launch_attrib.py`, the exact wave with pieces removed, on a 1-D ring at
T = 900).

*First order — energy projection.* The energy E is conserved, so its differential
dE is invariant under the flow linearised about the wave: it vanishes on every
Floquet mode with multiplier ≠ 1 and on the phase direction. A launch error δ
therefore moves the wave's amplitude by exactly dA = dE(δ)/E′(A), and its
frequency by W′(A)·dE(δ)/E′(A). Closed forms, with W₂ and c₂ the A² coefficients
of the frequency and second harmonic and √(b² + Q) (b = cβ sin sK) the same in
both directions:

    velocity at the linear frequency:   δW = −W₂² A⁴ / √(b² + Q)
    missing second harmonic:            δW = −32 W W₂ c₂² A⁴ / √(b² + Q)
    missing static shift:               dE(δ) = 0 identically — second order

(the uniform deviation has dE = −Σₙ u″ₙ = 0 on a travelling wave). These agree with
the exact projection at small A (velocity −0.000037 against −0.000038, second
harmonic −0.000092 against −0.000093 at A = 0.15).

*Second order — a three-frequency torus.* The launch leaves free oscillations of
the uniform (k = 0) and staggered (k = π) modes. The motion is solved as a torus
u = Σ c_mjl exp i(mθ + jφ₀ + lφ_π) by harmonic balance (every term at its own
lattice wavevector and frequency; Newton with an analytic Jacobian; converged in
truncation), matched to the launch by its uniform and staggered displacements and
its energy.

| A | static shift (derived / measured) | second harmonic | fundamental only | plain cosine |
|---|---|---|---|---|
| 0.15 | −0.000029 / −0.000030 | −0.000072 / −0.000071 | −0.000112 / −0.000108 | −0.000150 / −0.000146 |
| 0.20 | −0.000079 / −0.000080 | −0.000131 / −0.000131 | −0.000254 / −0.000252 | −0.000323 / −0.000321 |
| 0.30 | −0.000278 / −0.000278 | −0.000311 / −0.000313 | −0.000794 / −0.000786 | −0.000959 / −0.000954 |
| 0.40 | −0.000583 / −0.000574 | −0.000581 / −0.000582 | −0.001698 / −0.001681 | −0.002022 / −0.002014 |

(κ increments over the exact wave.) **Derived and measured agree within about 1%
from A = 0.15 to 0.4.** Directly, the exact wave plus an explicit free uniform
mode gives per-direction shifts within 0.04% (A = 0.3, amplitude 0.02).

*The mechanism.* The wave's second harmonic (k = π, frequency 2W) is
**near-resonant with the sum of the two free-mode frequencies**, 2W ≈ Ω₀ + Ω_π —
the **same (0, π) channel as the P-1 decay window** (PROVENANCE §6o, "P-1 — what
was found": the pump's four-wave decay k₀ + k₀ → 0 + π). The detuning is **0.04
for −k against 0.24 for +k** at A = 0.3, so a free uniform mode drives a staggered
one about seven times more strongly for −k. That direction asymmetry is the
static-shift piece's odd part, and it is why the two pieces interact. A torus that
leaves the staggered mode out is 2% wrong for −k and correct for +k.

**Beam fourth-order tests, redone with orbit-consistent launches (2026-09-25)**
(`kappa4_orbit_launch.py`; predictions committed before any beam run, commit
`9e2c4c5`). The launch is the second-order forced field plus every transverse
component's velocity at its second-order nonlinear frequency. **Plane-wave check:**
it removes **96%** of the launch effect, the error bars are **25 times smaller**,
and the residual, **+0.000039 at A = 0.3**, is the missing third harmonic. The test
compares κ_box(A)/κ_box(0.10) at A = 0.30 and 0.40 against (1 + r g(A))/(1 + r g(0.10)),
with g the plane wave's derived physical growth and r from each hypothesis.

- **H0 and S1 are excluded.** H0 fails every L = 4 beam with fill ≥ 0.38 (−14σ to
  −142σ); S1 fails every L = 4 beam (+3σ to +31σ at T = 900).
- **S2's status, exactly:** by the criterion committed in advance it **passed three
  of four L = 4 beams** (w = 1, 1.5, 2) and **failed w = 3, L = 4 by +18σ** (A = 0.4,
  T = 900; +10.7σ at 0.3). It is consistent there only under a band that allows the
  launch's third-harmonic residual, and that band was **chosen after seeing the
  data — post hoc** (`kappa4_orbit_compare.py`). Under it S2 fits all eight L = 4
  points and S1 none. [**CORRECTED 2026-09-25:** with the third harmonic added to the
  launch (plane-wave residual down from +0.000039 to −0.000006 at A = 0.3;
  `kappa4_orbit3_launch.py`), **S2 fails three of the four L = 4 beams** by the
  criterion committed in advance (commit `de96bc3`): w = 1.5 (−6.1σ, −11.1σ at
  A = 0.30, 0.40; T = 900), w = 2 (−11.7σ, −21.9σ) and w = 3 (−9.9σ, −21.3σ) grow
  **more** than S2 allows. The earlier "passed three of four" came from the old
  launch's missing third harmonic, which under-read the beams' growth. The failures
  at w = 1.5 and w = 2 are **robust**; w = 3 is **marginal** — a 0.05% miss, within
  about two of the estimated remaining launch residuals (each order added to the
  launch moves the plane wave by about a sixth of the previous step); w = 1 **does
  not discriminate** (±0.0017).]
- **Post hoc** (`kappa4_orbit3_reading_output.txt`, written after the run): the
  beams' growth is a **clean A² law** — the implied r is consistent between
  A = 0.30 and 0.40 (0.351/0.350 at w = 1.5, 0.547/0.549 at w = 2, 0.750/0.758 at
  w = 3) — and it lies **0.33, 0.50 and 0.68 of the way from S2 to S1** at fills
  0.38, 0.53 and 0.74. **The fourth-order cross terms are partly present, not
  zero.** Also post hoc: each fraction is close to 0.9 × the fill (0.34, 0.48, 0.66).
- **w = 2, L = 8 was excluded:** its κ changes 2.5% with record length at A = 0.10
  (−0.004463 at T = 300, −0.004350 at T = 900), from the slow secondary energy
  transfer seen in larger boxes (`kappa_side_gpu.py`); the L = 4 boxes agree between
  T = 300 and 900.

~~Open (§9): test S2 at w = 3, L = 4 with the third harmonic added to the launch,
and derive why the fourth-order cross terms cancel.~~ [**Superseded 2026-09-25:** the
test was done — S2 fails three of four — and the cross terms do not cancel.] Open
(§9): derive the beam's fourth-order cross kernel, with the measured r values as
the target.

**The PINNING is width-independent at every configuration tested** — that is the
invariance the experiment rests on, and it is unaffected.

- leading term from **linear spectroscopy alone** — c, β, k
- correction **proportional to β**, so the *normalised* drift is β-independent:
  |Δ/Δ₀| = 1.007227, 1.007190, 1.007262, 1.007051 across a tenfold β range
  [RETRACTED values, swapped seeding; corrected 0.998339, 0.998316, 0.998390,
  0.998316 — the collapse itself survives the fix]
- **the collapse is the experiment**: sweep amplitude at two or three couplings,
  the normalised curves must fall on one
- valid A ≲ 0.9; resolution required is ~10⁻⁴ for the pinning, **~10⁻⁵ for the
  A² law**

**Ordering splittings** (n = 2, n = 3): non-commuting axes give a definite
splitting matching independent Bloch-branch theory; commuting axes give zero.

**Script:** `pinned_asymmetry_reference.py`, `phi_gauge_u3_working.py`.

**Gyroscopic sign conventions — seed from each script's own dispersion
(2026-09-24).** The scalar gyro term appears with both signs in this archive:

| convention | gyro term | dispersion | +k root | scripts |
|---|---|---|---|---|
| platform | +cβ(v[n+1] − v[n−1]) | ω² + 2cβ sin k·ω − W² = 0 | **lower** | `phi_gauge_test.py`, `phi_gauge_delta.py`, `phi_gauge_nonlinear.py` (and via it `phi_gauge_closure.py`, `pinned_asymmetry_headline.py`), `phi_gauge_decaymap.py` (via `phi_gauge_delta.py`), `s2_universality.py`, `shape_zero_tests/p1/sim.py` |
| reference | +βc(v[n−1] − v[n+1]) | ω² − 2cβ sin k·ω − W² = 0 | **upper** | `pinned_asymmetry_reference.py`, `shape_zero_tests/joint3_kappa_stiffness.py`, `model.py` (scalar sector), `residual_selection_rule.py` |

The two are the same physics with k → −k, so |Δω| = 2cβ sin k either way. A
travelling-wave seed must take each direction's velocity from the dispersion
relation of **that script's** convention — e.g. `w_lin(direction·K, β)` in the
platform scripts — **never from a fixed "+k upper / −k lower" rule**, which is
right for one convention and is the §6o swap in the other. Both seeding errors
behind §6o were of this kind: `pinned_asymmetry_reference.py` swapped the roots;
`phi_gauge_nonlinear.py` used the β = 0 frequency for both. Check: the
counter-propagating admixture in the seeded mode should sit at the finite-record
floor (1.5×10⁻³ at A = 0.001, β = 0.05, T = 300 in `phi_gauge_nonlinear.py`);
the β = 0 seed leaves 1.25×10⁻² and the other direction's root 2.4×10⁻². The matrix-gauge scripts
(`phi_gauge_chiral.py`, `phi_gauge_u3*.py`) use the link matrices W and are not
covered by this table.

---

## 5b. JOINT observables — EM-like sector × base sector

**Four channels attempted; three stand, one withdrawn.** Numbering is fixed here
and used throughout:

| # | channel | status |
|---|---|---|
| **1** | holonomy × uniform stiffness | **passed** |
| **2** | pin null × uniform stiffness | **passed, derived** |
| **3** | κ × uniform stiffness | **passed** |
| **4** | pin × stiffness **gradient** | **WITHDRAWN as a law** (§5b.4) |

### 5b.1 Joint #1 — gauge clock × base potential

**What it is.** The gauge rotation through a segment depends on the branch
wavenumbers, which depend on ω, which is set by the **on-site well stiffness** —
a base-sector quantity. So changing the base changes the gauge holonomy. The
gauge sector is the clock; the base sector modulates it.

**Inputs fixed openly:** stiffness scale, g = 0.12, segment geometry, packet.
**No new fitted constant** — the prediction comes from `k_branch`/`U_segment`
as already written.

| | baseline | stiffness +10% | Δ |
|---|---|---|---|
| **measured** | 148.71° | 140.74° | **−7.96°** |
| **predicted** (`U_segment`) | 137.13° | 130.79° | **−6.34°** |
| sim − pred, absolute | +11.6° | +10.0° | — |

**The differential is the claim: −6.34° predicted against −7.96° measured**,
sign correct and ~20% residual on the shift. The **absolute** ~10° offset is the
known single-segment protocol floor and cancels in the difference.

**The sign is the non-trivial part.** Higher stiffness gives a *longer* dwell
(24.88 → 25.52 sites/v_g), so the naive reading predicts *more* rotation.
Measured is **less**. The ω-dependence inside `k_branch` outweighs the dwell
term — and the code already encoded that before the measurement was made.

**What this is not.** Not gravity: no field equation, no source, no G. It is an
index-of-refraction effect — the structural analogue of a clock rate varying
with local potential. What makes it worth having is that it is **checkable and
coupled**: neither sector alone produces the number.

**Status of the programme this marks.** Structure derived; inputs supplied
openly; a coupled observable predicted from structure and measured. That is
calibration-and-joint-prediction, which is where a physical theory normally
lives — not a failure to derive constants.

### 5b.2 Joint #2 — the pin null, and Joint #3 — κ

*Same stiffness scan at two amplitudes: A = 0.02 isolates the pin (#2), A = 0.30
the κ correction (#3). #2 is the only joint result with a derivation behind it
rather than a measurement alone.*

**Independent of the first**: different gauge sector (scalar U(1), not su(2)
segments), different readout (frequency asymmetry, not Bloch angle), same base
knob.

**Prediction, exact and parameter-free.** ω± = (∓B + √(B² + 4ω₀²))/2 with
B = 2cβ sin k. Stiffness enters **only** through ω₀², identically in both
branches, so **Δω = −B with no stiffness dependence.** Both frequencies move;
their difference does not.

| A | stiffness 0.90 / 1.00 / 1.10 | Δ from baseline |
|---|---|---|
| **0.02** (linear) | 1.000038 / 1.000033 / 1.000026 | **−1.2×10⁻⁵ — NULL** |
| **0.30** (nonlinear) | ~~1.008633 / 1.007192 / 1.006090~~ RETRACTED (swapped seeding) | ~~−2.5×10⁻³~~ |
| **0.02** (linear), corrected seeding | 0.999992 / 0.999993 / 0.999994 | **+2×10⁻⁶ — NULL** |
| **0.30** (nonlinear), corrected seeding | 0.998072 / 0.998316 / 0.998513 | **+4.4×10⁻⁴ — SIGNAL** |

*Corrected rows: `shape_zero_tests/joint3_kappa_stiffness.py` (stiffness factor f
multiplies the linear on-site stiffness √5; each direction seeded at its own root;
DOP853, weighted phase regression). At f = 1.00 its A = 0.30 ratio equals the fixed
`pinned_asymmetry_reference.py` β-sweep value to six digits.*

**Liveness check passes:** the branch frequencies move by **+0.109** across the
range, so the knob is connected and the null is not vacuous.

**The linear pin is protected** to 1.2×10⁻⁵ across a 20% stiffness range —
exactly as the algebra requires.

**κ is not.** Converting: **κ = 0.0959, 0.0799, 0.0677** at stiffness 0.90, 1.00,
1.10 — a **35% swing**, monotone, far outside noise. [**RETRACTED** — swapped
seeding, §5.]

**Joint #3, re-measured with correct seeding (2026-09-24):** **κ = −0.0214,
−0.0187, −0.0165** at stiffness 0.90, 1.00, 1.10 — a **26% spread**, monotone
(|κ| falls as stiffness rises, as the retracted values did), with the linear
ratio at 0.99999 throughout. Every value and the sign differ from the retracted
ones; **the qualitative result stands: the linear pin is protected against
stiffness, κ is not.** Script: `shape_zero_tests/joint3_kappa_stiffness.py`
(about a minute).

### 5b.3 The pin/κ split is structural, not merely geometric

| quantity | vs transverse geometry | vs base stiffness |
|---|---|---|
| **pinning** Δω = −2cβ sin k | **protected** | **protected** |
| **κ**, the A² coefficient | varies — w = 2: box-dependent, → 0 as the box grows (**CLOSED**, §5); w = 3 re-measured and **CLOSED** (width scan, §5) — the old w = 3 value and geometry table *superseded* | **varies, 26% over ±10%** (−0.0214 / −0.0187 / −0.0165) — ~~~35%, 0.0959 / 0.0799 / 0.0677~~ retracted |

*κ vs stiffness re-measured with correct seeding (`joint3_kappa_stiffness.py`);
the qualitative split stands. κ vs transverse geometry: w = 2 re-measured with
own-branch seeding and **CLOSED** (§5) — static fixed dilution, F = 2 − s, κ → 0 as
the box grows, no box-independent localised κ; the w = 3 value and the geometry
table were measured with the swapped seeding and remain unverified [since
SUPERSEDED by the width scan, §5: w = 3 CLOSED, the mechanism general across
widths 1.5–4].
The pinning rows are unaffected.*

Two independent knobs, same split. **The pinning carries the falsifiable content
of the U(1) sector; κ is a contingent coefficient** that must always be quoted
with its profile *and* its stiffness. [And, for a localised beam, its box: only the
plane-wave κ is a real coefficient — §5, κ(w = 2, side) CLOSED.]

### 5b.4 Joint #4 — WITHDRAWN as a law, retained as an observation

A **stiffness gradient** appeared to couple to the pin as Δ ∝ (∇s)² with
C = −1337 at 2% scatter over a linear ramp. **That law does not generalise and
is withdrawn.**

**The adiabatic contribution is provably zero.** Linearising,
ω² − 2βcω sin k − ω₀²(s) = 0 with ω₀²(s) = s√5 + 2c(1−cos k), so
ω(±k) = ±βc sin k + √(β²c²sin²k + ω₀²) and **Δω = 2βc sin k, independent of s.**
This holds **locally**, at every x — s enters only through ω₀², common to both
branches. **WKB therefore predicts zero pin shift for any profile.**

**And the measured effect contradicts the quadratic law.** For Gaussian bumps of
fixed amplitude and growing width, **max\|∇s\| falls monotonically
(1.4×10⁻² → 9.5×10⁻⁴) while the shift RISES** (−4.6×10⁻⁵ → −3.6×10⁻⁴), peaking
near σ ≈ 16 on N = 128 and falling by σ = 32. A law quadratic in the gradient
requires the shift to fall with the gradient. It does the opposite.

**Why every local-functional guess failed.** ⟨g²⟩ over the chain, Σg²/N², and
⟨\|g\|⟩² were each tested against four source rows; spreads 57%, 57%, 28%, none
constant. There is no local functional of ∇s because **the effect is not
governed by ∇s** — the σ-peak is resonance-like, consistent with non-adiabatic
mode mixing at a characteristic scale.

**What survives:** a non-uniform stiffness *does* perturb the pin, and the
adiabatic contribution is *proved* zero, so the mechanism is non-adiabatic mode
structure — **not a gravitational-redshift analogue.** It becomes a law only when
a mode-mixing calculation predicts the σ-peak.

### 5b.6 Joint #5 — a DERIVED law, source strength in, pin shift out

**δ(Δω) = −¼ · β · s² · S²**

The only pure number is **¼**, and it is derived, not fitted.

**Measured** (ε-continuation of the ±k eigenvalues — the one estimator that
survives, see §5b.4): C = −0.2507·β·s², constant to **0.3%** across a 2× range in
β and a 16× range in s². Quadratic in S to 0.1% for S ≤ 0.02, with a resolvable
S⁴ correction above.

**Derived:** the block kernel's IR limit is **K(p→0) = −β·s²/2** — verified to
ratio 1.0006 across the same parameter range. The fold over a unit-rms localised
profile gives δ = ½·K·S², since Parseval fixes Σ|η̂_p|² = rms² and the ½ is the
±p double count. Hence **C = −β·s²/4**.

| factor | origin |
|---|---|
| **β** | the pin's own antisymmetry — at β = 0 the dispersion is even in q and the shift vanishes identically |
| **s²** | the perturbation enters squared at second order |
| **¼** | ½ (kernel IR limit) × ½ (±p fold) |

**Shape-independence is explained by the same derivation.** The fold needs only
Σ|η̂_p|² and K is flat in the IR, so any localised mean-zero profile with the same
rms gives the same answer:

| shape | C |
|---|---|
| Gaussian σ = 8 | −0.06268 |
| Gaussian σ = 16 | −0.06256 |
| sech² w = 8 | −0.06277 |
| **two separated bumps** | −0.06284 |
| *linear ramp (different class)* | *−0.02807* |

Four profiles — different widths, different tails, **disconnected support** —
agree to **0.4%**. The ramp differs because it is discontinuous on the ring and
its spectrum reaches p where K is no longer flat.

**This retroactively explains the withdrawn ∇s law** (§5b.4): the ramp used to
calibrate it and the Gaussians it was applied to are in **different classes**.
The 8–11× miss and its tracking of spatial extent were class mismatch, not a
failed quadratic.

**⚠ −1/16 was a coincidence.** C = −0.0627 sits 0.3% from −1/16, and a parameter
scan was run *before* writing that up: C moves ∝ β and ∝ s², so the agreement was
an artifact of the default values. **−¼ survives the test −1/16 failed.**

**Still not Einstein:** no field equation, no stress-energy, no derived G. S is
posited. What is new is that the *response* to it is derived.

### 5b.6a Joint #5 is q = 1 ONLY — and the observable, not the geometry, is why

**C_q(N) fails to converge for every q ≥ 2, identically:**

| q | sides | C values | spread |
|---|---|---|---|
| **1** | 48, 64 | −0.06343, −0.06303 | **0.6%** |
| 2 | 10, 12 | −0.29050, +2.93361 | **244%** |
| 3 | 6, 8 | +1.92846, −0.18799 | **243%** |

q = 2 and q = 3 fail by the *same* amount and both flip sign. So this is **not**
about three dimensions being hard — **the observable only exists in one
dimension.** A single continued Bloch label is a stable quantity as N → ∞ at
q = 1 and at no higher q.

**At q = 3 specifically**, with every point resolved (shift ≪ gap) and
S-extrapolated at matched box fraction: C₃ = **−0.188, +1.458, −1.429** at sides
8, 12, 16. Each is well-defined *per box* — S-convergence is tight — and the
sequence does not approach a limit.

**The IR/dense split does not rescue it.** Partitioning the second-order sum at
\|p\| ≤ 2π/8 (the side-8 shell, rule fixed before looking): C_IR = **−0.0253**
at side 12 and **−0.0365** at side 16 — a 44% move, neither near the side-8
−0.188. C_IR + C_dense reproduces C_full to ~1%, so the additivity is sound and
the conclusion is not a method artifact. **Even the coarse sector is not a
constant**, and it carries only ~2% of the total: the result lives in the dense
modes.

**⚠ TWO MECHANISMS PROPOSED AND BOTH RETRACTED.**

*Degeneracy:* claimed the branch sits in a multiplet. **Wrong** — dense
diagnosis gives **multiplicity 1**, with clean k-content (2,0,0) at side 8 and
(3,0,0) at side 12. The "2 eigenvalues within 1e-9" was a duplicate listing.

*Collapsing gap:* claimed the probe gap falls as ~1/N in 3D. **Wrong** — those
values (5.8×10⁻⁴, 1.2×10⁻⁴) came from `eigs` near a shift returning near-duplicates.
**Dense spectra give O(10⁻²) at q = 3, not falling**, and comparable to q = 1:

| q | gaps across sides |
|---|---|
| 1 | 8.2×10⁻², 6.4×10⁻³, 6.4×10⁻³ |
| 2 | 3.8×10⁻², 6.9×10⁻³, 5.3×10⁻³ |
| 3 | 1.4×10⁻¹, 3.8×10⁻², 5.9×10⁻² |

**So C_q(N) not converging for q ≥ 2 is a measured fact without a named
mechanism.** The label is isolated and the coefficient still refuses a limit —
isolation was never sufficient for a universal number, only for a well-defined
per-box one. A plausible unproven account: the second-order sum runs over *many*
modes, and as the box changes the discrete **k**-grid under a fixed blob fraction
moves, sampling different parts of a 3-D kernel that is not IR-flat. Sign flips
with N look like that. No collapsing neighbour required.

**On (1,3):** the spectral behaviour singles out q = 1 and says **nothing about
how many** spatial dimensions follow — q = 2 and q = 3 fail identically. It is a
kinematic analogy, not a second route to the signature, which is derived
elsewhere from well-posedness (§2b of the ledger) and three intersecting
requirements.

### 5b.7 Residual sector — base-stable on q = 3

| check | status |
|---|---|
| C_r = 0 inert | **pass** — both q = 3 sizes and q = 1 |
| B ∝ C_r² | **pass** — ratios 3.88, 3.83, 3.85 for 2× C_r |
| drift | **pass** — ~10⁻⁷ throughout |
| **working range** | **C_r ≲ 0.05** |
| side dependence | ~20% on \|B\| — flagged |

Above C_r ≈ 0.1 the response turns over. That is the **fixed-T phase sampling**
of the residual oscillation (f_res ∝ C_r, §4b.1), not saturation and not a
failure.

**Do not quote a universal \|B\|** — the ~20% box dependence is the same class of
issue that made κ unquotable. **Colour stays on n = 3 / u(3), not on the
residual.**

### 5b.5 Standing summary — EM-like × base

| perturbation | pin | κ | holonomy |
|---|---|---|---|
| transverse geometry | **protected** | varies | — |
| **uniform** base stiffness | **protected**, null 1.2×10⁻⁵, derived | **varies 26%** (~~~35%~~ retracted, §5b.2) | **moves, sign predicted** |
| **non-uniform** base stiffness | perturbed by **non-adiabatic** mode physics; adiabatic shift proved zero | — | — |

**Joints #1, #2 and #3 carry.** #4 is an observation without a law.

**Open:** the ~20% shift residual on joint #1 (readout timing, Qt, ramp
weighting), and a mode-mixing calculation for the σ-peak if #4 is to be
recovered.

**Target for that calculation, now measured and box-independent:** the kernel
must peak at **σ\* = 2λ = 4π/k** — verified σ\* = 8 at both N = 64 and N = 128
(so not finite-size), doubling to 16 when k halves, with σ\*·k = 4π in all three
runs. It must also reproduce the **sign change**: at k = π/4 the shifts are ~30×
larger and flip sign. Predict those before fitting the scan.

## 6. Harness — mandatory

    from harness import calibrate_all, freq_phase, rank_abs
    calibrate_all()      # locks anything that fails

`harness.py` refuses to report from an uncalibrated routine. It currently
**locks `freq_fft`** at 0.4985 relative error against the exact Duffing shift,
while `freq_phase` passes at 0.0028.

**Non-negotiable, because:** a factor-of-ten error and a factor-of-2.6 error both
reached a document intended for a lab, and neither was caught internally.
Convergence in timestep tests the integrator, **not** the instrument reading it.

Also required: **absolute** floors on rank tests, a control that **can fail**,
and more than one sample before a system is characterised.

---

## 6b. The assembled model — `04_scripts/session/model.py`

The spec is implemented. **One force law parameterised by node size**, replacing
three scripts with three node types. Run it:

    cd 04_scripts/session && python3 model.py

| gate | measured |
|---|---|
| 1 harness calibrated | freq_phase pass, **freq_fft LOCKED**, rank_abs pass |
| 2 free propagation | energy drift **2.0×10⁻⁷** |
| 3 complex structure selected | chirality purity **0.9834** |
| 4 passivity ⟹ u(n), dim n² | **1, 4, 9, 16, 25** |
| 5 n=1 asymmetry, measured | **0.100003** vs −2cβ sin k = 0.100000 |
| 6 κ and the β-collapse | **κ = 0.0798 ± 0.00089** across a tenfold β range — RETRACTED (swapped seeding, §5); corrected −0.0184 ± 0.00033 |
| 7 u(2) ordering | sim-vs-pred **0.285°, 0.165°**; splitting 101.12 measured, 100.80 predicted |
| 7 u(3) ordering | sim-vs-pred **0.615°, 0.566°**; splitting **65.1166** measured, **64.9712** predicted |
| 8 Abelian control | **0.0169°** where theory says 0 |
| residual sector | present, C_r = 0 by default, inert to 4.5×10⁻¹⁹ |

**One script now reproduces every verified result the programme has** — u(1),
u(2), u(3), the pinned asymmetry, κ, the β-collapse, both ordering splittings and
the control. `U_segment` is generic in node size: spectral projectors, one per
eigenvalue, with the u(2) two-eigenvalue case falling out as the degenerate
instance of su(3)'s three.

**Gates run in order and stop at the first failure**, because a later gate means
nothing if an earlier one is broken.

`force()` is a single function. The residual term (§4b) would be **one more line
in it** — even in direction, respecting V = −2 log(1 − 4B), vanishing at B = 0.
That is now a concrete edit rather than an open design question.

## 7. Build order

| step | do | done when |
|---|---|---|
| 1 | import `harness`, call `calibrate_all()` | locked routines refuse |
| 2 | ring of N nodes, φ-well, elastic coupling | free packet propagates, drift < 10⁻⁶ |
| 3 | gyroscopic κ𝕁 | complex structure selected; chirality purity ~0.98 |
| 4 | n = 1 velocity coupling | Δω = −2cβ sin k reproduced |
| 5 | nonlinear regime | κ = 0.0799 reproduced, β-collapse holds — κ RETRACTED (§5); gate now reports −0.0184, collapse still holds |
| 6 | n = 2 nodes, Pauli W | ordering 59.86° vs 59.84° |
| 7 | n = 3 nodes, Gell-Mann W | ordering 65.12° vs 64.97°, control ~0 |
| 8 | segments at ≤ 20-site separation | Abelian control 0.0137° |

**Step 8 is not cosmetic.** The prediction has no free-evolution operator between
segments; at a 60-site gap the Abelian control reads 62° where theory says 0.

---

## 7b. Why the model stops at D8 — the tower says so

**D16 and above are not excluded material. They are the reason the build target
is D8**, and that is a verified result rather than a convention.

**The distillation operator.** Ask where composition survives at each level:

| algebra | configuration | \|LᵀL − \|a\|²I\| |
|---|---|---|
| 𝕊 (16) | both halves generic | 3.075 |
| 𝕊 (16) | one half zero — **octonionic** | **8.9×10⁻¹⁶** |
| A₅ (32) | both halves generic | 9.918 |
| A₅ (32) | Q = 0, P a generic **sedenion** | 3.638 |
| A₅ (32) | supported on the first 8 — **octonionic** | **3.6×10⁻¹⁵** |

**The composition locus of every level ≥ 16 is the octonions.** Not the
intermediate sedenions — a generic sedenion half still fails at level 32. So the
operator is a **projection, not a sequence**: it lands on D8 in one step from any
height, and D8 is its fixed point everywhere above.

**What that licenses, and what it does not.** The result is about
**composition**. It licenses: *nothing above adds structure to the composing
sector* — a model's octonionic fibre is complete at D8 and climbing adds no
composing structure.

**It does NOT license "nothing above matters."** The residual (§4b) is by
construction the part that does **not** compose, so the distillation says nothing
about it. Generalising a result about one sector into a claim about the tower was
an error and is corrected here.

**This matters for Standard Model reach.** C1S2 §8 item 4 establishes that **no
basis-independent 3 exists in 𝕆** — the three Fano lines through a point are a
single su(3) orbit, connected by finite elements at ~10⁻¹², so the discrete 3 is
a basis artifact. The conclusion filed was that *any generation structure
requires a genuinely new object.*

**The residual is that object**, and the generations question **was never asked
there.** It was asked at D8, answered negatively, and not re-asked above. So:

> A model consisting of the D8 fibre alone cannot carry a generation structure,
> by the programme's own D8 result. If generations are in scope, they must be in
> the residual sector or nowhere — and that has not been tested.

That is an open question, not a closed one, and §4b is in the model partly
because of it.

**The tower also saturates.** dim Der = 14 at dimensions 8, 16 and 32 alike
(Schafer 1954). Eigenvalue multiplicities of LᵀL are multiples of 4 at every
level — [4,8,4] at D16, all 4s at D32 (Biss–Christensen–Dugger–Isaksen 2009).
Block size does not grow; only the count does. Climbing buys repetition, not new
structure.

**Verified at D16, and standing:** non-homogeneous with orbits of dimension 11 in
S¹⁵ (codimension 4); four invariants Re(a), Re(q), |p|², p·q constant to 10⁻¹⁶;
zero divisors confined to (0, 0, ½, 0) with kernel exactly 4; that locus is
G₂/SU(2), stabiliser 3, bracket-closed at 1.4×10⁻¹⁵; the exact reduced metric,
drift and domain; det L = D₂² (Koebisu). Composition failure genuinely produces
non-round structure — σ_min 0.102–0.853 at D16 against 1.0000 exactly at D8.

**One D16 result is withdrawn**, and only one: the irregular spectrum, because Ω
is chosen and Colbois–Dryden–El Soufi makes any chosen Ω uninformative
(`LADDER_TWO.md` §6a).

---

## 8. Not in the build, and why

*These do not change the code you write. That is different from not being
established — see §7b.*

| piece | why not in the build |
|---|---|
| D16 structure (§7b) | **established**; it justifies the target rather than entering it |
| the reduced Laplacian **spectrum** | **withdrawn** — Ω is chosen, any Ω gives any spectrum |
| **generation structure (three-fold)** | **CLOSED NEGATIVELY — do not re-run.** The tower **doubles**: copies of the D8 content go 1, 2, 4, 8, 16 = 2^(k−3), every level a power of two. Three is unreachable at any height, not for want of searching but because doubling cannot produce it. Independently confirmed by eigenvalue multiplicities being multiples of 4 at every level (D8 [8], D16 [4,8,4], D32 all 4s). Together with C1S2 §8.4 — no basis-independent 3 in 𝕆, the three Fano lines being a single su(3) orbit at ~10⁻¹² — **neither D8 nor the residual carries generations.** The only three the programme owns is the **role triad**, which fixes node size n = 3; connecting node size to generation count is a separate and unsupported claim. |
| the reduced Laplacian and its spectrum | Ω is chosen, and any Ω gives any spectrum (Colbois–Dryden–El Soufi) |
| Nekhoroshev | practically inaccessible — signal 10⁹ below integrator drift |
| G₂-stabiliser su(3) | imported (Günaydin–Gürsey 1973); the trimer route is the programme's own |
| forced spectra vs data | failed on functional *form* — quadratic towers against linear Regge trajectories |

---

## 9. Open, and what each blocks

**Live items only.** Closed items are recorded in their own sections and in
`PROVENANCE.md`; they are not repeated here.

| question | blocks |
|---|---|
| **what fixes the fibre metric scale** | **the largest one — it blocks three numbers at once.** ℏ, c₈ and Λ are all functions of it (κ_ℏ ∝ 1/k, c₈ ∝ k², Λ ∝ 1/k). Nothing in the architecture supplies a length: structure constants are ±1, \|1\| = 1 is a norm, and 2, 6, 42, 3/8, 2π² are ratios. The φ-well does supply a length (√5) but it lives on the **radial** coordinate while ℂP² is the **angular** one, and the HK cone relation ties them only as k = m — which discretises the node mass without fixing its unit |
| **what fixes C_r** | the residual coupling *form* is determined (§4b.1); its *strength* is not |
| ~~fourth-order cross-modulation (κ, F at A = 0.30)~~ **DONE for the plane wave, 2026-09-25** (§5, "κ to fourth order"); the original entry: | the derived F is second order; it matches every sharp point except the two narrowest beams at A = 0.30, which it matches at A = 0.10 (§5, "The cross-modulation factor F, derived") |
| ~~beam fourth order with orbit-consistent launches~~ **DONE 2026-09-25** (§5): launch pieces derived, H0 and S1 excluded; the original entry: | the beam fourth-order tests (H0 excluded at large fill; S1 against S2 unresolved) used plain-cosine launches, so they mix physics with launch effects. Redo them with orbit-consistent (second-order) launches (`kappa_seed2_test.py`), derive the beam's fourth-order kernel, and derive the launch's static-shift and second-harmonic pieces, which are measured, not derived (§5) |
| ~~S2 at w = 3, L = 4, and why the cross terms cancel~~ **SUPERSEDED 2026-09-25** (§5): with the third harmonic in the launch S2 fails three of four L = 4 beams, and the cross terms are partly present; the original entry: | two things: (1) test S2 at w = 3, L = 4 with the third harmonic added to the launch — S2 failed that beam by +18σ under the committed criterion and fits it only under a post-hoc third-harmonic band; (2) derive why the fourth-order cross terms cancel, leaving only the box-wide component's own term (S2 is a surviving hypothesis, not a derivation) |
| **which J-compatibility bound applies at q = 3** (candidate, not adopted; §3) | κ ≥ 0.972 if the coupling segments conserve transverse momentum, κ ≥ 2.091 if not — decides the candidate floor on κ at the model's own dimension |
| **the beam's fourth-order cross kernel** | derive it — every fourth-order cross term between a beam's transverse components, on the lattice — with the measured r values as the target: 0.351/0.350 (w = 1.5), 0.547/0.549 (w = 2), 0.750/0.758 (w = 3) at L = 4, A = 0.30/0.40, which lie 0.33, 0.50, 0.68 of the way from the self-only limit S2 to the all-terms limit S1 (§5, measured with the third-harmonic launch, `kappa4_orbit3_launch.py`) |
| ~~verify the 0.39° attribution~~ | **CLOSED** — readout timing; the Abelian floor is exactly 0 under clearing readout at q = 1 and q = 3 |
| ~~wavenumber-averaged prediction at q = 3~~ | **CLOSED** — closes the gap to 0.04–0.17°; the q = 3 gauge sector is quantitatively verified |
| ~~save the q = 3 scripts~~ | **CLOSED** — all test scripts and results are in `shape_zero_tests/` in this repository, with both `model.py` versions pinned by content hash (`c49da46f` for q = 1 work, `948b09e8` for q = 3). Runs from the repository alone; `q3_kavg.py` reproduces its saved result with difference 0.0 |
| ~~permanent q = 3 gate~~ | **CLOSED** — `shape_zero_tests/q3_gate.py` (commit `ed32937`). u(2) and u(3), AB/BA plus Abelian floors, clearing readout, spectrum-averaged prediction; PASS needs every error < 1° and floors < 0.5°. On 260×8×8: **PASS at 0.10–0.18°**; the same runs scored with the single-wavenumber prediction **FAIL** (2.4–5.1°) — proof it can fail. Refuses a verdict if any window fails to clear or the lattice can wrap. 11.6 min wall on 4 workers. Slab kept at 8×8 to stay clear of the transverse-uniform trap |
| free-evolution operator between segments | would remove the ≤20-site gap constraint in §7 step 8. Note the profile-map test showed the q = 1 product residual is already sub-degree, so this matters at q = 3 or not at all |

### Closed this session

α (input, not predicted) · generations (tower doubles, never 3) · A-2 torsion
class (field-cubic, difference, odd in k) · the physical cycle (Gr(k,n), ℂP² at
n = 3) · D = 8 (arithmetic, 4 + 4) · n = 3 (role triad, plus SU(3)-in-6 as a
second route) · the q = 3 gate port · the spatial arrival model (optional,
0.08° gain) · the C_r saturation anomaly (oscillation, f_res ∝ C_r)

**Retracted this session:** ℏ = (μℓ_f²/T)/4 (generator normalisation is not a
metric scale) · the 1-D Bloch map failing at q = 3 (tube clipping) · G being
dimensionally impossible (hand arithmetic) · κ(w) as a publishable formula
(depends on the transverse domain, not just the beam) [κ(w = 2, side) re-measured
with own-branch seeding and CLOSED: static fixed dilution, κ_box = κ_pw·fill·F,
F ≈ 2 − s; κ → 0 as the box grows; no box-independent localised κ — §5]
