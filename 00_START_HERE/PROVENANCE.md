# Provenance Ledger — how each claim moved, and why

**Purpose.** The current value of a claim is in `CLAIM_INDEX.md`. This file
records the **trail**: what was claimed, what went wrong, what the corrected
value is, and which check caught it. When something later fails, this is how you
find where the mistake entered rather than re-deriving it.

**Rule.** No claim is silently updated. A value that changes gets an entry here
with its predecessor, the reason, and the catch.

---

## 1. κ — the A² coefficient in the pinned asymmetry

The most-corrected number in the programme, and the one that reached a
lab-facing document twice while wrong.

| version | value | why wrong | caught by |
|---|---|---|---|
| first | **0.30** | hand-arithmetic in prose: divided by −0.2 instead of the leading term −2 | external referee |
| second | **0.0305** | FFT-peak frequency estimator, biased 1.8–2.0× at short records | independent reimplementation |
| third | **0.0799** | RETRACTED 2026-09-24: reference seeded each direction with the other's root | P-1 investigation; confirmed with separate code (§6o) |
| **current** | **−0.0187** (β = 0.05, A = 0.30); **−0.0184 ± 0.00033** β-sweep | — | own-root seeding; agrees with second-order PT −0.0175 |

**Diagnostic that settled it:** `estimator_calibration.py` against the exact
Duffing shift 3ε/(8ω₀²) = 0.044263. Phase regression recovers it to **0.4%**;
FFT-peak is off by **2.0×** at T = 300 and by 1.8× on the quadratic case, in
*opposite* directions.

**Why the second error survived so long:** timestep convergence was checked and
passed — κ = 0.03109 from DT = 0.02 down to 0.0025. **Convergence tests the
integrator, not the instrument reading it.** Two internally consistent sweeps
also agreed, but they shared one biased estimator; internal consistency is not
accuracy.

**Reference implementation:** `pinned_asymmetry_reference.py` (DOP853,
rtol 10⁻⁹, T = 900, weighted phase regression).

---

## 2. u(3) ordering splitting

| version | measured | control | verdict |
|---|---|---|---|
| first | 44.891° | **14.987°** where theory says 0 | uninterpretable |
| corrected control | 65.876° | **36.43°** with different strengths | still failing |
| **current** | **65.12°** vs **64.97°** predicted | **0.0137°** | **passes** |

**Four defects, found in sequence, each isolated by a test that could fail:**

1. **Integrator** — velocity-Verlet is not symplectic for velocity-dependent
   forces (the gyroscopic κ𝕁v and directional Wv terms). Drift 7.4×10⁻¹ → 3×10⁻⁷
   with RK4.
2. **Readout** — single-site `argmax` extraction let packet spreading leak in.
   Replaced by the whole-lattice density matrix.
3. **Composition order** — the prediction composed `U(a)@U(b)` where the packet
   meets the *first* segment first. Flipping it dropped residuals from ~70° to
   7.6°.
4. **Inter-segment propagation** — the prediction has no free-evolution operator
   between segments. At a 60-site gap the Abelian control reads 62°; at 40 sites
   29°; **at 20 sites 0.0169°.** Residuals scale with the gap.

**The first "control" could not fail** — it ran the identical spec twice, so
0.0000° was a determinism check, not a commutativity check. A real control needs
the same axis at *different strengths* in both orders.

**Near-miss worth recording:** at the intermediate stage, measured 65.88° against
predicted 63.28° looked like agreement while sim-vs-pred was 70–113° in every
case. **Two numbers landing near each other while the underlying states are 100°
apart.** Comparing only the splittings would have declared success.

**Scripts:** `phi_gauge_u3_working.py` (applied), `phi_gauge_u3.py` (annotated
handoff, raises `NotImplementedError` by design).

---

## 3. A-2 leg (ii) — introduced, withdrawn, and twice replaced

Three distinct claims have travelled under one label. Separating them:

| claim | status | script |
|---|---|---|
| leg (ii) as originally written | **withdrawn** — reading-dependent | `a2_leg_ii.py` |
| **replacement: intrinsic torsion** | **verified, four classes** | `a2_intrinsic_torsion.py` |
| **holonomies preserve the algebra** | **verified**, residual 6.3×10⁻¹⁶ | in the C1S2 correction record |

**Why the original is reading-dependent, not simply false.** On the weak reading
(sign flips) |A| = 896 with ≥120 elements outside B — leg (ii) **fails**. On the
strong reading (colouring plus induced orientation) |A| = |B| = **1344 exactly**
and it **holds by construction**, because the role assignment was built to
determine the multiplication. Neither reading gives independent support.

**The replacement is a real result.** Of **147** torsion components, **98** are
absorbable by re-choosing the G₂-compatible connection and **49** are not,
splitting into **four irreducible classes: 1 + 7 + 14 + 27**, identical across
all 16 valid algebras. So "is the octonion structure background or dynamical" is
**not** yes/no — it has exactly four independent switches, 2⁴ = 16 possible
partial closures. Any replacement for leg (ii) must say which of the four it
turns off.

**The holonomy result is also real and was nearly lost.** The claim that *no*
transport built from multiplications preserves the algebra is **false**: g₂ sits
inside the span of the commutators [L_a, L_b] at **6.3×10⁻¹⁶**. Single generators
are orthogonal to g₂ and do not preserve; **closed-loop holonomies can.**

**Filing defect this exposes:** both positive results sat *inside* a retraction
paragraph, where they read as part of the withdrawal. A verified claim with a
machine-precision residual should never be filed only as a footnote to a
retraction.

---

## 4. Base signature

| version | claim | why wrong |
|---|---|---|
| C1S §14 | *"the ladder cannot be Lorentzian"* | persistence constrains the **fibre**; compact Euclidean fibres over an indefinite base are ordinary gauge theory |
| **current** | (1,3), **conditional on a base existing** | — |

**Standing caveat:** C1S2 §§1–3 derive the base's signature, dimension, Lovelock
form and Maxwell — all **given that a base exists**, which nothing supplies. The
invariant torus (q = 1) is excluded because so(1) = 0 gives no centrifugal term.

---

## 5. su(3) — two routes, one forgotten

| route | source | status |
|---|---|---|
| G₂ stabiliser of a unit imaginary octonion | Günaydin–Gürsey 1973 | **imported**, not the programme's own |
| **trimer, Rank 3, passivity ⟹ u(3)** | spec v5.3 line 113 | **the programme's own**, and it was overlooked for the whole D8 derivation |

**n = 3 is the role triad** and it is forced: blocks of a Steiner triple system
have cardinality three because there are three roles (Formal Proofs §3). The
algebra corroborates — at n = 2 the associator is **1.1×10⁻¹⁶** (observer and
observed with no observation; nothing is carried) against **0.81** on the full
algebra.

**How it was missed:** a search of eight platform files for four keywords
reported u(3) absent. It was in the spec.

---

## 6. Withdrawn, and not to be re-derived

| claim | status | source |
|---|---|---|
| Wick coefficient **28** | **withdrawn** — control test failed | `REVISION_2_BASE.md` |
| soft-region mechanism | **withdrawn** | `REVISION_2_BASE.md` |
| Ladder Two irregular spectrum | **withdrawn** — Ω is chosen; Colbois–Dryden–El Soufi makes any Ω uninformative | `LADDER_TWO.md` §6a |
| forced spectra matching data | **failed** — mismatch of functional *form* | `INPUT_LEDGER.md` §5 |
| Nekhoroshev route | **closed practically** — signal 10⁹ below integrator drift | `HIERARCHIES.md` §3 |

---

## 6b. Corrections banked in the input-derivation pass

| claim | was | now |
|---|---|---|
| invariant antisymmetric forms on the structure sphere | script reported **0**, contradicting its own conclusion | **1** — relative-only rank threshold on an identically-zero matrix (max singular value 5.3×10⁻¹⁶) |
| residual coupling form | C_r·mul(g, **x**) — does work, F·v = 12.6 | C_r·mul(g, **v**) — F·v = 2.2×10⁻¹⁵ |
| C_r "saturation" | B insensitive to a fourfold coupling change | **not saturation** — B *oscillates*; a fixed readout time samples an arbitrary phase. f_res ∝ C_r |
| Kaluza–Klein 15 = 10 + 4 + 1 | cited as the gravity–EM relation | **holds only for a circle fibre.** S² gives 21 = 10 + 3 + 8. The KK gauge group is Isom(fibre), not the passivity u(n) |
| dimensionless inputs | stated as a **theorem forbidding** dimensionful outputs | a **scoping statement** about the current principle set. Adding inputs is legitimate; the test is conditions vs parameters |

| **which cycle is physical** | two candidates, n = 4 vs n = 8, unsettled | **settled**: structure manifold is Gr(k,n); n = 3 gives ℂP², so **n_flux = 4**. Gr(1,2) = S² recovers the n = 2 case as a control |
| **G from Kaluza–Klein** | claimed derived, then claimed **dimensionally short by L³** | **the shortfall claim was wrong.** Both were errors: the second used the c ≠ 1 form of [G_D] *and* took D = 5 for a **four**-dimensional fibre. With [G_D] = L^(D−3)/M and D = 8, the reduction is consistent. What is open is c₈, and whether ℓ_f = ℓ |
| input count | three → four | **three or four**, pending whether the cone identifies ℓ_f with ℓ |

**Failure mode, corrected:** the lesson is *not* "a negative read as a positive"
— that was itself a wrong diagnosis. It is **do dimensional arithmetic in code
before drawing a structural conclusion from it.** The exponents were written by
hand in prose, the same class of error as the factor-of-ten κ.

**And a process note:** the erroneous condemnation was caught only because it was
challenged before commit. A negative verdict deserves the same verification as a
positive one.

**New, derived:** ℏ = (μℓ_f²/T)/**4**, with the cycle fixed by the node size — J² = −I forces
tr(JᵀJ) = 4 at zero spread over 400 samples, rescaling breaks J² = −I, and ℝ²
gives n = 4 as a dimension-sensitivity control.

## 6c. The three externally proposed lemma sets

Each was checked rather than accepted. **All three conclusions survived; all
three arguments needed correction.**

| proposal | conclusion | correction required |
|---|---|---|
| **ℏ, three lemmas** | ℏ = (μℓ_f²/T)/4 — **holds** | Lemma A used Fubini–Study (area π), Lemma C the round convention (4π); **the 1/4 was π/4π, a convention mismatch.** The model's own su(3) normalisation tr(λ_aλ_b) = 2δ_ab gives n = 4 directly |
| **Lemma B, nodal mass** | ℓ never enters — **holds, and is the important part** | the chain multiplied by T to "convert to an action"; **a U(1) moment map already *is* an action**, μωℓ_f² = μℓ_f²/T = M·L. The extra T over-counts by one length |
| **D = 8 by holonomy** | D = 8 — **holds, by arithmetic** | neither dim 7 nor 8 preserves both a complex structure and a 3-form. **SU(3) in dim 6 does** — which is the trimer node, giving a *second route to n = 3* rather than to D |

**Net gain:** the ℂP³ corollary excludes the n = 8 flux route structurally; ℓ_f/ℓ
is shown **irrelevant to ℏ** rather than left open; and n = 3 now has two
independent derivations.

## 6d. Near-misses recorded and refused

Numbers in this model that land close to physical targets, kept visible so they
are not rediscovered and written up.

| number | model source | target | proximity | verdict |
|---|---|---|---|---|
| **exp(42)** = 1.74×10¹⁸ | ‖c‖², the nonzero structure-constant count | 10¹⁷ electroweak–Planck | factor 17 | **numerology** — no mechanism puts ‖c‖² in an exponent |
| **128** | the orientation count (16 valid of 128) | α⁻¹ = 137.036 | 7% | **refused** — no rule selects it; proximity after seeing the target |
| 147, 168, 189 | incidence-data products | 137.036 | 7–38% | **refused**, same reason |
| **β = 1/4** | B-2 Fisher–Rao candidate | α = 1/137 | factor 34 | not α; a lattice gyroscopic ratio is not a QFT coupling |

| **ℏ = (μℓ_f²/T)/4** | flux quantum on ℂP¹ ⊂ ℂP² | a derived ℏ | — | **RETRACTED** — `tr(λλ)=2` is a *generator* normalisation, not a *metric scale*. Integrality permits n ∈ {1,2,4,8,…} and selects none |

**The ℏ retraction is the second time today a result of mine was wrong in both
directions** — G was claimed derived, then claimed dimensionally impossible, and
both were wrong; ℏ was claimed unfixed (correctly, in C1S §9), then claimed
derived, and the original was right. **A reversal of a prior cautious assessment
deserves more scrutiny than a new claim, not less.**

**The rule that catches all four:** compute the factor from a stated rule
*before* comparing to the target. Every entry above fails that test, and each was
found while looking at the target rather than at the model.

## 6e. The q = 3 port

| claim | status |
|---|---|
| pinned asymmetry at q = 3 | **survives** — 0.100001 vs theory 0.100000 |
| **κ at q = 3** | **does NOT survive** — 0.0799 → 0.0168 (w=2), 0.0311 (w=3), and depends on transverse width [all measured with swapped seeding: 0.0799 RETRACTED. κ(w = 2, side) re-measured: −0.00447 → −0.00047 over side 8–32, **measured and open**; w = 3 and the geometry table UNVERIFIED — §6o] |
| gates 5–8 in `model.py` | measured at **q = 1**; 7–8 not yet re-run at q = 3 |

**A vacuous test was caught and is now documented.** The first q = 3 run gave
results **bit-identical** to q = 1 at every β — because the seed was a
transverse-uniform plane wave whose transverse Laplacian is identically zero, so
the 3D problem reduced exactly to 1D. **Identical numbers from a supposedly
different configuration are a red flag, not a confirmation.** Third instance in
one session of a test that could not fail.

**A bug that produced a wrong conclusion:** `model.py`'s `energy()` computed the
**1D** gradient on a q-dimensional lattice, making q = 3 appear non-convergent at
9.5×10⁻³ *independent of timestep*. Insensitivity to DT was the tell — truncation
error falls with DT. Fixed; drift is 3.8×10⁻⁸ falling to 3.7×10⁻¹¹, proper RK4.
**The 3D base worked the whole time; the measurement did not.**

## 6f. The q = 3 gate port — five traps

Every q = 3 number in this thread was wrong at least once before it was right.
The traps, in the order they were hit: **flat indexing** (rolls addressed the
last spatial axis), **flat segment placement**, a **transverse-uniform seed**
(3D reduced exactly to 1D, bit-identical output), **overlapping segments** (12-long
ramps placed 3 apart), and **wrap-around** (modular position cannot detect it —
cumulative displacement is required).

**The unifying signature: identical or exactly-zero results from configurations
that should differ.** An equal-strength ordering control makes the two specs the
same array; it cannot fail, and it returned 0.000° three separate times.

| **1-D Bloch map at q = 3** | reported as failing by ~10–12×, "a prediction limitation" | **RETRACTED — it holds to 1.7°.** The discrepancy was tube clipping (rotation falls 0.885 → 0.582 → 0.270 as the gauge region narrows below the packet's transverse support), plus mid-segment readout, plus a flat `np.roll` in one working copy |

**Retracted below:** the matched single-segment comparison (same ramp, cumulative-exit readout,
no wrap either side): q = 1 gives 4.0°/5.6°/6.5° and q = 3 gives 41.5°/62.0°/77.1°
at g = 0.08/0.12/0.15 — a stable ~10–12× ratio scaling with g, with the g → 0
control at exactly 0.00° in both. **The 1-D Bloch map is adequate at q = 1 and
fails by an order of magnitude at q = 3.**

**Final q = 3 gate status:** free evolution, residual selection rule, gate 8
(0.39°), single-segment map (~2°) all **pass**; gate 7 **provisional pass**
(~3° per order, ~6° split error — decomposed as ~1.4° kinematic from unequal
\|g\|, removable by equal strengths, plus **~4–5° that survives equal strengths**
because the spatial profile arriving at the second segment is order-dependent
when the axes do not commute; a pure product U_B·U_A carries no spatial
information between segments and cannot represent it); gate 6 (κ) is **geometry-dependent** and is the only result the base
change actually alters.

**What survived:** gates 5 and 8 and the residual selection rule pass at q = 3;
gate 6 (κ) is geometry-dependent; gate 7 is unverified because the 1-D Bloch
segment map does not describe a localised packet on a 3-D base — measured as
4.0°/15.9°/44.9° at g = 0.01/0.04/0.12 against 0.01°/0.02°/0.05° at q = 1, with
the g → 0 control at exactly 0.00° in both.

## 6g. A-2 closed

The residual/torsion selection rule was re-measured on a live instrument.
**Baseline Δω = 0.100020 (ratio 1.0002)** — the original
`residual_selection_rule.py` returned 0.000000 for every case *including the
baseline*, so it never demonstrated its instrument was alive.

**The class: field-cubic, nearest-neighbour difference, odd under k → −k.**
Even (f = ε·x³) leaves Δω pinned to 10⁻⁶–10⁻⁵; odd (f = ε·(x₊−x₋)·x²) shifts it
by +0.61 at ε = 0.005. The observable is **Δω**, not energy drift.

**A stand-in built on spatial sin/cos parity tests a different operator class**
and its negative result carries no information about A-2. Retired.

## 6h. First joint observable

**Gauge holonomy vs on-site stiffness.** +10% stiffness gives a measured
Δ = **−7.96°** against a predicted **−6.34°** from `k_branch`/`U_segment` with no
new fitted constant. Sign correct; ~20% residual on the shift; the ~10° absolute
offset is the single-segment protocol floor and cancels in the difference.

**Why the sign matters:** higher stiffness gives a *longer* dwell, so the naive
reading predicts *more* rotation. Measured is less — the ω-dependence in
`k_branch` outweighs the dwell term, and the code encoded that before the
measurement.

**Not gravity** — no field equation, no source, no G. An index-of-refraction
effect. Its value is that it is **coupled**: neither sector alone produces it.

**Second joint observable — pin vs base stiffness.** Independent channel:
scalar U(1) rather than su(2) segments, frequency asymmetry rather than Bloch
angle. **Linear pin NULL to 1.2×10⁻⁵** across a 20% stiffness range, exactly as
the algebra requires (stiffness enters both branches identically and cancels in
the difference). **Liveness check passes** — branch frequencies move by +0.109,
so the null is not vacuous. **κ SIGNAL: 0.0959 → 0.0799 → 0.0677**, a 35% swing.
[κ values RETRACTED — swapped seeding, §6o. Re-measured: −0.0214, −0.0187,
−0.0165, a 26% spread; the linear pin null holds (0.99999). The result stands.]

**Joint #4 WITHDRAWN as a law** (numbering: #1 holonomy, #2 pin null, #3 κ, #4 gradient). A stiffness gradient appeared to couple as
Δ ∝ (∇s)², C = −1337 at 2% scatter — but only for a linear ramp. For Gaussian
bumps, **max\|∇s\| falls while the shift rises**, peaking near σ ≈ 16, which a
quadratic-in-gradient law forbids. Three local functionals were tested (spreads
57%, 57%, 28%); none constant. **The adiabatic shift is provably zero** — s
enters ω₀² identically in both branches and cancels locally — so the mechanism is
non-adiabatic mode mixing, not a redshift analogue. Retained as an observation.

**σ\* is box-independent:** 8 at both N = 64 and N = 128, doubling to 16 when k
halves, σ\*·k = 4π throughout — so **σ\* = 2λ**, carrier-locked, not a finite-size
artifact. A recovery of #4 must predict that peak and the sign change at
k = π/4 without fitting.

**A process note:** the withdrawal came from *deriving* the expected law rather
than testing a fourth functional. Searching functionals until one fits is
curve-fitting on the two profiles in hand — the exp(42) failure mode in another
costume.

**The pin/κ split is structural.** Two independent knobs — transverse geometry
and base stiffness — and the pinning is protected against both while κ varies
under both. The pinning carries the falsifiable content; κ is contingent.

## 6i. Joint #5 derived; residual certified

**Joint #5: δ(Δω) = −¼·β·s²·S², derived.** Kernel IR limit K = −β·s²/2 (ratio
1.0006 across 2× in β, 16× in s²); fold gives C = K/2. Shape-independent to 0.4%
across four localised profiles including **disconnected support**. Measured by
ε-continuation, the only estimator that survived §5b.4.

**Joint #5 is q = 1 only, and two mechanisms were proposed and retracted.**
C_q(N) fails to converge for **every** q ≥ 2, identically — spread 244% at q = 2
and 243% at q = 3, both flipping sign, against 0.6% at q = 1. So the observable
only exists in one dimension; it is not about 3-D being hard.

| retracted | why |
|---|---|
| "the branch sits in a degenerate multiplet" | dense diagnosis gives **multiplicity 1**; the "2 within 1e-9" was a duplicate listing |
| "the probe gap collapses as ~1/N in 3-D" | those 10⁻⁴ gaps came from `eigs` near a shift returning near-duplicates; **dense spectra give O(10⁻²), not falling** |

**Both were built on sparse `eigs` artifacts, and the same artifact class had
already been caught once** (the multiplicity-2 duplicate). I built an explanation
on it a second time. Dense diagonalisation is authoritative for spectral
questions; a windowed solver is not.

**What survives:** C₃(N) = −0.188, +1.458, −1.429 at sides 8/12/16, every point
resolved and S-extrapolated — a measured non-limit **without** a named mechanism.
The IR/dense split does not rescue it (C_IR moves 44% and carries ~2% of the
total), and additivity checks to ~1%, so that is not a method artifact either.

**Two near-misses refused on the way.** −1/16 sits 0.3% from the default-parameter
C — a parameter scan run *before* writing showed C ∝ β·s², so the agreement was an
artifact. And σ\* = 2λ / 4λ / ∝1/k were all withdrawn when ε-continuation showed
the spectral response is monotone.

**The three "confirmations" of the σ peak shared one fault.** FFT and Prony are
both phase-slope fits on a projection; the coherent block used max-overlap
selection at full ε. All three carry the same mode-purity / label-swap disease. I
called Prony an "independent estimator" — it was not. Only ε-continuation, which
follows eigenvectors stepwise from ε = 0, is free of it.

**Residual certified base-stable on q = 3:** inert at C_r = 0, B ∝ C_r² (ratios
3.83–3.88), drift ~10⁻⁷, working range C_r ≲ 0.05. The turnover above 0.1 is
fixed-T phase sampling of the residual oscillation, not saturation. ~20% box
dependence on \|B\| — do not quote a universal value.

## 6j. The cone — repaired, not retracted

§0a had flagged the cone as particle mechanics that fails to transfer to the
field model (null: 5.9554 vs 5.9555). **Wrong on two counts:** it tested spatial
circulation on a lattice when the cone is the **node's internal ℝ²** (per the D2
rung), and it ran at amplitude 1 when the platform runs at 10⁻³.

At the right object and regime: D2 turning points reproduced to three digits in
the linear limit (L conserved to 6×10⁻⁹), and r_min/r₀ = 0.5000 with L drift
3×10⁻⁴ at platform amplitude. C1's elementwise u∘u is anisotropic but that only
bites at \|u\| ≳ 0.1.

**Process note:** the fix came from reading the ladder's history (D2 first)
before testing at q = 3. The earlier failure skipped that step.

## 6k. First machine-verified result, and an error it exposed in C1

**Lean 4 / Prove2Me:** "The passivity-admissible couplings have dimension
n² = dim u(n)" — goal and milestones M1–M4 proved with no `sorry`, relying only
on Lean's three standard axioms (`#print axioms`). Statements published publicly
and launched for moderator review 2026-09-23. The proof: every admissible
coupling corresponds one-to-one with an ordinary n×n real matrix X, via its
symmetric and antisymmetric halves slotted into [[S, −K], [K, S]].

**ERROR IN C1 FORMAL PROOFS, THEOREM 6.1.** It writes the neighbour coupling
Σᵢ⟨vᵢ, W(vᵢ₊₁ − vᵢ₋₁)⟩ and states passivity forces W **skew**-symmetric. For
that expression the correct answer is **symmetric**: reindexing gives
P = Σᵢ vᵢ(W − Wᵀ)vᵢ₊₁. Verified numerically — symmetric W gives |P| ≤ 6×10⁻¹⁵,
skew gives 61.6. "Skew" is the rule for the **on-site** gyroscopic term K·v
(skew gives 2×10⁻¹⁵, symmetric gives 95). The theorem also maps skew matrices
to Hermitian ones, which is inconsistent on its own terms (skew + J-commuting is
anti-Hermitian).

**Consequence:** symmetric J-commuting matrices are **Hermitian** — iu(n), not
u(n). They match u(n) in dimension; the Lie algebra appears only after
multiplying by i. The mission title was changed from "admits exactly u(n)" to
"dimension n² = dim u(n)" *before* submission, which avoided publishing the
slip as a permanent statement. `model.py` and the mission are correct; the
document is not.

**Caveat for any future formalisation of the premise — CORRECTED.** The model's
coupling has **one matrix per link**, Wᵢvᵢ₊₁ − Wᵢ₋₁vᵢ₋₁, and for that form the
premise needs **N ≥ 3** for a sharper reason than first recorded:

- **N = 1** is vacuous: the power is zero for every W.
- **N = 2** is **false, not vacuous**: the power reduces to v₀·(D − Dᵀ)v₁ with
  D = W₀ − W₁, so two links sharing the **same non-symmetric** matrix do exactly
  zero work (verified: 0.000, against ~33 at N = 3, 4).

An earlier note here called N ≤ 2 vacuous; that is true only for a single shared
W. The general identity P = Σᵢ vᵢ·(Wᵢ − Wᵢᵀ)vᵢ₊₁ holds for every N (7×10⁻¹⁵).
Proved and submitted as Prove2Me mission 2 (goal and M1–M3 accepted, In review
2026-09-23), with the N = 2 counterexample also proved locally.

## 6l. Mission 3 — the pinned asymmetry, proved exactly

Prove2Me mission 3, "The propagation asymmetry does not depend on the on-site
stiffness": ω(q) − ω(−q) = 2βc·sin q for the upper-branch frequency, with a
milestone showing ω solves the dispersion relation and a corollary that any two
stiffnesses give the same asymmetry. Goal, M1, M2 and corollary proved with no
`sorry`; In review 2026-09-23. Scope stated in the mission: linear order and
uniform stiffness only — the nonlinear coefficient κ and non-uniform stiffness
both break the protection, as measured (§5b.2, §5b.4 of MODEL_SPEC).

**Running total: three missions, 13 theorems, all machine-verified.**

**Missions 4a and 4b (2026-09-24):** the passivity and asymmetry results extended
to a lattice with any number of axes, so they cover q = 3 directly; each reduces
to missions 2 and 3 at q = 1. 4b adds a protection that only exists above one
dimension — the asymmetry is independent of transverse wavenumbers — and its
formula was checked against the eigenvalues of an actual q = 3 lattice
(1.8×10⁻¹⁴) before drafting. The L = 2 counterexample is proved for every q ≥ 1.
All ten theorems accepted; both missions In review.

**Running total: five missions, 23 theorems, all machine-verified.**

**First publication (2026-09-24): mission 3 approved by Prove2Me moderator
Shuze Chen.**
The review cited the self-contained mathematics, M1 tying the formula to the
dispersion relation, M2 showing the radicand is even, and — explicitly — that the
description "states plainly that only the linear asymmetry on a uniform lattice is
covered." The scope limits written into the description were read as intended.

## 6m. J-compatibility — the one premise, classified CHOSEN

Hypothesis tested (Claude, this session): J-compatibility follows from requiring
the coupling to preserve the D2 isotropy, via minimality. **Half right.** The
J-rotations are the cone's phase, and "commutes with J" is exactly "conserves
total phase charge." **But it fails at three points:** minimality favours W = aI
(dimension 1), not n²; "preserve the ladder's symmetry" is ambiguous at n ≥ 2
(O(2n) or U(n) give dimension 1; only the single U(1) generated by J gives n²,
and that U(1) is selected by the gyroscopic term, not by D2); and D2's symmetry
is per node, whereas any nonzero coupling can only conserve a global phase.

Counts (numerically verified): symmetric W commuting with nothing / J / U(n) /
O(2n) have dimension 3, 1, 1, 1 at n = 1; 10, 4, 1, 1 at n = 2; 21, 9, 1, 1 at
n = 3.

**Status: CHOSEN** — the principle "the coupling conserves total phase charge."
Physically natural (it is the standard global-symmetry step of gauge theory) and
it extends conservativity from energy to phase, but it is new content.

**Open, testable:** a rotating-wave derivation. Averaging a general symmetric W
over one internal phase cycle leaves exactly (W − JWJ)/2. If the model's
dynamics suppress the n(n+1) extra directions by ~g/ω, J-compatibility is
derived approximately rather than chosen.

**Also corrected:** MODEL_SPEC §3 no longer says passivity alone gives u(n) or
u(4); the erratum now covers C1 Theorem 2.10.

**Then partly derived.** The rotating-wave route failed as framed (the ratio at
the model's k₀ = π/2 levels off at ~0.08, not ∝ g/ω), but the experiment found
the real mechanism: suppression happens when the **opposite-chirality channel is
closed** at the packet frequency. A dispersion calculation then predicted the
model's own parameters close it for k₀ < 1.45 and open it above; the prediction
was stated before rerunning at π/4 and 3π/4 and **held on both sides**. The gates'
k₀ = π/2 sits just past the threshold, which is why the first test saw no
suppression. Status upgraded to **derived at long wavelength, chosen at short**.

## 6n. q = 3 readout check — a guard that was the error

`run_until_exit()`, written this session specifically to prevent readout errors,
certifies exit by centroid position and **fires with 98% of the packet still in
the segment windows** at q = 3. Every q = 3 gate number in §4d was read that way.

Under a clearing readout (< 10⁻⁶ in every window, 320×12×12 full slab):
**the Abelian floor is exactly 0** for u(2) and u(3) — the "~1.4° kinematic" row
and gate 8's 0.39° were both the packet read mid-exit, not unequal group
velocities. **The centroid readout's apparently excellent agreement (u(3) split
error 0.06°) was a coincidence of mid-exit timing** — a near-miss agreement from
a faulty instrument, the pattern this ledger exists to catch. **The real split
error is 4.4–5.1°**, and a single segment misses by 2.70°: intrinsic to the
single-wavenumber prediction.

**Provenance gap:** the original q = 3 runs were interactive and unsaved, so
§4d's numbers cannot be reproduced; the rerun rebuilt the protocol and gets a
u(2) split of ~106° where §4d quoted ~30°. Old-vs-new comparisons are on
identical runs, so the conclusions hold; the archived absolute values do not.

**Lesson:** configurations for headline numbers must be saved as scripts, not
run interactively.

**Then closed.** Averaging the prediction over the packet's wavenumber content
(prediction stated before running) brings the q = 3 errors from 2.7–5.1° to
0.04–0.17°: u(2) split 105.89° measured vs 105.85° predicted, u(3) 60.54° vs
60.71°. **Retracted:** the claim that the ~4–5° was irreducible because "a pure
product carries no spatial information." The product works mode by mode; the
single-carrier approximation was the error. The q = 3 gauge sector is
quantitatively verified to ~0.1°, matching q = 1.

**Saved.** All scripts and results are committed to the public repository
(commit `0badafb`, `shape_zero_tests/`), with both model versions pinned by
SHA-256 and reproduction verified from the repository alone. Four printed-only
checks remain regenerable by rerunning their scripts; the q = 3 old-vs-new table
is now a saved file.

**Made permanent.** `q3_gate.py` (commit `ed32937`) passes at 0.10–0.18° with the
averaged prediction and fails at 2.4–5.1° with the single-wavenumber one, on the
same runs. **Its own no-wrap guard had a bug** — it ignored the packet's width on
the backward side and accepted a 240-site lattice that a direct run showed
failing (a stray wave re-entered the second segment at t ≈ 350). Fixed in commit
`e410714`. Another instance of the day's pattern — a guard is only trustworthy
once tested against a case known to be bad.

## 6o. κ = 0.0799 retracted; P-1 decay window not supported (2026-09-24)

**How it was found.** Investigating why `phi_gauge_decaymap.py` predicts the P-1
window moving down while its measured map moves up, the sign of the amplitude
term was checked against MODEL_SPEC §5. The script's second-order PT gives the
asymmetry correction as +0.0349 βA², i.e. κ = −0.0175; §5 quoted +0.0799.

**κ — what was wrong.** `pinned_asymmetry_reference.py` has gyro term
βc(v[n−1] − v[n+1]), opposite in sign to the platform scripts. In its lattice +k
is the **upper** root (B + d)/2; the code seeded "+" with (−B + d)/2 and "−" with
(B + d)/2 — each direction with the other's root. Its measured Δω is therefore
+0.1007 at β = 0.05, A = 0.30, and its own main block (d0 = −0.1 hard-coded) could
not have printed 0.0799; the quoted value is (|Δ/Δ₀| − 1)/A². The O(β) velocity
mismatch seeds a counter-rotating admixture read in the same FFT bin, whose
frequency pulling biased κ — the mechanism `phi_gauge_delta.py` Part 2 records
for v5.2. Signature in hindsight: the phase residual grew in proportion to β
(0.009 → 0.076 over the β-sweep); after the fix it is flat at 0.0054–0.0057.

| run, `pinned_asymmetry_reference.py` | old (swapped) | new (own root) |
|---|---|---|
| amplitude sweep, β = 0.05, A = 0.10 / 0.20 / 0.30 / 0.40 | +0.0773 / +0.0783 / +0.0799 / +0.0821 | −0.0176 / −0.0179 / −0.0187 / −0.0200 |
| β-sweep, A = 0.30, β = 0.02 / 0.05 / 0.10 / 0.20: \|Δ/Δ₀\| | 1.007227 / 1.007190 / 1.007262 / 1.007051 | 0.998339 / 0.998316 / 0.998390 / 0.998316 |
| β-sweep κ | +0.0798 ± 0.00089 | **−0.0184 ± 0.00033** |
| linear, A = 0.02, β = 0.05: \|Δω\| − 2cβ sin k | +3.1×10⁻⁶ | −7×10⁻⁷ |

Corrected κ = −0.0187 at β = 0.05, A = 0.30 was confirmed independently with
separate code. `model.py` gate 6 also took abs() of the drift, so it could not
report a negative κ; it is now signed. A search for the same swap (a direction
seeded with the other direction's root) found no other instance:
`phi_gauge_delta.py`, `phi_gauge_decaymap.py` and `s2_universality.py` use the
platform sign and seed with w_lin(direction·K), which is correct.
`phi_gauge_nonlinear.py` (and so `pinned_asymmetry_headline.py`,
`phi_gauge_closure.py`) seeds both directions at the β = 0 frequency — an O(β)
mismatch but not a swap; not changed here, and its superseded 0.0305 carries
that bias as well as the estimator bias. [Since fixed the same day — see
"`phi_gauge_nonlinear.py` re-seeded" below.] The two frozen `model.py` snapshots in
`shape_zero_tests/model_versions/` are left as recorded.

**Unaffected:** the linear pinned asymmetry Δω = −2cβ sin k and the Lean missions
(Prove2Me 2, 3, 4a, 4b), all linear-order. The β-collapse survives the fix.
**Unverified pending re-measurement:** the κ(w, side) investigation (§5, §6e),
measured with the swapped seeding; no script for it is in this repository.
[Since re-measured for w = 2 — see "κ(w = 2, side) re-measured" below; now
**measured and open**. Other widths remain unverified.]

**κ versus stiffness re-measured (same day).** `shape_zero_tests/joint3_kappa_stiffness.py`,
seeding each direction at its own root, gives κ = −0.0214, −0.0187, −0.0165 at
stiffness 0.90, 1.00, 1.10 (26% spread; retracted: 0.0959, 0.0799, 0.0677, 35%)
with the linear ratio 0.999992 / 0.999993 / 0.999994. Its unit-stiffness A = 0.30
ratio, 0.998316, equals the fixed reference β-sweep value. MODEL_SPEC §5b.2
Joint #3 and §5b.3 carry the corrected values; the qualitative result — the linear
pin is protected against stiffness, κ is not — stands.

**`phi_gauge_nonlinear.py` re-seeded (same day).** Each direction now starts at
its own linear root, `w_lin(direction·K, β)`. That script uses the platform gyro
sign, so +k is the **lower** root there — the reverse of the reference; a fixed
"+k upper" rule would have re-created the swap (MODEL_SPEC §5, sign-convention
table). Counter-propagating admixture at A = 0.001, β = 0.05: 1.25×10⁻² with the
β = 0 seed, 1.5×10⁻³ with the own root (the finite-record floor), 2.4×10⁻² with
the other direction's root. Hypothesis beforehand: linear results move < 10⁻⁵,
nonlinear ones may move. Both held.

| output | old (β = 0 seed) — RETRACTED | new (own root) |
|---|---|---|
| `phi_gauge_nonlinear.py`, β = 0 rows and β = 0.05, A = 0.001 row | 2.00878 / 2.10878 / −0.10000 | unchanged to 5 decimals |
| same, β = 0.05, Δω at A = 0.1 / 0.2 / 0.3 / 0.4 | −0.10003 / −0.10012 / −0.10028 / −0.10050 | −0.09998 / −0.09993 / −0.09983 / −0.09968 |
| same, ω(+k) / ω(−k) at A = 0.4 | 1.99237 / 2.09287 | 1.99278 / 2.09246 (centre 2.04262 unchanged) |
| residual/A², β = 0.05, A = 0.1–0.4 (PINNED_ASYMMETRY_TEST §3b) | −0.002978 … −0.003109 | +0.001757 … +0.001996 |
| coeff/β at β = 0.02 / 0.05 / 0.10 / 0.20 (§3b) | −0.06146 / −0.06091 / −0.06302 / −0.05891 | +0.03670 / +0.03707 / +0.03510 / +0.03721 |
| `phi_gauge_closure.py` fit | δ = −0.0598·β, max residual 1.28×10⁻⁴ (1.4%) | **δ = +0.0357·β**, 8.35×10⁻⁵ (1.5%) |
| closure δ(0.05) (its docstring) | −0.0031 (printed −0.00303) | +0.00179 |
| closure parity sum | −3.7×10⁻¹⁴ | −2.0×10⁻¹⁴ |
| `pinned_asymmetry_headline.py` drift/A², A = 0.1 / 0.2 / 0.3 / 0.4 | 0.0662 / 0.0471 / 0.0307 / 0.0311 | +0.0180 / −0.0029 / −0.0203 / −0.0205 |
| headline coefficient | 0.044 ± 0.015 (mean over all A) | **−0.0205** (fit over A ≥ 0.3) |

The closure fit now agrees with second-order PT (+0.0349·β) to 2%, and
κ = −δ/(2c β sin k) = −0.0179 agrees with the reference β-sweep (−0.0184 ± 0.00033).
The sign of the platform-script correction was an artifact of the seed.

`pinned_asymmetry_headline.py` also changed how it fits. Its T = 300 readout does
not resolve the drift at A ≤ 0.2: halving the record moves the reading by as much
as the drift (it prints this check per amplitude), and averaging those points gave
−0.006 ± 0.016 even after the re-seed. It now fits only A ≥ 0.3, where the drift
exceeds twice the record-length check. It had been printing 0.044, not the 0.0305
AUDIT.md attributed to it; that claim is corrected. `phi_gauge_decaymap.py` saves
its map next to itself instead of to `/home/claude`, and reproduces
`shape_zero_tests/p1/t0.txt`.

**κ(w = 2, side) re-measured (same day) — measured and open.** `shape_zero_tests/kappa_readout_test.py` (β = 0.05, A = 0.30, q = 3, transverse
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
  and steepens at the end (≈ side⁻³ from 24 to 32). **No κ(w) is published.**
- **Mechanism unresolved.** It is **not seeding**: new/old is −0.25 at every side
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

Status: κ(w = 2, side) is **measured and open**. The other transverse widths
(w = 3, the geometry table) were measured with `pinned_asymmetry_reference.py`'s
single-carrier seed and remain **unverified**.

**P-1 — what was found** (annotated in `shape_zero_predictions_v1.md`):

1. `beta_res()` in `phi_gauge_decaymap.py` holds w(0) + w(π) at the linear value
   and lets only the pump soften, giving −0.0991 A² exactly — an incomplete
   resonance condition. A Hill/Floquet analysis of the exact travelling wave puts
   the (0, π) band midpoint at ≈ 0.063 + 0.08 A², moving up.
2. The measured map is an artifact of the plain-cosine start (it seeds the q = 0
   and q = π product modes at O(A²); retention is independent of the noise to
   four digits and is 0.9998 from the exact wave) and of the fixed T = 400
   readout (the dip moves from β = 0.075 to 0.0675 with readout time) —
   MODEL_SPEC §4d.1 trap 6.
3. A clean start shows broad instability across β = 0.05–0.10 at A = 0.3–0.4
   (other pair channels grow as fast or faster), not a window.
4. The reverse-direction (−k) protection holds on every channel checked.

The investigation scripts and outputs are in `shape_zero_tests/p1/`, with a README
mapping each file to the finding it produced.

**Catch:** a sign comparison between a prediction script and the spec, made
while chasing a different discrepancy. Failure mode: #2 below in a new form —
the instrument was calibrated, the *preparation* was not. Also a case of
recurring mode #1's cousin: a quoted value that the script's own main block
could not have printed.

## 7. Recurring failure modes

Each has produced at least two errors in this programme.

1. **A number written in prose rather than emitted by a script.** → κ = 0.30.
2. **An uncalibrated instrument.** → κ = 0.0305; four estimator failures in one
   session, two caught only by external reruns.
3. **A control that cannot fail.** → the first u(3) control ran the identical
   spec twice.
4. **One sample read as the system.** → a single Hénon–Heiles initial condition
   reversed the conclusion; a single trajectory reported line spectra at every
   energy.
5. **A relative-only rank threshold.** → rank 67 against a true 22; spurious
   associator spans at dimensions 1 and 2.
6. **Lookup by whichever terms occurred to the searcher.** → u(3), the
   cone-at-D8, the Hénon–Heiles scope limit. This is what `CLAIM_INDEX.md` exists
   to fix.
7. **A positive result filed inside a retraction.** → the holonomy result and the
   intrinsic-torsion replacement.
