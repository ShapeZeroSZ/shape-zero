# The D8 Rung — Synthesis

Standalone account of what the octonionic rung is, as established in the C1S
package. Readable without the scripts. Every number cited names the script that
produces it.

---

## 1. What the rung is

The arena advances with the ladder — D1 scalar, D2 ℝ², D4 ℝ⁴, D8 ℝ⁸, verified
directly in the C1 rung scripts. So the D8 base is 𝕆, and Im(𝕆) = ℝ⁷ after the
deletion Proposition 2.8 already uses at D3. **G₂ acts on the base as well as
the fibre.**

The state lives on **S⁷**, the unit octonions, chosen over flat 𝕆 on the
criterion "which one can it actually close": every measured feature
distinguishing D8 from D4 lives on the loop and none of it exists in the vector
space.

| ingredient | status | script |
|---|---|---|
| target S⁷ = Spin(7)/G₂ | verified, not assumed | `sigma_topology.py` |
| target metric | **forced** up to one scale (isotropy irreducible → Schur) | `coset_audit.py` |
| tangent labelling by Im(𝕆) | **canonical**, `v ↦ L_v`, equivariant to 10⁻¹⁵ | `coset_audit.py` |
| π₁ = π₂ = 0 | no solitons, no winding sectors | `sigma_topology.py` |

---

## 2. Three things D8 has that D4 cannot

From `z1_d8_dynamics.py`:

**The loop–group gap.** In ℍ the unit loop is 3-dimensional and generates 3 —
gap zero. In 𝕆 a 7-dimensional loop generates **28 = so(8)**, gap 21. Under G₂
that decomposes as g₂ ⊕ 7 ⊕ 7 (Casimir 4.0×14 and 2.0×14, the latter two
isomorphic copies), with the loop directions disjoint from g₂ at 3.8×10⁻¹⁶.
**Non-associativity manufactures the automorphism group out of the loop.**

**Associative sectors.** Artin: associator 9.1×10⁻¹³ inside any 2-generated
subalgebra, 57.5 generic. The seven Fano lines are regions where the dynamics is
D4-like.

**Bracketing as a degree of freedom.** Product spread over parenthesisations is
2.2×10⁻¹⁶ within a line and grows outside it (0.77, 0.90, 1.11 for n = 3, 4, 5).
Identically zero at D4 for every sequence.

Separately: the associator is **0 or exactly 4**, no other value, over all basis
triples.

---

## 3. The term census

Cubic four-derivative invariants on the corrected base. Index parity —
7 = 3 + 2 + 2 — means **every one carries exactly one φ**, so there are no
non-octonionic terms at this order.

| | count | script |
|---|---|---|
| invariants | **22** | `d8_census_clean.py` |
| divergences (null) | **8** | |
| **entering the field equations** | **14** | |

Stable at tolerances 10⁻⁸ through 10⁻¹², confirmed by explicit projection (rank
14 outside the divergence span, residual at 86% of the leading scale).

The cubic **three**-derivative term — available only because G₂ acts on the base,
and identically zero on any 2-dimensional base — exists, is a genuinely new
invariant (rank 4 against 3 for the traces), and is a **null Lagrangian**: EL
vanishes at 3.4×10⁻¹⁴, because the total antisymmetry that permits it forces c
against a symmetric second derivative. `d8_cubic_term.py`

**Support dependence** (`d8_crossing_trigger.py`): 2 dimensions → 0 terms, 3 → 9,
5 → 22. The 14 do not form a clean irreducible class — ten of them require the
channels of J to mix (`d8_channels.py`). 𝕆 and S⁷ give identical counts.

---

## 4. The law, and how it is selected

Built from conservativity and minimality alone, with no reference to sectors:
ψ̇ = ψ·a. `z1_d8_flow.py`

| law | motion's imaginary support | terms | frequencies |
|---|---|---|---|
| ψ̇ = ψa + aψ (anticommutator) | 2 | 0 | 1 |
| ψ̇ = ψa − aψ (commutator) | 3 | 6 | 1 |
| ψ̇ = ψa + bψ (generic) | 5 | **14** | **2** |

Closed forms verified to 4×10⁻¹⁶: the anticommutator is 2ψ₀a − 2⟨ψ⃗,a⟩, lying in
span{1,a}; the commutator is 2ψ⃗×a, purely imaginary. **Neither parameter-free
law is a D8 flow** — one is D2 motion re-embedded, the other D4.
`z1_d8_minimality.py`

**Plurality selects the generic flow.** Each parameter-free law is a single
periodic angle — one dynamical degree of freedom, 6/6 trials. The generic flow
has two, 6/6 at adequate resolution. Plurality states there is more than one, so
it excludes both; and minimality's own wording admits what plurality forces:
*"unless forced by a prior principle or by the purchased plurality assumption."*
`z1_d8_plurality.py`

**Confinement was not imposed.** It emerged — nothing in the construction
mentions Fano lines, Artin, or associativity.

---

## 5. What the principle audits found

| principle | applications across D1–D8 |
|---|---|
| conservativity | D1 first integral, D2 Noether reduction, D4 passivity — **repeatedly** |
| persistence | **D1 only** |
| minimality | **never**, until D8 |

Every elimination in the formal proofs document is performed by conservativity,
Noether, Hurwitz/Frobenius, or passivity. The selection principle *minimality* is
stated in Section 1 and never cited in a lemma, theorem or proof; the word
recurs only as "role minimality," a distinct postulate.

**This matters for Section 1's own justification.** The three principles are
called "earned, not postulated" because "each selects structure inside the arena
before they are used as generative rules." That criterion cannot support
minimality, which selects nothing until D8. Its earned status derives from D8,
not from the arena work.

Recommended: a footnote saying so. Not removal — the D8 result needs minimality,
and it earned its place there by finally biting.

---

## 6. Ledger

**Forced:** the arena; the target and its metric up to scale; the canonical
tangent labelling; the term census; the loop–group gap; the associative sectors;
the associator quantisation; the law.

**Permitted but not forced:** the a–b angle. Admitted by minimality's exemption
clause rather than forbidden. Unobservable in every measured outcome — the sweep
gives 5/14 regardless of its value.

---

## 7. Where the thread stops, and why

The selected law is **linear**: M = R_a + L_b is antisymmetric, so the flow is
exp(Mt) in SO(8) — exactly integrable, constant frequencies.

That closes the last lead. The a–b angle controls the frequency ratio, which
sweeps monotonically over [1.04, 23.9] as the angle runs 5° to 150°. The golden
ratio is attainable near 76° — **and so is every other value in the range**.
Attainability is not selection. KAM-style persistence would select the most
irrational ratio, but KAM needs a perturbation, and an exactly integrable flow on
a compact target offers none.

The only available source of nonlinearity is the 14 terms themselves, which are
field-theory terms requiring ψ(x,t). **D8 has no field theory with time.** Every
remaining step needs structure the program does not yet contain.

---

## 8. Correction history

Recorded because the rate bears on how much weight the results carry, and
because two of the largest corrections reshaped the thread rather than adjusting
it.

- **The base.** Five sessions were computed on a 2-dimensional base taken from
  `phi_gauge_wilson.py`, a D2-level script. The rung scripts show the arena
  advancing 1→2→4→8. Everything about instability, the coupling bound
  λ < 2√(μc₄), and the "no window" conclusion belongs to a theory with a 2D
  base, which the program does not have.
- **The step function.** "No intermediate regime" was an artifact of scanning
  only unions of lines, which skip 4 points. Four points containing one line
  give 13.
- **The crossing threshold**, corrected three times: associativity, then line
  containment, then support dimension. Each earlier version was a structured
  sample read as general.
- **Coverage logic.** The (3,4) census first tested containment rather than
  coverage and printed the opposite conclusion.
- Plus: an aliasing amplitude, a pure-cubic scaling prediction, log-vs-log², an
  apex-artefact diagnosis, a linear quadrature, a Wick normalisation and its
  diagnosis, a rank threshold with no absolute floor, and a span-versus-generated-
  algebra conflation.

The recurring failure mode is a structured sample read as general. It produced
the two largest corrections and three of the smaller ones.

A second recurring fault: a **numerical rank threshold that was relative only**.
When a matrix is identically zero up to floating-point noise, sv[0] is itself
~10⁻¹⁴, everything clears tol·sv[0], and the reported rank is spurious. It gave
67 against a full-space value of 22 on a 2-dimensional support, and 0 where the
true answer was 1 on the D4 structure sphere. Fixed at source in
`d8_channels.py` with an absolute floor alongside the relative one. **All
headline counts reproduce unchanged** — 22/8/14, 9/3/6, 20/7/13 and the
structural zeros — because the cases that mattered were either genuinely
non-zero or exactly zero from structural cancellation rather than noise.

---

## 9. B-4: what the quantisation route actually gives

`z1_d4_structure_sphere.py`. Geometric quantisation's integrality condition
needs a closed 2-cycle, and Theorem 2.9 supplies one — the structure sphere,
SO(4)/U(2) = S², sampled here in both duality components (self-dual and
anti-self-dual, each spanning 3 ambient dimensions). so(4) = su(2) ⊕ su(2)
verified, the two ideals commuting at 0.000e+00.

The isotropy action leaves **exactly one** invariant antisymmetric form
(ε, invariant to 2.2×10⁻¹⁶) and **exactly one** invariant symmetric form, with
the traceless symmetric form failing invariance at 2.0 as a control. So metric
and symplectic structure are **both forced up to scale and compatible** — the
sphere is Kähler with no independent choice. Same Schur argument that fixed the
D8 target metric.

**What integrality then does is discretise that single scale**: flux must be an
integer multiple of 2πℏ. It does **not** produce a value for ℏ — ℏ is the unit,
not the output. The honest ledger entry: ℏ remains underived, and the route
removes the *continuum* of symplectic normalisations, leaving a discrete tower.
Smaller than "ℏ from geometry," and it holds.

## 10. E1: hidden agents are not what the residual detects

`e1_hidden_agent.py`. The queued E1 extension is "two coupled agents, objective
inference via integrability residual." Tested with a jointly variational system —
z-dot = −grad E at every coupling, no curl anywhere — where agent 1 sees only
(z₁, ż₁) and asks whether ż₁ is a function of z₁.

| | 3e-2 | 1e-2 | 3e-3 | 1e-3 | 3e-4 |
|---|---|---|---|---|---|
| g = 0 | 0.0442 | 0.0165 | 0.0056 | 0.0021 | 0.0008 |
| g = 0.35 | 0.0483 | 0.0196 | 0.0074 | 0.0032 | 0.0013 |
| ratio | 1.09 | 1.19 | 1.33 | 1.52 | **1.71** |

202,000 pairs at the smallest ε, control (the joint state) converging correctly.
The coupling excess is **real and monotone** — but every curve goes to zero.
**No plateau.**

**Mechanism.** A contracting flow collapses its reachable set onto a graph over
z₁: all trajectories descend to the same minimum, so the visited (z₁, z₂) lie on
a surface where z₂ is a function of z₁. Coupling slows the approach — that is the
finite-ε excess — but does not prevent it. The hidden agent is **asymptotically
invisible**, and improving the instrument destroys the evidence.

**What this separates.** A curl term is a property of the vector field, present
at every scale; the integrability residual finds it. A hidden variable in a
contracting flow only mimics that at coarse resolution. **The extension treats
these as one detection problem and they are not** — a residual-based detector
finds non-variational dynamics and not hidden agents.

Four earlier attempts failed and shaped the design: binning (measured bin-width
variation; caught by the control scoring 0.27 on a system that is a vector field
by construction), a single-trajectory limit (a gradient flow never revisits a
state), an ensemble with all pairs (swamped by within-trajectory pairs, which
carry no multivaluedness), and adding a curl term to stop contraction (it did
not).

## 11. B-7 closes in the negative

`b7_boundedness_earned.py`. `b7_boundedness.py` showed the E_G-shaped envelope
follows from **boundedness of the bond coupling alone** — three unrelated bounded
families gave the same plateau at (count × per-element ceiling), independent of
functional form. That left one proposition: is boundedness earned?

**It is not, and persistence argues against it.** A saturating bond has a ceiling,
and a ceiling is an escape energy: above it nothing holds the configuration.
An unbounded confining bond has no escape energy at all.

| energy | max\|x\| at T = 200 / 400 / 800 | verdict |
|---|---|---|
| 0.99 | 3.035 / 3.035 / 3.035 | bounded |
| 1.01 | 30.7 / 59.0 / 115.5 | **escapes** |
| 10.0 | 848.6 / 1697.1 / 3394.2 | **escapes** |

Ratio at 4× the integration time: 1.000 below the ceiling, → 4.000 above it —
free streaming. The threshold is exactly s². Control: the unbounded bond confines
at every energy tested including E = 500.

So persistence, the one principle that could have supplied boundedness, **favours
the unbounded coupling**. An earlier sketch had this backwards, reading D1's
bounded-motion requirement as an argument for a bounded coupling when a bounded
coupling is precisely what permits unbounded motion.

**Consequence for the Penrose contact.** The E_G correspondence — including the
derivation of the linear-in-N counting rule that Orch-OR must assume — rests
entirely on a saturating coupling. It is not refuted but demoted: it holds for
structures with bounded couplings, and the ladder has no reason to be one. A
reason internal to the program rather than to the physics.

Recorded miss: escape was first tested as |x| > 1e6, which free-streaming
trajectories never reach in finite T. Detection criterion, not prediction.

## 12. External document note

An independently produced write-up built a cosmological framework on the sector split. Its
one correct ingredient — the associator being 0 or exactly 4 — is independently
confirmed here. Its central input is wrong: it uses 2 sector-local and 12
cross-sector terms against measured values of 6 and 8. Dark matter, the Hubble
ratio and the PMNS loop factor all carry N_cross = 12; with 8 the dark-matter
figure misses Planck by 33%. Its verification script contradicts its own text
(computes λ = 0.227025 where the text claims 0.22429), and its associator test
assigns values from a lookup rather than computing them, with a sorting bug
misclassifying 3 of 7 lines.

The failure mode is worth naming: it reads a **classification** result as a
**physical inventory**. The 14 are candidate terms in an action — no Lagrangian,
no field equations, no time direction. Treating them as degrees of freedom
carrying energy, mass and cosmological abundance is a category leap taken
silently in one sentence, and everything downstream inherits it.
