# Shape Zero — Savepoint

A coherent stopping point. Every file has a stated status; nothing here is
presented as sound that isn't.

**Read `INPUT_LEDGER.md` first.** It carries the input count, the falsification
list, and the scope statement. Everything else is downstream of it.

---

## Documents

| file | what it is | status |
|---|---|---|
| **INPUT_LEDGER.md** | the ledger: one given, one purchase, what's forced, what would falsify it | **current** |
| **PINNED_ASYMMETRY_TEST.md** | the bench experiment — zero-parameter, calibrated, resolution requirements stated | **current**, two errata recorded |
| **HIERARCHIES.md** | where scale separation can come from; the arithmetic route verified | **current** |
| **BUILD_CHECKLIST.md** | phased plan for the computational model | **current**, nothing ticked |
| **C1S2.md** | base, gravity, EM, D16 map | **current with caveats** — §§1–3 conditional on a base existing |
| **LADDER_TWO.md** | the quasi-periodic branch | **current**, §4 partly superseded (marked inline), §6 result withdrawn |
| **C1S_SYNTHESIS.md** | the earlier synthesis | **§14 superseded** — titled "the ladder cannot be Lorentzian," overturned by C1S2 §1 |

---

## Scripts

| file | status |
|---|---|
| **greene_production.py** | **sound** — Greene residues + K_c, calibrates to 0.971635 |
| **greene_fixed.py** | **sound** — independent Jacobian, agrees to 5 figures |
| **estimator_calibration.py** | **sound** — settled the κ dispute against known Duffing shift |
| **d1_given_irreducible.py** | **sound** — removals verified before conclusions read |
| **d1_scope_limit.py** | **sound** — Hénon–Heiles scope limit on Lemma 2.1, section-sampled |
| **d16_spectrum_v2.py** | **sound as an instrument**; its Ω-weighted output carries no information (see LADDER_TWO §6a) |
| **base_signature_test.py** | sound; P3 flagged MISS on a threshold artefact, noted in-file |
| **base_lorentzian_forced.py** | sound |
| **pinned_asymmetry_headline.py** | **SUPERSEDED NUMBER** — emits 0.0305 via a biased estimator; correct value 0.082 |
| **residual_selection_rule.py** | **UNSOUND SOURCE, VALID RESULT** — its own lattice returned zeros; result obtained elsewhere |
| **nekhoroshev_form.py** | **INCONCLUSIVE BY CONSTRUCTION** — superseded by the Greene route |

All three problem scripts carry a banner at the top of the file.

---

## Audit findings

**One cross-document contradiction, resolved.** `pinned_asymmetry_headline.py`
computed 0.0305 while `PINNED_ASYMMETRY_TEST.md` states 0.082. The script is now
banner-marked. **The script itself should be rebuilt on the phase-regression
estimator** — that is the first Phase 0 item in the checklist.

**Two errata recorded in-document**, both in the lab-facing file:

1. **Factor of ten.** A headline gave 0.30 where the tables said 0.0305 — from
   dividing by −0.2 instead of the leading term −2. Caught by an external
   reviewer. An experimentalist would have reported a spurious failure.
2. **Factor of 2.6.** 0.0305 → 0.082, traced to the FFT-peak estimator being
   biased 1.8–2.0× at short records. Caught by an independent reimplementation,
   settled by calibration against the Duffing shift.

**Both errors were in the one document intended to reach a lab, and neither was
caught internally.** That is the strongest argument in this package for the
independent-implementation step being non-optional.

**Prior-literature findings, all confirmed after the fact:**

| result | prior work |
|---|---|
| Der = g₂ at every Cayley–Dickson level | Schafer 1954 |
| fourfold eigenspace multiplicity, [4,8,4] | Biss–Christensen–Dugger–Isaksen 2009 |
| zero-divisor condition | Moreno 1998 |
| locus as V₂(ℝ⁷), invariant metrics | Biss–Dugger–Isaksen; Reggiani |
| det L_x closed form | Koebisu, arXiv:2512.13002 — the 33-term polynomial is that paper's D₂² |

**One finding was lost and has been recovered.** A Hénon–Heiles result bounding
Lemma 2.1 was computed in an earlier session, reported in conversation, and never
written to a script or document. It is now `d1_scope_limit.py` and
`INPUT_LEDGER.md` §1a. Recovering it required correcting a single-initial-condition
version that reported the opposite conclusion — **one trajectory is not the
system**, the same error class as the instrument failures.

**Novelty assessment failed three times.** Web search was available throughout
and used once. The standing rule: **search before writing a section, not after.**

---

## What is verified, in one place

- **Bench prediction, zero-parameter:** Δω = −2cβ sin(k)·[1 + 0.082 A²], with
  β-independent normalised drift (the collapse protocol), valid A ≲ 0.9,
  resolution requirements stated per claim
- **Hierarchy mechanism:** convergent denominators q_n ~ φⁿ (growth 1.618 to four
  digits); Greene residues falling 1540× from q = 2 to 13, faster than
  exponential; two implementations agreeing to five figures; K_c → 0.971635
- **The given is irreducible** within the model — three verified removals, three
  failures of periodicity
- **q = 3** from three intersecting requirements; the invariant torus excluded
  because so(1) = 0
- **Odd-sector residual couplings forbidden outright**, r\* = 0, stable under a
  40× even background
- **Colour and isospin as transverse halves** of one symmetric-space split,
  residuals 10⁻¹⁵

## What is closed negatively

- Quantum dimensional transmutation — no RG flow, ladder is classical
- Nekhoroshev — practically inaccessible, signal 10⁹ below integrator error
- Forcing Ω on the D16 quotient — four attempts, last closed by theorem
- Forced spectra against measured spectra — mismatch of functional form, not
  magnitude

## What remains open

- Does a base exist? (§§1–3 of C1S2 are conditional on it)
- Is the reduced Laplacian construction new? (searched, not located, weak evidence)
- Phase 2.1 of the checklist: does passivity still force the right structure at D8?
- Three untested hierarchy mechanisms (HIERARCHIES §7)
