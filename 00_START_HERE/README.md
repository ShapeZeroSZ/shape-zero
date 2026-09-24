# START HERE

You have received a research program in progress. This document tells you what it
is, what is established, what is not, and in what order to read. Assume no prior
context; none is needed.

**Read this file completely before opening anything else.** Several documents in
here contain claims that later documents overturn, and the map below says which.

---

## 1. What this is, in one paragraph

Shape Zero is an attempt to derive physical structure from a small set of
selection principles rather than to fit it. The construction proceeds up a
"ladder" of rungs D1, D2, D4, D8 — real numbers, complex, quaternions,
octonions — where each rung is selected by principles rather than chosen. It
derives structure that already exists (gauge algebras, representation content,
field-equation form). **It has not derived any number that physics does not
already have**, and §5 of the input ledger explains why that limit is
structural rather than incidental.

## 2. The single most important thing to understand

**Every input to the construction is dimensionless.** Existence is a posit;
conservativity says a quantity vanishes; persistence is a topological condition
on an orbit; minimality and plurality are counts. Rescale every quantity in the
theory and all five read identically.

**Therefore no derived quantity can carry a unit.** This is a theorem, not a gap,
and it is why the program produces pure numbers — 1, 2, 3; 2 and 6; 42; 16 of
128; 14; 3/8 — and no masses, no couplings, no scales. Any claim that the
construction predicts a dimensionful quantity is wrong on its face.

## 2b. If you are here to BUILD

Read **`00_START_HERE/MODEL_SPEC.md`**. It is the buildable core assembled in
build order — state, on-site force, complex structure, coupling and gauge class,
lattice, predictions, harness, and an eight-step build sequence with a
done-when condition for each. Every piece is traced to its source and its
script, and §8 says what is deliberately excluded and why.

The other documents are the research record. The spec is the machine.

## 3. Reading order

1. **`03_current/INPUT_LEDGER.md`** — the ledger. One given, one purchase, what
   is forced, what would falsify it, and what has already been tested and failed.
   *Everything else is downstream of this.*
2. **`03_current/PINNED_ASYMMETRY_TEST.md`** — the one bench experiment. Zero
   free parameters once a linear measurement is made. This is the program's only
   currently falsifiable prediction.
3. **`03_current/HIERARCHIES.md`** — where scale separation can come from. One
   mechanism verified, two routes closed.
4. **`03_current/BUILD_CHECKLIST.md`** — the phased plan for a computational
   model, with definitions of done and failure modes.
5. **`03_current/C1S2.md`** and **`03_current/LADDER_TWO.md`** — deeper results,
   both carrying internal caveats. Read the caveats.
6. **`02_synthesis/`** — earlier syntheses. **Superseded in places** (see §6).
7. **`01_source/`** — the original program documents and spec.

## 3b. The index and the checker — use these before searching by hand

Every failure in this project has been a **lookup** failure. The corpus holds
**844 numeric claims across 29 documents and 88 scripts** — nobody holds that in
memory, and searching by whatever terms occur to you reports absence when the
thing is present.

    python3 00_START_HERE/build_index.py        # regenerate CLAIM_INDEX.{json,md}
    python3 00_START_HERE/check_consistency.py  # find contradictions

`CLAIM_INDEX.md` lists every numeric claim with file, line, section and status.
`check_consistency.py` reports: the same quantity given different values in
different documents; scripts referenced but absent; blocked scripts whose numbers
are quoted anyway; documents with no script attribution. It found a live
contradiction on its first run.

**Rule: regenerate the index after any change, and run the checker before
circulating anything.**

## 3c. The provenance ledger

`00_START_HERE/PROVENANCE.md` records **how each corrected claim moved and
why** — predecessor value, reason it was wrong, and which check caught it.
`CLAIM_INDEX.md` gives current values; PROVENANCE gives the trail.

It also separates claims that travelled under one label. **A-2 leg (ii)** is the
worst case: the original is *withdrawn* (reading-dependent), its **intrinsic
torsion replacement is verified** (147 components, 98 absorbable, 49 in four
classes 1+7+14+27), and a third result — **holonomies preserve the algebra**,
g₂ ⊂ span[L_a,L_b] at 6.3×10⁻¹⁶ — is also verified. Both positive results were
filed *inside* a retraction paragraph and read as part of the withdrawal.

**Rule: no claim is silently updated.** A value that changes gets an entry with
its predecessor, the reason, and the catch.

## 4. Directory map

| directory | what it holds |
|---|---|
| `01_source/` | the original program: open threads, the zero-ladder document, predictions, cover notes, spec |
| `02_synthesis/` | earlier consolidations, including the addendum delta targeting the spec |
| `03_current/` | the current documents — read these first |
| `04_scripts/rungs/` | the ladder's own source: `z1_d1_rung.py`, `z1_d4_rung.py`, `z1_d8_attempt.py`, `z1_hinge.py`, and others |
| `04_scripts/platform/` | `phi_gauge_*.py` — the lattice simulation where the bench prediction lives |
| `04_scripts/session/` | scripts produced in the most recent work, all sound |
| `04_scripts/superseded/` | scripts kept for the record, each banner-marked with why |
| `04_scripts/c1s/` | the C1S supplemental package (42 scripts) |

## 5. What is established

Each of these is reproducible from a script in this archive.

- **The given is irreducible.** Lemma 2.1 needs time, energy, space and
  boundedness; removing any one destroys periodicity. Three verified removals,
  three failures. (`session/d1_given_irreducible.py`)
- **…and it has a scope limit.** "Bounded conservative motion is periodic" is
  **false** in two dimensions. Hénon–Heiles at E = 0.155 gives spectral
  concentration 0.41 with 196 peaks at one section point and 0.998 with 7 peaks
  at another — regular and chaotic regions coexist at the same energy.
  (`session/d1_scope_limit.py`)
- **q = 3** is the unique spatial dimension admitted by three intersecting
  requirements: a centrifugal term must exist (q ≥ 2), it must beat the
  attraction (q ≤ 3), and gravity must have local degrees of freedom (q ≥ 3).
- **A hierarchy mechanism exists at D1.** Convergent denominators grow as φⁿ
  (measured growth ratio 1.618), and Greene residues fall **403,000×** from
  q = 2 to q = 21 — faster than exponential. Calibrated: K_c descends to 1.035
  against the known 0.971635. (`session/greene_production.py`, `greene_fixed.py`)
- **A zero-parameter bench prediction.** Δω = −2cβ sin(k)·[1 + κA²] with κ
  β-independent to four digits. Reference implementation:
  `session/pinned_asymmetry_reference.py`.
- **Colour and isospin are transverse halves** of one symmetric-space split of
  su(3), residuals 10⁻¹⁵.
- **All three gauge factors from one theorem.** Passivity gives u(n) for a node
  of n dimers, dimension n², verified n = 1…5. U(1), SU(2), SU(3) at node sizes
  1, 2, 3. **n = 3 is the role triad** — blocks of a Steiner triple system have
  cardinality three because there are three roles, and the node carries one
  block. u(3) is dynamically measured: ordering splitting 65.12° against 64.97°
  predicted, Abelian control 0.0169°.

## 6. What is superseded — READ BEFORE TRUSTING ANY DOCUMENT

| document | status |
|---|---|
| `02_synthesis/C1S_SYNTHESIS.md` **§14** | titled *"the ladder cannot be Lorentzian"* — **overturned** by `03_current/C1S2.md` §1 |
| `03_current/C1S2.md` **§§1–3** | base signature, dimension, Lovelock, Maxwell — all **conditional on a base existing**, which is *not* derived |
| `03_current/LADDER_TWO.md` **§4** | partly superseded, marked inline |
| `03_current/LADDER_TWO.md` **§6** | its irregular spectrum result is **withdrawn** — see §6a there |
| `04_scripts/superseded/*` | three scripts, each banner-marked at the top of the file |

## 7. Prior literature — attribution

Several results in this program were independently derived and are **already
published**. Novelty assessment failed three times before this was caught.

| result | prior work |
|---|---|
| Der = g₂ at every Cayley–Dickson level | Schafer 1954 |
| fourfold eigenspace multiplicity, [4,8,4] | Biss–Christensen–Dugger–Isaksen 2009 |
| sedenion zero-divisor condition | Moreno 1998 |
| zero-divisor locus as V₂(ℝ⁷); invariant metrics | Biss–Dugger–Isaksen; Reggiani |
| det L_x closed form | Koebisu, arXiv:2512.13002 |
| sin²θ_W = 3/8 | Georgi–Glashow 1974 (tree-level GUT value) |
| octonions → su(3), Standard Model algebra dimensions | Günaydin–Gürsey 1973; Dixon; Furey |

**Standing rule: search the literature before writing a section, not after.**

## 8. Process rules, learned the hard way

These are not style preferences. Each was established by a failure that reached a
document intended for other people.

1. **Calibrate every instrument against a known answer before quoting its
   output.** `04_scripts/session/harness.py` enforces this — it refuses to
   report from a routine that has not passed calibration, and currently **locks**
   the FFT-peak frequency estimator at 0.4985 relative error.
2. **No number in prose that was not emitted by a script.** A hand-written
   headline once carried a factor-of-ten error into a lab-facing document.
3. **Convergence tests the integrator, not the instrument.** A coefficient was
   confirmed converged in timestep while being wrong by 2.6× from estimator bias.
4. **One trajectory is not the system.** A single initial condition reversed the
   Hénon–Heiles conclusion until several section points were sampled.
5. **Internal consistency is not accuracy.** Two sweeps agreeing means little if
   they share one biased instrument.

## 9. What is open

- **Does a base exist?** C1S2 §§1–3 are conditional on it. This is the largest
  open question in the program.
- **Does passivity still force the right structure at D8?** Unverified, and it is
  Phase 2.1 of the build checklist — the item most likely to fail.
- **Is the reduced Laplacian construction new?** Searched, not located, weak
  evidence either way.
- **Three untested hierarchy mechanisms** — rarity weights, thermodynamic
  penalties, attractor spectra (`HIERARCHIES.md` §7).

## 10. What is closed, negatively

Recorded so nobody re-runs them.

- **Quantum dimensional transmutation** — no RG flow; the ladder is classical.
- **Nekhoroshev** — practically inaccessible. A GPU run of ~4,455 s per family
  produced secular rates nine orders of magnitude *below* the integrator's own
  energy drift.
- **Forcing a conformal factor on the D16 quotient** — four attempts, the last
  closed by theorem: L_aᵀL_a is exactly the identity on the invariant gradient
  span.
- **Forced spectra against measured spectra** — mismatch of functional *form*,
  not magnitude. Compact homogeneous spaces give quadratic towers; lepton mass
  ratios and Regge trajectories do not live in quadratic towers.

---

**If you read only one thing:** `03_current/INPUT_LEDGER.md`. If you act on only
one thing: `03_current/PINNED_ASYMMETRY_TEST.md`, which is a real experiment a
lab could run.
