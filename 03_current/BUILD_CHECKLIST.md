# Build Checklist — Computational Model of the Dynamics

**Goal.** One object with one state vector and one evolution law, whose
simulation reproduces the verified results and can be extended.

**The blocker, stated once.** The work currently contains three disconnected
kinds of thing:

| | what it is | has a lattice? | has propagating state? |
|---|---|---|---|
| **rungs** (`z1_d*`) | algebra studies | no | no |
| **platform** (`phi_gauge_*`) | a real simulation | yes | yes |
| **D1** | single oscillator | no | one degree of freedom |

Every testable prediction lives in the platform. Every structural result lives in
the rungs. **Nothing joins them.** The build is therefore: extend the platform
from D4 to D8 — a lattice of octonions carrying the selected flow. That
simulation has never been run and it is the first object that would deserve the
name "model of the dynamics."

Each item below has a **definition of done** and a **failure mode**. An item is
not done because code runs; it is done when its stated check passes.

---

## Phase 0 — Harness (do first, everything depends on it)

### 0.1 Calibration framework — **GREEN**
- [x] Every measurement routine has a paired known-answer case that runs with it
- [x] Frequency estimation calibrated against the Duffing shift 3ε/(8ω₀²)
- [x] Rank/nullity tests use an **absolute** floor, not relative-only
- [x] Any rate or width detector calibrated before its output is quoted

**Done when:** the harness refuses to report a number from an uncalibrated
routine. **MET** — `04_scripts/session/harness.py`:

    freq_phase   PASS            worst rel err 0.0028
    freq_fft     FAIL — LOCKED   worst rel err 0.4985
    rank_abs     PASS

`freq_fft` raises `UncalibratedError` on call. Calling any routine before
`calibrate_all()` also raises.

**Failure mode this prevents:** four instrument failures occurred in one session;
two were caught only by external reruns. A factor-of-ten error and a factor-of-2.6
error both reached a lab-facing document.

### 0.2 Headline discipline — **GREEN**
- [x] No number appears in prose that was not emitted by a script
- [x] Summary formulas are printed by the script that produces the table
- [x] Guards on quantities a reader would act on

**Done when:** every quoted figure has a script and a line number. **MET** —
document states κ = 0.0799; `pinned_asymmetry_reference.py` emits
**0.0798 ± 0.00089** from the β-sweep. The rebuilt `pinned_asymmetry_headline.py`
imports the harness and carries a factor-ten guard.

*Cost of getting here:* three successive values (0.30 → 0.0305 → 0.0799), two
external catches, and one estimator calibration. The first two were never
emitted by a calibrated script.

---

## Phase 1 — State definition

### 1.1 What lives at a node
- [ ] Decide: octonion in ℝ⁸, unit octonion on S⁷, or a pair (position, velocity)
- [ ] Reconcile with the rungs (S⁷, `z1_d8_flow.py`) and the platform
      (real displacement about φ, `phi_gauge_nonlinear.py`)

**The conflict:** the rung flow ψ̇ = ψa + bψ preserves |ψ|, so it lives on S⁷.
The platform's node is a displacement in a φ-well and does **not** preserve norm.
These are different state spaces and the coupling term is undefined until one is
chosen.

**Done when:** a single state type is written down and both the rung flow and the
platform coupling are expressible in it.

**Failure mode:** if no reconciliation exists, that is itself a result — it means
the rungs and the platform are about different systems, and the program should
say so rather than paper over it.

### 1.2 Norm handling
- [ ] If the state is on S⁷, decide how the φ-well nonlinearity acts
- [ ] If the state is in ℝ⁸, verify the selected D8 flow still applies

---

## Phase 2 — Coupling

### 2.1 The octonionic analogue of antisymmetric velocity coupling
- [ ] Write down the D8 inter-node term
- [ ] **Verify passivity still forces the right structure at D8**

**This can fail and it is the most informative item on the list.** At D4,
passivity forces the coupling matrix skew, hence u(2) — that is a derived result
and the whole synthetic-U(1) prediction rests on it. Whether the analogous
argument survives at D8 is **unverified**. It may force something larger, or
nothing.

**Done when:** the D8 coupling is written and the passivity argument is either
carried through or shown to fail.

### 2.2 Recover D4 as a limit
- [ ] Restricting the octonion to a quaternion subalgebra must reproduce the
      existing platform exactly
- [ ] Check the pinned asymmetry survives: Δω = −2cβ sin(k)

**Done when:** the D8 code with a quaternionic initial condition reproduces
`phi_gauge_nonlinear.py` to integrator precision.

**Failure mode this prevents:** a new simulation that does not contain the old
one as a special case is not an extension, it is a different model.

---

## Phase 3 — Base

### 3.1 Choose the lattice dimension deliberately
- [ ] 1D chain (matches the current platform and the bench test), **or**
- [ ] 3D lattice (matches `INPUT_LEDGER.md` §2b, where three intersecting
      requirements admit only q = 3)

**The tension, stated:** a lattice *is* a base. Building one chooses one. §2b
shows the ladder's own principles admit no q other than 3 for bound motion — so a
1D chain is outside what the construction permits, though it is fine as a lab
analogue and is where the falsifiable prediction lives.

**Done when:** the choice is made explicitly and its status recorded — analogue
platform, or claimed model of the dynamics. These are different claims.

### 3.2 Boundary conditions
- [ ] Periodic (current) or open; record which, since the D16 work showed
      boundary treatment silently determines spectra

---

## Phase 4 — Validation

### 4.1 Reproduce what is already verified
- [ ] D1 harmonic ratios 1, 2, 3 on the nonlinear potential
- [ ] Pinned asymmetry with κ = 0.0799 (phase-regression estimator, β-sweep)
- [ ] The A² law and the β-collapse
- [ ] D8 flow: one frequency parameter-free, two generic, all 14 terms

**Done when:** all four reproduce from the single unified code.

### 4.2 Produce something new
- [ ] Whatever the D8 lattice does that the D4 lattice does not

**Unknown in advance, and that is the point.** This is the first item on the list
whose outcome is not already determined.

---

## Not on this list, deliberately

**The AI connection.** There is a narrow real version — the ladder studies what
structures support stable, trackable, composable state, which is a question about
computation. But nothing here bears on intelligence specifically, and the
resemblance is the kind that feels explanatory while predicting nothing. Worth
pursuing separately, with its own falsifiable claims, not as an interpretation
layer on this work.

**SUPERSEDED — the residual IS in the model.** This entry predates
`MODEL_SPEC.md` §4b. What is closed is Ladder Two's *spectrum* route (Ω is
chosen; any Ω gives any spectrum). What is **not** closed and **is** in the
model: the residual scalar B with its exact potential V = −2 log(1 − 4B), the
rank-opening result (quotient rank 3 at B = 0, rank 4 for B > 0, null direction
= ∇B), and the odd/even selection rule. The residual composes under the D8
action — composition returns the moment one factor is octonionic, verified to
3.3×10⁻¹⁶ from either side and exactly 0.00e+00 for the isometry at D64, D256
and D1024. See MODEL_SPEC §4b and §7b.

**Hierarchy mechanisms beyond the arithmetic one.** §7 of `HIERARCHIES.md` lists
three untested candidates. They are cheap but they are not blockers.

---

## Order of work

**0 → 1 → 2 → 3 → 4.** Phase 0 first and without exception: every failure in the
session that produced this list was an instrument failure, and the two most
serious reached documents intended for other people.

Phase 2.1 is the item most likely to fail, and failing early is worth more than a
model built on an unverified coupling.
