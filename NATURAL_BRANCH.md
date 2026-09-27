# NATURAL_BRANCH — charter of the `natural-physics` branch

*Branched from `main` at `646a238` (2026-09-27). `main` is left untouched; anything here is
merged back only if it legitimately closes an item on `main`.*

## Purpose

To adapt the model **openly** toward known natural physics. `main` records what the model
derives from its own principles; this branch is allowed to add ingredients because nature has
them — but every addition is declared, costed and kept out of the evidence it was chosen to
produce.

## Rules

1. **Every adaptation is logged** in the ledger below: what was added, why, which principle or
   physical fact selected it, and how many parameters it adds.
2. **Every fitted or selected parameter is counted** — including values chosen to match a known
   fact and discrete choices that could have gone another way (a functional form, a sign, a
   dimension) when they were selected rather than forced.
3. **A fact used to select an adaptation cannot count as a prediction.** If Maxwell's equations
   are the reason a term was added, recovering Maxwell's equations is a consistency check, not
   evidence.
4. **The branch claims only what exceeds its inputs.** The conditions-versus-parameters count
   (below) is kept current with every adaptation. Conditions > parameters, all satisfied, is
   evidence; conditions ≤ parameters is fitting (MODEL_SPEC §0b; `03_current/INPUT_LEDGER.md` §3).
5. **Hypotheses before work.** For each target, the hypotheses are committed before the scoping
   or the build they test; results are reported against them, including the ones that fail.
6. **Merging back.** A change returns to `main` only if it closes a `main` item on `main`'s own
   terms (a derivation from `main`'s principles, a correction, a measurement); an adaptation that
   is justified only by agreement with nature stays here.

## Targets, in order

1. **Electromagnetism through dynamical u(1) links.**
2. **Gravity through couplings that respond to energy while conserving it.**
3. **Electromagnetism and gravity together** (added 2026-09-27). **From here on the joint count
   is the branch's primary measure**; the separate counts of targets 1 and 2 are kept as
   components. Why:
   - **Gravity couples to all energy, including electromagnetic field energy.** The equivalence
     principle is a statement that different kinds of energy fall alike, so it needs more than
     one kind of energy to be tested. Scoped alone, target 2 can test it only within matter.
   - **The two targets may share selections.** A single universal wave speed for photons,
     gravity and matter, or the q = 3 base, would be counted once, not twice. Counting them
     separately would overstate the cost.
   - **Gravity may supply stable lumps where the node well alone cannot.** Coleman's criterion
     fails for the adopted node form (`natural/EM_SCOPE.md` C5). Self-gravitating lumps exist
     (`natural/GRAVITY_SCOPE.md` §4), and they would give electromagnetism the static charges it
     otherwise lacks.
4. **The residual as the missing part** (added 2026-09-27). This is a Le Verrier / Pauli-style
   hypothesis: an unseen residual whose properties are fixed by what gravity needs and `main` lacks
   (both CP-G pilots), tested only by consequences nobody asked for.
   - **The frozen property list:** `natural/RESIDUAL_HYPOTHESES.md`.
   - **Its check against `main`'s principles and its consequences:** `natural/RESIDUAL_SCOPE.md`.
   - **Scoping only, not built, not adopted.**

## Candidate principles — under test, not adopted

- **CP-G (added 2026-09-27): gravity is a consequence, not an ingredient.** The lattice should
  produce gravitational behaviour from couplings its own principles allow — local, passive,
  J-compatible, including energy-dependent couplings and fluctuation-induced interactions from
  non-isolation's populated levels (`main` P0) — with no added gravitational field.
  - **Tested inside-out** (`natural/INSIDE_OUT_HYPOTHESES.md`, `natural/INSIDE_OUT_SCOPE.md`). An
    external teacher gravity is kept in place as training wheels, and the lattice learns internal
    couplings that reproduce its behaviour without it.
  - **The teacher is the repaired spin-2 form** — linearised general relativity, PPN γ = 1 — not
    the lapse alone, which `natural/JOINT_PREDICTIONS.md` shows is falsified.
  - **Accounting:** the training set counts as selections; only held-out results count as
    evidence.
  - **Not adopted.** Nothing enters the ledger until a held-out test passes.
  - **Status after the first pilot (2026-09-27): NOT SUPPORTED around the empty vacuum** — the vacuum the pilot tested. [**Amended 2026-09-27:** **untested around the populated background that `main`'s P0 (non-isolation) requires**, since there the vacuum is never empty. A second pilot tests it: `natural/CPG_PILOT2_PREDICTIONS.md`.] **Second pilot (2026-09-27): NOT SUPPORTED around the populated background either** (`natural/CPG_PILOT2_REPORT.md`): the conserved densities are gapless, but they carry no static long-range response and no power-law correlations, and the one long-range effect — a sink's depletion — follows the lump's radius profile, not its energy. The first pilot's record: The CPU pilot (`natural/cpg_pilot.py`,
    predictions 45c4c8f) finds no gapless mode in `main`: the smallest |ω| is 1.0864, and a static
    influence decays with ξ = 0.71 sites. So IO3 holds. The one inside route, a ring node's
    Goldstone, is recorded as a **fork, not adopted** (`natural/RING_FORK.md`). It is scalar and
    derivatively coupled, so it gives no inverse-square attraction between masses, and it would cost
    `main` its chirality branches and κ\*. The GPU inside-out program is not worth running as a test.

## The gravity arc — summary (closed 2026-09-27)

*This section is added, not a replacement: the entries above stand as written. Sources:
`natural/CPG_PILOT_PREDICTIONS.md`, `cpg_pilot_output.txt`, `RING_FORK.md`,
`CPG_PILOT2_PREDICTIONS.md`, `CPG_PILOT2_REPORT.md`, `RESIDUAL_HYPOTHESES.md` and
`RESIDUAL_SCOPE.md`.*

**1. CP-G (gravity as a consequence, not an ingredient): NOT SUPPORTED — on both backgrounds.**
- **Around the empty vacuum** (first pilot): every mode is gapped. The smallest |ω| is 1.0864, and a
  static influence is Yukawa with ξ = 0.71 sites. The one inside route with a massless mode, the
  ring node, is recorded as a fork, not adopted: its Goldstone is scalar and derivatively coupled.
- **Around the populated background that P0 requires** (second pilot; 17 hits, 12 misses against
  99131c0):
  - the conserved densities are gapless, but their transport is ballistic — a two-quasiparticle
    continuum |ω| ≤ v_max q, with no diffusion and no sound within reach;
  - there is **no long-range static response**: Yukawa with the populated operator Q + ⟨|ψ_B|⟩, and
    a range *shorter* than in the vacuum (0.7220 against 0.7233);
  - there are no power-law correlations;
  - the one long-range effect, a lump draining the background, has a **sink strength that follows
    the lump's radius profile |u|, not its energy**. a- and b-lumps whose energies differ 1.89×
    drain identically, and an A′ packet feels the depletion only as a scalar 1/r³ refraction.

**2. Target 4 (the residual as a Le Sage-type shadow medium): FALSIFIED as scoped.** The frozen
property list R1–R7 was committed at f20137d, and the falsifiers at 6badcc3, before evaluation.
- **Decisive:**
  - **F2:** under R2 the residual absorbs itself, with a mean free path c²h/(4πG) ≤ 11 km at the
    shielding bound h ≤ 10⁻²² m²/kg (Eckhardt 1990, lunar laser ranging). R4 needs ≳ 10¹³ m.
  - **F4:** drag below the orbit bound needs the shadow speed u ≥ 5×10²¹ c, but
    −3×10⁻¹⁵ < (c_g − c)/c < 7×10⁻¹⁶ (GW170817).
  - **F7:** under A′, absorption follows |u|, so composition dependence is O(1), against
    MICROSCOPE's η ≲ 10⁻¹⁵.
  - **F8:** light is deflected like a Newtonian corpuscle (γ-equivalent 0), against
    γ = 1 ± 2×10⁻⁵ (Cassini).
  - **F9:** the clock shift follows the ∝ 1/r² deficit, not the potential, against redshift ∝ ΔΦ
    to 10⁻⁴–10⁻⁵ (GP-A; Galileo 5 and 6).
  - **F11:** the shadow dynamics has no tensor polarisations, against the LIGO/Virgo polarisation
    tests and binary-pulsar damping.
- **Survivable only through counted escapes:**
  - **F5:** heating ≥ 1.3×10⁴³ W/kg against ≈ 8×10⁻¹² W/kg (Earth's heat budget). Needs R5's
    hidden channel.
  - **F6:** unreplenished depletion gives |Ġ/G| ≥ 1.3×10⁻¹¹ yr⁻¹ against ≲ 10⁻¹³ yr⁻¹ (LLR).
    Makes R7 compulsory.
  - **F10:** a uniform energy density ≥ 8.4×10³⁴ J/m³ ≈ 10⁴⁴ ρ_crit c². Needs the residual to be
    exempt from gravitating, which contradicts R2.

**3. The core reason.** A shadow needs net, sustained, one-way absorption. `main`'s lattice is
closed, conservative (passive) and fully populated (P0), and such a system re-emits what it absorbs:
in the steady state absorption equals emission everywhere (Kirchhoff), so it **cannot hold a steady
shadow**. Every gravity-like effect it does produce is transient (the pilot's sink), or scalar and
short-range.

**4. Conclusion: in this model, gravity must be an ingredient.** Its form is the **spin-2 form of
target 3** — linearised general relativity, γ = 1, tensor waves — added as a counted selection. Its
count, from the table below:
- 5 parameters plus a state;
- 7 discrete selections (the 6 of the joint scope, plus spin-2);
- ~9 conditions;
- **8 firm predictions**: the joint 6, frame dragging, and the κ-gradient force HELD at −0.23%. [**Corrected 2026-09-27:** **7 firm predictions** — the joint 6 plus frame dragging. The κ-gradient force (HELD, −0.23%) is a **verified consequence**, not an independent prediction: `main` recorded it (UNIVERSAL_RELATIONS row 12) as implied by row 5 plus ray kinematics, and did not raise its count.]
Light bending and tensor waves do not count: they motivated the selection. The joint theory is
still below 1:1 with selections counted. **Not built, not adopted** — the ledger stays empty.

**5. Open question — recorded as a question, not a claim.**
- Every route to emergent gravity tried here failed for the same missing thing: a **one-way sink,
  i.e. irreversibility**.
  - The empty vacuum has nothing to absorb.
  - The populated background absorbs only transiently.
  - A shadow medium needs a drive that a passive lattice forbids.
- **Irreversibility is also what the model lacks for an arrow of time.** Its dynamics is
  conservative and time-reversal symmetric, apart from the gyroscopic κ, which reverses under
  T together with the chirality.
- **The one widely respected derivation of emergent gravity**, Jacobson's (T. Jacobson,
  "Thermodynamics of Spacetime: The Einstein Equation of State", PRL 75, 1260 (1995)), obtains the
  Einstein equations from δQ = T dS applied to local causal horizons. That is a thermodynamic
  premise: entropy and heat flow across a one-way boundary.
- **Open:**
  - whether an irreversible ingredient — a coarse-graining, a horizon, or a genuine sink — would
    turn `main`'s gapless conserved densities into a source of long-range gravity;
  - whether the same ingredient would supply the arrow of time;
  - and what it would cost in the count.
- **Nothing here is scoped, tested or adopted.**

## The count

*Primary measure from target 3 on: the **joint** count of targets 1 and 2 (rows marked joint).*

**Baseline, inherited from `main`** (`UNIVERSAL_RELATIONS.md`, "The count — under the continuum
reading"): three genuine parameters **ĉ, κ̂, β̂**; about **six independent conditions**; strictly,
**four genuine predictions** neither used to choose a parameter nor excluded (the sin k departure
from linear Doppler, the coefficient ¼, its shape independence, ν → 0) — **three if ν → 0 is
kinematic**.

| after | parameters | conditions | strict predictions | note |
|---|---|---|---|---|
| baseline (`main` 646a238) | 3 | ~6 | 4 (3) | inherited |
| *projected:* target 1 as scoped, if built | 5 (+2 discrete selections) | ~7 (8) | 5 (6) | **not built, not adopted** — `natural/EM_SCOPE.md` §5 |
| *projected:* target 2 as scoped, if built | 4 (G free; c_g selected; β̂ → a state) + ≥ 4 discrete selections | ~7 | 5 | **not built, not adopted** — `natural/GRAVITY_SCOPE.md` §5; no improvement; equivalence principle violated in the κ sector unless a selection removes it |
| ***joint*** *(projected, primary measure):* targets 1 + 2 as scoped, if built | **5** + a state (+ **6 (+1)** discrete selections) | ~9 (10) | **6 (7)** | **not built, not adopted** — `natural/JOINT_SCOPE.md` (d); beats the separate sum (6 / 7 (+1) / 5); fitting level against the baseline with selections counted; the κ-sector equivalence-principle violation of target 2 alone is removed, the coupling being forced by gauge invariance |
| ***joint, after the candidate review*** (`natural/JOINT_PREDICTIONS.md`) | 5 + a state (+ 6 discrete selections) | ~9 | **6 firm** (+2 open: κ-gradient force, lump scaling) | **FALSIFIED as scoped:** two genuine predictions fail against nature — light deflection / Shapiro at PPN γ = 0 (half the observed bending), and scalar rather than tensor gravitational radiation |
| ***joint, repaired by a spin-2 selection*** *(projected)* | 5 + a state (+ **7** discrete selections) | ~9 | **7 firm** (frame dragging added; light bending and tensor waves do not count, having motivated the selection) (+2 open) [**Note 2026-09-27:** of the two open items, the κ-gradient force has since HELD, but as a verified consequence of row 5 plus ray kinematics (`main` row 12), so this row stays at **7 firm**; lump scaling remains open] | neutral repair: +1 selection, +1 prediction; still below 1:1 with selections counted |
| ***joint, after the κ-gradient test*** (`natural/KGRAD_HYPOTHESIS.md`) | 5 + a state (+ 6 discrete selections as scoped; 7 repaired) | ~9 | **7 firm** as scoped (κ-gradient force HELD, −0.23%) (+1 open: lump scaling); 8 repaired [**Corrected 2026-09-27:** **6 firm** as scoped, **7** repaired; the κ-gradient force is a verified consequence (`main` row 12: row 5 + ray kinematics), not counted as a prediction] | as scoped still **falsified** (γ = 0, scalar radiation). The κ-gradient force holds on `main`'s own lattice without any adaptation, so it belongs to `main`'s count; proposed as a merge candidate |
| ***target 4: the residual as the missing part*** (`natural/RESIDUAL_HYPOTHESES.md`, `natural/RESIDUAL_SCOPE.md`) | 5 + a state **+ 3 (Π, u, h)** (+ 6 discrete selections **+ 7 (R1–R7) + 2 escapes = 15**) | ~9 (+1: G, used as input — not a prediction) | **7 firm** as scoped, **+0** from target 4 [**Corrected 2026-09-27:** **6 firm** as scoped (κ-gradient force a verified consequence, not a prediction), +0 from target 4] | **FALSIFIED as scoped.** The list's own consequences fail at F2 (self-absorption: λ ≤ 11 km at the LLR shielding bound), F4 (drag needs gravity's speed ≥ 5×10²¹ c; GW170817), F7 (A′ absorbs by \|u\|, not energy), F8 (γ = 0), F9 (redshift ∝ 1/r²) and F11 (no tensor waves). F5, F6 and F10 each need a counted escape. Nine conflicts with `main`'s principles (A′ ×4, passivity ×3, P0 ×2) |
| ***correction (2026-09-27): the κ-gradient force as a verified consequence*** (`main` UNIVERSAL_RELATIONS row 12) | unchanged | unchanged | **joint as scoped: 6 firm** (not 7); **spin-2 repaired: 7 firm** (not 8); **target 4 as scoped: 6 firm** (not 7). The κ-gradient force (HELD, −0.23%) is listed as a **verified consequence** of row 5 plus ray kinematics, as on `main`, which did not raise its count | the figures in the rows above are kept as first written, with the correction noted beside each |

## Adaptation ledger

| # | date | what was added | why | selected by (principle or physical fact) | parameters added | status |
|---|---|---|---|---|---|---|
| — | — | *(none yet)* | | | | |
