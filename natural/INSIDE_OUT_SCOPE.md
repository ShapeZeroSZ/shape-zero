# CP-G, inside-out: scoping report

*Scoping only; nothing is built. Hypotheses were committed first: `INSIDE_OUT_HYPOTHESES.md`
(53ccd07), IO1–IO8, all OPEN until the pilot or a run tests them. The candidate principle CP-G is
in the charter, under test and not adopted.*

## The method in one paragraph

Put an external **teacher gravity** on `main`'s lattice — linearised general relativity, PPN γ = 1.
Record how matter and waves behave under it on a stated **training set**. Let the lattice learn
**internal couplings**, drawn from a library its principles allow, that reproduce that behaviour
**with the teacher removed**. Extract the learned mechanism **sparsely**, so that it can be read.
Then judge it only on a **held-out set** never used in training. CP-G — gravity is a consequence,
not an ingredient — passes only if held-out behaviour comes out right from couplings the lattice
already permits.

## 1. The teacher (IO1)

- **Form.** A symmetric perturbation h_μν on the q = 3 lattice, sourced in harmonic gauge by
  □h̄_μν = −16πG T_μν, with T_μν summed over every sector.
- **Coupling.** −½h_μν T^μν. Matter sees the lapse through h₀₀, the shift through h₀ᵢ, and the
  spatial metric through hᵢⱼ. With the shared wave speed, γ = 1.
- **Not the lapse alone.** A lapse-only teacher is falsified (JOINT_PREDICTIONS, candidate 1): it
  would train a student to γ = 0.
- **Its status.** The teacher is a selection — spin-2, G, the shared speed — used only as training
  wheels. It never counts as evidence, and it is removed before any held-out test.
- **Its limit.** Being linear, it fixes γ but not PPN β. So it defines orbit closure only at
  Newtonian order, not the 1PN precession.

## 2. The library — what the principles allow (IO2)

Every term must be **local** (on-site or nearest-neighbour), **passive** (derivable from a
Hamiltonian, or gyroscopic in the antisymmetric/transposed form — EM_SCOPE C1, C7) and
**J-compatible** (invariant under the node phase rotation, so charge is conserved).

| family | form (schematic) | plays the role of | passivity |
|---|---|---|---|
| L1 | on-site stiffness K(1 + a₁ε̄ᵢ), ε̄ᵢ a local energy density | lapse, h₀₀ | Hamiltonian term |
| L2 | elastic coupling c(1 + a₂ε̄ᵢⱼ) on each bond | spatial metric, hᵢⱼ — changes the local wave speed | Hamiltonian term |
| L3 | velocity links with coefficient a₃·(local momentum density) | shift, h₀ᵢ | gyroscopic, transposed form |
| L4 | phase-invariant neighbour nonlinearities, a₄\|ψᵢ\|²\|ψᵢ − ψⱼ\|², … | contact interactions | Hamiltonian term |
| L5 | effective two-body couplings from integrating out the populated background (P0 non-isolation) at a stated fluctuation level | fluctuation-induced forces | Hamiltonian (effective) |

- **Energy dependence** is allowed in L1–L3, provided the total Hamiltonian stays conserved: the
  modulation must itself be part of a Hamiltonian, not an external drive.
- **L5** is where "populated levels" enter. Every mode of `main` carries a background (P0), and
  integrating it out gives induced interactions between lumps — Sakharov's route to induced
  gravity, in lattice form.

## 3. Training (IO4)

**Training set, stated in advance**, each item a counted selection:
- **T1** free fall of slow scalar packets at several stiffnesses K;
- **T2** free fall of both chirality branches at κ\*;
- **T3** gravitational redshift of a packet's frequency at rest, at several heights.

**Nothing about light, waves, two-body forces or orbits is in training.**

**Two ways to fit:**
- **(a) Local residual regression.** Record teacher trajectories. At each site and time, form the
  residual r = ü_teacher − F_main(u, v) and regress it onto the library features evaluated on the
  same fields. This needs no backpropagation and is cheap.
- **(b) Differentiable simulation.** Simulate the student with the library coefficients as
  parameters, and fit them by backpropagation through time against the teacher's trajectories.
  It is more faithful — it fits behaviour, not instantaneous forces — and needs GPU memory
  (checkpointing).

**A built-in diagnostic for IO3.** In (a), the residual depends on distant sources through h, but
the features are local. So the fraction of residual variance explained, as a function of distance
from the source, measures directly whether a local mechanism can carry the influence. It should
fall off within a few ξ if IO3 holds.

## 4. Sparse extraction

- **SINDy style:** sequential thresholded least squares over the library coefficients a₁ … a₅.
  Report the Pareto front (terms against residual) and pick the knee.
- **Symbolic regression, following Cranmer et al. (2020), "Discovering Symbolic Models from Deep
  Learning with Inductive Biases":**
  - a message-passing network with **local** messages (neighbour to site) is trained on the teacher
    data — locality is the inductive bias the principles demand;
  - then symbolic regression (PySR) is run on the learned messages;
  - the extracted expressions are checked for passivity and J-compatibility, and discarded if they
    fail.
- **Expected outcome (IO4).** With free-fall training only, the sparsest fit is L1 alone — a lapse,
  γ = 0. A γ = 1 student needs L2 in a fixed ratio to L1, which free fall does not determine.

## 5. Held-out tests — the only evidence

| test | measured how | teacher value | predicted for the student (hypotheses) |
|---|---|---|---|
| **H-γ** light bending | deflection of a wave packet passing a static lump, against impact parameter | γ = 1 (4GM/b) | γ = 0 if the student is L1 alone (IO4); undetermined otherwise |
| **H-pol** wave polarisation | angular pattern and detector response of radiation from an oscillating quadrupolar lump | tensor, + and × | scalar/vector only (IO5) |
| **H-r** two-body law at q = 3 | force between two static lumps against separation, fitted exponent | inverse-square | Yukawa, range ≲ ξ (IO3) |
| **H-orb** closing orbits | precession per orbit of a light lump bound to a heavy one | closed, at Newtonian order | not closed — follows H-r (IO6) |

**These tests are held out — never shown to the fit.** Each is run with the teacher removed.

## 6. Weinberg–Witten (IO5)

- **The theorem.** A Lorentz-invariant theory cannot contain a massless particle of spin > 1 that
  carries a Lorentz-covariant, conserved stress tensor. So such a graviton cannot be a composite of
  the theory's own fields in the naive sense.
- **The lattice is not Lorentz-invariant**, so the theorem does not apply at the lattice scale. But
  the shared wave speed makes the infrared effectively Lorentz-invariant, and there the theorem
  applies.
- **So an emergent graviton needs one of two escapes:**
  - (a) infrared Lorentz violation that survives — constrained by the shared-speed selection and by
    observation;
  - (b) an emergent linearised-diffeomorphism gauge symmetry, so the graviton's stress tensor is not
    a covariant local operator — as in general relativity itself, and in lattice models with
    emergent gauge structure.
- **Neither is in L1–L5.** The theorem therefore predicts the H-pol failure independently of IO3.
  An honest positive result would need the library extended with a gauge structure — an
  ingredient, or at least a selection, to be counted.

## 7. The accounting

- **Inputs:**
  - the teacher's form (1 selection);
  - T1–T3 (3 selections);
  - the library families actually used (up to 5 selections);
  - every non-zero learned coefficient (a fitted parameter);
  - any fluctuation level assumed in L5 (a parameter).
- **Evidence:** H-γ, H-pol, H-r and H-orb only, each pass counting once.
- **Predicted:** at least 4 selections plus at least 1 parameter, against 0 passes. **CP-G fails
  its own test at H-r first** (IO3) — unless the pilot finds a long-range mediator in `main`'s own
  fields, the one outcome that would be real evidence.
- **Under the charter:** only held-out passes enter the joint count, and nothing enters the ledger
  unless CP-G is adopted.

## 8. Compute, and whether to run it on Colab

**Step 0 — CPU pilot, recommended first; minutes.**
- **What it tests:** q = 1 and 2, no teacher. Place a source (a lump or a localised energy excess)
  and measure how far each library term L1–L5 carries a static influence.
- **It decides IO3.** If every influence decays within a few ξ, H-r and H-orb cannot pass, and the
  GPU program is not worth running as a test of CP-G.
- **Pilot expectation (IO3):** decay lengths ≲ 1 site for L1–L4, and ≲ ξ/2 for L5.

**Step 1 — only if the pilot finds a long-range carrier: a self-contained Colab GPU script.**
- **Teacher trajectories:** matter plus the 10 h_μν components on 48³–64³, float64. On earlier
  Colab runs a 64³ box took about 5 minutes per batch of four runs to T = 300 (the width scan). A
  training set of about 100 trajectories is therefore about 2–4 hours on a T4, or about 1 hour on
  an A100.
- **Local residual regression (a) and SINDy:** minutes, run in the same script on GPU or CPU.
- **Differentiable training (b):** backpropagation through about 10⁴ steps on 48³ needs gradient
  checkpointing; an A100 (40 GB) is preferable; about 1–3 hours.
- **Symbolic regression (PySR)** on the learned messages: CPU, tens of minutes.
- **Held-out tests:** four families of runs at 64³, about 1–2 hours on a T4.
- **Form:** one self-contained script, like the earlier `*_gpu.py` ones — PyTorch float64, NumPy
  fallback, validation gates first (the teacher reproduces γ = 1 and the Newtonian limit before any
  training), printed predictions, then training, extraction and held-out tests. **Yes, it should be
  a self-contained Colab script — but after the pilot, not before.**

**Recommendation: run the CPU pilot (Step 0) first.** It is cheap, it tests the hypothesis most
likely to decide CP-G, and its outcome says whether the GPU program is a test or a formality.
