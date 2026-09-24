# Shape Zero

A research archive for a lattice model in which gauge structure arises from a
single rule. Each node carries internal oscillators with a complex structure J;
neighbouring nodes are coupled by velocity-dependent link matrices. Requiring
the couplings to do no net work (passivity), and to respect J, fixes the space
of allowed couplings to the dimension of u(n) — giving u(1), u(2) and u(3) at
node sizes n = 1, 2, 3.

Independent research, developed with AI-directed computation. Not peer-reviewed.
Canonical location: <https://github.com/ShapeZeroSZ/shape-zero>
**The correction history is kept on purpose**: every retracted or superseded
claim is marked where it appears and traced in `00_START_HERE/PROVENANCE.md`.

---

## Status at a glance

### Machine-verified (Lean 4, published on [Prove2Me](https://prove2.me), captain ShapeZero)

| # | result | status |
|---|---|---|
| 1 | Real symmetric 2n×2n matrices commuting with J form a space of dimension **n² = dim u(n)**, for every n | proved, in review |
| 2 | On a ring of N ≥ 3 sites, the per-link neighbour coupling does no net work for all motions **iff every link matrix is symmetric** | proved, in review |
| 3 | The propagation asymmetry is exactly **2βc·sin q**, independent of the on-site stiffness | **proved, approved and published** |
| 4a | Result 2 on a periodic lattice with **any number of axes** | proved, in review |
| 4b | Result 3 in any dimension, also independent of **transverse** wavenumbers | proved, in review |
| 5 | A Steiner triple system with at least one point admitting a **role colouring has exactly 7 points** | proved, in review |

Each mission states what it does **not** prove. In particular: why couplings
commute with J is not derived by passivity (see "Measured" below), and mission 5
proves the point count, not that the system is the Fano plane.

### Measured (simulation, with the checks that could have failed them)

- **Non-commuting gauge ordering** matches its independent prediction to
  0.2–0.45° on a one-dimensional base and 0.10–0.18° on a three-dimensional base
  (the latter a standing gate, `q3_gate.py`, shown to fail against a wrong
  prediction). Commuting segments give exactly zero, as the algebra requires.
- **J-compatibility is enforced by the dynamics for long wavelengths.** A
  coupling that breaks J is suppressed completely at first order when the
  opposite-chirality channel is closed (k₀ < 0.46π at the model's parameters),
  and not suppressed above that. Predicted from the dispersion relation before
  the confirming runs.
- **The pinned asymmetry** is protected against uniform stiffness to 10⁻⁵; its
  nonlinear coefficient κ is not. [Re-measured with correct seeding: κ = −0.0214,
  −0.0187, −0.0165 at stiffness 0.90, 1.00, 1.10, linear ratio 0.99999 throughout
  (`joint3_kappa_stiffness.py`) — the claim stands. The earlier values, built on
  κ = 0.0799, are **RETRACTED**:
  the reference script seeded each direction with the other's root. Corrected
  κ = −0.0187 at unit stiffness (β = 0.05, A = 0.30; −0.0184 ± 0.00033 over the
  β-sweep), from `pinned_asymmetry_reference.py` and `model.py` gate 6. The
  linear pinning and the Lean missions are unaffected. See `PROVENANCE.md` §6o.]
- **The node's internal cone structure** (centrifugal barrier) holds at the
  model's operating amplitude to 3×10⁻⁴.

### Open

- **No scale is derived.** The model is dimensionless; ℏ, G and Λ are not
  predicted, and a length or mass unit is an input.
- J-compatibility for short wavelengths remains a chosen premise.
- A per-mode coefficient in three dimensions does not converge with box size,
  without a named mechanism.
- Formal proof that every seven-point Steiner triple system is the Fano plane.

### Retracted — see `PROVENANCE.md`

Among others: a claimed derivation of ℏ; a claimed irreducible 4–5° error in
three dimensions (it was a single-wavenumber approximation); two proposed
mechanisms for the three-dimensional non-convergence; a local gradient law.

---

## Where to start

| document | what it is |
|---|---|
| `00_START_HERE/MODEL_SPEC.md` | **the current specification** — read this first |
| `00_START_HERE/PROVENANCE.md` | every correction, with what caught it |
| `01_source/` | original source documents (C1 Formal Proofs, v5.3 spec) |
| `01_source/proofs/ERRATUM_Theorem_6.1.md` | **known errors in the C1 Formal Proofs** (§3 and §6), with corrections |
| `02_synthesis/C1_ERRATA.md` | the earlier errata list for C1 |
| `04_scripts/session/model.py` | the assembled model; `python3 model.py` runs its gates |

The C1 Formal Proofs contain errors that were found by formalizing its results:
most importantly, Theorem 6.1 states that passivity forces **skew**-symmetric
couplings where the correct condition is **symmetric**. Read the erratum
alongside it.

## Tests

The scripts behind the measured results above, with their saved outputs and the
exact model versions they ran against (pinned by hash), are in
`shape_zero_tests/`. Its README maps every reported number to the script and
settings that produced it. The three-dimensional gauge gate is
`shape_zero_tests/q3_gate.py`.

---

## License

- **Documents** (`.md`, `.pdf`, `.docx`, `.txt`): Creative Commons Attribution
  4.0 International (CC BY 4.0) — see `LICENSE-docs`.
- **Code** (`.py` and other source files): MIT — see `LICENSE`.

You may use, share and build on any of it, for any purpose, provided you credit
the source. Copyright © 2026 Shape Zero LLC.
