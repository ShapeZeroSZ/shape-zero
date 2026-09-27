# Target 3 — the joint theory's candidate genuine predictions

*Report only; nothing built, nothing run for this file. It draws on `EM_SCOPE.md`,
`GRAVITY_SCOPE.md` and `JOINT_SCOPE.md` and their checks. A candidate is **genuine** if it is a
consequence of target 3 that was not used to select any ingredient, parameter or coupling
(charter rule 3). Ranked by how cleanly each could fail.*

## Summary

- **The joint theory as scoped is already falsified by known physics, twice.**
  1. **Light deflection.** Its gravity is a lapse only, so light falls exactly like matter (J1).
     That is PPN γ = 0: light is bent by half the observed amount and the Shapiro delay is halved,
     where Cassini measures γ − 1 = (2.1 ± 2.3)×10⁻⁵.
  2. **Gravitational radiation.** Its gravity radiates scalar (and, with a dynamical shift, vector)
     modes and no tensor modes. That contradicts the observed tensor polarisation of gravitational
     waves.
  - Both are genuine — neither was used to select anything — and both fail.
- **The repair is to make gravity Lorentz-covariant spin-2.** That adds the spatial metric, and
  the same selection is what ties the shift to G for frame dragging. But it costs one more
  selection, and the facts that now motivate it (light bending, tensor waves) cannot then be
  counted.
- **Surviving candidates that could fail cleanly:**
  - the **κ-gradient force**: a charge-dependent acceleration, parameter-free and testable in a
    cheap simulation on `main`'s own lattice;
  - the **self-gravitating lump scaling** R ∝ G⁻²N⁻¹;
  - the **lattice photon dispersion shape**, sharing one lattice with matter's sin k.
- **Charged matter:** the softening node nonlinearity would bind it, at +1 parameter and one
  selection, and it conflicts with `main`'s P9 premise.

## Ranked candidates

### 1. Light deflection and the Shapiro delay: PPN γ = 0 — **FAILS against nature**

- **What:** light crossing a static lapse potential bends by 2GM/(bc²), the Newtonian value,
  not the observed 4GM/(bc²); the Shapiro delay is half.
  - J1 measured light's transverse acceleration equal to matter's: 0.976 against 0.992, the gap
    a finite-width effect that closes with packet width.
  - With no spatial-metric perturbation, that is γ = 0.
- **New or known:** it contradicts known physics — Eddington 1919, VLBI, and Cassini's
  γ = 1 to 2×10⁻⁵.
- **Depends on:** the lapse-only structure (T⁰⁰ coupling, no γᵢⱼ) and the shared wave speed. Both
  are counted selections; neither was selected with light bending in mind.
- **Test:** already failed. A lattice check would be a ray-deflection run in a 1/r lapse field.
- **Cleanliness:** the cleanest possible — the measurement exists.

### 2. Gravitational radiation: scalar (and vector), no tensor modes — **FAILS against nature**

- **What:**
  - a dynamical lapse radiates **scalar** (breathing) waves; a dynamical shift adds **vector**
    modes;
  - nothing radiates the + and × tensor modes;
  - a scalar theory also predicts monopole and dipole emission from binaries, and a quadrupole
    rate differing from general relativity's.
- **New or known:** it contradicts known physics — LIGO–Virgo polarisation tests favour pure tensor
  over pure scalar (GW170814, three detectors), and binary-pulsar decay matches general
  relativity's quadrupole formula to about 0.1%.
- **Depends on:** the massless dynamical Φ (a counted selection, made for long range) and the
  dynamical shift. Its radiative content was not used to select anything.
- **Test:** already failed. On the lattice, the polarisation content of radiation from an
  oscillating lump.
- **Cleanliness:** very clean.

### 3. The κ-gradient force — a charge-dependent acceleration (genuine; **HELD** — `KGRAD_HYPOTHESIS.md`: a_a − a_b within 0.23% of c·κ′/ω_rot on `main`'s lattice)

- **What:** a static, non-uniform κ(x) is exactly a static potential A₀(x) = κ(x)/(2e), with the
  local shift K′ = K + κ²/4. It follows the same algebra as EM_SCOPE §4, with the cross term
  −eA₀(x)u̇·𝕁u giving a local gyroscopic coefficient 2eA₀(x).
- **Prediction** (eikonal, per branch, ω_s(k, x) = ω_rot(k, x) ∓ κ(x)/2): a-branch and b-branch
  packets at rest accelerate
  - **oppositely:** a_a − a_b = c·κ′/ω_rot, with ω_rot = √(K + κ²/4) = 1.5723 at κ\* and
    c/ω_rot = 0.636;
  - **plus a common part** from ∂ω_rot/∂x, −c·κκ′/(4ω_rot²) per branch.
  - The difference is parameter-free given κ, K and c.
- **New or known:** the form (opposite charges accelerating oppositely in an electric field) is
  known physics, and the "non-uniform rotation = electric field" reading is known from synthetic
  gauge fields. The model-specific content is the coefficient: a Larmor gradient acts with charge
  1/2 per unit κ.
- **Depends on:** the identity κ = 2eA₀ — derived (EM_SCOPE §4), not selected. The acceleration
  difference needs no dynamical links at all, so it is testable **on `main`'s lattice** with κ made
  site-dependent.
  - If it holds, it is a derivation on `main`'s own terms, **a candidate to merge back** (charter
    rule 6).
- **Test:** a 1-D two-component lattice with κ(x) = κ\* + κ′x, a- and b-branch packets at rest,
  fitted centroid accelerations — like G2, a few seconds of computing.
- **Cleanliness:** high — sharp numbers, a cheap run, and no free parameter.

### 4. Self-gravitating lump structure: R ∝ G⁻² N⁻¹ (genuine; **OPEN**)

- **What:** under node form A′, the non-relativistic energy of a lump of N quanta is
  kinetic + A′ repulsion + gravity:
  aN/R² + bN^{3/2}R^{−3/2} − GN²ω²/(√(2π)R).
  - **For weak gravity** (G ≪ b²/a) the kinetic term is negligible at equilibrium, and the reduced
    functional has an exact scaling symmetry: **R ∝ G⁻² N⁻¹**. GRAVITY_SCOPE G4 shows the G⁻² at
    fixed N (R = 4.4, 1.5×10³, 8.4×10⁶, 8.4×10¹⁰ for G = 1 … 10⁻⁶).
  - **For stronger gravity** the kinetic term takes over: R ∝ G⁻¹N⁻¹, the boson-star law.
  - **The crossover** is at G ~ b²/a.
  - **Stability:** in the Newtonian regime the energy is bounded below at small R (kinetic and A′
    positive), so the ground state is stable. The compactness GM/R ~ G³N²ω/b² grows with N; see
    candidate 7.
- **New or known:** new. The G⁻² law comes from A′'s non-analytic |ψ|³ term; nothing natural
  corresponds. Boson stars are unobserved, and ordinary stars have different equations of state.
- **Depends on:** A′, a `main` premise selected for other reasons (§1a: the cone reading and gauge
  symmetry); G (free); and the lapse coupling. The scaling law was not used to select anything.
- **Test:** a simulation — needs building a 3-D lattice with Φ from Poisson's equation. Find
  stationary lumps at two G and two N and fit the exponents. No natural measurement.
- **Cleanliness:** clean in simulation (sharp exponents); untestable in nature.

### 5. The lattice photon dispersion shape (genuine; counted already)

- **What:** ω² = c_γ² Σᵢ 4 sin²(kᵢ/2) beyond the one point that fixes c_γ (EM_SCOPE C3, exact).
  - Along an axis the group velocity is c_γ cos(k/2).
  - At fixed |k| the dispersion depends on direction (lattice anisotropy).
  - **One lattice** carries both photons and matter, so the photon's departure from linear
    dispersion and matter's sin k departure (`main`'s row 1) are fixed by the same spacing ℓ — a
    cross-sector tie.
- **New or known:** new in the model's terms. In nature, Lorentz-invariance tests bound ℓ to below
  about the Planck length, so no departure is expected there.
- **Depends on:** nearest-neighbour plaquettes. It is the same for Wilson and Villain, so it does
  not rest on that selection. The shared speed fixes c_γ.
- **Test:** an engineered realisation (a lattice of coupled resonators, circuit-QED arrays)
  measuring photon dispersion and matter's sin k on the same platform. It cannot fail in nature
  unless ℓ is known.
- **Cleanliness:** clean in a realisation; weak in nature.

### 6. Frame dragging (genuine only with a further selection)

- **What:** a moving mass induces nonreciprocity, Δω = 2N sin k, with the shift N sourced by the
  momentum current.
- **The selection that ties the shift to G:** Lorentz covariance of the gravitational sector —
  lapse and shift as components of one metric under boosts.
  - Its minimal realisation is **linearised general relativity (spin-2)**. That also brings the
    spatial metric, the same selection that repairs candidates 1 and 2.
  - **Not counted yet:** it is the "(+1)" in the joint count's discrete selections.
- **New or known:** it reproduces known physics — the Lense–Thirring effect, measured by Gravity
  Probe B and LAGEOS.
- **Count:** with the spin-2 selection, frame dragging counts as +1 prediction for +1 selection:
  **net zero**. Light bending and tensor waves would **not** count, because they now motivate that
  selection (rule 3).
- **Test:** a lattice simulation of a rotating lump and Δω nearby; in nature, already measured.
- **Cleanliness:** moderate — it would pass by construction of a GR-like sector.

### 7. Horizons (not a prediction until another choice is made)

- **What:** the linear lapse N = 1 + Φ, with Φ ≈ −GM/R, reaches zero when a lump is compact
  enough, GM/R ~ 1.
  - In the linear-lapse Hamiltonian Σ(1 + Φ)h, a negative lapse makes the matter energy negative,
    so the energy is **unbounded below**: a collapse instability, not a horizon.
  - An exponential lapse, e^Φ (Nordström-like), never reaches zero, so **no horizons** — against
    the black holes observed (EHT, LIGO).
- **Status:** it depends on the lapse's nonlinear completion, **which is not yet selected**. Any
  choice made to produce horizons would be a selection. Not a genuine prediction now, and either
  natural choice already has a problem.
- **Cleanliness:** not testable until the choice is made.

### Already tested — the equal fall of the chirality branches (genuine; **HELD**, reproduces known physics)

Gauge invariance forces gravity onto the gauge-invariant energy (J4). Both branches then fall at
0.9959 in every gauge, against 0.691 / 1.309 for the inadmissible lab-type coupling. It is counted
in the joint strict predictions; its content is the equivalence principle, a known fact.

## Charged matter — what binds it if λ ~ 10⁴⁰?

- **Gravity cannot**: charged lumps bind only if λ = e²/(4πGω²) < 1 (J3).
- **The softening node nonlinearity would supply the binding.** Replace A′'s |ψ|³/3 by
  −λ_s|ψ|⁴/4 + σ|ψ|⁶/6. Coleman's criterion then holds (GRAVITY_SCOPE G4: min 2V/r² − K = −0.19),
  so Q-balls exist without gravity. **Gauged (charged) Q-balls** survive their own Coulomb
  repulsion below a critical charge, set by e and the self-coupling.
- **Cost:**
  - **+1 dimensionless parameter** (σ/λ_s² — one of the two coefficients is absorbed by the
    amplitude unit);
  - **+1 selection**: the softening sign, selected by the fact that charged matter binds, so the
    existence of charged lumps cannot count.
  - Consequences such as the Q-ball charge–mass relation would be genuine, but they buy back at
    most the parameter: **net zero or worse.**
- **Conflicts with `main`:**
  - **P9:** the premise "a quadratic on-site force, applied radially" is replaced — a premise
    change;
  - **C38:** "the well contributes zero dimensionless parameters" becomes false (+1);
  - the J-sector nonlinear results (the certification slopes, §1a) would need re-running.
  - **Not in conflict:** phase conservation, J-compatibility, the D2 rung, κ\* (linear) and P-3's
    zero self-precession (whole-node isotropy kept); persistence as bounded motion holds with the
    sextic.
  - **The P9 note does not decide it.** A smooth isotropic restoring force fixes the |ψ|²ψ power,
    not its sign. Only the softening, pendulum-like sign binds.

## The count

| | numeric parameters | discrete selections | strict predictions, firm | provisional (open) | **failed against nature** |
|---|---|---|---|---|---|
| baseline (`main` 646a238) | 3 | — | 4 (3) | — | — |
| joint, as scoped (JOINT_SCOPE (d)) | 5 + a state | 6 (+1) | 6 — baseline 4, photon dispersion shape, chirality-branch equal fall | 2 — κ-gradient force, lump scaling law | **2 — light deflection / Shapiro (γ = 0), scalar gravitational radiation** |
| joint, repaired by the spin-2 selection | 5 + a state | 7 | 7 — the above, plus frame dragging | 2 | 0 — light bending and tensor waves now motivate the selection and do not count |

- **As scoped, the joint theory is falsified** — two genuine predictions fail against known
  gravitational physics.
- **Repaired,** it gains one prediction (frame dragging) for one selection, which is neutral.
- **If both open candidates hold:** 9 strict predictions against 5 numeric parameters + 7 discrete
  selections (12 inputs). **Still below 1:1 with selections counted**, as the charter requires.
- **Cleanest next step:** candidate 3, the κ-gradient force. It is cheap, parameter-free and
  testable on `main`'s own lattice, and **it could close a `main` item if it holds.**
