# Shape Zero — overview

**The claim.** Shape Zero is a lattice model in which each node carries internal oscillators with a complex structure J and neighbouring nodes are coupled by velocity-dependent link matrices. Requiring the couplings to do no net work (passivity) and to respect J fixes the space of allowed couplings to the dimension of u(n) — u(1), u(2) and u(3) at node sizes n = 1, 2, 3 (`README.md`; `00_START_HERE/MODEL_SPEC.md` §3).

**Universal relations** — the results that hold independently of the genuine parameters, and how many conditions they place on them: [`UNIVERSAL_RELATIONS.md`](UNIVERSAL_RELATIONS.md).

**Scope.** The current target is a closed theory of a passive lattice with complex structure J and emergent u(n). The model is dimensionless and takes its units as inputs (`MODEL_SPEC.md` §1b); it is not presented as a theory of nature, and it makes no claims about consciousness or about deriving ℏ, G or Λ.

## Machine-proved (Lean 4, [Prove2Me](https://prove2.me); list in `00_START_HERE/PROVENANCE.md` §6l)

| # | result | link |
|---|---|---|
| 1 | Symmetric 2n×2n matrices commuting with J form a space of dimension n² = dim u(n) | [mission](https://prove2.me/missions/ebeea7d4-28c4-4613-a9a9-6a0754d83355) |
| 2 | On a ring of N ≥ 3 sites, a per-link coupling does no net work iff every link matrix is symmetric | [mission](https://prove2.me/missions/69ababbe-614d-4808-8441-4a41fa2ecba1) |
| 3 | The propagation asymmetry is exactly 2βc·sin q, independent of the on-site stiffness | [mission](https://prove2.me/missions/98b43e35-82c7-4c3a-a64c-04a8fb7c057f) |
| 4a | Result 2 on a periodic lattice with any number of axes (L ≥ 3) | [mission](https://prove2.me/missions/c78c5d4f-4dd8-4f4d-b27f-83349255f95f) |
| 4b | Result 3 in any dimension, also independent of transverse wavenumbers | [mission](https://prove2.me/missions/9ff04037-d5c4-4f85-9134-da4857002c2f) |
| 5 | A nonempty Steiner triple system with a role colouring has exactly 7 points | [mission](https://prove2.me/missions/d322e356-9907-4f54-ac53-198182d579ae) |
| 6 | Every Steiner triple system on 7 points is the Fano plane | [mission](https://prove2.me/missions/41c20aa4-1fd8-4027-9122-cc980db17bc9) |
| 7 | The octonionic flow p ↦ p·e₁ + (cos θ e₁ + sin θ e₂)·p has frequencies 0, 2, 2 sin(θ/2) (in review) | [goal](https://prove2.me/theorems/0850ee71-7133-4033-b33f-f964f8b18a0f) |
| 8 | Every force −a(x − r₁)(x − r₂), r₁ ≠ r₂, is z″ = −(z² − 1) in other units (in review) | [goal](https://prove2.me/theorems/f9298e70-0c29-4f54-a448-291860e866e1) |

## Measured (simulation; each result from the script named)

- **Non-commuting gauge ordering at q = 3** matches its independent linear prediction to 0.011–0.020° at amplitude 10⁻³ — `shape_zero_tests/q3_gate.py` (`MODEL_SPEC.md` §9).
- **J-compatibility is enforced by the dynamics**: at κ_g = κ_g\* the J-breaking channel is closed at every travelling wavenumber — `shape_zero_tests/jcompat_kappa.py`, `jcompat_gates.py` (`MODEL_SPEC.md` §3).
- **The linear pinned asymmetry is independent of uniform stiffness** to 10⁻⁵ (ratio 0.99999) — `shape_zero_tests/joint3_kappa_stiffness.py` (`README.md`).
- **The plane-wave amplitude coefficient** κ_A = −0.01748 at small amplitude, derived by perturbation theory — `shape_zero_tests/kappa_pw4_pt.py` — and measured −0.0176 at A = 0.10 — `04_scripts/session/pinned_asymmetry_reference.py` (`MODEL_SPEC.md` §5).
- **A localised beam's κ_A depends on its box** and has no box-independent value: κ_box = κ_A·fill·F, derived and tested — `shape_zero_tests/kappa_cross_pt.py`, `kappa_cross_compare.py` (`MODEL_SPEC.md` §5).
- **Unit equivalence**: runs at K = √5, 2, 1, 7.3 with ĉ, κ̂_g, β̂ held fixed agree to ≤ 1.3×10⁻¹⁴ — `shape_zero_tests/scale_invariance.py` (`MODEL_SPEC.md` §1b).

## Chosen or fitted

- **Parameters** (`MODEL_SPEC.md` §1b; `03_current/INPUT_LEDGER.md` §2d): ĉ = c/K = 1/√5 = 0.4472, with c = 1 chosen; κ̂_g = κ_g/√K, where only the floor κ̂_g ≥ 2ĉ/√(1 + 2ĉ) = 0.6498 is derived and the model runs at that floor (κ_g = κ_g\* = 0.9717); β̂ = βc/√K = 0.0334, with β = 0.05 chosen, not derived.
- **Premises**: the quadratic on-site force −(x² − x − 1), whose constants are units (`MODEL_SPEC.md` §1, §1b); the node form, the well on the whole node's radius (§1a); J-compatibility at every wavelength (§3).

## Open

- **A derived scale**: nothing internal fixes the fibre metric scale; a length or mass unit is an input (`MODEL_SPEC.md` §9; `03_current/SCALE_SCOPING.md`).
- **J-compatibility as a consequence** of the dynamics rather than an adopted principle (`MODEL_SPEC.md` §3).
- ~~**The q ≥ 2 stiffness-coupling coefficient C_q(N)**: the pin shift δ(Δω) = −¼βs²S² (joint #5) converges with box size at q = 1 (0.6%) but not at q ≥ 2 (spread 244% at q = 2, 243% at q = 3, with sign flips), and no mechanism is named (`MODEL_SPEC.md` §5b.6a; `PROVENANCE.md` §6i).~~ [2026-09-27: mechanism named, limit shown not to exist — the kernel grows as 1/p⊥² transverse to the probe.]
- ~~**The correct infinite-volume observable for a localised stiffness bump at q ≥ 2**: the second-order coefficient C_q(N) has no limit there (`MODEL_SPEC.md` §5b.6a, "MECHANISM NAMED").~~ [2026-09-27: answered — the golden-rule scattering rate, which carries the pin's nonreciprocity.]
- ~~**q = 3 rate convergence to be confirmed**: the scattering rate is converged at q = 2 and −2.6% from its limit at side 96 at q = 3 (`MODEL_SPEC.md` §5b.6a).~~ [CLOSED 2026-09-27: within 0.16% at side 256.]
- **A continuum limit**, stated or explicitly refused: ĉ is a length unit only in that limit, which is not taken (`MODEL_SPEC.md` §1b).
- **A prediction with more independent conditions than parameters** (`03_current/SCALE_SCOPING.md`, "Success").

## Run the model

    git clone https://github.com/ShapeZeroSZ/shape-zero
    cd shape-zero/04_scripts/session
    pip install numpy scipy
    python3 model.py

Requires Python 3 with NumPy and SciPy. `model.py` runs eleven build gates in order and stops at the first failure; all pass, in about 16 minutes on one core (`04_scripts/session/model.py`).

## Glossary

K = √5, linear on-site stiffness · c, neighbour elastic coupling · ĉ = c/K · κ_g, gyroscopic ratio (the intra-node term κ_g 𝕁u̇); κ̂_g = κ_g/√K; κ_g\* its derived floor · β, lattice gyroscopic coupling; β̂ = βc/√K · κ_A, nonlinear amplitude coefficient of the pinned asymmetry (plane wave) · C_q(N), stiffness-coupling coefficient of joint #5 at q axes and box side N, with s and S as in `MODEL_SPEC.md` §5b.6 · q, number of spatial axes · n, node size (gauge class u(n)) · A, amplitude · θ_ab, angle between the two D8 generators.
