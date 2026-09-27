# Universal relations — a census

*Recorded 2026-09-27. A relation is listed if it holds independently of the genuine parameters —
ĉ = c/K, κ̂ = κ/√K (with its floor) and β̂ = βc/√K (`00_START_HERE/MODEL_SPEC.md` §1b) — or depends on them
through fewer combinations than it constrains. "Nature" means a physical realisation of this lattice
(the records call κ "measurable in a lab now" by the Larmor splitting, and β̂ fixable by one Δω
measurement); the model is not presented as a theory of nature (`OVERVIEW.md`, "Scope").*

| # | relation | independent of | status | checked by | comparable with a measurement |
|---|---|---|---|---|---|
| 1 | Δω = 2βc sin k, i.e. Δω/√K = 2β̂ sin k | ĉ, κ̂, the well — depends on β̂ only | **proved** (Lean, published) | Prove2Me mission 3; joint #2 (null to 2×10⁻⁶ over ±10% stiffness) | **yes** — one k fixes β̂; the sin k shape at every other k is a prediction [**NOTE 2026-09-27:** "independent of κ̂" holds because the β sector contains no κ — it is scalar (`04_scripts/session/pinned_asymmetry_reference.py`). With β placed on spinning (κ) nodes, the precession branch alone gives Δω = 2βc sin k·(1 − κ/√(κ² + 4Q)) + O(β³), which depends on κ̂ and ĉ; only the **two-branch sum rule** survives: Δω summed over both positive branches = 2·(2βc sin k) for any κ, K and anisotropy (found, then verified; branch `realisation-gyroscopic`, `REALISATION.md` §4, with its verification script)] [**CONTINUUM NOTE 2026-09-27** (MODEL_SPEC §1c): at long wavelength rows 1–2 are the **Doppler kinematics of a uniformly drifting medium** (V = βc): Δω = 2V·k is automatically independent of K, c and the transverse wavenumbers. Their genuine content is the **lattice's sin k departure from linear Doppler**.] |
| 2 | the same, independent of transverse wavenumbers, in any dimension | transverse k, dimension | **proved** | mission 4b | **yes** [**CONTINUUM NOTE 2026-09-27** (MODEL_SPEC §1c): transverse independence is Doppler kinematics at long wavelength — Δω depends on k·V only; see row 1's note.] |
| 3 | q = 1 pin shift δ(Δω) = −¼βs²S², shape-independent to 0.4% | the bump's shape and width (localised class); the pure number ¼ | **derived + measured** | joint #5 records; `shape_zero_tests/joint5_cq.py`, `joint5_kernel.py` | **yes**, in a 1-D realisation (at q ≥ 2 there is no such limit; the observable there is the scattering rate, `MODEL_SPEC.md` §5b.6a) |
| 4 | κ̂ ≥ κ̂\* = 2ĉ/√(1 + 2ĉ) | a function of ĉ only; the well drops out | **derived** (a floor), checked | `jcompat_*` tests; PREMISE_LEDGER C14 | **yes** — an inequality between the Larmor splitting and the band shape [**CONTINUUM NOTE 2026-09-27** (MODEL_SPEC §1c): the floor is the **no-resonance condition in the rotating frame** — J-breaking terms rotate at κ there, and an a-wave reaches the b-branch at equal lab frequency iff κω ≤ c(1 − cos k₀); this agrees with the recorded closed-iff rule at all 3600 points of a 60 × 60 (κ, k₀) grid.] |
| 5 | Larmor splitting ω_b − ω_a = κ exactly | ĉ, β̂, the well | **derived** | MODEL_SPEC §3 | yes — but it **defines** κ̂ (a calibration, not a test) [**CONTINUUM NOTE 2026-09-27** (MODEL_SPEC §1c): this is **Larmor's theorem** — κ is a rotating frame at μ = κ/2, and the splitting is 2μ.] |
| 6 | every two-root quadratic well is the same well in other units | — (removes parameters) | **proved**, in review | mission 8; `shape_zero_tests/scale_invariance.py` | no — a statement about coordinates |
| 7 | D8 frequency ratio 1/sin(θ/2) | all but θ_ab | **proved**, in review | mission 7; `shape_zero_tests/d8_closed_form.py` | no — the lattice does not realise the D8 flow (MODEL_SPEC §9) |
| 8 | P-3: a circular spinor wave does not precess, Ω = 0 at every amplitude (node form A′) | every parameter and amplitude | **derived + measured** (10⁻¹⁸) | `shape_zero_tests/nodewell_test.py p3` | yes — but **excluded from the count** (below) |
| 9 | wide-bump nonreciprocity limit ν = (Γ(+k) − Γ(−k))/Γ → 0 at k = π/2, q ≥ 2 | bump shape; ĉ, β̂ enter only at finite width | **derived + computed** | `shape_zero_tests/joint5_rate2.py` | yes, in principle |
| 10 | symmetric + J-commuting couplings have dimension n² (mission 1); passivity ⇔ symmetric links (missions 2, 4a); roles force Fano (missions 5, 6) | every parameter (structural) | **proved** | missions 1, 2, 4a, 5, 6 | no — constraints on the model's form, not a measured number |
| 11 | small-amplitude reduction to the linear gauge dynamics | amplitude, in the limit | **measured** (certified) | `shape_zero_tests/certify_gates.py` | no — an internal consistency check |
| 12 | **κ-gradient force:** in a static κ gradient, packets at rest on the two branches accelerate apart with a_a − a_b = c·∂ₓ(ω_b − ω_a)/ω̄, ω̄ = (ω_a + ω_b)/2 — the electric-field analogue, branches as opposite charges | the well; K and κ enter only through the measured branch frequencies | **derived + measured** (−0.23%; predictions committed first) | `shape_zero_tests/kgrad_test.py`; MODEL_SPEC §1c | **yes** — in a realisation with a position-dependent Larmor splitting |

## The count

Setting aside the structural and coordinate results (6, 7, 10) and the internal check (11), the
census holds about **eight independent conditions against the three genuine parameters** ĉ, κ̂, β̂:
the sin k functional form beyond the one point that fixes β̂; the independence of Δω from the stiffness
and from ĉ and κ̂; its independence from transverse wavenumbers; the coefficient ¼; its shape
independence; the κ̂ floor (an inequality); P-3's absence; and the ν → 0 limit.

- **Two are used as calibrations:** Δω at one k fixes β̂; the Larmor splitting defines κ̂.
- **ĉ is fixable by the band shape** (the dispersion across the zone); no census condition fixes it.
- **Used to choose a parameter: only the κ̂ floor**, which set the operating value κ = κ\*. The other
  values (c = 1, β = 0.05) were set, not fitted.
- **Not used to choose anything — genuine predictions:** the sin k form, the stiffness and transverse
  independence of the asymmetry, the coefficient ¼ with its shape independence, and the ν → 0 limit.
- **P-3 is excluded** from the predictions: its absence of self-precession is a consequence of node form
  A′, which was selected under the selection rule, and what that selection produced "cannot be counted as
  evidence for the model" (MODEL_SPEC §1a, "ADOPTED 2026-09-27").

**Row 12 (proposed 2026-09-27, merge candidate).** A new, independent condition, not used to choose
anything. If adopted, the counts in both sections below rise by one: the strict genuine predictions go
from four to **five** (four if ν → 0 is kinematic), against three parameters. The earlier counts are
kept as written.

## The count — under the continuum reading (2026-09-27)

*Added after MODEL_SPEC §1c; the count above is kept as written.* Taking rows 1–2 at long
wavelength as the Doppler kinematics of a uniform drift, two of the eight conditions above stop
being independent: **the independence of Δω from the stiffness (and from ĉ and κ̂ — vacuous, the
β sector has no κ)** and **its independence from transverse wavenumbers** are properties of any
uniformly drifting medium. What remains:

| # | condition | status under this reading |
|---|---|---|
| 1 | the sin k departure from linear Doppler, beyond the one point that fixes β̂ | genuine; lattice-scale. It tests that the drift coupling is nearest-neighbour — the shape is the Fourier symbol of that coupling |
| 2 | the coefficient ¼ (q = 1 pin shift) | genuine; not affected by this reading |
| 3 | its shape independence | genuine; not affected |
| 4 | the κ̂ floor | a condition (an inequality), now read as rotating-frame no-resonance; **used to choose κ**, not a prediction |
| 5 | P-3's absence | a condition, **excluded** (a consequence of the selected form A′) |
| 6 | ν → 0 for wide bumps at k = π/2 | counted provisionally; **open whether it is the moving-frame statement seen from the lattice** (MODEL_SPEC §9) |

**The new count: about six independent conditions against three parameters** (ĉ, κ̂, β̂) —
down from eight. It **still exceeds three.** But the stricter count, genuine predictions
neither used to choose a parameter nor excluded, is **four** (sin k departure, ¼, shape
independence, ν → 0), and **three if ν → 0 turns out to be kinematic** — equal to the number of
parameters, not above it. The calibrations are unchanged (Δω at one k fixes β̂; the Larmor
splitting — Larmor's theorem — defines κ̂; the band shape fixes ĉ). Note also that in the J sector
without links κ̂ enters only through ĉ′ = ĉ/(1 + κ̂²/4) (MODEL_SPEC §1c).

