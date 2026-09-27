# Universal relations — a census

*Recorded 2026-09-27. A relation is listed if it holds independently of the genuine parameters —
ĉ = c/K, κ̂ = κ/√K (with its floor) and β̂ = βc/√K (`00_START_HERE/MODEL_SPEC.md` §1b) — or depends on them
through fewer combinations than it constrains. "Nature" means a physical realisation of this lattice
(the records call κ "measurable in a lab now" by the Larmor splitting, and β̂ fixable by one Δω
measurement); the model is not presented as a theory of nature (`OVERVIEW.md`, "Scope").*

| # | relation | independent of | status | checked by | comparable with a measurement |
|---|---|---|---|---|---|
| 1 | Δω = 2βc sin k, i.e. Δω/√K = 2β̂ sin k | ĉ, κ̂, the well — depends on β̂ only | **proved** (Lean, published) | Prove2Me mission 3; joint #2 (null to 2×10⁻⁶ over ±10% stiffness) | **yes** — one k fixes β̂; the sin k shape at every other k is a prediction [**NOTE 2026-09-27:** "independent of κ̂" holds because the β sector contains no κ — it is scalar (`04_scripts/session/pinned_asymmetry_reference.py`). With β placed on spinning (κ) nodes, the precession branch alone gives Δω = 2βc sin k·(1 − κ/√(κ² + 4Q)) + O(β³), which depends on κ̂ and ĉ; only the **two-branch sum rule** survives: Δω summed over both positive branches = 2·(2βc sin k) for any κ, K and anisotropy (found, then verified; branch `realisation-gyroscopic`, `REALISATION.md` §4, with its verification script)] |
| 2 | the same, independent of transverse wavenumbers, in any dimension | transverse k, dimension | **proved** | mission 4b | **yes** |
| 3 | q = 1 pin shift δ(Δω) = −¼βs²S², shape-independent to 0.4% | the bump's shape and width (localised class); the pure number ¼ | **derived + measured** | joint #5 records; `shape_zero_tests/joint5_cq.py`, `joint5_kernel.py` | **yes**, in a 1-D realisation (at q ≥ 2 there is no such limit; the observable there is the scattering rate, `MODEL_SPEC.md` §5b.6a) |
| 4 | κ̂ ≥ κ̂\* = 2ĉ/√(1 + 2ĉ) | a function of ĉ only; the well drops out | **derived** (a floor), checked | `jcompat_*` tests; PREMISE_LEDGER C14 | **yes** — an inequality between the Larmor splitting and the band shape |
| 5 | Larmor splitting ω_b − ω_a = κ exactly | ĉ, β̂, the well | **derived** | MODEL_SPEC §3 | yes — but it **defines** κ̂ (a calibration, not a test) |
| 6 | every two-root quadratic well is the same well in other units | — (removes parameters) | **proved**, in review | mission 8; `shape_zero_tests/scale_invariance.py` | no — a statement about coordinates |
| 7 | D8 frequency ratio 1/sin(θ/2) | all but θ_ab | **proved**, in review | mission 7; `shape_zero_tests/d8_closed_form.py` | no — the lattice does not realise the D8 flow (MODEL_SPEC §9) |
| 8 | P-3: a circular spinor wave does not precess, Ω = 0 at every amplitude (node form A′) | every parameter and amplitude | **derived + measured** (10⁻¹⁸) | `shape_zero_tests/nodewell_test.py p3` | yes — but **excluded from the count** (below) |
| 9 | wide-bump nonreciprocity limit ν = (Γ(+k) − Γ(−k))/Γ → 0 at k = π/2, q ≥ 2 | bump shape; ĉ, β̂ enter only at finite width | **derived + computed** | `shape_zero_tests/joint5_rate2.py` | yes, in principle |
| 10 | symmetric + J-commuting couplings have dimension n² (mission 1); passivity ⇔ symmetric links (missions 2, 4a); roles force Fano (missions 5, 6) | every parameter (structural) | **proved** | missions 1, 2, 4a, 5, 6 | no — constraints on the model's form, not a measured number |
| 11 | small-amplitude reduction to the linear gauge dynamics | amplitude, in the limit | **measured** (certified) | `shape_zero_tests/certify_gates.py` | no — an internal consistency check |

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
