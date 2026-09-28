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
| 9 | wide-bump nonreciprocity limit ν = (Γ(+k) − Γ(−k))/Γ → 0 at k = π/2, q ≥ 2 | bump shape; ĉ, β̂ enter only at finite width | **derived + computed** | `shape_zero_tests/joint5_rate2.py` | yes, in principle [**NOTE 2026-09-28** (MODEL_SPEC §5b.6a, "CORRECTED 2026-09-28"; `shape_zero_tests/joint5_rate3.py`, predictions committed first): **kinematic, not independent** — a wide bump's rate is a dwell-time law, Γ ∝ 1/\|v_g\|, and ν → 0 at π/2 is the stationary point of row 1's sin k shape (dΔω/dk = 2βc cos k = 0); confirmed at π/4, π/3, 2π/3 within 2×10⁻⁵. Its leading correction −βΩ(q − 1)J is **not universal** (31.5% spread across shapes against the pre-registered 2%).] |
| 10 | symmetric + J-commuting couplings have dimension n² (mission 1); passivity ⇔ symmetric links (missions 2, 4a); roles force Fano (missions 5, 6) | every parameter (structural) | **proved** | missions 1, 2, 4a, 5, 6 | no — constraints on the model's form, not a measured number |
| 11 | small-amplitude reduction to the linear gauge dynamics | amplitude, in the limit | **measured** (certified) | `shape_zero_tests/certify_gates.py` | no — an internal consistency check |
| 12 | **κ-gradient force:** in a static κ gradient, packets at rest on the two branches accelerate apart with a_a − a_b = c·∂ₓ(ω_b − ω_a)/ω̄, ω̄ = (ω_a + ω_b)/2 — the electric-field analogue, branches as opposite charges | the well; K and κ enter only through the measured branch frequencies | **derived + measured** (−0.23%; predictions committed first) — **a verified consequence of row 5 plus ray kinematics and the band curvature; not independent** | `shape_zero_tests/kgrad_test.py`; MODEL_SPEC §1c; PROVENANCE §6t | yes, in a realisation with a position-dependent Larmor splitting — but it would test ray kinematics, not a new condition |
| 13 | **universal refraction:** a weak D ≤ 8 packet in a populated tower senses one stiffness shift dK = √s, the same for both chiralities, every colour and every node size (b/a ≤ 3.4×10⁻⁷, D16–D128 to 10⁻¹⁵); dw = √s/(2ω_a + κ); in a gradient the wavenumber is pushed away from the denser region, dp/dt = −∂ₓ√s/(2ω_a + κ) | chirality, colour, node size; which upper components carry s | **measured** (predictions committed first) — **a verified consequence of node form A′'s single shared radius; not independent** | `shape_zero_tests/tower_chirality_matched_test.py`; PREMISE_LEDGER C51; PROVENANCE §6w | no — it follows from a form selected under the selection rule |
| 14 | **product rule:** ω_a(k)·ω_b(k) = Q(k) = K + 2c(1 − cos k) in every internal eigen-direction of a link, for any link strength, generator and κ; with the splitting rule ω_b − ω_a = κ + 2cgh sin k. **β obeys the same product rule** (scalar sector: ω(k)·ω(−k) = Q(k)) | g, H, κ̂, β̂ | **derived** (exact per-mode equation, residual 9×10⁻¹¹ on `model.py`'s force) + **measured** (links change the product by ≤ 5×10⁻⁸ against the link-free control) — **a verified consequence: Vieta's formula, and its k-dependence is the band shape already named as ĉ's calibration; not independent** | `shape_zero_tests/link_scoping_checks.py`; MODEL_SPEC §1c, "The link sector" | yes, as a **realisation diagnostic** — it distinguishes velocity-linear links (product unchanged) from ordinary unitary links (product changed) |
| 15 | **q ≥ 2 direction locking and windings:** uniform non-commuting links on different axes lock the internal eigenbasis to the propagation direction (axis at atan2(sin k_y, sin k_x)) and close the internal splitting at (0,0), (π,0), (0,π), (π,π) with windings +1, −1, −1, +1 | every parameter and the link strength | **derived** (algebra), not simulated — `model.py` has links on axis 0 only — **a verified consequence of row 2 (per-axis sin k) plus the su(2) algebra (row 10); not independent** | MODEL_SPEC §1c, "The link sector"; `shape_zero_tests/link_scoping_predictions.txt` | in principle — it would test the per-axis form and the link algebra, not a new condition |

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

**Row 12 (added 2026-09-27) does not change either count.** It is implied by row 5 applied locally,
standard ray (eikonal) kinematics and the band curvature 1/m\* = c/ω̄. A packet at rest accelerates at
ẍ = −(∂²ω/∂k²)(∂ω/∂x); with ω_b = ω_a + κ(x) at every k, the branches share 1/m\*, and their
x-derivatives differ by κ′ (PROVENANCE §6t). It is recorded as a **verified consequence**. The strict
genuine predictions stay at **four** (three if ν → 0 is kinematic). [2026-09-28: it is kinematic; see "The count — corrected accounting".]

**Row 13 (added 2026-09-27) does not change either count.** Universal refraction follows from node form
A′'s single shared radius — every D ≤ 8 component feels the same \|u\|, so a background enters every
weak packet as the same stiffness √s — and A′ was selected under the selection rule, so, like P-3
(row 8), what that selection produced is not counted as evidence. It is recorded as a **verified
consequence** (PREMISE_LEDGER C51).

**Rows 14 and 15 (added 2026-09-28) change neither count.** Row 14, the product rule, is Vieta's
formula — in ω² + (linear term)ω − Q(k) = 0 the product of the roots is the constant term, which any
coupling linear in velocity (κ, β or links) cannot change — and its k-dependence is the band shape already
named as ĉ's calibration, whose cos k form tests the same nearest-neighbour range as the sin k form, so
counting it would double-count; β obeys the same rule. It is useful as a **realisation diagnostic**,
distinguishing velocity-linear links from ordinary unitary ones. Row 15 follows from row 2 and the su(2)
algebra. Both are recorded as **verified consequences** (MODEL_SPEC §1c, "The link sector").

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
parameters, not above it. [**CORRECTED 2026-09-28:** ν → 0 is kinematic (row 9's note), so three remain — but comparing predictions with parameters was the wrong accounting: the calibrations are conditions too. See "The count — corrected accounting".] The calibrations are unchanged (Δω at one k fixes β̂; the Larmor
splitting — Larmor's theorem — defines κ̂; the band shape fixes ĉ). Note also that in the J sector
without links κ̂ enters only through ĉ′ = ĉ/(1 + κ̂²/4) (MODEL_SPEC §1c).

## The count — corrected accounting (2026-09-28)

*Added after the row-9 test (MODEL_SPEC §5b.6a, "CORRECTED 2026-09-28"); the counts above are kept as
written.* **The criterion** is **more satisfied conditions than parameters, with at least one condition not
used to choose a parameter** — the over-determination test that `03_current/SCALE_SCOPING.md`, "Success",
states for the fibre scale ("more conditions than it spends parameters"), applied here to the lattice. So the conditions are counted
**including the calibrations**; comparing only the uncalibrated predictions with the parameters, as the
sections above did, double-counts the parameters.

| condition | role |
|---|---|
| Δω at one k | **calibration** — fixes β̂ |
| the Larmor splitting | **calibration** — fixes κ̂ |
| the band shape | **calibration** — fixes ĉ |
| the sin k departure from linear Doppler (row 1, beyond the calibrating point) | **satisfied, not used** |
| the coefficient ¼ (row 3) | **satisfied, not used** |
| its shape independence (row 3) | **satisfied, not used** |

**Six conditions against three parameters — over-constrained by three**, and each genuine prediction
is the excess. Not counted: the κ̂ floor (used to choose κ), P-3 (a consequence of the selected form A′),
row 9 (kinematic — contained in row 1's sin k shape), and the verified consequences (rows 12–15).

**What the over-constraint is.** The three genuine predictions follow from the model's own equations, so
**within the model they are exact and cannot fail**. They can genuinely fail **only in a physical
realisation** — a lattice whose couplings are measured, not assumed. The over-constraint is therefore
**as a theory of realisable lattices**. Rows 1–2 are testable as described on the branch
`realisation-gyroscopic` (`REALISATION.md` §5): a chain of motors-off pendula with bond rotors.

