# Anomaly A — the open first-order mechanism: candidates and predictions

*Committed 2026-09-29 before any comparison with the recorded slopes.*

The candidates are computed by `ampA_mechanisms.py`, with output in `ampA_mechanisms_predictions.txt` and `.json`. Design choices are ours. No parameter was fitted.

## Disclosures
- The recorded (A′) slopes were known before this work.
- I saw the model outputs before committing.
- Two bugs were found and fixed after the first outputs, and before this commit.
  - **Non-unitary Born term.** The first-order Born term was added to the state at A = 10⁻³. There the self-phase over the readout time is about 0.1 rad, so the non-unitary second-order part swamped the 10⁻⁴-level effects.
    - The symptom was a "post-segment" contribution that grew with readout time.
    - That contradicts the exact result derived below: the post-segment term must be zero, and the slopes cannot depend on readout time.
    - Fix: the model runs at A = 10⁻⁸ and is scaled linearly.
  - **Angle precision.** `model.angle` uses arccos, which quantised the tiny angles. It was replaced by 2 arcsin(|â − b̂|/2).
- The readout times are the recorded protocol values (`nodewell_1d.json`, `q3_gate_runs_260x8_nodeA1.json`). They are amplitude-independent settings, not slopes.

## Exact statements derived first

**(E1) First-order structure.** Linearising −(√5 + |u|)u about the linear solution u₀ gives, at first order in amplitude, the linear lattice plus the scalar potential V(x, t) = |u₀(x, t)|. The potential is common to every internal component. So every first-order mechanism is the interplay of a scalar potential that moves with the packet and the direction-dependent propagation in the link segments.

**(E2) The smooth control.**
- Under the smooth diagnostic the potential is |u₀|² ∝ A², so every candidate below vanishes by construction.
- The recorded smooth slopes (≤ 1×10⁻⁵) are therefore passed by all of them. This control does not discriminate between candidates.

**(E3) No first-order effect after the last segment.** Outside the segments the propagator acts identically on every internal component. A first-order kick −iεVψ therefore changes ρ = Σₓψψ† by −iεΣVψ_aψ_b* + iεΣVψ_aψ_b* = 0. Only potential acting **before** a segment can change the Bloch vector, so no candidate may depend on the readout time.

**(E4) u(3) is the two-level problem.** Axes 0 and 1 of u(3) (λ₁, λ₂) act only on {e₀, e₁}. (A′) is U(n)-invariant, so e₂ is never populated. u(3) is the u(2) problem with strengths (0.15, 0.10) or (0.15, 0.15) in the Gell-Mann metric.

**(E5) Equal-strength symmetry.**
- For equal strengths, any mechanism that only changes each segment's rotation angle about its own axis gives equal AB and BA per-order deviations. The z-rotation by 90° followed by complex conjugation maps AB to BA and fixes e₀.
- So an AB/BA asymmetry at q = 3 u(3) (0.15/0.15) **requires a first-order rotation about the commutator axis** (λ₃).

## Candidates and their derived contributions

**ANGLE — the per-direction Peierls angle.**
- In-segment phase at Ω = ω + δω with Q fixed, spectrum-averaged per mode, δω = A_eff/(2ω+κ) at each ramp site's crossing time.
- Pattern: per-segment rotation-angle changes. At q = 1 the u(2)/u(3) split signs are − / +.
- By (E5) it **cannot** produce the equal-strength AB/BA asymmetry.

**K1 — envelope averaging.**
- Ray theory with symmetric weights gives the first-order weight as A_eff = ⟨F³⟩/⟨F²⟩ at each site crossing, which is the one used in ANGLE. The per-site, time-integrated and density-weighted forms all coincide for a rigid envelope.
- The alternative, the peak amplitude, gives a uniform factor of 1.22 (q = 1) and 1.67 (q = 3) **upward**.
- It cannot produce a sign change or the asymmetry.

**K2 — dwell time from the amplitude-dependent group velocity.**
- This is already inside ANGLE: the phase is the splitting at fixed k (∝ Ω/(2Ω+κ)) times the dwell (∝ (2Ω+κ)), which gives gΩh per site.
- Adding it separately would double-count.
- If the dwell did not change (envelope at the linear group velocity), ANGLE would be scaled by 1 − 2ω/(2ω+κ) = **0.230**, a uniform factor.
- Check made before commit, on a free packet with no links: it slows by 0.87× of the plane-wave estimate over t = 0–100 and 0.73× over t = 100–200 (energy-density centroid). So the dwell change is present, and **K2 contributes nothing beyond ANGLE**.

**K3 — wavenumber shift at fixed frequency in finite segments.**
- (a) WKB: identical to ANGLE at k₀ = π/2. The h²-dependent part vanishes exactly for these generators, whose populated eigenvalues are ±1.
- (b) Non-adiabatic ramp: the exact plane-wave transfer-matrix phase against the WKB sum gives d(relative phase)/dV ratios of **1.0000** at g = 0.08, 0.10, 0.12 and 0.15.
- **Contribution: zero.**

**K4 (BORN) — the shared radius acting on eigen-channels displaced by the first segment's group delays, and on the chirped packet, before the second segment.**
- Method: first-order Born integral of −iVψ/(2ω+κ), with each segment applied as an instantaneous per-mode transfer at its centre-crossing time. By (E3) the post-segment term is zero.
- Pattern: rotations that are **not** about the segment axes, including the commutator axis. So it can change signs and produce the AB/BA asymmetry.
- Its approximation (instantaneous transfers) is worst when the packet overlaps both segments at once. That is the case at q = 1 (packet width 8, segments 20 apart); at q = 3 the width is 3.

## Predicted slopes (degrees per 10⁻³)

| | ANGLE | K1 peak | BORN | ANGLE + BORN |
|---|---|---|---|---|
| q=1 u(2) split | −0.0143 | −0.0175 | −0.0002 | **−0.0145** |
| q=1 u(2) per-order AB / BA | 0.0273 / 0.0221 | 0.0334 / 0.0271 | 0.0007 / 0.0008 | **0.0270 / 0.0214** |
| q=1 u(3) split | +0.0275 | +0.0337 | −0.0011 | **+0.0264** |
| q=1 u(3) per-order AB / BA | 0.0251 / 0.0290 | 0.0307 / 0.0354 | 0.0023 / 0.0004 | **0.0236 / 0.0289** |
| q=3 u(2) split | −0.0064 | −0.0107 | −0.0083 | **−0.0147** |
| q=3 u(2) per-order AB / BA | 0.0160 / 0.0126 | 0.0267 / 0.0211 | 0.0052 / 0.0033 | **0.0190 / 0.0135** |
| q=3 u(3) split | −0.0115 | −0.0191 | +0.0131 | **+0.0017** |
| q=3 u(3) per-order AB / BA | 0.0138 / 0.0138 | 0.0230 / 0.0230 | 0.0147 / 0.0164 | **0.0191 / 0.0284** |
| floors q=1 u(2) / u(3) | 0.00005 / 0.00005 | | | **0.00001 / 0.00016** |
| floors q=3 u(2) / u(3) | 0.0005 / 0.0005 | | | **0.0001 / 0.0016** |

What each candidate predicts:
- **ANGLE + BORN:**
  - q = 3 u(3) split turns **positive**.
  - AB/BA asymmetry with **BA > AB** (ratio 1.49).
  - At q = 1 the per-order slopes stay at about 1.0× ANGLE, so **no candidate here produces the q = 1 factor of about 0.57**.
- **K1 peak:** 1.22× (q = 1) or 1.67× (q = 3), upward.
- **K2:** 0 beyond ANGLE; the no-dwell variant is 0.23×.
- **K3:** 0.

## Comparison criteria

These were fixed now, after seeing the model outputs.
- **Slopes.** A slope is **accounted for** when its sign matches and |pred − meas| ≤ 0.3|meas| + 0.002.
- **Pattern tests.**
  - The q = 1 factor is accounted for if the measured/predicted per-order ratio is within 0.85–1.15.
  - The q = 3 u(3) sign is accounted for if the signs match.
  - The AB/BA asymmetry is accounted for if the larger order matches and the ratio BA/AB is within ±30% of the measured one.
- **Vector test.** Compare the direction of the predicted first-order Bloch-vector change with the recorded vector-fit slope co₁ (cosine, per order).
