# Anomaly A — where is the common size factor ≈ 0.6? Masked first-order runs: predictions

*Committed 2026-09-29, before any masked run.*

Script: `ampA_masked.py`. The A′ nonlinearity is switched on **IN** (the ramp sites of both segments,
every transverse site), **OUT** (everywhere else) or **ALL** (everywhere), each against the linear run
of the same geometry and readout time. Each slope comes from a single run at A = 10⁻³. Design choices
are ours.

In addition (ours) there is a q = 1 geometry with the segments further apart: 60 and 110 instead of
60 and 80, run ALL and linear only, read at T + 110.

## Disclosures
- The recorded slopes and the mechanism model's outputs (8b61310, fccf291) are known.
- The measured/model factors from fccf291 are known: q = 1 0.57–0.60, q = 3 u(2) 0.60–0.72, q = 3 u(3) 0.83.
- The separated-geometry model numbers below were computed before committing. They use the same
  machinery, with `segs` added to `ampA_mechanisms.Model` and no change to its committed outputs.
- Readout times are the recorded protocol values.
- Tolerances were set now.

## What the angle law plus K4 predicts for each mask

**Ray theory, derived now, for the IN and OUT split.**
- **IN.** With the potential inside a segment only, a slice enters at its linear frequency. At fixed
  frequency the potential shifts every eigen-direction's wavenumber by −V/(2c√(1 + t_j²)). For
  eigenvalues ±1, t_j² is equal, so the relative phase is **exactly zero** at WKB level.
- **OUT.** With the potential outside only, a slice carries Ω = ω + V/(2ω + κ) into the segment
  (frequency is conserved across the static boundary), and the Peierls angle there uses Ω. So the
  **whole angle law lives in OUT**. In the ray picture: the channel that dwells longer in the segment
  spends less time accumulating the self-phase outside.
- **K4** (the shared radius acting on displaced channels before segment 2) also acts outside.

**Predicted:**
- **IN ≈ 0.** Each per-order |IN| ≤ 0.15 × |ALL|, and |IN split| ≤ 0.15 × max per-order ALL.
- **OUT ≈ ALL.**
- **Additivity** (first order): the Bloch-vector changes satisfy |dco_IN + dco_OUT − dco_ALL| ≤ 0.05 |dco_ALL|.
- **ALL reproduces the recorded runs at 10⁻³.** The force is identical, and the per-order values
  should be within 5% of the recorded vector-fit slopes.

**Model values for OUT and ALL** (degrees per 10⁻³; from 8b61310, and computed now for the separated geometry):

| | split | per-order AB / BA |
|---|---|---|
| q=1 u(2) | −0.0145 | 0.0270 / 0.0214 |
| q=1 u(3) | +0.0264 | 0.0236 / 0.0289 |
| q=3 u(2) | −0.0147 | 0.0190 / 0.0135 |
| q=3 u(3) | +0.0017 | 0.0191 / 0.0284 |
| q=1 separated u(2) | −0.0141 | 0.0264 / 0.0203 |
| q=1 separated u(3) | +0.0242 | 0.0212 / 0.0285 |

## Which pattern implicates what

The **factor** is the projection ratio dco_meas·dco_model/|dco_model|², per order.

**(a) The instantaneous-segment approximation.** The model treats each segment as an instantaneous
per-mode transfer, and that is worst when the packet overlaps both segments at once.
- **Predicts:** the factor moves toward 1 as the segments separate relative to the packet width.
  - q = 1 separated geometry: factor ≥ 0.85, against 0.57–0.60 in the certification geometry.
  - q = 3 (width 3, segments 20 apart): factor ≥ 0.9.
- **Caveat:** the recorded q = 3 factors are already known (0.60–0.72 for u(2), 0.83 for u(3)), and
  they argue against (a) there. The separated q = 1 geometry is the clean test of (a).

**(b) A further mechanism inside the segments.**
- **Predicts:**
  - IN is substantially nonzero (|IN| > 0.15 |ALL|) and opposes OUT (the cosine of their Bloch-vector changes is < 0);
  - OUT alone is close to the model (factor 0.85–1.15);
  - the factor is unchanged in the separated geometry.
- **Reading:** the missing ~0.4 comes from in-segment dynamics that WKB ray theory says cancels, so
  it is a mechanism in the dynamics, not an artefact.

**(c) A further mechanism, or mis-sizing, outside the segments.**
- **Predicts:**
  - IN ≈ 0;
  - OUT ≈ ALL ≈ 0.6 × the model;
  - the factor is unchanged in the separated geometry.
- **Reading:** the outside physics (the dwell/self-phase partition, or K4's size) is wrong in the
  model, not because of the instantaneous approximation.

**(d) Non-additivity.** IN + OUT ≠ ALL beyond 5% would mean the effect is not first order, which
contradicts the linear fits. Not expected.

**Mixed patterns** will be reported as mixed.
