# Anomaly A — the 2/p law from the conserved quantities: derivation and the p = 6 test

*Committed 2026-10-01, before any run.*

Script: `ampA_whitham.py`. `predict` has been run; its output is `ampA_whitham_predictions.txt` and
`.json`. Design choices are ours. The p = 6 well, −(√5 + |u|⁴)u, is a **diagnostic only**: it is
defined in the script, not in `model.py`, and is not a change of premise.

## Disclosures
- Everything measured so far is known:
  - packet factors 0.666–0.667 (p = 3) and 0.498–0.500 (p = 4);
  - stationary factors 0.9999 / 1.0000 (20030b1).
- The candidate (E/I) is yours.
- I saw the model outputs before committing.
- Tolerances were set now.

## (1) Derivation (Whitham's averaged Lagrangian)

**Averaged Lagrangian.** For a circular a-wave of amplitude a, wavenumber k and frequency Ω, with the
well ∝ |u|^p:

  𝓛 = ½(Ω² + κΩ − Q(k)) a² − a^p/p

**Quantities derived from it:**

| quantity | expression |
|---|---|
| dispersion (∂𝓛/∂a = 0) | Ω² + κΩ = Q + a^{p−2}, so δΩ = a^{p−2}/D at fixed k, D = 2Ω + κ |
| action density | 𝒜 = ∂𝓛/∂Ω = D a²/2 |
| action flux | 𝓑 = −∂𝓛/∂k |
| energy density | ℰ = Ω𝒜 − 𝓛 |
| energy flux | Ω𝓑 |

On-shell, **𝓛 = (½ − 1/p) a^p**, which vanishes only for a linear wave.

**Result 1, the energy per action:** ℰ/𝒜 = Ω − 𝓛/𝒜 = Ω − (1 − 2/p)δΩ = **ω_lin + (2/p)δΩ**.

**Result 2, packet totals.** A localised packet crossing a static, passive segment conserves its total
energy E and total action I. The action is the U(1) phase charge, which A′ conserves at every order.
At first order:

  E/I = ω_lin + (2/p)·⟨a^{p−2}⟩_{a²}/D

- The weight is ⟨F^p⟩/⟨F²⟩, exactly the envelope weighting the eikonal angle law uses.
- So E/I differs from the eikonal carried frequency (the action-weighted local frequency shift) by
  exactly **2/p**, at every width and in any dimension.
- Numerical check of this identity on lattice packets (algebra only, not the slope test):
  (E/I − ω_lin)/⟨δΩ⟩ = 0.6666, 0.5001 and 0.3333 for p = 3, 4 and 6.

**Result 3, stationary wave: a correction to the request's expectation.**
- **E/I does not reduce to the local frequency for a stationary nonlinear wave.** ℰ/𝒜 = Ω − 𝓛/𝒜 ≠ Ω,
  because 𝓛 ≠ 0 on-shell.
- What reduces exactly to Ω is the ratio of the conserved **fluxes**: (energy flux)/(action flux) = Ω,
  for linear and nonlinear waves alike. The stationary transmission problem conserves exactly those
  fluxes.

**The hypothesis H_W** (the formal version of the candidate). The segment's response to the
nonlinearity is set by the frequency that the conserved quantities fix:
- for a localised packet, the ratio of the conserved totals, E/I, which gives factor **2/p**;
- for a stationary wave, the ratio of the conserved fluxes, which gives factor **1**.

H_W reproduces all four measurements so far: 2/3, 1/2, and 1 and 1.

**What H_W does not supply:** a dynamical argument that the segment's rotation is governed by E/I.
The action-weighted mean of the field's temporal spectrum is the full shift (the self-phase rate
measured earlier, 0.01928 against 0.0192), not E/I. That link stays open, whatever the test below
gives.

## (2) The p = 6 test

**Packet.** Single segment (u(2), g = 0.12), widths 16 and 32. The effect is fourth order in amplitude,
so it is measured at A = 0.05 and 0.035. Predicted rotation changes, degrees:

| width | A | eikonal (1) | **H_W (1/3)** | 1/2 | 2/3 |
|---|---|---|---|---|---|
| 16 | 0.05 | 0.0001005 | **0.0000335** | 0.0000503 | 0.0000670 |
| 32 | 0.05 | 0.0001024 | **0.0000341** | 0.0000512 | 0.0000682 |
| 16 | 0.035 | 0.0000241 | **0.0000080** | | |
| 32 | 0.035 | 0.0000246 | **0.0000082** | | |

**Stationary wave.** Flat-top, A = 0.05, V = 6.25×10⁻⁶. H_W (flux ratio) predicts **0.0001775°**,
factor 1. "Totals everywhere" would give 1/3.

**Criteria:**
- **p = 6 packet gives 1/3:** factor 0.333 ± 0.03 at both widths.
- **Fourth order:** the A = 0.05/0.035 ratio of the rotation changes is (0.05/0.035)⁴ = 4.165 ± 0.10.
- **Direction:** cosine to the eikonal vector ≥ 0.99.
- **p = 6 stationary:** factor 1.00 ± 0.05.

**The derivation "holds"** (as a conservation-law account) if the packet gives 1/3 and the stationary
wave gives 1. The dynamical link noted above remains open even then.

## Correction pending for the record (unchanged)

When anomaly A is next recorded, the fccf291 reading "K4 flips the q = 3 u(3) split sign" is marked
**corrected**: the positive split comes mainly from the in-segment narrow-width term.
