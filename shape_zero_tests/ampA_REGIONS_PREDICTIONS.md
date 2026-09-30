# Anomaly A — where the outside factor lives, what the in-segment term is: predictions

*Committed 2026-09-30, before any run.*

Script: `ampA_regions.py` (`predict` has already been run: model numbers are in
`ampA_regions_predictions.txt` and `.json`; `run` and `evaluate` come later). Design choices are ours.

## Disclosures
- Everything measured so far is known: the recorded slopes (fccf291) and the masked IN/OUT/ALL
  runs (fb51f1b). In particular:
  - the OUT factor is 0.58–0.65;
  - the q = 3 IN term is 29–47% of ALL;
  - the q = 1 IN term is 4–10% of ALL.
- I saw the model outputs before committing.
- Tolerances were set now.
- The width scan (item 3b) is ours; you did not ask for it.
- The IN, OUT, ALL and linear runs of the certification geometry are reused from
  `ampA_masked_runs.json`: same code, same geometry.

## (3) First-principles re-derivation: how the self-phase becomes the frequency carried into a segment

**Setup.** At first order, A′ adds to the a-branch Hamiltonian the perturbation δH = V/(2Ω + κ), with
V = |u₀| the local node radius. Inside a segment the perturbation depends on direction:
δH_j = V/(2Ω + κ + 2c g w h_j sin k), which is ≈ (V/D)(1 − g w h_j v₀).

**Eikonal theorem.** The first-order phase change is δS_j = −∫δH dt along the unperturbed ray of
channel j that ends at the readout point.

**Channel geometry.**
- A segment splits the packet into eigen-channels with group delays
  τ_j = Σ_sites g w h_j/(1 + t_j²) (from dk_j/dΩ).
- The rays of different channels that end at the same point coincide after the segment.
- Before the segment, a delayed channel's ray starts ahead by v₀τ_j. So it crosses every earlier
  boundary τ_j sooner.

**Result per region R.** Region R contributes τ_j·[V(t_out,R) − V(t_in,R)]/D to the relative phase.
The launch region contributes τ_j·V(t_exit)/D.

Inside a channel's own segment, the dwell excess and the direction-dependent δH cancel at O(g),
exactly for eigenvalues ±1. The regions therefore contribute:

| region | segment 1's channels | segment 2's channels |
|---|---|---|
| BEFORE (x < segment-1 start) | V(t₁ in) | V(t₁ in) |
| IN | 0 | V(t₁ out) − V(t₁ in) |
| BETWEEN | 0 | V(t₂ in) − V(t₁ out) |
| AFTER | 0 | 0 (E3) |
| total | V(t₁ in) | V(t₂ in) |

The total is the angle law.

**Weighting over the envelope.**
- The relative phase is weighted by the coherence F_j F_j′ ≈ F² at the readout point, so the
  first-order weight is **⟨F³⟩/⟨F²⟩ at each boundary-crossing time**:
  - 1-D Gaussian: √(2/3) A_peak = 0.816 A_peak;
  - 3-D Gaussian: (2/3)^{3/2} A_peak = 0.544 A_peak.
- The same result holds in 1-D and 3-D. The weight is a property of the envelope, not of the
  dimension.
- The chirp term, v₀τ ∫∂_ξV dt weighted by F², vanishes for symmetric envelopes.

**Predicted size factor** (measured/derived, projected onto the derived direction): **1.00 in 1-D and
in 3-D**, in OUT and in ALL.

**Stationary limit.** For a stationary plane wave the angle law is exact: the transfer matrix gives a
ratio of 1.0000 (K3b, 8b61310). So a factor different from 1 in the dynamics can only come from what
separates a transient packet from a stationary wave.

### (3b) Width scan (ours)

Single segment, u(2), g = 0.12. Predicted rotation change, degrees per 10⁻³:

| width | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| predicted | 0.01556 | 0.02165 | 0.02306 | 0.02318 |

Predicted factor: 1.00 at every width. Accounted for if 0.85 ≤ factor ≤ 1.15.

How to read the outcome:
- If the factor is about 0.6 at every width, including 32, the eikonal conversion fails for transient
  packets by a width-independent factor. Since the stationary limit is exact, that would be a further
  mechanism in the dynamics.
- If the factor moves toward 1 as the width grows, it is a non-eikonal width effect.
- If the factor is 1 here, the 0.6 belongs to the two-segment certification geometry.

## (1) OUT split into BEFORE, BETWEEN and AFTER (q = 1 and q = 3, certification geometry)

**Predicted:**
- **AFTER = 0**, as E3 requires: |dco_AFTER| ≤ 0.05 |dco_ALL|.
- **BEFORE carries the angle law.** Model magnitudes, deg per 10⁻³:
  - q = 1: 0.0272 / 0.0222 (u(2)) and 0.0289 / 0.0334 (u(3));
  - q = 3: 0.0166 / 0.0136 and 0.0162 / 0.0162.
- **BETWEEN:**
  - small from ray bookkeeping: q = 1 ≤ 0.0001; q = 3 0.0008–0.0013;
  - plus K4's between-segment part in the instantaneous model: q = 3 0.0016–0.0109, q = 1 0.0013–0.0025.
- Additivity: BEFORE + BETWEEN + AFTER = OUT within 5%, as a check.
- Where the factor lives: under the derivation, the BEFORE factor (projection onto the model's BEFORE
  vector plus K4 pre) is 1.00 ± 0.15.
  - If BEFORE is about 0.6, the conversion fails where the angle law is generated.
  - If BEFORE is about 1 while BETWEEN or AFTER carry a contribution of opposite sign, the missing 0.4
    is a mechanism located there.

## (2) q = 3, transversely uniform packet: IN only (with linear and ALL)

- **Transverse hypothesis** (your framing): if the q = 3 in-segment term comes from the packet's
  transverse structure, it vanishes here: |dco_IN| ≤ 0.15 |dco_ALL|, as at q = 1.
- **Scalar ray bookkeeping** predicts a tiny IN for this packet too (≤ 0.001 against ALL about 0.024),
  and also for the transversely structured packet (0.0002–0.0004). So the ray bookkeeping **does not
  produce** the measured q = 3 IN term (29–47%).
- **Reading:**
  - IN ≈ 0 here means the in-segment term needs transverse structure, i.e. it comes from the
    transverse (Qt ≠ 0) modes.
  - IN ≠ 0 here means it is not transverse. The transversely uniform slab is exactly the 1-D problem
    with x-width 3, so it would then be a narrow-width (non-eikonal) effect.

## Correction pending for the record

When anomaly A is next recorded, the reading in fccf291's report, "K4 flips the q = 3 u(3) split
sign", is marked **corrected**: the masked runs show that the positive q = 3 u(3) split comes mainly
from the in-segment contribution (IN +0.0069 against OUT +0.0008), not from K4.
