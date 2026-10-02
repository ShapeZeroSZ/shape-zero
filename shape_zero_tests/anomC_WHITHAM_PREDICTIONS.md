# Re-ranking the open items in light of the Whitham result; anomaly C tested — predictions

*Committed 2026-10-02, before any run.*

Script: `anomC_whitham.py`. `predict` has been run; its output is `anomC_whitham_predictions.txt` and
`.json`. Design choices are ours.

## Disclosures
- The recorded anomaly-C reading is known (`kappa4_orbit3_reading_output.txt`, post hoc in its own
  right): fractions **+0.33, +0.50, +0.68** at w = 1.5, 2, 3 (L = 4), and −0.05 (not discriminating)
  at w = 1, with the post-hoc pattern "≈ 0.9 × fill".
- So at w = 1, 1.5, 2 and 3 these predictions are **not out of sample**. w = 2.5 and w = 4 (L = 4) have
  never been run and are the out-of-sample test.
- I saw the model numbers before committing.
- scipy was installed in this container to run the existing anomaly-C code (`kappa_pw4_pt.py`).
- A timing run reproduced the recorded w = 3, A = 0.10, T = 300 value (−0.012975).

## (1) Does C's width-dependent fraction follow from the totals/fluxes rule? Derivation

**What the beam is.** It is extended and time-steady along propagation (a box-periodic travelling
wave whose transverse structure fills an L = 4 box). In Whitham terms it is a **stationary
multi-phase wave**: each transverse component is a plane wave of its own phase.

**What the rule says:**
- Stationary waves are governed by the conserved **fluxes**, whose ratio is the frequency.
- So the measured κ_box is the beam's true frequency asymmetry at every order, and **no 2/p factor
  enters C**.
- A totals (E/I) reading would put a factor 2/p on each order: 1/2 on the second (quartic) and 1/3 on
  the fourth (sextic). That is **already excluded** by the recorded second order, where F₂ from
  frequency perturbation theory matches the beams (P3b, MODEL_SPEC §5).

**What the rule's envelope weighting gives for the fourth-order fraction:**
- **W-multi:** Whitham averaging over independent phases gives the **dephased** sextic moment, which
  is S1's definition. So **f = 1** at every width.
- **W-locked:** a single common phase (a phase-locked beam), weighted by the actual envelope, gives
  X = ⟨e⁶⟩/⟨e²⟩³. This puts f at **2.2–4.6**, beyond S1.

**Conclusion of the derivation (committed before running):**
- The rule fixes the *ratio* question (fluxes, no 2/p).
- Its envelope weighting gives only the two coherence limits, f = 1 and f ≈ 2.2.
- It **does not produce** a fraction between S2 and S1. C's fraction needs input the conservation laws
  do not supply: which fourth-order cross terms are resonant.
- If any rule-derived candidate fits the out-of-sample widths, this conclusion is refuted.

## Predicted fractions (f = 0 is S2, f = 1 is S1)

| w | fill | W-multi | W-locked | K-transfer (ours) | 0.9 × fill (post-hoc pattern) |
|---|---|---|---|---|---|
| 1.0 | 0.192 | 1.000 | 4.642 | 0.691 | 0.173 |
| 1.5 | 0.376 | 1.000 | 2.403 | 0.664 | 0.338 |
| 2.0 | 0.535 | 1.000 | 2.183 | 0.655 | 0.481 |
| 2.5 | 0.653 | 1.000 | 2.166 | 0.652 | 0.587 |
| 3.0 | 0.736 | 1.000 | 2.185 | 0.651 | 0.662 |
| 4.0 | 0.836 | 1.000 | 2.228 | 0.650 | 0.753 |

**K-transfer** is ours, not from the rule. It carries the second-order cross-term survival
f₂ = (F₂ − P₂)/(2 − 2P₂) to fourth order.

**Runs.** The recorded L3 launch (`kappa4_orbit3_launch.run_case`), L = 4, T = 900, A = 0.10, 0.30 and
0.40, at all six widths. The measured fraction comes from r = (ratio − 1)/(g(A) − ratio·g(0.10)), both
amplitudes, with σ propagated from the κ error bars.

**Criterion.** A candidate fits a width if |f_pred − f_meas| ≤ 2σ_f at both amplitudes. At w = 4 the
S1−S2 gap is small (Δr = 0.021), so σ_f will be large there. That is reported, not hidden.

## (2) Re-ranking (a priori, committed now; revisited after the runs)

1. **A, open item 1 (a mechanism linking E/I to the segment rotation): first.**
   - The Whitham result points here directly. The same field obeys the frequency (stationary,
     factor 1) or E/I (packet, 2/p), and the two regimes are sharply separated: factor 2/p is
     width-independent from 8 to 32, yet the long flat-top read in its plateau gives 1.
   - Decisive tests are cheap:
     - read the same flat-top run with the whole-packet readout instead of the plateau window;
     - scan the plateau length.
   - Totals predict that the whole-packet readout moves toward 2/p as the edges' share grows, while
     the plateau stays at 1.
2. **D (tower heating's uneven deceleration, and its suppression by mean size): second.**
   - The populated-tower packets are localised, so the totals regime applies to them.
   - The suppression by the upper levels' mean size is the kind of quantity a conserved-totals
     account can address.
   - But the rule gives no explicit prediction for D without a derivation that has not been done.
3. **C: third.** The rule's content for C is negative: no 2/p factor, and coherence limits that bracket
   but do not produce the fraction. C stays a fourth-order kernel calculation.
4. **A, open item 2 (the narrow-packet in-segment term): last.** Whitham theory is the slow-modulation
   (eikonal) limit, so by construction it cannot address a narrow-packet effect.
