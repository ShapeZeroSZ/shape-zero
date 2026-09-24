#!/usr/bin/env python3
"""
kappa_cross_compare.py -- the derived F (kappa_cross_pt.py, P3a and P3b) against
every measured F. Run only after the predictions were recorded
(kappa_cross_pt_output.txt).

Measured F = kappa / (kappa_pw fill) +- error, copied from the raw outputs:
  kappa_resolution_test_raw.txt        w/L = 1/4, w = 1..6
  kappa_widthscan_gpu_colab_raw.txt    w/L = 1/8, 1/12, 1/16
  kappa_extended_gpu_colab_raw.txt     w = 2, L = 12..80 ([0, 300] column)

SHARP, fixed in advance: error bar <= 0.25. MATCH: |pred - meas| <= 2 error bars.
"""

import kappa_cross_pt as PT

MEAS = {  # (w, L): (F, err, source)
    (1.0, 4): (1.15, 0.01, "resolution"), (2.0, 8): (1.23, 0.03, "resolution"),
    (3.0, 12): (1.32, 0.04, "resolution"), (4.0, 16): (1.33, 0.07, "resolution"),
    (5.0, 20): (1.29, 0.10, "resolution"), (6.0, 24): (1.34, 0.16, "resolution"),
    (1.5, 12): (2.01, 0.12, "widthscan"), (2.0, 16): (2.22, 0.23, "widthscan"),
    (3.0, 24): (2.38, 0.65, "widthscan"), (4.0, 32): (1.61, 0.53, "widthscan"),
    (2.0, 24): (2.81, 0.89, "widthscan"), (3.0, 36): (1.99, 0.83, "widthscan"),
    (4.0, 48): (2.78, 1.66, "widthscan"),
    (1.5, 24): (2.86, 0.86, "widthscan"), (2.0, 32): (2.03, 0.63, "widthscan"),
    (3.0, 48): (2.60, 1.85, "widthscan"), (4.0, 64): (4.82, 3.09, "widthscan"),
    (2.0, 12): (1.77, 0.08, "extended"), (2.0, 20): (2.50, 0.52, "extended"),
    (2.0, 28): (2.49, 1.13, "extended"), (2.0, 36): (2.22, 0.86, "extended"),
    (2.0, 40): (2.26, 1.16, "extended"), (2.0, 48): (2.43, 1.83, "extended"),
    (2.0, 64): (3.22, 3.64, "extended"), (2.0, 80): (4.59, 5.91, "extended"),
}
SHARP = 0.25


def main():
    print("=" * 86)
    print("DERIVED F vs MEASURED F   (pull = (pred - meas) / err;  * = sharp point, err <= 0.25)")
    print("=" * 86)
    print("      w     L      meas F          2 - s  pull     P3a diag  pull     P3b triads  pull")
    tally = {"2-s": [0, 0], "P3a": [0, 0], "P3b": [0, 0]}
    for (w, L) in PT.POINTS:
        F, e, _ = MEAS[(w, L)]
        fill, s = PT.geometry(L, w)
        preds = {"2-s": 2 - s, "P3a": PT.F_diag(L, w), "P3b": PT.F_triads(L, w)}
        sharp = e <= SHARP
        cells = []
        for k, p in preds.items():
            pull = (p - F) / e
            if sharp:
                tally[k][1] += 1
                tally[k][0] += abs(pull) <= 2
            cells.append(f"{p:6.3f} {pull:+6.1f}")
        print(f"   {'*' if sharp else ' '} {w:4.1f}  {L:3d}   {F:5.2f} +- {e:4.2f}    "
              + "    ".join(cells), flush=True)
    print()
    for k, (m, n) in tally.items():
        print(f"  {k:4s}: matches {m} of {n} sharp points")


if __name__ == "__main__":
    main()
