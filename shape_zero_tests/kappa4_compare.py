#!/usr/bin/env python3
"""
kappa4_compare.py -- the committed predictions (kappa4_predictions.txt, commit
0e7f269) against every sharp point measured at A = 0.30.

Measured box kappa and error bar, verbatim from the raw outputs:
  kappa_resolution_test_raw.txt     w = 1..6 at w/L = 1/4
  kappa_widthscan_gpu_colab_raw.txt w = 1.5, L = 12 and w = 2, L = 16
  kappa_extended_gpu_colab_raw.txt  w = 2, L = 12
F = kappa / (kappa_pw * fill), kappa_pw = -0.01869, error e / |kappa_pw * fill|,
as those scripts compute it. Pull = (pred - meas) / err; match = |pull| <= 2.

usage:  python3 kappa4_compare.py
"""

import kappa4_predict as P

MEAS = {(1.0, 4): (-0.00415, 0.00004), (2.0, 8): (-0.00447, 0.00011),
        (3.0, 12): (-0.00478, 0.00013), (4.0, 16): (-0.00482, 0.00024),
        (5.0, 20): (-0.00470, 0.00036), (6.0, 24): (-0.00487, 0.00060),
        (1.5, 12): (-0.00184, 0.00011), (2.0, 16): (-0.00204, 0.00021),
        (2.0, 12): (-0.00289, 0.00014)}
KPW = -0.01869


def main():
    print("=" * 84)
    print("COMMITTED PREDICTIONS vs SHARP POINTS at A = 0.30   (pull = (pred - meas)/err)")
    print("=" * 84)
    print("     w     L     meas F          F2 (A->0)  pull   |  H0     pull  |  S1     pull  |  S2     pull")
    tally = {"F2": 0, "H0": 0, "S1": 0, "S2": 0}
    for (w, L), (k, e) in MEAS.items():
        fill, s, F2, o = P.predict(L, w)
        F = k / (KPW * fill)
        Fe = e / abs(KPW * fill)
        row = f"   {w:4.1f}  {L:3d}   {F:.3f} +- {Fe:.3f}    {F2:.3f}  {(F2-F)/Fe:+5.1f}  "
        tally["F2"] += abs((F2 - F) / Fe) <= 2
        for h in ("H0", "S1", "S2"):
            pr = o[h][1]
            pull = (pr - F) / Fe
            tally[h] += abs(pull) <= 2
            row += f"|  {pr:.3f} {pull:+5.1f}  "
        print(row)
    n = len(MEAS)
    print(f"\n  matched within 2 error bars:  F2 alone {tally['F2']}/{n}   H0 {tally['H0']}/{n}   "
          f"S1 {tally['S1']}/{n}   S2 {tally['S2']}/{n}")


if __name__ == "__main__":
    main()
