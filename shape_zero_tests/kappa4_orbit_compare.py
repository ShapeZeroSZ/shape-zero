#!/usr/bin/env python3
"""
kappa4_orbit_compare.py -- POST-HOC reading of kappa4_orbit_measure_output.txt,
written after the measurement; the committed test is kappa4_orbit_predictions.txt
(commit 9e2c4c5) and its pulls are in the measurement output.

The one known systematic of the L2 launch: it omits the third harmonic, so on the
plane wave (kappa4_orbit_validate_output.txt) it reads the growth from A = 0.10
short of the exact wave -- ratio 1.01177 against 1.01371 at 0.30, 1.02285 against
1.02610 at 0.40. How much of that a beam inherits is not derived, so each
hypothesis is given as a BAND between two limits: the beam's growth is r times
the plane wave's growth as the L2 launch delivers it (g_L2), or r times the
physical growth (g). A band that misses the measurement by more than two error
bars fails even with the systematic allowed for.

Only the L = 4 boxes are used: their kappa is independent of record length
(T = 300 and 900 agree), while w = 2, L = 8 changes by 2.5% between T = 300 and
900 even at A = 0.10 -- the slow secondary energy transfer seen in L >= 8 boxes
(kappa_side_gpu.py) -- which no fourth-order hypothesis is about.

usage:  python3 kappa4_orbit_compare.py
"""

import kappa4_predict as P4

KE = {0.10: -0.017510, 0.30: -0.017750, 0.40: -0.017967}      # exact wave
KL2 = {0.10: -0.017505, 0.30: -0.017711, 0.40: -0.017905}     # L2 launch, plane wave
# T = 900 beam ratios kappa_box(A)/kappa_box(0.10) and error bars (measurement output)
MEAS = {(1.0, 4): {0.30: (0.99990, 0.00174), 0.40: (0.99999, 0.00177)},
        (1.5, 4): {0.30: (1.00378, 0.00026), 0.40: (1.00717, 0.00028)},
        (2.0, 4): {0.30: (1.00618, 0.00016), 0.40: (1.01180, 0.00016)},
        (3.0, 4): {0.30: (1.00856, 0.00011), 0.40: (1.01648, 0.00012)}}


def main():
    print("=" * 90)
    print("POST-HOC: each hypothesis as a band allowing the L2 launch's third-harmonic residual")
    print("=" * 90)
    for A in (0.30, 0.40):
        gp = KE[A] / KE[0.10] - 1
        gl = KL2[A] / KL2[0.10] - 1
        print(f"\n  A = {A:.2f}: plane-wave growth from 0.10 -- physical {gp:.5f}, L2 launch {gl:.5f}")
        print("     w     L    measured              H0 band           S1 band            S2 band")
        for (w, L), m in MEAS.items():
            _, _, _, o = P4.predict(L, w)
            val, err = m[A]
            row = f"   {w:4.1f}  {L:3d}   {val:.5f} +- {err:.5f}  "
            for h in ("H0", "S1", "S2"):
                r = o[h][0]
                lo, hi = 1 + r * gl, 1 + r * gp
                miss = max(lo - val, val - hi, 0.0) / err
                row += f"  [{lo:.5f}, {hi:.5f}] {'ok' if miss <= 2 else f'x{miss:.0f}'}"
            print(row)


if __name__ == "__main__":
    main()
