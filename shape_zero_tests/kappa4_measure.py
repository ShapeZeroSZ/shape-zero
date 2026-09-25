#!/usr/bin/env python3
"""
kappa4_measure.py -- the out-of-sample test of kappa4_predictions.txt (commit
0e7f269): two never-measured large-fill beams, w = 3, L = 4 and w = 2, L = 4,
at A = 0.10 and 0.30, with the plane wave measured at each amplitude.

Physics, seed (linear, Fourier space), integrator, readout and error bar:
kappa_resolution_test.py, imported, with only the amplitude changed -- the
protocol behind every measured F. Validation: the plane wave must reproduce
-0.01755 (A = 0.10) and -0.01869 (A = 0.30).

F = kappa_box / (kappa_pw * fill), error bars combined in quadrature; the ratio
F(0.30)/F(0.10) carries both in quadrature (conservative: the two runs share
the geometry, so part of their error is common).

Criterion, set in advance (kappa4_predictions.txt): a hypothesis passes if its
predicted F(0.30)/F(0.10) is within two combined error bars of the measured one.

usage:  python3 kappa4_measure.py
"""

import math

import kappa_resolution_test as RT
import kappa4_predict as P

CASES = [(3.0, 4), (2.0, 4)]
REF_PW = {0.10: -0.01755, 0.30: -0.01869}


def main():
    print("=" * 84)
    print("OUT-OF-SAMPLE: large-fill beams at A = 0.10 and 0.30")
    print("=" * 84)
    res = {}
    ok = True
    for A in (0.10, 0.30):
        RT.A_NL = A
        tP, SP, _ = RT.run(8, None, "plane wave")
        kp, rp, _, ep = RT.kappa_of(tP, SP)
        good = abs(kp - REF_PW[A]) < 2e-5 and abs(rp - 1) < 2e-5
        ok &= good
        print(f"  VALIDATION A = {A:.2f}: plane wave {kp:+.5f} (ref {REF_PW[A]:+.5f})   "
              f"{'PASS' if good else 'FAIL'}", flush=True)
        for w, L in CASES:
            t, S, d = RT.run(L, w, "localised")
            k, r, _, e = RT.kappa_of(t, S)
            fill, _ = RT.geometry(L, w)
            F = k / (kp * fill)
            Fe = abs(F) * math.hypot(e / abs(k), ep / abs(kp))
            res[(w, L, A)] = (k, e, F, Fe, r)
    if not ok:
        print("  VALIDATION FAILED -- stopping.")
        return
    print("\n     w     L     A     kappa_box              lin ratio   F")
    for (w, L, A), (k, e, F, Fe, r) in res.items():
        print(f"   {w:4.1f}  {L:3d}  {A:4.2f}   {k:+.6f} +- {e:.6f}   {r:.6f}   {F:.4f} +- {Fe:.4f}")
    print("\n  AGAINST THE COMMITTED PREDICTIONS")
    print("     w     L    measured F(0.30)   H0      S1      S2     |  measured ratio      H0      S1      S2")
    for w, L in CASES:
        _, _, F2, o = P.predict(L, w)
        _, _, F1, F1e, _ = res[(w, L, 0.10)]
        _, _, F3, F3e, _ = res[(w, L, 0.30)]
        ratio = F3 / F1
        re = ratio * math.hypot(F1e / F1, F3e / F3)
        line = f"   {w:4.1f}  {L:3d}   {F3:.4f} +- {F3e:.4f}  "
        line += "  ".join(f"{o[h][1]:.4f}" for h in ("H0", "S1", "S2"))
        line += f"  |  {ratio:.4f} +- {re:.4f}  "
        line += "  ".join(f"{o[h][2]:.4f}" for h in ("H0", "S1", "S2"))
        print(line)
        verdict = []
        for h in ("H0", "S1", "S2"):
            pull = (o[h][2] - ratio) / re
            verdict.append(f"{h} {pull:+.1f} {'PASS' if abs(pull) <= 2 else 'FAIL'}")
        print("        ratio pulls: " + "   ".join(verdict)
              + f"    (F2 = {F2:.4f}; measured F(0.10) = {F1:.4f} +- {F1e:.4f})")


if __name__ == "__main__":
    main()
