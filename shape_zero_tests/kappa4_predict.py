#!/usr/bin/env python3
"""
kappa4_predict.py -- predictions of F at A = 0.30 (and F(0.30)/F(0.10)) from the
fourth-order analysis. Written and committed BEFORE any comparison.

WHAT IS DERIVED (kappa_pw4_pt.py, kappa_pw4_seed.py, kappa_seed2_test.py):
  * the plane-wave ORBIT: kappa(A) = -0.017480 - 0.002947 A^2 + ... (exact
    harmonic balance; an orbit-seeded run reproduces it to every digit);
  * the measured plane-wave growth (linear-seed protocol: -0.01755, -0.01792,
    -0.01869 at A = 0.1, 0.2, 0.3) is mostly NOT the orbit: the linear seed
    leaves out the forced static shift and second harmonic. With them added
    (second-order seed) the plane wave gives -0.01787 +- 0.00006 at 0.30,
    against orbit + derived velocity renormalisation -0.01791;
  * the narrow beams' box kappa barely moves with amplitude or seed.

F divides the box kappa by the PLANE-WAVE kappa measured with the same protocol,
so F(A) = F2 * (kappa2 / kappa_pw(A)) * (1 + g_box(A)), where
  F2        second-order F (kappa_cross_pt.py P3b, A -> 0, [0, 300] readout),
  kappa2    -0.017480 (derived),
  kappa_pw  the protocol's plane-wave kappa (MEASURED on the plane wave only:
            -0.01755 at 0.10, -0.01869 at 0.30), g_pw(A) = kappa_pw/kappa2 - 1,
  g_box     the beam's own relative growth, = r * g_pw(A), with r from
            the geometry. Every fourth-order term is quartic in the component
            amplitudes |a_j|^2 = (A^2/4)|c_j|^2, so relative to the second-order
            term it carries one extra factor of the fill; the kernel is not
            derived here, so two limits and a null are stated:
    H0  r = 0                       no beam growth (the hypothesis as given)
    S1  r = fill (6 - 3 P2 - 6 s + 4 s^2) / F2
                                    incoherent local sextic: all quartic terms
                                    survive, with the combinatorics of <|psi|^6>
                                    over dephased components (= 1 for a plane wave)
    S2  r = fill s^2 / F2           self only: only the box-wide component's own
                                    fourth-order term, weight |c0|^4 = (fill s)^2
  s = |c0|^2 / fill, P2 = sum p_j^2 with p_j = |c_j|^2 / fill.
Caveat: the narrow-beam data (w/L = 1/4) were SEEN before these were written and
lean toward S2/H0; that preference is post hoc. The new configurations below are
the out-of-sample test.

usage:  python3 kappa4_predict.py > kappa4_predictions.txt
"""

import numpy as np

import kappa_cross_pt as X

KAPPA2 = -0.017480
KPW = {0.10: -0.01755, 0.30: -0.01869}          # plane wave, protocol, measured

SHARP = [(1.0, 4), (2.0, 8), (3.0, 12), (4.0, 16), (5.0, 20), (6.0, 24),
         (1.5, 12), (2.0, 16), (2.0, 12)]
NEW = [(3.0, 4), (2.0, 4)]


def geom(L, w):
    e = X.env2d(L, w)
    c = np.fft.fft2(e) / (L * L)
    p = np.abs(c) ** 2
    fill = float(p.sum())
    pn = p / fill
    return fill, float(pn[0, 0]), float((pn ** 2).sum())


def predict(L, w):
    fill, s, P2 = geom(L, w)
    F2 = X.F_triads(L, w)
    r = {"H0": 0.0,
         "S1": fill * (6 - 3 * P2 - 6 * s + 4 * s * s) / F2,
         "S2": fill * s * s / F2}
    out = {}
    for h, rr in r.items():
        F = {}
        for A, kp in KPW.items():
            g = kp / KAPPA2 - 1
            F[A] = F2 * (KAPPA2 / kp) * (1 + rr * g)
        out[h] = (rr, F[0.30], F[0.30] / F[0.10])
    return fill, s, F2, out


def main():
    print("=" * 86)
    print("PREDICTIONS -- F at A = 0.30 and F(0.30)/F(0.10); written before comparison")
    print("=" * 86)
    print(f"  kappa2 = {KAPPA2}; plane-wave protocol kappa: 0.10 {KPW[0.10]}, 0.30 {KPW[0.30]}")
    print(f"  kappa2/kappa_pw(0.30) = {KAPPA2/KPW[0.30]:.4f}")
    for title, pts in (("SHARP POINTS (measured at A = 0.30, width scan / resolution test)", SHARP),
                       ("NEW CONFIGURATIONS (never measured; to be measured at A = 0.10 and 0.30)", NEW)):
        print(f"\n  {title}")
        print("     w     L    fill     s      F2(P3b)  |   r: H0    S1     S2   |"
              "  F(0.30): H0     S1      S2    |  F(.30)/F(.10): H0     S1      S2")
        for w, L in pts:
            fill, s, F2, o = predict(L, w)
            print(f"   {w:4.1f}  {L:3d}  {fill:.4f}  {s:.3f}   {F2:.4f}  |  "
                  f"{o['H0'][0]:.3f}  {o['S1'][0]:.3f}  {o['S2'][0]:.3f}  |  "
                  f"{o['H0'][1]:.4f}  {o['S1'][1]:.4f}  {o['S2'][1]:.4f}  |  "
                  f"{o['H0'][2]:.4f}  {o['S1'][2]:.4f}  {o['S2'][2]:.4f}", flush=True)
    print("\n  Criteria set in advance for the NEW configurations: a hypothesis passes if its")
    print("  F(0.30)/F(0.10) is within two combined error bars of the measured ratio; the")
    print("  absolute F(0.30) is reported against each but also carries F2's own error.")


if __name__ == "__main__":
    main()
