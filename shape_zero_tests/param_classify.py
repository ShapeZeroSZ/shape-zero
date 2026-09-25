#!/usr/bin/env python3
"""
param_classify.py -- evidence for classifying the lattice parameters c and beta
(INPUT_LEDGER.md §2d, "kind" column).

THE RESCALING ARGUMENT (for c). The lattice equation, in model units, is
    u'' = -(sqrt5 u + u^2) + c lap u + beta c (v[n-1] - v[n+1])
with the phi-well coefficients sqrt5 and 1 fixed by the force law -(x^2 - x - 1)
(MODEL_SPEC §1). Rescale time t = tau / lam and amplitude u = a w:
    w'' = -(sqrt5/lam^2) w - (a/lam^2) w^2 + (c/lam^2) lap w + (beta c/lam) (...)
Keeping the well's two coefficients fixed forces lam = 1 and a = 1: the well
fixes both the time and the amplitude unit, so no rescaling is left to absorb
c. On a lattice the spacing is one site and cannot be rescaled either. So
c/sqrt5 is a dimensionless parameter the lattice physics can depend on. (In
the long-wavelength continuum limit c(1 - cos k) -> c k^2/2 and c could be
absorbed into the length unit; the results here are at k = pi/2, far from it.)
The table below shows the dependence directly, on the derived plane-wave kappa.

THE beta CHECK. The pinned asymmetry is exactly 2 c beta sin k for every beta
(Prove2Me missions 3, 4b; MODEL_SPEC §3), and kappa is beta-independent at
leading order. The table gives the derived kappa at three beta values to show
how far that holds.

Both tables use kappa_pw4_pt.py's harmonic balance (orbit, not a launch).

usage:  python3 param_classify.py
"""

import math

import kappa_pw4_pt as P


def main():
    print("=" * 78)
    print("PARAMETER CLASSIFICATION -- derived plane-wave kappa, k = pi/2")
    print("=" * 78)
    c0, b0 = P.C, P.BETA
    print("\n  kappa against the elastic coupling c (beta = 0.05):")
    print("     c      c/sqrt5    W(+K)       kappa2       kappa4")
    for c in (0.5, 1.0, 2.0):
        P.C, P.BETA = c, b0
        P.TH = abs(2 * c * b0 * math.sin(P.K))
        _, k2, k4 = P.kappa_analytic()
        print(f"   {c:4.1f}    {c / P.SQ5:.3f}    {P.W_lin(+1):.5f}    {k2:+.6f}    {k4:+.6f}")
    print("\n  kappa against the lattice gyroscopic coupling beta (c = 1):")
    print("     beta     kappa2       relative to beta = 0.05")
    ref = None
    for b in (0.05, 0.02, 0.10, 0.20):
        P.C, P.BETA = c0, b
        P.TH = abs(2 * c0 * b * math.sin(P.K))
        k2 = P.kappa_analytic()[1]
        ref = k2 if ref is None else ref
        print(f"    {b:5.2f}    {k2:+.6f}    {100 * (k2 / ref - 1):+.2f}%")
    P.C, P.BETA = c0, b0
    P.TH = abs(2 * c0 * b0 * math.sin(P.K))
    print("\n  reading: kappa depends on c (c/sqrt5 is a physical ratio); kappa is")
    print("  beta-independent at leading order only -- an O(beta^2) drift of a few")
    print("  percent over beta = 0.02-0.20, inside the measured beta-sweep's spread")
    print("  (|D/D0| = 0.998339, 0.998316, 0.998390, 0.998316; MODEL_SPEC §5).")


if __name__ == "__main__":
    main()
