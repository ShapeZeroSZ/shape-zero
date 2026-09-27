#!/usr/bin/env python3
"""
cpg_pilot.py -- CP-G pilot: is every field of main gapped, and how far can a force reach?
Predictions committed first: natural/CPG_PILOT_PREDICTIONS.md (45c4c8f).

P1  Mode census from MAIN's force: 04_scripts/session/model.py (identical to main), linearised about
    its vacuum by numerical Jacobians (amplitude 1e-7), u'' = A u + B u'; frequencies from the
    first-order system. Smallest |omega| per sector over the lattice's Brillouin zone.
P2  Range: the static response of main's force to a point source. (a) At side 12 (q = 3), solve
    A u = -s with A from model.py's own force and compare with the FFT of 1/Q(k) -- the check that
    main's static operator is Q. (b) At side 64, fit ln(r G) along an axis for the decay length xi.
"""

import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M

EPS = 1e-7


def jacobians(lat):
    n = lat.N * lat.D
    zero = np.zeros((lat.N, lat.D))
    A = np.zeros((n, n)); B = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = EPS
        E = e.reshape(lat.N, lat.D)
        A[:, i] = (lat.force(E, zero) - lat.force(-E, zero)).ravel() / (2 * EPS)
        B[:, i] = (lat.force(zero, E) - lat.force(zero, -E)).ravel() / (2 * EPS)
    return A, B


def min_omega(lat):
    A, B = jacobians(lat)
    n = A.shape[0]
    Mx = np.block([[np.zeros((n, n)), np.eye(n)], [A, B]])
    lam = np.linalg.eigvals(Mx)
    growth = np.max(lam.real)
    return float(np.min(np.abs(lam.imag))), float(growth)


def p1():
    print("P1  mode census from model.py's linearised force (smallest |omega| over the zone)")
    print("    sector                                         N    smallest |w|   predicted   max Re(lambda)")
    cases = [
        ("J sector n=1, kappa*, q=1", dict(n=1, N=200), 1.0864),
        ("J sector n=2, kappa*, q=1", dict(n=2, N=60), 1.0864),
        ("J sector n=3, kappa*, q=1", dict(n=3, N=40), 1.0864),
        ("J sector n=1, kappa*, q=3 (side 6)", dict(n=1, N=216, q=3, shape=6), 1.0864),
        ("scalar beta sector, beta=0.05, q=1", dict(n=1, N=200, kappa=0.0, gyro_scalar=0.05), 1.4935),
        ("kappa=0 nodes, n=8 (tower configuration), q=1", dict(n=8, N=12, kappa=0.0), 1.4953),
    ]
    worst = 1e9
    for name, kw, pred in cases:
        lat = M.Lattice(**kw)
        w, g = min_omega(lat)
        worst = min(worst, w)
        print(f"    {name:46s} {lat.N:4d}   {w:.4f}        {pred:.4f}      {g:+.1e}", flush=True)
    print(f"    global smallest |omega| = {worst:.4f}  -> "
          f"{'NO gapless mode' if worst > 1e-3 else 'a GAPLESS mode exists'}")


def p2():
    print("\nP2  range of a static force from main's static operator")
    # (a) side 12: main's own static operator vs 1/Q
    lat = M.Lattice(n=1, N=12 ** 3, q=3, shape=12)
    A, _ = jacobians(lat)
    s = np.zeros((lat.N, lat.D)); s[0, 0] = 1.0
    u = np.linalg.solve(A, -s.ravel()).reshape(lat.N, lat.D)[:, 0].reshape(12, 12, 12)
    k = 2 * np.pi * np.fft.fftfreq(12)
    kk = np.meshgrid(k, k, k, indexing="ij")
    Q = M.SQ5 + 2 * M.C * sum(1 - np.cos(x) for x in kk)
    G12 = np.fft.ifftn(1.0 / Q).real
    print(f"    (a) side 12: max |u_main - FFT(1/Q)| = {np.max(np.abs(u - G12)):.1e}  "
          f"(main's static operator is Q)")
    # (b) side 64: decay length along an axis
    L = 64
    k = 2 * np.pi * np.fft.fftfreq(L)
    kk = np.meshgrid(k, k, k, indexing="ij")
    Q = M.SQ5 + 2 * M.C * sum(1 - np.cos(x) for x in kk)
    G = np.fft.ifftn(1.0 / Q).real
    r = np.arange(3, 9)
    g = np.array([G[i, 0, 0] for i in r])
    slope = np.polyfit(r, np.log(r * g), 1)[0]
    xi_pred = 1 / math.acosh(1 + M.SQ5 / (2 * M.C))
    print(f"    (b) side 64: fitted decay length xi = {-1/slope:.4f} sites  (predicted {xi_pred:.4f}, "
          f"{100*(-1/slope/xi_pred-1):+.1f}%)")
    print(f"        G(r)/G(1) along the axis: r = 2: {G[2,0,0]/G[1,0,0]:.2e}, 5: {G[5,0,0]/G[1,0,0]:.2e}, "
          f"10: {G[10,0,0]/G[1,0,0]:.2e}")
    print(f"        single-field exchange: Yukawa, range {-1/slope:.2f} sites; "
          f"two-field (fluctuation-induced): range <= {-1/(2*slope):.2f} sites")


if __name__ == "__main__":
    print("=" * 78)
    print("CP-G PILOT -- is every field of main gapped? (predictions: CPG_PILOT_PREDICTIONS.md)")
    print("=" * 78)
    p1(); p2()
