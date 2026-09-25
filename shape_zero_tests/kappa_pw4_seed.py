#!/usr/bin/env python3
"""
kappa_pw4_seed.py -- does the SEED move the plane-wave kappa at fourth order?

kappa_pw4_pt.py derives the exact travelling-wave orbit's kappa(A) = kappa2 +
kappa4 A^2 with kappa4 = -0.00295. The measured plane-wave kappa grows much
faster (-0.01755, -0.01792, -0.01869 at A = 0.1, 0.2, 0.3; kappa_cross_amplitude
_output.txt). The measurement starts from the LINEAR seed (u = A cos, v on the
linear branch, no harmonics), which differs from the orbit at O(A^2).

This script integrates the plane wave on a 1-D ring of N = 8 sites (a
transverse-uniform wave on the side-8 cube is exactly this problem) with the
integrator and readout of kappa_resolution_test.py -- RK4, dt = 0.01, record
every 0.1, T = 300, uniform readout of mode N/4, the same kappa_of -- from:
  LIN   the linear seed (must reproduce kappa_resolution_test.py's plane wave)
  ORB   the exact harmonic-balance orbit with fundamental amplitude A
  ORB0  the orbit's fundamental only, with the orbit frequency (harmonics and
        static shift removed) -- isolates the harmonic content from the
        velocity mismatch
and prints kappa(A) for each against the orbit prediction.

usage:  python3 kappa_pw4_seed.py
"""

import math
import numpy as np

import kappa_resolution_test as RT
import kappa_pw4_pt as P

N = 8
PHI = RT.PHI


def force(x, v):
    return (-(x * x - x - 1.0) + RT.C * (np.roll(x, 1, 1) + np.roll(x, -1, 1) - 2 * x)
            + RT.BETA * RT.C * (np.roll(v, 1, 1) - np.roll(v, -1, 1)))


def integrate(x, v):
    """x, v: (cases, N). Returns t, S with S[:, j] the mode-N/4 series of case j."""
    m = N // 4
    h = RT.DT
    n = int(round(RT.T_RUN / h))
    t_rec = [0.0]
    rec = [np.fft.fft(x, axis=1)[:, m] / N]
    for step in range(n):
        k1v = force(x, v); k1x = v
        x2 = x + 0.5 * h * k1x; v2 = v + 0.5 * h * k1v
        k2v = force(x2, v2); k2x = v2
        x3 = x + 0.5 * h * k2x; v3 = v + 0.5 * h * k2v
        k3v = force(x3, v3); k3x = v3
        x4 = x + h * k3x; v4 = v + h * k3v
        k4v = force(x4, v4); k4x = v4
        x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        if (step + 1) % RT.REC_EVERY == 0:
            t_rec.append((step + 1) * h)
            rec.append(np.fft.fft(x, axis=1)[:, m] / N)
    return np.array(t_rec), np.array(rec)


def seed_lin(A, s):
    n = np.arange(N)
    W = P.W_lin(s)
    return PHI + A * np.cos(s * RT.K * n), s * 0 + A * W * np.sin(s * RT.K * n)


def seed_orbit(A, s, harmonics=True):
    sol = P.hb_exact(s, A)
    W = sol[0]
    c = np.zeros(P.M_HARM + 1)
    c[0], c[1], c[2:] = sol[1], A / 2, sol[2:]
    th = s * RT.K * np.arange(N)
    if not harmonics:
        c = c.copy(); c[0] = 0; c[2:] = 0
    u = c[0] + sum(2 * c[m] * np.cos(m * th) for m in range(1, P.M_HARM + 1))
    # u = U(theta), theta = s K n - W t  ->  du/dt = -W U'(theta)
    v = sum(2 * m * W * c[m] * np.sin(m * th) for m in range(1, P.M_HARM + 1))
    return PHI + u, v


def kappa_for(seedfn, A, **kw):
    """kappa from the four cases (lin +, lin -, nl +, nl -), as kappa_of expects."""
    xs, vs = [], []
    for a in (RT.A_LIN, A):
        for s in (+1, -1):
            x, v = seedfn(a, s, **kw)
            xs.append(x); vs.append(v)
    RT.A_NL = A
    t, S = integrate(np.array(xs), np.array(vs))
    k, r, res, e = RT.kappa_of(t, S)
    return k, e, r


def main():
    print("=" * 74)
    print("PLANE-WAVE KAPPA: linear seed vs exact orbit (1-D ring, N = 8, T = 300)")
    print("=" * 74)
    _, k2, k4 = P.kappa_analytic()
    print("     A     LIN seed            ORB (exact orbit)   ORB0 (fund. only)   orbit theory")
    for A in (0.10, 0.20, 0.30, 0.40):
        kl, el, _ = kappa_for(seed_lin, A)
        ko, eo, _ = kappa_for(seed_orbit, A)
        k0, e0, _ = kappa_for(seed_orbit, A, harmonics=False)
        kth = P.kappa_exact(A)[0]
        print(f"   {A:4.2f}   {kl:+.5f}+-{el:.5f}   {ko:+.5f}+-{eo:.5f}   "
              f"{k0:+.5f}+-{e0:.5f}   {kth:+.5f}", flush=True)
    print("\n  measured on the side-8 cube (kappa_cross_amplitude_output.txt, LIN seed):")
    print("     A = 0.10 -0.01755   0.20 -0.01792   0.30 -0.01869")


if __name__ == "__main__":
    main()
