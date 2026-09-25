#!/usr/bin/env python3
"""
kappa_seed2_test.py -- is the amplitude growth of the measured kappa a property
of the wave, or of the LINEAR SEED?

kappa_pw4_pt.py derives the exact plane-wave orbit: kappa4 = -0.00295, so the
orbit's kappa at A = 0.30 is -0.01775, while the linear-seed protocol measures
-0.01869. kappa_pw4_seed.py shows the orbit-seeded run reproduces the orbit to
every digit, and that most of the difference comes from the static shift and
second harmonic the linear seed leaves out.

This script adds the SECOND-ORDER FORCED RESPONSE to the linear seed -- for any
beam, per pair of transverse components, each at its own lattice frequency:
  u1 = sum_j a_j e^{i(k_j.n - W_j t)} + c.c.     (the linear seed, a_j = A c_j / 2)
  u2 = sum_{jk} [-a_j a_k / D(k_j+k_k, W_j+W_k)] e_j e_k + c.c.       (sum modes)
     + sum_{jk} [-2 a_j a_k* / D(k_j-k_k, W_j-W_k)] e_j e_k*           (difference/DC)
(u1^2 = [sum + c.c.] + 2 sum_{jk} a_j a_k* e_j e_k*), with velocities from the
same frequencies -- and measures kappa with the kappa_resolution_test.py
integrator and readout, beside the plain linear seed, for the plane wave and the
two narrow beams at A = 0.10 and 0.30.

HYPOTHESIS, stated before running: with the second-order seed the plane wave's
kappa at 0.30 falls to near the orbit value (about -0.0179), while the narrow
beams' box kappa moves little -- so the narrow-beam F "gap" at A = 0.30 is mostly
the linear seed's artifact in the PLANE-WAVE normalisation.

usage:  python3 kappa_seed2_test.py
"""

import math
import numpy as np

import kappa_resolution_test as RT

SQ5 = math.sqrt(5.0)
_seed_lin = RT.seed


def D(kx, ky, kz, W):
    Q = SQ5 + 2 * RT.C * ((1 - np.cos(kx)) + (1 - np.cos(ky)) + (1 - np.cos(kz)))
    return -W * W + Q + 2 * RT.C * RT.BETA * W * np.sin(kx)


def Wb(kx, ky, kz):
    b = RT.C * RT.BETA * np.sin(kx)
    Q = SQ5 + 2 * RT.C * ((1 - np.cos(kx)) + (1 - np.cos(ky)) + (1 - np.cos(kz)))
    return b + np.sqrt(b * b + Q)


def second_order(side, width, amp, sign):
    """(du, dv): the forced second-order field at t = 0, shape (side,)*3."""
    L = side
    env = RT.envelope(side, width)[0]                 # transverse slice (L, L)
    c = np.fft.fft2(env) / (L * L)
    a = amp * c / 2
    q = 2 * np.pi * np.fft.fftfreq(L)
    kx = sign * RT.K
    Wj = Wb(kx, q[:, None], q[None, :])               # (L, L)
    U = np.zeros((L, L, L), complex)                  # Fourier coefficients (kx, qy, qz)
    V = np.zeros((L, L, L), complex)
    ix_sum = int(round((2 * kx) / (2 * np.pi) * L)) % L
    idx = np.arange(L)
    for jy in range(L):
        for jz in range(L):
            aj = a[jy, jz]
            if abs(aj) < 1e-18:
                continue
            ky = (jy + idx[:, None]) % L              # sum: q_j + q_k
            kz = (jz + idx[None, :]) % L
            Ws = Wj[jy, jz] + Wj
            rs = -aj * a / D(2 * kx, q[ky], q[kz], Ws)
            np.add.at(U, (ix_sum, ky, kz), rs)
            np.add.at(V, (ix_sum, ky, kz), -1j * Ws * rs)
            dy = (jy - idx[:, None]) % L              # difference: q_j - q_k
            dz = (jz - idx[None, :]) % L
            Wd = Wj[jy, jz] - Wj
            rd = -2 * aj * np.conj(a) / D(0.0, q[dy], q[dz], Wd)
            np.add.at(U, (0, dy, dz), rd)
            np.add.at(V, (0, dy, dz), -1j * Wd * rd)
    # the sum part carries "+ c.c."; the difference part is already real in total
    Us = np.zeros_like(U); Vs = np.zeros_like(V)
    Us[ix_sum] = U[ix_sum]; Vs[ix_sum] = V[ix_sum]
    Ud = U.copy(); Vd = V.copy(); Ud[ix_sum] = 0; Vd[ix_sum] = 0
    if ix_sum == 0:
        raise RuntimeError("2K folds onto kx = 0")
    n3 = L ** 3
    du = 2 * np.real(np.fft.ifftn(Us) * n3) + np.real(np.fft.ifftn(Ud) * n3)
    dv = 2 * np.real(np.fft.ifftn(Vs) * n3) + np.real(np.fft.ifftn(Vd) * n3)
    return du, dv


def seed2(side, width, amp, sign):
    x0, v0 = _seed_lin(side, width, amp, sign)
    du, dv = second_order(side, width, amp, sign)
    return x0 + du, v0 + dv


def measure(A, cases, seedfn):
    RT.seed = seedfn
    RT.A_NL = A
    out = {}
    tP, SP, _ = RT.run(8, None, "plane wave")
    kp, _, _, ep = RT.kappa_of(tP, SP)
    out["pw"] = (kp, ep)
    for w, L in cases:
        t, S, _ = RT.run(L, w, "localised")
        k, _, _, e = RT.kappa_of(t, S)
        out[(w, L)] = (k, e)
    RT.seed = _seed_lin
    return out


def main():
    print("=" * 78)
    print("LINEAR SEED vs SECOND-ORDER SEED -- plane wave and the two narrow beams")
    print("=" * 78)
    # check: for the plane wave the second-order field must equal the orbit's
    # c0 + 2 c2 cos(2 theta) (kappa_pw4_pt.py analytic, per A^2)
    du, dv = second_order(8, None, 1.0, +1)
    print(f"  plane-wave check: static {du[0,0,0]+du[1,0,0]:.5f}/2 -> c0 = "
          f"{0.5*(du[0,0,0]+du[1,0,0]):+.5f} (analytic -0.22361); "
          f"2nd harmonic 2c2 = {0.5*(du[0,0,0]-du[1,0,0]):+.5f} (analytic +0.04328)")
    cases = [(1.0, 4), (2.0, 8)]
    rows = {}
    for A in (0.10, 0.30):
        for lbl, fn in (("linear", _seed_lin), ("2nd-order", seed2)):
            rows[(A, lbl)] = measure(A, cases, fn)
    print("\n     A    seed         kappa_pw             kappa_box w=1 L=4      kappa_box w=2 L=8"
          "        F(w=1)   F(w=2)")
    for (A, lbl), r in rows.items():
        kp = r["pw"][0]
        f1 = r[(1.0, 4)][0] / (kp * RT.geometry(4, 1.0)[0])
        f2 = r[(2.0, 8)][0] / (kp * RT.geometry(8, 2.0)[0])
        print(f"   {A:4.2f}  {lbl:10s}  {kp:+.5f}+-{r['pw'][1]:.5f}   "
              f"{r[(1.0,4)][0]:+.6f}+-{r[(1.0,4)][1]:.6f}   {r[(2.0,8)][0]:+.6f}+-{r[(2.0,8)][1]:.6f}"
              f"    {f1:.3f}    {f2:.3f}")
    print("\n  orbit theory (kappa_pw4_pt.py): kappa_pw 0.10 -0.01751, 0.30 -0.01775")
    print("  second-order F (kappa_cross_pt.py, P3b): w=1 L=4 1.225, w=2 L=8 1.318")


if __name__ == "__main__":
    main()
