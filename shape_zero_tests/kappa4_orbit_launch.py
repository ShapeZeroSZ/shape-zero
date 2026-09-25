#!/usr/bin/env python3
"""
kappa4_orbit_launch.py -- the beam fourth-order tests redone with
ORBIT-CONSISTENT launches (MODEL_SPEC §9).

THE LAUNCH (L2). A localised beam has no exact travelling solution to launch on,
so the launch is made consistent with the nonlinear wave to second order:
  * the linear Fourier-space seed (kappa_resolution_test.py), plus
  * the second-order forced field -- static shift and second harmonic for every
    pair of transverse components, each at its own lattice frequency
    (kappa_seed2_test.second_order), plus
  * every component's velocity at its SECOND-ORDER NONLINEAR frequency,
    W_j + dW_j, with dW_j from the second-order kernel of kappa_cross_pt.py:
        dW_j = [ |a_j|^2 (4/sqrt5 + 2/D(2k_j, 2W_j))
                 + sum_{k != j} 4|a_k|^2 (1/D(k_j+k_k, W_j+W_k)
                                          + 1/D(k_j-k_k, W_j-W_k) + 1/sqrt5) ] / D_W(k_j)
    (the diagonal, dephased form; a_j = A c_j / 2).
For the plane wave this is the exact wave to O(A^2) in the field and O(A^2) in
the frequency; what it leaves out (third harmonic, O(A^4) field corrections) is
checked against the exact wave below (VALIDATION), not assumed small.

THE TEST. With launch effects removed, a beam's box kappa grows with amplitude
only through physics. The plane wave's physical growth is derived
(kappa_pw4_pt.py): g(A) = kappa_wave(A)/kappa2 - 1. The beam's own growth is
r * g(A), with r as in kappa4_predict.py (H0: 0; S1: fill (6 - 3P2 - 6s + 4s^2)/F2;
S2: fill s^2/F2). Predicted: kappa_box(A) / kappa_box(0.10) = (1 + r g(A)) / (1 + r g(0.10)).
The plane wave does not enter the ratio, so its own launch residual cannot
masquerade as beam physics.

usage:  python3 kappa4_orbit_launch.py predict     (writes nothing; prints the predictions)
        python3 kappa4_orbit_launch.py validate    (plane wave: L2 launch vs exact wave)
        python3 kappa4_orbit_launch.py measure     (the beams)
"""

import math
import sys

import numpy as np

import kappa_resolution_test as RT
import kappa_seed2_test as S2
import kappa_pw4_pt as PW
import kappa4_predict as P4

SQ5 = math.sqrt(5.0)
_seed_lin = S2._seed_lin
BEAMS = [(1.0, 4), (2.0, 8), (1.5, 4), (2.0, 4), (3.0, 4)]
AMPS = (0.10, 0.30, 0.40)
KAPPA2 = -0.017480


def dW_components(side, width, amp, sign):
    """Second-order nonlinear frequency shift of every transverse component."""
    L = side
    env = RT.envelope(side, width)[0]
    c = np.fft.fft2(env) / (L * L)
    a2 = np.abs(amp * c / 2) ** 2
    q = 2 * np.pi * np.fft.fftfreq(L)
    qy, qz = np.meshgrid(q, q, indexing="ij")
    kx = sign * RT.K
    Wj = S2.Wb(kx, qy, qz)
    DWj = -2 * Wj + 2 * RT.C * RT.BETA * math.sin(kx)
    dW = np.zeros((L, L))
    for iy in range(L):
        for iz in range(L):
            w = Wj[iy, iz]
            cross = 4 * a2 * (1 / S2.D(2 * kx, q[iy] + qy, q[iz] + qz, w + Wj)
                              + 1 / S2.D(0.0, q[iy] - qy, q[iz] - qz, w - Wj)
                              + 1 / SQ5)
            cross[iy, iz] = 0.0
            selfterm = a2[iy, iz] * (4 / SQ5 + 2 / S2.D(2 * kx, 2 * q[iy], 2 * q[iz], 2 * w))
            dW[iy, iz] = (selfterm + cross.sum()) / DWj[iy, iz]
    return dW


def seed_L2(side, width, amp, sign):
    x0, v0 = _seed_lin(side, width, amp, sign)
    du, dv = S2.second_order(side, width, amp, sign)
    # velocity correction: component (kx, q) of the fundamental runs at W + dW
    idx0 = np.indices((side,) * 3)[0].astype(float)
    psi = amp * RT.envelope(side, width) * np.exp(1j * sign * RT.K * idx0)
    dW = dW_components(side, width, amp, sign)
    dom = np.broadcast_to(dW[None, :, :], (side,) * 3)
    dv1 = np.fft.ifftn(-1j * dom * np.fft.fftn(psi)).real
    return x0 + du, v0 + dv + dv1


def growth(A):
    return PW.kappa_exact(A)[0] / KAPPA2 - 1


def predict():
    print("=" * 86)
    print("PREDICTIONS -- kappa_box(A) / kappa_box(0.10) with orbit-consistent (L2) launches")
    print("=" * 86)
    g = {A: growth(A) for A in AMPS}
    print("  plane-wave physical growth g(A) = kappa_wave(A)/kappa2 - 1 (kappa_pw4_pt.py): "
          + ", ".join(f"A={A:.2f}: {g[A]:.5f}" for A in AMPS))
    print("     w     L    fill     s     F2     |  r: H0    S1     S2   |  ratio at 0.30: H0      S1       S2"
          "     |  ratio at 0.40: H0      S1       S2")
    out = {}
    for w, L in BEAMS:
        fill, s, F2, o = P4.predict(L, w)
        r = {h: o[h][0] for h in ("H0", "S1", "S2")}
        rat = {(h, A): (1 + r[h] * g[A]) / (1 + r[h] * g[0.10]) for h in r for A in (0.30, 0.40)}
        out[(w, L)] = rat
        print(f"   {w:4.1f}  {L:3d}  {fill:.4f}  {s:.3f}  {F2:.4f}  |  {r['H0']:.3f}  {r['S1']:.3f}  "
              f"{r['S2']:.3f}  |  " + "  ".join(f"{rat[(h, 0.30)]:.5f}" for h in ("H0", "S1", "S2"))
              + "   |  " + "  ".join(f"{rat[(h, 0.40)]:.5f}" for h in ("H0", "S1", "S2")))
    print("\n  Criterion, set in advance: a hypothesis passes a beam at an amplitude if its ratio")
    print("  is within two error bars of the measured one (error bars of the two kappa_box")
    print("  values combined in quadrature). The test only discriminates where hypotheses")
    print("  differ by more than about four error bars; that is reported, not assumed.")
    return out


def run_case(side, width, A, T):
    RT.seed = seed_L2
    RT.A_NL = A
    RT.T_RUN = T
    t, S, d = RT.run(side, width, "plane wave" if width is None else "localised")
    k, r, res, e = RT.kappa_of(t, S)
    RT.seed = _seed_lin
    return k, e, r, d


def validate():
    print("=" * 86)
    print("VALIDATION -- plane wave, L2 launch against the exact wave (kappa_pw4_pt.py)")
    print("=" * 86)
    for T in (300.0, 900.0):
        for A in AMPS:
            k, e, r, d = run_case(8, None, A, T)
            ke = PW.kappa_exact(A)[0]
            print(f"   T = {T:4.0f}  A = {A:.2f}:  L2 {k:+.6f} +- {e:.6f}   exact wave {ke:+.6f}   "
                  f"residual {k - ke:+.6f}   (plain cosine at 0.30: -0.018709)", flush=True)


def measure():
    pred = predict()
    print("\n" + "=" * 86)
    print("MEASURED -- beams with L2 launches")
    print("=" * 86)
    for T in (300.0, 900.0):
        print(f"\n  T = {T:.0f}")
        print("     w     L     A      kappa_box              ratio to A = 0.10        pulls: H0     S1     S2")
        for w, L in BEAMS:
            base = None
            for A in AMPS:
                k, e, r, d = run_case(L, w, A, T)
                if A == 0.10:
                    base = (k, e)
                    print(f"   {w:4.1f}  {L:3d}  {A:4.2f}   {k:+.6f} +- {e:.6f}", flush=True)
                    continue
                rat = k / base[0]
                re = abs(rat) * math.hypot(e / k, base[1] / base[0])
                pulls = [(pred[(w, L)][(h, A)] - rat) / re for h in ("H0", "S1", "S2")]
                print(f"   {w:4.1f}  {L:3d}  {A:4.2f}   {k:+.6f} +- {e:.6f}   {rat:.5f} +- {re:.5f}        "
                      + "  ".join(f"{p:+5.1f}" for p in pulls), flush=True)


if __name__ == "__main__":
    {"predict": predict, "validate": validate, "measure": measure}[sys.argv[1]]()
