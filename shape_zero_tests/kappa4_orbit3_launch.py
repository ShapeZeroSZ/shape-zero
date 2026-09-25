#!/usr/bin/env python3
"""
kappa4_orbit3_launch.py -- S2 at L = 4 with the THIRD HARMONIC added to the
orbit-consistent launch (MODEL_SPEC §9: S2 failed w = 3, L = 4 by +18 sigma under
the committed criterion and fitted it only under a post-hoc third-harmonic band).

THE LAUNCH (L3) = the L2 launch of kappa4_orbit_launch.py (linear seed, second-
order forced field, velocities at the second-order nonlinear frequencies) plus the
THIRD-HARMONIC forced field. With u1 = sum_j (a_j e_j + c.c.) and the second-order
sum modes s_jk = -a_j a_k / D(k_j+k_k, W_j+W_k), the third-order source 2 u1 u2
has third-harmonic part 2 sum_{l,j,k} a_l s_jk e_l e_j e_k + c.c., so
    u3 = sum_{l,j,k} [-2 a_l s_jk / D(k_l+k_j+k_k, W_l+W_j+W_k)] e_l e_j e_k + c.c.,
velocity from the same frequency. For the plane wave this is the orbit's
c3 = -2 c1 c2 / D3 (checked below). Third-order terms that fall back on the
fundamental's own wavevectors are the frequency shifts already in the velocities
and are not added as a field.

CHECK on the plane wave against the exact wave (kappa_pw4_pt.py), beside the L2
residual (+0.000039 at A = 0.3).

TEST (predictions written and committed before any beam run): S2 for all four
L = 4 beams, kappa_box(A) / kappa_box(0.10) = (1 + r g(A)) / (1 + r g(0.10)),
r = fill s^2 / F2, g the plane wave's derived physical growth. Criterion as
before: pass if within two error bars (the two kappa_box error bars combined).

usage:  python3 kappa4_orbit3_launch.py check | predict | measure
"""

import math
import sys

import numpy as np

import kappa_resolution_test as RT
import kappa_seed2_test as S2
import kappa_pw4_pt as PW
import kappa4_orbit_launch as O
import kappa4_predict as P4

BEAMS = [(1.0, 4), (1.5, 4), (2.0, 4), (3.0, 4)]
AMPS = (0.10, 0.30, 0.40)


def third_order(side, width, amp, sign):
    """(du, dv) of the third-harmonic forced field at t = 0."""
    L = side
    env = RT.envelope(side, width)[0]
    c = np.fft.fft2(env) / (L * L)
    a = (amp * c / 2).ravel()
    q = 2 * np.pi * np.fft.fftfreq(L)
    iy, iz = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    iy, iz = iy.ravel(), iz.ravel()
    kx = sign * RT.K
    W = S2.Wb(kx, q[iy], q[iz])
    keep = np.nonzero(np.abs(a) > 1e-18)[0]
    ix3 = int(round(3 * kx / (2 * np.pi) * L)) % L
    U = np.zeros((L, L, L), complex)
    V = np.zeros((L, L, L), complex)
    for j in keep:
        for k in keep:
            ys, zs = (iy[j] + iy[k]) % L, (iz[j] + iz[k]) % L
            Wjk = W[j] + W[k]
            s_jk = -a[j] * a[k] / S2.D(2 * kx, q[ys], q[zs], Wjk)
            for l in keep:
                y3, z3 = (ys + iy[l]) % L, (zs + iz[l]) % L
                W3 = Wjk + W[l]
                r = -2 * a[l] * s_jk / S2.D(3 * kx, q[y3], q[z3], W3)
                U[ix3, y3, z3] += r
                V[ix3, y3, z3] += -1j * W3 * r
    n3 = L ** 3
    return 2 * np.real(np.fft.ifftn(U) * n3), 2 * np.real(np.fft.ifftn(V) * n3)


def seed_L3(side, width, amp, sign):
    x0, v0 = O.seed_L2(side, width, amp, sign)
    du, dv = third_order(side, width, amp, sign)
    return x0 + du, v0 + dv


def run_case(side, width, A, T):
    RT.seed = seed_L3
    RT.A_NL = A
    RT.T_RUN = T
    t, S, d = RT.run(side, width, "plane wave" if width is None else "localised")
    k, r, res, e = RT.kappa_of(t, S)
    RT.seed = O._seed_lin
    return k, e


def check():
    print("=" * 86)
    print("CHECK -- third-harmonic field, and the plane wave with the L3 launch")
    print("=" * 86)
    du, _ = third_order(8, None, 1.0, +1)
    n = np.arange(8)
    c3_field = np.mean(du[:, 0, 0] * np.cos(3 * RT.K * n)) # = c3 (2 c3 cos, projected)
    _, _, _, cc = PW.analytic(+1)
    print(f"  plane wave, per A^3: third-harmonic amplitude {c3_field:+.6f}  "
          f"(analytic orbit c3 {cc['c3']:+.6f})")
    print("\n     T      A     L3 launch              exact wave    residual     (L2 residual)")
    L2res = {0.10: 0.000004, 0.30: 0.000039, 0.40: 0.000062}
    for T in (300.0, 900.0):
        for A in AMPS:
            k, e = run_case(8, None, A, T)
            ke = PW.kappa_exact(A)[0]
            print(f"   {T:4.0f}   {A:4.2f}   {k:+.6f} +- {e:.6f}    {ke:+.6f}    {k - ke:+.6f}     "
                  f"({L2res[A]:+.6f})", flush=True)


def predict():
    print("=" * 86)
    print("PREDICTIONS -- S2, all four L = 4 beams, L3 launch; before any beam run")
    print("=" * 86)
    g = {A: O.growth(A) for A in AMPS}
    print("  g(A): " + ", ".join(f"{A:.2f}: {g[A]:.5f}" for A in AMPS))
    print("     w     L    fill     s      F2      r(S2)   |  ratio 0.30   ratio 0.40")
    out = {}
    for w, L in BEAMS:
        fill, s, F2, o = P4.predict(L, w)
        r = o["S2"][0]
        rat = {A: (1 + r * g[A]) / (1 + r * g[0.10]) for A in (0.30, 0.40)}
        out[(w, L)] = rat
        print(f"   {w:4.1f}  {L:3d}  {fill:.4f}  {s:.3f}  {F2:.4f}   {r:.4f}   |  {rat[0.30]:.5f}      {rat[0.40]:.5f}")
    print("\n  Criterion: S2 passes a beam at an amplitude if the measured ratio is within two")
    print("  error bars (the two kappa_box error bars combined in quadrature).")
    return out


def measure():
    pred = predict()
    print("\n" + "=" * 86)
    print("MEASURED -- L = 4 beams, L3 launch")
    print("=" * 86)
    for T in (300.0, 900.0):
        print(f"\n  T = {T:.0f}")
        print("     w     L     A      kappa_box              ratio to 0.10          S2 pred    pull")
        for w, L in BEAMS:
            base = None
            for A in AMPS:
                k, e = run_case(L, w, A, T)
                if A == 0.10:
                    base = (k, e)
                    print(f"   {w:4.1f}  {L:3d}  {A:4.2f}   {k:+.6f} +- {e:.6f}", flush=True)
                    continue
                rat = k / base[0]
                re = abs(rat) * math.hypot(e / k, base[1] / base[0])
                p = pred[(w, L)][A]
                print(f"   {w:4.1f}  {L:3d}  {A:4.2f}   {k:+.6f} +- {e:.6f}   {rat:.5f} +- {re:.5f}   "
                      f"{p:.5f}   {(p - rat) / re:+5.1f}  {'PASS' if abs(p - rat) <= 2 * re else 'FAIL'}",
                      flush=True)


if __name__ == "__main__":
    {"check": check, "predict": predict, "measure": measure}[sys.argv[1]]()
