#!/usr/bin/env python3
"""
kappa_pw4_attrib.py -- where does the linear seed's extra plane-wave growth come
from? Two numerical experiments on the 1-D ring (N = 8, kappa_pw4_seed.py
integrator = kappa_resolution_test.py's, T = 300 and 900).

1. ATTRIBUTION at A = 0.30: start from the exact orbit (kappa_pw4_pt.py) with
   pieces removed -- the static shift c0, the second harmonic c2, the third and
   higher harmonics, or all of them (fundamental only). Read at T = 300 and 900:
   a shift that is the same at both is a frequency shift, not a transient.
2. THE LEADING CROSS-MODULATION FORMULA against a direct test: the orbit plus an
   explicit free mode -- uniform (k = 0) or staggered (k = pi), zero velocity,
   chosen amplitude beta -- and each direction's frequency shift against
   kappa_pw4_pt.xpm(). Shows which part the formula captures.

usage:  python3 kappa_pw4_attrib.py
"""

import math
import numpy as np

import kappa_resolution_test as RT
import kappa_pw4_pt as P
import kappa_pw4_seed as S

N = S.N


def orbit(A, s, drop=()):
    sol = P.hb_exact(s, A)
    W = sol[0]
    c = np.zeros(P.M_HARM + 1)
    c[0], c[1], c[2:] = sol[1], A / 2, sol[2:]
    for m in drop:
        c[m] = 0
    th = s * RT.K * np.arange(N)
    u = c[0] + sum(2 * c[m] * np.cos(m * th) for m in range(1, P.M_HARM + 1))
    v = sum(2 * m * W * c[m] * np.sin(m * th) for m in range(1, P.M_HARM + 1))
    return u, v, W


def seed_drop(A, s, drop=()):
    u, v, _ = orbit(A, s, drop)
    return RT.PHI + u, v


def main():
    print("=" * 78)
    print("1. ATTRIBUTION, A = 0.30: the exact orbit with pieces removed")
    print("=" * 78)
    A = 0.30
    cases = (("orbit", ()), ("no c0", (0,)), ("no c2", (2,)),
             ("no c3 and up", tuple(range(3, P.M_HARM + 1))),
             ("fundamental only", tuple([0] + list(range(2, P.M_HARM + 1)))))
    for T in (300.0, 900.0):
        RT.T_RUN = T
        for lbl, drop in cases:
            k, e, _ = S.kappa_for(seed_drop, A, drop=drop)
            print(f"   T = {T:4.0f}   {lbl:17s} kappa {k:+.5f} +- {e:.5f}", flush=True)
    RT.T_RUN = 300.0

    print("\n" + "=" * 78)
    print("2. ORBIT + EXPLICIT FREE MODE (A = 0.30): measured shift vs leading formula")
    print("=" * 78)
    n = np.arange(N)
    xs, vs, meta = [], [], []
    for s in (+1, -1):
        u, v, W = orbit(A, s)
        for sector, beta in (("none", 0.0), ("k=0", 0.02), ("k=0", 0.04),
                             ("k=pi", 0.004), ("k=pi", 0.008)):
            du = beta * (np.ones(N) if sector == "k=0" else (-1.0) ** n)
            xs.append(RT.PHI + u + du); vs.append(v.copy()); meta.append((s, sector, beta, W))
    t, Sr = S.integrate(np.array(xs), np.array(vs))
    fr = {}
    for j, (s, sector, beta, W) in enumerate(meta):
        fr[(s, sector, beta)] = RT.fit(Sr[:, j], t)[0]
    for s in (+1, -1):
        base = fr[(s, "none", 0.0)]
        Wo = P.hb_exact(s, A)[0]
        print(f"  direction {s:+d}: orbit frequency measured {base:.8f}, harmonic balance {Wo:.8f}")
        for sector, beta in (("k=0", 0.02), ("k=0", 0.04), ("k=pi", 0.004), ("k=pi", 0.008)):
            if sector == "k=0":
                pred = P.xpm(s, Wo, 0.0, 0.0, 0.0, 5 ** 0.25, (beta / 2) ** 2)
            else:
                pred = P.xpm(s, Wo, math.pi, 0.0, 0.0, math.sqrt(P.SQ5 + 4), (beta / 2) ** 2)
            print(f"     {sector:5s} beta {beta:.3f}: measured shift {fr[(s, sector, beta)] - base:+.3e}"
                  f"   leading formula {pred:+.3e}")


if __name__ == "__main__":
    main()
