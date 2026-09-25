#!/usr/bin/env python3
"""
kappa_launch_attrib.py -- the launch pieces of the plane-wave kappa against
amplitude, so that each piece's ORDER in A can be read before it is derived.

Same ring, integrator and readout as kappa_pw4_seed.py (N = 8, RK4 dt = 0.01,
record every 0.1, uniform readout of mode N/4, kappa_resolution_test.kappa_of),
read over T = 900. Launches, each for both directions at A and at the linear
reference amplitude 0.02:
  orbit       the exact travelling wave (kappa_pw4_pt.py harmonic balance)
  no c0       the exact wave without its static shift
  no c2       the exact wave without its second harmonic
  fund only   the exact wave's fundamental only, velocity at the ORBIT frequency
  linear      the plain cosine, velocity at the LINEAR frequency (the protocol)
Increments are reported against the orbit: a piece of order A^2 in kappa (a true
fourth-order term) has increment / A^2 constant; one of order A^4, increment / A^4.

usage:  python3 kappa_launch_attrib.py
"""

import numpy as np

import kappa_resolution_test as RT
import kappa_pw4_pt as P
import kappa_pw4_seed as S

N = S.N
AMPS = (0.15, 0.20, 0.30, 0.40)
LAUNCHES = ("orbit", "no c0", "no c2", "fund only", "linear")


def launch(kind, A, s):
    if kind == "linear":
        return S.seed_lin(A, s)
    sol = P.hb_exact(s, A)
    W = sol[0]
    c = np.zeros(P.M_HARM + 1)
    c[0], c[1], c[2:] = sol[1], A / 2, sol[2:]
    if kind == "no c0":
        c[0] = 0.0
    elif kind == "no c2":
        c[2] = 0.0
    elif kind == "fund only":
        c[0] = 0.0
        c[2:] = 0.0
    th = s * RT.K * np.arange(N)
    u = c[0] + sum(2 * c[m] * np.cos(m * th) for m in range(1, P.M_HARM + 1))
    v = sum(2 * m * W * c[m] * np.sin(m * th) for m in range(1, P.M_HARM + 1))
    return RT.PHI + u, v


def main():
    RT.T_RUN = 900.0
    xs, vs, index = [], [], []
    for A in AMPS:
        for kind in LAUNCHES:
            for a in (RT.A_LIN, A):
                for s in (+1, -1):
                    x, v = launch(kind if a == A else "orbit", a, s)
                    xs.append(x); vs.append(v)
            index.append((A, kind))
    t, Sr = S.integrate(np.array(xs), np.array(vs))
    res = {}
    for j, (A, kind) in enumerate(index):
        RT.A_NL = A
        k, r, _, e = RT.kappa_of(t, Sr[:, 4 * j:4 * j + 4])
        res[(A, kind)] = (k, e)
    print("=" * 84)
    print("LAUNCH PIECES OF THE PLANE-WAVE KAPPA AGAINST AMPLITUDE (1-D ring, N = 8, T = 900)")
    print("=" * 84)
    print("     A     orbit       increment over the orbit:  no c0      no c2      fund only   linear")
    for A in AMPS:
        ko = res[(A, "orbit")][0]
        row = f"   {A:4.2f}  {ko:+.6f}                        "
        row += "  ".join(f"{res[(A, k)][0] - ko:+.6f}" for k in LAUNCHES[1:])
        print(row)
    print("\n  error bars (weighted - unweighted fit):")
    for A in AMPS:
        print(f"   {A:4.2f}  " + "  ".join(f"{k}: {res[(A, k)][1]:.6f}" for k in LAUNCHES))
    print("\n  increment / A^2 (constant for a fourth-order piece) and / A^4 (sixth order):")
    for A in AMPS:
        ko = res[(A, "orbit")][0]
        r2 = "  ".join(f"{(res[(A, k)][0] - ko) / A**2:+.5f}" for k in LAUNCHES[1:])
        r4 = "  ".join(f"{(res[(A, k)][0] - ko) / A**4:+.4f}" for k in LAUNCHES[1:])
        print(f"   {A:4.2f}   /A^2: {r2}    /A^4: {r4}")


if __name__ == "__main__":
    main()
