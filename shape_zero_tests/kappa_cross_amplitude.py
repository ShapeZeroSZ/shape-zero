#!/usr/bin/env python3
"""
kappa_cross_amplitude.py -- diagnostic for the narrow-beam failure of the
derived F (kappa_cross_compare_output.txt).

The derivation (kappa_cross_pt.py) is second order in amplitude (A -> 0). The
measured F at w/L = 1/4 are at A = 0.30, and the directly measured kernel
(kappa_cross_kernel_output.txt) sits 3-4% below PT at large q_perp at A1 = 0.30
but on PT at A1 = 0.10.

HYPOTHESIS, stated before running: the remaining narrow-beam gap is an amplitude
effect beyond second order. Re-measured at A = 0.10 (with kappa_pw at the same
A), F at w = 1, L = 4 and w = 2, L = 8 moves toward the PT values (P3b: 1.225,
1.318; P3a: 1.226, 1.299).

Physics, seed, readout and error bar: kappa_resolution_test.py, imported, with
the nonlinear amplitude changed.
"""

import kappa_resolution_test as RT


def main():
    print("=" * 74)
    print("NARROW-BEAM F AGAINST AMPLITUDE")
    print("=" * 74)
    rows = []
    for A in (0.30, 0.20, 0.10):
        RT.A_NL = A
        tP, SP, _ = RT.run(8, None, "plane wave")
        kp, _, _, ep = RT.kappa_of(tP, SP)
        for w, L in ((1.0, 4), (2.0, 8)):
            t, S, d = RT.run(L, w, "localised")
            k, r, _, e = RT.kappa_of(t, S)
            fill, s = RT.geometry(L, w)
            F = k / (kp * fill)
            Fe = abs(F) * ((e / abs(k)) ** 2 + (ep / abs(kp)) ** 2) ** 0.5
            rows.append((A, w, L, kp, k, F, Fe, d))
    print("\n     A     w    L    kappa_pw     kappa_box      F")
    for A, w, L, kp, k, F, Fe, d in rows:
        print(f"   {A:4.2f}  {w:3.1f}  {L:3d}   {kp:+.5f}    {k:+.6f}    {F:.3f} +- {Fe:.3f}")
    print("\n   PT (A -> 0): w = 1: P3a 1.226, P3b 1.225;  w = 2: P3a 1.299, P3b 1.318")


if __name__ == "__main__":
    main()
