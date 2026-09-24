#!/usr/bin/env python3
"""
joint3_kappa_stiffness.py -- kappa versus on-site stiffness, CORRECT seeding

Re-measurement of MODEL_SPEC 5b.2 Joint #3 after the seeding retraction
(PROVENANCE 6o).

THE SEEDING BUG. The force includes beta*c*(v[n-1] - v[n+1]). A plane wave
e^{i(kn - wt)} then satisfies  w^2 - 2*beta*c*w*sin(k) - w0^2 = 0,  so the
+k wave has the UPPER root and the -k wave the LOWER root. The original runs
seeded each direction with the other direction's root. That mixes in a
counter-propagating component and produced kappa = +0.0799 (and +0.0959,
+0.0677 across stiffness). Seeded correctly, kappa is about -0.019.

WHAT THIS MEASURES. At stiffness factor f (the linear on-site stiffness is
sqrt(5)*f; the cubic well shape is unchanged), the propagation asymmetry
|w(+k) - w(-k)| is measured at A = 0.02 (linear) and A = 0.30, normalised by
the exact linear value 2*c*beta*sin(k), and kappa is the A^2 coefficient.

EXPECTED OUTPUT (recorded 2026-09-24):
    stiffness   ratio A=0.02   ratio A=0.30   kappa
      0.90      0.999992       0.998072       -0.0214
      1.00      0.999993       0.998316       -0.0187
      1.10      0.999994       0.998513       -0.0165

READING. The linear ratio stays at 0.99999 at every stiffness: the pin is
protected. kappa changes steadily with stiffness, a ~26% spread over +-10%:
the nonlinear correction is not protected. The qualitative Joint #3 result
stands; every value and the sign differ from the retracted swapped-seeding
numbers (+0.0959, +0.0799, +0.0677).

Runtime: a few minutes. numpy + scipy.
"""

import numpy as np
from scipy.integrate import solve_ivp

PHI = (1 + np.sqrt(5)) / 2
S5 = np.sqrt(5.0)
C = 1.0
K = np.pi / 2
BETA = 0.05
N = 64
T = 600.0


def roots(f):
    """(upper, lower) linear frequencies at stiffness factor f."""
    B = 2 * C * BETA * np.sin(K)
    w2 = S5 * f + 2 * C * (1 - np.cos(K))
    up = (B + np.sqrt(B * B + 4 * w2)) / 2
    lo = (-B + np.sqrt(B * B + 4 * w2)) / 2
    return up, lo


def frequency(sgn, A, wd, f):
    """Evolve a travelling wave in direction sgn, seeded at frequency wd, and
    return its measured frequency from the phase of its Fourier mode."""
    n = np.arange(N)
    x0 = PHI + A * np.cos(K * n)
    v0 = sgn * A * wd * np.sin(K * n)

    def eom(t, z):
        x = z[:N]
        v = z[N:]
        return np.concatenate([
            v,
            -(S5 * f) * (x - PHI) - (x - PHI) ** 2
            + C * (np.roll(x, -1) + np.roll(x, 1) - 2 * x)
            + BETA * C * (np.roll(v, 1) - np.roll(v, -1)),
        ])

    s = solve_ivp(eom, (0, T), np.concatenate([x0, v0]), method='DOP853',
                  rtol=1e-10, atol=1e-13, t_eval=np.linspace(0, T, 4000))
    m = int(round(K * N / (2 * np.pi)))
    F = np.fft.fft(s.y[:N], axis=0) / N
    ph = np.unwrap(np.angle(F[m]))
    w = np.abs(F[m])
    g = w > 0.05 * w.max()
    Am = np.vstack([s.t[g], np.ones(g.sum())]).T
    Wm = np.diag(w[g])
    sol, *_ = np.linalg.lstsq(Wm @ Am, Wm @ ph[g], rcond=None)
    return abs(sol[0])


def main():
    print("JOINT #3 -- kappa vs stiffness, CORRECT seeding")
    print("(+k seeded at the upper root, -k at the lower root)\n")
    print("   stiffness   ratio A=0.02   ratio A=0.30   kappa")
    kappas = []
    for f in (0.90, 1.00, 1.10):
        up, lo = roots(f)
        th = up - lo
        r = {}
        for A in (0.02, 0.30):
            r[A] = abs(frequency(+1, A, up, f) - frequency(-1, A, lo, f)) / th
        k = (r[0.30] - r[0.02]) / (0.30 ** 2 - 0.02 ** 2)
        kappas.append(k)
        print(f"     {f:.2f}      {r[0.02]:.6f}       {r[0.30]:.6f}       {k:+.4f}")
    spread = (max(kappas) - min(kappas)) / abs(np.mean(kappas)) * 100
    print(f"\n   kappa spread across stiffness: {spread:.0f}%")
    print("   linear ratio ~0.99999 at every stiffness -> the pin is protected")
    print("   kappa changes with stiffness            -> the nonlinear term is not")


if __name__ == "__main__":
    main()
