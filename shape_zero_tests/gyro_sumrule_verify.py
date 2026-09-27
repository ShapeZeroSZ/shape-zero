#!/usr/bin/env python3
"""
gyro_sumrule_verify.py -- verification of the TWO-BRANCH SUM RULE (REALISATION.md sec 4).

STATUS: FOUND, THEN VERIFIED -- NOT PREDICTED. The rule was noticed in the output of
gyro_beta.py (2026-09-27) and derived afterwards; this script checks the derivation.

THE RULE. For a uniform linear lattice of two-component nodes u in R^2 with identical
isotropic inertia, any symmetric positive-definite stiffness D(k) with D(-k) = D(k)*
(any K, any c, any anisotropy c_par != c_perp, oblique bonds allowed), any gyroscopic
ratio kappa, and the scalar beta term -beta c (u'[n+1] - u'[n-1]) along axis 0:

    [w_a(+k) - w_a(-k)] + [w_b(+k) - w_b(-k)] = 2 * (2 beta c sin k_0)

summed over the two positive-frequency branches a and b.

DERIVATION (trace). Plane wave u = a exp(i(k.n - wt)), u' = -i w u:
    -w^2 a = -D a + kappa J (-i w a) - beta c (-i w)(2i sin k_0) a
    w^2 a = D a + w (i kappa J + b) a,          b = 2 beta c sin k_0.
With z = (a, w a) this is w z = A z, A = [[0, I], [D, i kappa J + b I]], so the four roots
at k sum to tr A = tr(i kappa J) + 2b = 2b (J is traceless; D does not enter).
At -k, b -> -b: the roots sum to -2b.  The field is real, so (w, a) at k gives
(-w*, a*) at -k; the roots are real (a gyroscopic system with D > 0 is stable), so the
roots at -k are the negatives of the roots at k, and the positive roots at -k are minus
the negative roots at k. Hence
    sum_pos w(k) - sum_pos w(-k) = sum_pos w(k) + sum_neg w(k) = tr A = 2b.
Only the gyroscopic/velocity matrix enters the trace; K, c, the anisotropy and kappa drop
out. (With a non-identity but uniform isotropic inertia m, b -> b/m.)

CHECKS
  1. random draws: k_0, transverse k_1, K, c_par, c_perp (incl. negative, the magnet case),
     an oblique bond angle (off-diagonal D), kappa in [0, 30], beta: |sum - 2b| at machine
     precision.
  2. the slow branch alone, isotropic case (circular modes, i kappa J -> -kappa):
     w^2 + (kappa - b) w - Q = 0, so exactly
     d_omega_a = b + [sqrt((kappa - b)^2 + 4Q) - sqrt((kappa + b)^2 + 4Q)] / 2
               = b (1 - kappa/sqrt(kappa^2 + 4Q)) + O(b^3).
  3. kappa = 0: each positive branch alone gives b (the scalar sector's census row 1).

usage:  python3 gyro_sumrule_verify.py
"""
import numpy as np

J = np.array([[0, -1], [1, 0]])


def D_of(k0, k1, K, cpar, cperp, th):
    """2-D square lattice; axis-0 bonds at angle th to the node's x axis (oblique), axis-1 bonds
    rotated by 90 degrees. Each bond: stiffness c_par along the bond, c_perp across it."""
    def bond(phi):
        e = np.array([np.cos(phi), np.sin(phi)])
        P = np.outer(e, e)
        return cpar * P + cperp * (np.eye(2) - P)
    return (K * np.eye(2) + 2 * (1 - np.cos(k0)) * bond(th)
            + 2 * (1 - np.cos(k1)) * bond(th + np.pi / 2))


def roots(k0, k1, K, cpar, cperp, th, kap, beta, c=1.0):
    D = D_of(k0, k1, K, cpar, cperp, th)
    b = 2 * beta * c * np.sin(k0)
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [D, 1j * kap * J + b * np.eye(2)]])
    w = np.linalg.eigvals(A)
    return np.sort(w.real), np.abs(w.imag).max()


def main():
    rng = np.random.default_rng(20260927)
    worst, worst_im, n = 0.0, 0.0, 0
    while n < 5000:
        k0, k1 = rng.uniform(-np.pi, np.pi, 2)
        K = rng.uniform(0.2, 5)
        cpar = rng.uniform(0, 2)
        cperp = rng.uniform(-0.4, 1.5) * cpar
        th = rng.uniform(0, np.pi)
        kap, beta = rng.uniform(0, 30), rng.uniform(0, 0.3)
        if np.linalg.eigvalsh(D_of(k0, k1, K, cpar, cperp, th)).min() <= 0.05:
            continue
        wp, ip = roots(k0, k1, K, cpar, cperp, th, kap, beta)
        wm, im = roots(-k0, -k1, K, cpar, cperp, th, kap, beta)
        s = wp[wp > 0].sum() - wm[wm > 0].sum()
        worst = max(worst, abs(s - 2 * 2 * beta * np.sin(k0)))
        worst_im = max(worst_im, ip, im)
        n += 1
    print(f"1. sum rule, {n} random draws (2-D, oblique, anisotropic, kappa 0-30):")
    print(f"   worst |sum_pos d_omega - 2*(2 beta c sin k0)| = {worst:.2e}   "
          f"(largest imaginary part of any root {worst_im:.1e})")

    K, beta, w2, w2a = np.sqrt(5), 0.05, 0.0, 0.0
    for kap in np.linspace(0, 20, 41):
        for k in np.linspace(0.1, 3.0, 30):
            wp, _ = roots(k, 0, K, 1, 1, 0, kap, beta)
            wm, _ = roots(-k, 0, K, 1, 1, 0, kap, beta)
            Q = K + 2 * (1 - np.cos(k))
            b = 2 * beta * np.sin(k)
            d = wp[2] - wm[2]
            exact = b + 0.5 * (np.sqrt((kap - b) ** 2 + 4 * Q) - np.sqrt((kap + b) ** 2 + 4 * Q))
            w2 = max(w2, abs(d - exact))
            w2a = max(w2a, abs(d - b * (1 - kap / np.sqrt(kap ** 2 + 4 * Q))))
    print(f"2. slow branch, isotropic: worst |d_omega_a - exact form| = {w2:.2e};"
          f"  first-order form b(1 - kappa/sqrt(kappa^2 + 4Q)) off by at most {w2a:.2e} (O(b^3), beta = 0.05)")

    w3 = 0.0
    for k in np.linspace(0.1, 3.0, 30):
        for cperp in (1.0, 0.0, -0.25):
            wp, _ = roots(k, 0.7, K, 1, cperp, 0.3, 0.0, beta)
            wm, _ = roots(-k, -0.7, K, 1, cperp, 0.3, 0.0, beta)
            w3 = max(w3, np.abs((wp[2:] - wm[2:]) - 2 * beta * np.sin(k)).max())
    print(f"3. kappa = 0: worst |d_omega - 2 beta c sin k| on either branch = {w3:.2e}")


if __name__ == "__main__":
    main()
