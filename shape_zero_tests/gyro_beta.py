#!/usr/bin/env python3
"""
gyro_beta.py -- the scalar beta asymmetry placed on GYROSCOPIC (kappa) nodes, with
isotropic or J-breaking (anisotropic) bond stiffness (REALISATION.md sec 4).

WHY. The model's beta sector is scalar and contains no kappa
(04_scripts/session/pinned_asymmetry_reference.py); census row 1 (UNIVERSAL_RELATIONS.md)
is exact there. A gyroscope platform has kappa on every node, and its springs or magnets
have unequal longitudinal and transverse stiffness (the J-breaking bond term). This
script asks what happens to d_omega = omega(+k) - omega(-k) then.

THE SYSTEM (linear, uniform 1-D chain, node u in R^2, bonds along x):
    u'' = -D(k) u + kappa J u' - beta c (u'[n+1] - u'[n-1])
    D(k) = diag(K + 2 c_par (1 - cos k), K + 2 c_perp (1 - cos k))
c_perp = c_par is the J-compatible (isotropic) coupling; c_perp = 0 is an unstretched
spring (Nash et al. 2015); c_perp = -c_par/4 is the point-dipole idealisation of the
magnets (REALISATION.md sec 3, ours, not the papers').
Plane wave u = a exp(i(kn - wt)):  w^2 a = D a + w (i kappa J + b) a,  b = 2 beta c sin k,
solved as the linear eigenproblem of A = [[0, I], [D, i kappa J + b I]].

RESULT (found here, 2026-09-27, not predicted): on the slow (precession) branch
d_omega = b + [sqrt((kappa-b)^2 + 4Q) - sqrt((kappa+b)^2 + 4Q)]/2
        = b (1 - kappa / sqrt(kappa^2 + 4Q)) + O(b^3) in the isotropic case (last column,
first-order form) -- it depends on
kappa-hat and c-hat and falls ~ 1/kappa-hat^2 -- while the two positive branches' d_omega
always add to 2b. Verified in gyro_sumrule_verify.py.

usage:  python3 gyro_beta.py
"""
import numpy as np

J = np.array([[0, -1], [1, 0]])


def roots(k, K, cpar, cperp, kap, beta, c=1.0):
    """The four real frequencies at wavenumber k, sorted."""
    D = np.diag([K + 2 * cpar * (1 - np.cos(k)), K + 2 * cperp * (1 - np.cos(k))])
    b = 2 * beta * c * np.sin(k)
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [D, 1j * kap * J + b * np.eye(2)]])
    return np.sort(np.linalg.eigvals(A).real)


def main():
    K, k, beta = np.sqrt(5), np.pi / 2, 0.05
    b = 2 * beta * np.sin(k)
    Q = K + 2 * (1 - np.cos(k))
    print(f"b = 2 beta c sin k = {b:.5f}   (K = sqrt5, c = 1, k = pi/2, beta = 0.05)")
    print(f"{'kappa-hat':>9} {'c_perp/c_par':>12} {'dw slow':>9} {'dw fast':>9} {'sum':>9}"
          f" {'b(1-kap/sqrt(kap^2+4Q))':>24}")
    for kap in [0.0, 0.6498 * np.sqrt(K), 2.0, 5.0, 20.0]:
        for cperp in [1.0, 0.0, -0.25]:
            wp, wm = roots(k, K, 1.0, cperp, kap, beta), roots(-k, K, 1.0, cperp, kap, beta)
            d = wp - wm
            slow, fast = d[2], d[3]          # positive roots, ascending
            pred = b * (1 - kap / np.sqrt(kap ** 2 + 4 * Q))
            print(f"{kap / np.sqrt(K):9.3f} {cperp:+12.2f} {slow:9.5f} {fast:9.5f} {slow + fast:9.5f}"
                  f" {pred:24.5f}")


if __name__ == "__main__":
    main()
