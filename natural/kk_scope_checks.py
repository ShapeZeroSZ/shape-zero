#!/usr/bin/env python3
"""
kk_scope_checks.py -- checks for the Kaluza-Klein reading (predictions: KK_HYPOTHESES.md, 88cf23d).

KC1  model.py's force at a single node (N = 1, no neighbours) against the Euler-Lagrange acceleration of
     the KK-form Lagrangian L = |v|^2/2 + A0 v.(JJ u) + A0^2 |u|^2/2 - (sqrt5 + kappa^2/4)|u|^2/2 - |u|^3/3,
     i.e. r^2 (theta' + A0)^2 / 2 along the U(1) orbit:  a = -2 A0 JJ v + (A0^2 - sqrt5 - kappa^2/4) u - |u| u.
     Both signs of A0 = +-kappa/2, n = 1 and n = 3, 200 random states.
KC2  linearised spectrum of a single node about the vacuum (r = 0): the distinct |omega| values; is there a
     third (radion-like) mode besides the two chirality branches?
KC3  gauge-invariant masses and the charge-to-mass ratio: w_a + kappa/2, w_b - kappa/2, sqrt(sqrt5 + kappa^2/4),
     and 1 / w_rot against 2 / (2 w_a + kappa).
"""
import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M

K, ka = M.SQ5, M.KAPPA
rng = np.random.default_rng(7)
print("=" * 78)
print("KK SCOPE CHECKS (predictions: KK_HYPOTHESES.md, 88cf23d)")
print("=" * 78)
print("KC1  model.py force vs the KK-form Euler-Lagrange acceleration (single node)")
for n in (1, 3):
    lat = M.Lattice(n=n, N=1, well="node")
    for A0 in (-ka / 2, +ka / 2):
        err, scale = 0.0, 0.0
        for _ in range(200):
            u = rng.normal(size=(1, 2 * n)) * rng.uniform(0.01, 1.0)
            v = rng.normal(size=(1, 2 * n)) * rng.uniform(0.01, 1.0)
            fm = lat.force(u, v)
            r = np.linalg.norm(u)
            fk = -2 * A0 * (v @ lat.JJ.T) + (A0 ** 2 - K - ka ** 2 / 4) * u - r * u
            err = max(err, np.max(np.abs(fm - fk))); scale = max(scale, np.max(np.abs(fm)))
        print(f"    n = {n}, A0 = {A0:+.6f}: max |f_model - f_KK| = {err:.2e}  (max |f| = {scale:.2f})")

print("\nKC2  linearised spectrum of one node about r = 0 (distinct |omega|)")
EPS = 1e-7
for n in (1, 3):
    lat = M.Lattice(n=n, N=1, well="node")
    d = 2 * n
    A = np.zeros((d, d)); B = np.zeros((d, d)); z = np.zeros((1, d))
    for i in range(d):
        e = np.zeros((1, d)); e[0, i] = EPS
        A[:, i] = ((lat.force(e, z) - lat.force(-e, z)) / (2 * EPS)).ravel()
        B[:, i] = ((lat.force(z, e) - lat.force(z, -e)) / (2 * EPS)).ravel()
    lam = np.linalg.eigvals(np.block([[np.zeros((d, d)), np.eye(d)], [A, B]]))
    w = np.unique(np.round(np.abs(lam.imag), 6))
    print(f"    n = {n}: {2*d} eigenvalues, distinct |omega| = {list(w)}, max Re = {np.max(np.abs(lam.real)):.1e}")

print("\nKC3  gauge-invariant masses and charge-to-mass")
wa = (-ka + math.sqrt(ka * ka + 4 * K)) / 2; wb = wa + ka
wr = math.sqrt(K + ka * ka / 4)
print(f"    w_a = {wa:.10f}, w_b = {wb:.10f}")
print(f"    w_a + kappa/2 = {wa+ka/2:.12f}, w_b - kappa/2 = {wb-ka/2:.12f}, sqrt(sqrt5 + kappa^2/4) = {wr:.12f}")
print(f"    |q|/m = 1/w_rot = {1/wr:.12f}; 2/(2 w_a + kappa) = {2/(2*wa+ka):.12f}; diff {abs(1/wr-2/(2*wa+ka)):.1e}")
print(f"    lab-frame q/m: a {1/wa:.4f}, b {1/wb:.4f} (ratio {wb/wa:.4f}) -- gauge-dependent")
