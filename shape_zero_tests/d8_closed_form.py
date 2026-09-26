#!/usr/bin/env python3
"""
d8_closed_form.py -- the D8 flow's frequencies in closed form (2026-09-26).

The generic D8 flow (04_scripts/rungs/z1_d8_plurality.py) is psi' = psi a + b psi,
linear: psi' = M psi with M = R_a + L_b, where R_a x = x a and L_b x = b x on the
octonions (the D8 rung's own table, z1_d8_flow.oct_table). For unit imaginary a, b at
angle theta, M is antisymmetric and its eigenvalues are +-i w with
    w in {0, 2 sin(theta/2), 2}
so the ratio of the two nonzero frequencies is exactly 1/sin(theta/2) (phi at
theta = 76.345 deg, sqrt2 at 90 deg, 2 at 60 deg). This script builds M for
(i) a fixed pair swept over theta and (ii) random pairs at random angles, prints the
frequencies with their multiplicities, and checks them against the closed form.

usage:  python3 d8_closed_form.py
"""
import importlib.util
import os

import numpy as np

RUNGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "rungs")
spec = importlib.util.spec_from_file_location("fl", os.path.join(RUNGS, "z1_d8_flow.py"))
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)
E = fl.oct_table(fl.ch.oriented_lines())


def flow_matrix(a, b):
    return np.array([fl.mul(x, a, E) + fl.mul(b, x, E) for x in np.eye(8)]).T


def freqs(M):
    """Nonnegative frequencies with multiplicity (each +-i w pair counted once; 0 per zero eigenvalue)."""
    w = np.sort(np.abs(np.linalg.eigvals(M).imag))
    out, i = [], 0
    while i < len(w):
        j = i
        while j < len(w) and abs(w[j] - w[i]) < 1e-8:
            j += 1
        m = j - i
        out.append((round(float(w[i]), 10), m if w[i] < 1e-8 else m // 2))
        i = j
    return out


def unit_imag(rng):
    v = np.zeros(8); v[1:] = rng.normal(size=7)
    return v / np.linalg.norm(v)


def main():
    rng = np.random.default_rng(2026)
    a = unit_imag(rng)
    e = unit_imag(rng); e -= (e @ a) * a; e /= np.linalg.norm(e)
    worst = 0.0
    print("theta (deg)   frequencies (value, multiplicity)                 2 sin(theta/2)   ratio   1/sin(theta/2)")
    for th in (5, 30, 60, 76.345, 90, 120, 150, 175):
        t = np.radians(th)
        M = flow_matrix(a, np.cos(t) * a + np.sin(t) * e)
        assert np.abs(M + M.T).max() < 1e-12
        f = freqs(M)
        nz = [w for w, _ in f if w > 1e-8]
        pred = sorted({0.0, 2 * np.sin(t / 2), 2.0})
        worst = max(worst, max(min(abs(w - p) for p in pred) for w, _ in f))
        print(f"{th:10.3f}   {str(f):50s} {2 * np.sin(t / 2):14.6f} {max(nz) / min(nz):8.5f} {1 / np.sin(t / 2):14.5f}")
    print("\nrandom pairs (a, b independent unit imaginary):")
    for _ in range(6):
        a2, b2 = unit_imag(rng), unit_imag(rng)
        t = np.arccos(np.clip(a2 @ b2, -1, 1))
        f = freqs(flow_matrix(a2, b2))
        pred = sorted({0.0, 2 * np.sin(t / 2), 2.0})
        worst = max(worst, max(min(abs(w - p) for p in pred) for w, _ in f))
        print(f"   theta = {np.degrees(t):7.3f} deg: {f}   predicted 2 sin(theta/2) = {2 * np.sin(t / 2):.6f}")
    print(f"\nlargest deviation from {{0, 2 sin(theta/2), 2}}: {worst:.1e}")


if __name__ == "__main__":
    main()
