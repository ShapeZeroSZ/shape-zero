#!/usr/bin/env python3
"""
ring_spectrum.py -- reading (B) (ring_model.py): linear spectrum around the ring,
consistency checks, bounded motion, and the predictions for model.py's gates on it.

LINEAR SPECTRUM. Around the ring point psi0 = phi (any phase), write the displacement
as radial rho and tangential eta. Linearising F = -(x^2 - x - 1) psi/x at x = phi:
radial stiffness V''(phi) = sqrt5, tangential stiffness 0 (V'(phi) = 0). The lattice
term acts on both as c lap; the gyroscopic term kappa JJ v couples them:
    rho'' = -sqrt5 rho + c lap rho - kappa eta',     eta'' = c lap eta + kappa rho'.
Plane waves, s = 2c sum_a (1 - cos k_a), Omega = w^2:
    Omega^2 - (sqrt5 + 2s + kappa^2) Omega + s (sqrt5 + s) = 0.
At k = 0: Omega = 0 (the phase mode is MASSLESS -- a Goldstone mode of the
spontaneously broken phase symmetry) and Omega = sqrt5 + kappa^2 (the gapped
"optical" branch). For small k the lower branch is acoustic,
w = v k with v^2 = c sqrt5 / (sqrt5 + kappa^2). There are no circular chirality
branches: the modes are radial/tangential ellipses, and w and -w are symmetric.

kappa* WAS DERIVED for chirality branches around the origin (w^2 + kappa w = Q and its
mirror). Its analogue here, the kappa that separates the two branches (acoustic top
below the optical bottom), is computed below for comparison only.

usage:  python3 ring_spectrum.py predict | run
"""
import math
import os
# model.py's J sector defaults to the radial well since 2026-09-26; this script's
# "model.py" results are the ELEMENTWISE form, so pin it (before model is imported).
os.environ["SZ_J_WELL"] = "elementwise"
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_ring as MRING           # built by ring_model.py build
import model_radial as MRAD          # built by radial_model.py build
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as MS

K = math.sqrt(5.0)
PHI = (1 + K) / 2
KS = float(MS.KAPPA)


def branches(s, kap):
    b = K + 2 * s + kap * kap
    d = np.sqrt(b * b - 4 * s * (K + s))
    return np.sqrt(np.maximum((b - d) / 2, 0)), np.sqrt((b + d) / 2)


def separation_kappa(q, c=1.0):
    """kappa at which the acoustic top (s = 4qc) meets the optical bottom (s = 0)."""
    f = lambda kap: branches(4 * q * c, kap)[0] - branches(0.0, kap)[1]
    lo, hi = 0.0, 50.0
    if f(lo) < 0:
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def numeric_spectrum(kap, N=32):
    """Eigenfrequencies of the ring copy linearised at u = 0 (finite-difference Jacobian)."""
    lat = MRING.Lattice(n=1, N=N, kappa=kap)
    D = 2 * N
    J = np.zeros((D, D))
    eps = 1e-7
    z = np.zeros((N, 2))
    for i in range(D):
        up = z.copy().reshape(-1); up[i] += eps
        um = z.copy().reshape(-1); um[i] -= eps
        J[:, i] = ((lat.force(up.reshape(N, 2), z) - lat.force(um.reshape(N, 2), z)) / (2 * eps)).reshape(-1)
    G = np.zeros((D, D))                       # velocity part: kappa JJ v
    for i in range(N):
        vv = np.zeros((N, 2)); vv[i, 0] = 1
        G[:, 2 * i] = (lat.force(z, vv) - lat.force(z, np.zeros_like(vv))).reshape(-1)
        vv = np.zeros((N, 2)); vv[i, 1] = 1
        G[:, 2 * i + 1] = (lat.force(z, vv) - lat.force(z, np.zeros_like(vv))).reshape(-1)
    A = np.block([[np.zeros((D, D)), np.eye(D)], [J, G]])
    ev = np.linalg.eigvals(A)
    return np.sort(np.abs(ev.imag))[::2]


def predict():
    print("=" * 90)
    print("READING (B) -- predictions (committed before any run)")
    print("=" * 90)
    print("  LINEAR SPECTRUM (analytic): Omega^2 - (sqrt5 + 2s + kappa^2) Omega + s(sqrt5 + s) = 0")
    for kap in (0.0, KS):
        lo0, hi0 = branches(0.0, kap)
        lo1, hi1 = branches(4.0, kap)
        print(f"     kappa = {kap:.4f}: k = 0: w = {lo0:.4f} (phase mode), {hi0:.4f}; k = pi: {lo1:.4f}, {hi1:.4f};"
              f" acoustic speed v = {math.sqrt(K / (K + kap * kap)):.4f}")
    print("     numeric check: the linearised ring copy's eigenfrequencies (N = 32) equal the formula to < 1e-6")
    for q in (1, 3):
        print(f"     branch-separation kappa (the analogue of kappa*, for comparison only), q = {q}, c = 1: "
              f"{separation_kappa(q):.6f}   [kappa* = 0.971737 (q = 1), 2.090698 (q = 3 general)]")
    print("\n  BOUNDED MOTION (single node, n = 1, kappa*, launched with kinetic energy 5 at the equilibrium):")
    print("     elementwise model.py: the per-component well is unbounded below (u -> -inf past the saddle")
    print("       u = -sqrt5, barrier 5 sqrt5/6 = 1.863): the node ESCAPES, |u| > 100 within T = 200")
    print("     radial (A): V = sqrt5 r^2/2 + r^3/3 >= 0, confining: max |u| < 5 over T = 200")
    print("     ring (B): V(x) = x^3/3 - x^2/2 - x on x >= 0, confining: max |u| < 5 over T = 200")
    print("\n  GATES on the ring copy (u = displacement from psi0 = (phi, 0); every gate reports):")
    print("     1 PASS, unchanged (harness)")
    print("     2 PASS (energy conserved; drift < 1e-6), drift value may change")
    print("     3 FAIL: chirality purity < 0.95 -- the circular packet is not a mode about the ring (radial")
    print("       part gapped at sqrt(sqrt5 + kappa^2), tangential part acoustic); readout at the isotropic omega")
    print("     4, 5, 6 PASS, unchanged (algebra; reference scripts independent of Lattice's force)")
    print("     7 u(2), u(3) FAIL: the independent prediction uses the isotropic chirality branch, which")
    print("       does not exist about the ring; per-order errors >= 1 deg")
    print("     8 PASS (commuting sigma_x segments commute with the per-dimer ring dynamics when every")
    print("       dimer sits at the same ring point): floor < 3 deg")
    print("     9 PASS, unchanged value: it inspects the initial displacement only (never evolves) -- under")
    print("       (B) it splits the displacement u, not the node state psi0 + u")
    print("     10 PASS (drift < 1e-6), values change")
    print("     11 not predicted (no derivation)")
    print("  GATES on the literal copy (psi = u, no offset): the gates' states sit at the origin, which is")
    print("     not an equilibrium (|F| = 1 outward); gate 2 FAILS (drift >= 1e-6 or the packet leaves the")
    print("     small-amplitude regime); gates 3 and 7 FAIL.")


def run():
    print("MEASURED")
    for kap in (0.0, KS):
        num = numeric_spectrum(kap)
        N = 32
        ks = 2 * np.pi * np.arange(N) / N
        s = 2 * (1 - np.cos(ks))
        lo, hi = branches(s, kap)
        ana = np.sort(np.concatenate([lo, hi]))
        print(f"  spectrum, kappa = {kap:.4f}: max |numeric - analytic| = {np.abs(np.sort(num) - ana).max():.2e}; "
              f"lowest four {np.round(np.sort(num)[:4], 5)}")
    for name, Mod in (("elementwise model.py", MS), ("radial (A)", MRAD), ("ring (B)", MRING)):
        lat = Mod.Lattice(n=1, N=1, kappa=KS)
        u = np.zeros((1, 2)); v = np.array([[-math.sqrt(10.0), 0.0]])
        mx, t, esc = 0.0, 0.0, None
        while t < 200:
            u, v, _ = lat.run(u, v, 0.5); t += 0.5
            r = float(np.linalg.norm(u))
            if not np.isfinite(r) or r > 100:
                esc = t; break
            mx = max(mx, r)
        print(f"  bounded motion, {name}: " + (f"ESCAPED (|u| > 100) at t = {esc:.1f}" if esc else
                                                f"bounded, max |u| = {mx:.3f} over T = 200"))


if __name__ == "__main__":
    {"predict": predict, "run": run}[sys.argv[1]]()
