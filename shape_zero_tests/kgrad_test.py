#!/usr/bin/env python3
"""
kgrad_test.py -- candidate 3 of natural/JOINT_PREDICTIONS.md on MAIN's own lattice:
does a static kappa gradient push the two chirality branches apart like an electric field?

Hypothesis and numerical predictions committed first on branch natural-physics:
natural/KGRAD_HYPOTHESIS.md (0b12001); first run there (3c1a445). Re-run here on main.

Lattice: 04_scripts/session/model.py (identical to main), Lattice(n = 1, q = 1): its force law
(node form A'), DT and RK4 step. The only change is kappa -> a static per-site kappa(x) =
kappa* + kappa' (x - x0), applied exactly where model.py applies kappa (f += kappa v JJ^T).
No dynamical links. Packets at rest (k = 0), amplitude 1e-3, pure a- or b-branch in model.py's
convention (a: psi ~ e^{-i w_a t}, w_a^2 + kappa w_a = Q; b: psi ~ e^{+i w_b t}, w_b = w_a + kappa),
each launched at its LOCAL branch frequency.

Predicted (kappa' = 4e-4): a_a = +8.79e-5, a_b = -1.665e-4, a_a - a_b = +2.544e-4 = c kappa'/w_rot;
reversed for kappa' < 0; zero for kappa' = 0.
"""

import math
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M

KSTAR = M.KAPPA
N, SIGMA, AMP, T = 1400, 30.0, 1e-3, 300.0


class KGradLattice(M.Lattice):
    def __init__(self, kappa_x):
        super().__init__(n=1, N=N, kappa=KSTAR, well="node")
        self.kx = kappa_x

    def force(self, u, v, W=None, Wm=None):
        k0 = self.kappa
        self.kappa = 0.0                      # main's force without its uniform kappa ...
        f = super().force(u, v, W, Wm)
        self.kappa = k0
        return f + self.kx[:, None] * (v @ self.JJ.T)   # ... plus kappa(x) exactly where main puts kappa


def run(kprime, branch):
    x = np.arange(N); x0 = N // 2
    kx = KSTAR + kprime * (x - x0)
    lat = KGradLattice(kx)
    env = AMP * np.exp(-0.5 * ((x - x0) / SIGMA) ** 2)
    wa = 0.5 * (-kx + np.sqrt(kx ** 2 + 4 * M.SQ5))      # local k = 0 branch frequencies
    wb = wa + kx
    psi = env.astype(complex)
    dpsi = (-1j * wa * psi) if branch == "a" else (+1j * wb * psi)
    u = np.stack([psi.real, psi.imag], 1)
    v = np.stack([dpsi.real, dpsi.imag], 1)

    def energy(u, v):
        e = 0.5 * np.sum(v * v, 1) + 0.5 * M.SQ5 * np.sum(u * u, 1) + np.linalg.norm(u, axis=1) ** 3 / 3
        b = 0.5 * M.C * np.sum((np.roll(u, -1, 0) - u) ** 2, 1)
        return e + 0.5 * (b + np.roll(b, 1))
    E0 = energy(u, v).sum()
    ts, xs, ph = [], [], []
    steps = int(round(T / M.DT))
    for s in range(steps + 1):
        if s % 25 == 0:
            e = energy(u, v)
            ts.append(s * M.DT); xs.append(np.sum(x * e) / np.sum(e))
            ph.append(math.atan2(u[x0, 1], u[x0, 0]))
        if s == steps:
            break
        u, v = lat.step_rk4(u, v, None, None)
    ts, xs = np.array(ts), np.array(xs)
    a = 2 * np.polyfit(ts, xs, 2)[0]
    drift = abs(energy(u, v).sum() - E0) / E0
    # rotation rate of psi at the centre: -w_a for branch a, +w_b for branch b
    rate = np.polyfit(ts[:40], np.unwrap(np.array(ph[:40])), 1)[0]
    return a, drift, rate


def main():
    wr = math.sqrt(M.SQ5 + KSTAR ** 2 / 4)
    print("=" * 78)
    print("KAPPA-GRADIENT FORCE ON MAIN'S LATTICE (model.py, n = 1, q = 1, no dynamical links)")
    print("=" * 78)
    print(f"  kappa* = {KSTAR:.6f}, w_rot = {wr:.5f}, c/w_rot = {M.C/wr:.4f}")
    res = {}
    for kp in (4e-4, -4e-4, 0.0):
        for br in ("a", "b"):
            res[(kp, br)] = run(kp, br)
            a, dr, rate = res[(kp, br)]
            print(f"    kappa' = {kp:+.0e}  branch {br}:  a = {a:+.4e}   energy drift {dr:.1e}   "
                  f"centre rotation rate {rate:+.4f}", flush=True)
    wa0 = 0.5 * (-KSTAR + math.sqrt(KSTAR ** 2 + 4 * M.SQ5))
    print("\nVALIDATION")
    ok = True
    for br, want in (("a", -wa0), ("b", wa0 + KSTAR)):
        rate = res[(0.0, br)][2]
        g = abs(rate - want) < 1e-3
        print(f"  branch {br} is pure: rotation rate {rate:+.4f} vs {want:+.4f}   {'PASS' if g else 'FAIL'}")
        ok &= g
    worst = max(v[1] for v in res.values())
    print(f"  energy conserved (kappa does no work): worst drift {worst:.1e}   {'PASS' if worst < 1e-5 else 'FAIL'}")
    ok &= worst < 1e-5
    ctrl = max(abs(res[(0.0, b)][0]) for b in "ab")
    print(f"  control kappa' = 0: |a| <= {ctrl:.1e}   {'PASS' if ctrl < 1e-6 else 'FAIL'}")
    ok &= ctrl < 1e-6
    if not ok:
        print("  VALIDATION FAILED"); return
    print("\nRESULT (predictions from KGRAD_HYPOTHESIS.md, committed 0b12001 before this run)")
    print("   kappa'     a_a (pred)                a_b (pred)                 a_a - a_b (pred)          common (pred)")
    for kp in (4e-4, -4e-4):
        aa, ab = res[(kp, "a")][0], res[(kp, "b")][0]
        d_p = M.C * kp / wr
        c_p = -M.C * KSTAR * kp / (4 * wr * wr)
        d, cm = aa - ab, 0.5 * (aa + ab)
        print(f"   {kp:+.0e}  {aa:+.4e} ({d_p/2+c_p:+.4e})   {ab:+.4e} ({-d_p/2+c_p:+.4e})   "
              f"{d:+.4e} ({d_p:+.4e}) {100*(d/d_p-1):+.2f}%   {cm:+.3e} ({c_p:+.3e}) {100*(cm/c_p-1):+.1f}%")


if __name__ == "__main__":
    main()
