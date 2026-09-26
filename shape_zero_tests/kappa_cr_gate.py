#!/usr/bin/env python3
"""
kappa_cr_gate.py -- gate 11's residual test with kappa = kappa* and C_r on together
(never run before: model.py's gate 11 uses kappa = 0). Predictions:
kappa_cr_predictions.txt (committed first, 53ce955).

Runs gate 11's setup (n = 8, sedenion table cd(4), g = the gate's random imaginary octonion,
amp 1e-3 packet plus 0.3 x copy into the upper half, T = 20, N = 512, q = 1, 3) at
kappa in {0, kappa*} x C_r in {0, 0.05, 0.20}; reports B, energy drift, the U(1) phase
charge drift, and the commutator of JJ with L_g.

usage:  python3 kappa_cr_gate.py   (run from anywhere)
"""
import importlib.util
import os
import sys

import numpy as np

SESSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session")
sys.path.insert(0, SESSION)
import model as M

sp = importlib.util.spec_from_file_location("v2", os.path.join(SESSION, "d16_spectrum_v2.py"))
v2m = importlib.util.module_from_spec(sp); sp.loader.exec_module(v2m)
E16 = v2m.cd(4)
rg = np.random.default_rng(5)
gv = np.zeros(16); gv[1:8] = rg.normal(size=7); gv /= np.linalg.norm(gv)


def charge(u, v, kap):
    psi = u[:, 0::2] + 1j * u[:, 1::2]
    dps = v[:, 0::2] + 1j * v[:, 1::2]
    return float(np.sum(np.imag(np.conj(psi) * dps) - 0.5 * kap * np.abs(psi) ** 2))


def main():
    JJ = M.rho(1j * np.eye(8))
    Lg = np.einsum("ijk,i->kj", E16, gv)
    print(f"||[JJ, L_g]|| / ||L_g|| = {np.linalg.norm(JJ @ Lg - Lg @ JJ) / np.linalg.norm(Lg):.3f}"
          f"   (L_g antisymmetric: {np.abs(Lg + Lg.T).max():.1e}; JJ L_g + L_g JJ: "
          f"{np.linalg.norm(JJ @ Lg + Lg @ JJ) / np.linalg.norm(Lg):.3f})")
    print(f"{'q':>2} {'kappa':>8} {'C_r':>5} {'B mean':>10} {'energy drift':>12} {'charge N0':>11} {'dN/N0':>10}")
    rows = []
    for q, N in ((1, 512), (3, 512)):
        for kap in (0.0, M.KAPPA):
            for Cr in (0.0, 0.05, 0.20):
                lr = M.Lattice(n=8, N=N, q=q, kappa=kap, C_r=Cr, tower=(E16, gv))
                uu, vv = lr.packet(amp=1e-3)
                uu[:, 8:] += 0.3 * uu[:, :8]
                N0 = charge(uu, vv, kap)
                uu, vv, dq = lr.run(uu, vv, T=20.0)
                B = lr.residual_B(uu).mean()
                dN = charge(uu, vv, kap) / N0 - 1
                rows.append((q, kap, Cr, B, dq, dN))
                print(f"{q:2d} {kap:8.5f} {Cr:5.2f} {B:10.3e} {dq:12.2e} {N0:11.3e} {dN:10.2e}", flush=True)
    for q in (1, 3):
        for kap in (0.0, M.KAPPA):
            inert = all(abs(B) < 1e-12 for qq, k, c, B, _, _ in rows if qq == q and k == kap and c == 0)
            alive = all(abs(B) > 1e-6 for qq, k, c, B, _, _ in rows if qq == q and k == kap and c > 0)
            b1, b2 = [B for qq, k, c, B, _, _ in rows if qq == q and k == kap and c > 0]
            print(f"gate 11 at q = {q}, kappa = {kap:.5f}: {'PASS' if inert and alive else 'FAIL'}; "
                  f"B(0.20)/B(0.05) = {b2 / b1:.2f} (C_r^2 would give 16)")


if __name__ == "__main__":
    main()
