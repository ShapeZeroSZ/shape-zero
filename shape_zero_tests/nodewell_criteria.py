#!/usr/bin/env python3
"""nodewell_criteria.py -- A' (whole-node radial well) against the node-form criteria:
U(1) phase charge at large amplitude (all orders) and bounded motion at high energy, n = 2, 3,
with A (per dimer) alongside. usage: python3 nodewell_criteria.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M

def charge(lat, u, v):
    J = lat.JJ
    return float(np.einsum("na,ab,nb->", v, J, u) - 0.5 * lat.kappa * (u * u).sum())

rng = np.random.default_rng(3)
for n in (2, 3):
    for well in ("radial", "node"):
        lat = M.Lattice(n=n, N=64, well=well)
        u = 0.3 * rng.normal(size=(64, 2 * n)); v = 0.3 * rng.normal(size=(64, 2 * n))
        Q0 = charge(lat, u, v); u2, v2, dE = lat.run(u, v, 50.0)
        print(f"n={n} {well:6s}: amplitude ~0.3, T=50: charge rel drift {charge(lat, u2, v2)/Q0-1:+.1e}, energy drift {dE:.1e}")
        lat1 = M.Lattice(n=n, N=1, well=well)
        u = np.zeros((1, 2 * n)); v = rng.normal(size=(1, 2 * n)); v *= np.sqrt(2 * 5.0) / np.linalg.norm(v)  # kinetic energy 5
        umax = 0.0
        for _ in range(200):
            u, v, _ = lat1.run(u, v, 0.5); umax = max(umax, np.abs(u).max())
        print(f"          single node, kinetic energy 5, T=100: max |u| = {umax:.3f} (bounded if finite and O(1))")
