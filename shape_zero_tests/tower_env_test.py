#!/usr/bin/env python3
"""tower_env_test.py -- the tower as an environment under node form (A'), predictions in
tower_env_predictions.txt (committed first, 6e49d2d). Node n = 8, 16, 32 (D16, D32, D64); octonion
half = components 0-7, upper = the rest. q = 1 ring N = 128, kappa*, packet amplitude 0.05 in the
first dimer, T = 2000. E1: octonion half only. E2/E3: upper seeded at 10% in its first dimer.
Energy per part: quadratic energy (kinetic + sqrt5 |u|^2/2 + c grad^2/2) of that part's components;
charge per part: sum v.JJ u - (kappa/2)|u|^2 over that part.
usage: python3 tower_env_test.py"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M

N, AMP, T, DTS = 128, 0.05, 2000.0, 2.0


def parts(lat, u, v):
    out = []
    for sl in (slice(0, 8), slice(8, lat.D)):
        uu, vv = u[:, sl], v[:, sl]
        J = lat.JJ[sl, sl]
        grad = ((np.roll(uu, -1, 0) - uu) ** 2).sum()
        E = 0.5 * (vv * vv).sum() + 0.5 * M.SQ5 * (uu * uu).sum() + 0.5 * M.C * grad
        Q = float(np.einsum("na,ab,nb->", vv, J, uu) - 0.5 * lat.kappa * (uu * uu).sum())
        out.append((E, Q))
    return out


def run(n, seed, mirror=False):
    lat = M.Lattice(n=n, N=N, well="node")
    u, v = lat.packet(amp=AMP, n0=N // 2, width=8.0, per_mode=True)
    if seed and mirror:
        # POST HOC (added after the proportional seed gave a degenerate, direction-preserving
        # solution): a counter-propagating packet at n0 = N/4, mirrored so k -> -k, in the first
        # upper dimer, so the two parts' profiles differ and cross.
        u2, v2 = lat.packet(amp=AMP, n0=N // 4, width=8.0, per_mode=True)
        u[:, 8:10] = seed * u2[::-1, 0:2]; v[:, 8:10] = seed * v2[::-1, 0:2]
    elif seed:
        u[:, 8:10] = seed * u[:, 0:2]; v[:, 8:10] = seed * v[:, 0:2]
    (E0l, Q0l), (E0u, Q0u) = parts(lat, u, v)
    ts, fl, maxup = [], [], 0.0
    t = 0.0
    while t < T - 1e-9:
        u, v, _ = lat.run(u, v, DTS); t += DTS
        (El, Ql), (Eu, Qu) = parts(lat, u, v)
        ts.append(t); fl.append(Eu / (El + Eu)); maxup = max(maxup, np.abs(u[:, 8:]).max())
    fl = np.array(fl); f0 = E0u / (E0l + E0u)
    dev = fl - f0
    res = dict(maxup=maxup, dQl=Ql / Q0l - 1, dQu=(Qu / Q0u - 1) if Q0u else Qu, f0=f0,
               exch=np.abs(dev).max() / max(1 - f0, 1e-30), sign_changes=int(np.sum(np.diff(np.sign(dev[np.abs(dev) > 0])) != 0)) if np.any(dev) else 0)
    # first near-return: after the first maximum of |dev|, first time |dev| < 5% of max|dev|
    if np.abs(dev).max() > 0:
        k = int(np.argmax(np.abs(dev) > 0.5 * np.abs(dev).max()))
        back = np.where(np.abs(dev[k:]) < 0.05 * np.abs(dev).max())[0]
        res["t_return"] = ts[k + back[0]] if len(back) else None
    else:
        res["t_return"] = None
    return res


def main():
    for label, seed, mir in (("E1 octonion half only", 0.0, False), ("E2/E3 upper seeded at 10%", 0.1, False),
                             ("E3b (post hoc) upper seeded at 10%, counter-propagating packet", 0.1, True)):
        print(label)
        for n in (8, 16, 32):
            t0 = time.time(); r = run(n, seed, mir)
            print(f"  D{2 * n:<3d}: max |u_upper| {r['maxup']:.3e}; charge drift octonion half {r['dQl']:+.1e}, "
                  f"upper {r['dQu']:+.1e}; energy fraction in upper: initial {r['f0']:.4e}, max change "
                  f"{r['exch']:.2e} of the octonion half's share; sign changes {r['sign_changes']}; "
                  f"first near-return t = {r['t_return']}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
