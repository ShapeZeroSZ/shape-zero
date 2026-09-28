#!/usr/bin/env python3
"""
irrev_q3.py -- target 5, I4 at q = 3: the distance law of the tower's depletion around an irreversibly absorbing lump.
Predictions committed first: natural/IRREV_HYPOTHESES.md (1c9e5bf).

Side 32 periodic cube, n = 6 (comps 0-3 = the D <= 8 part, lump in comp 0; comps 4-5 = tower at A_U = 0.005, main's
incoherent_upper), kappa*, node form A'. Model G, DETERMINISTIC limit (noise off), Gamma = 10, C = 1000, K_th = 1,
T0 = 1e-5; reference = the same run with Gamma = 0 (common random numbers). The tower energy density is time-averaged over
t in [10, 33] (the ballistic cone v_max t reaches the half-box, 16, at t = 33). Deficit d(r) = <e_G> - <e_ref>, radially
averaged about the lump; the box-mean level is subtracted; fit ln|d(r) - mean| against ln r for r = 2..8.
"""
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpg_pilot2 as P  # noqa: E402

M = P.M
SIDE, NC, A_U = 32, 6, 0.005
TOWER = range(4, NC)
DT = M.DT
LOW = slice(0, 8)
UP = slice(8, 2 * NC)


def e3(lat, u, v, sl):
    uu, vv = u[:, sl], v[:, sl]
    y = uu.reshape(lat.shape + (-1,))
    g = np.zeros(lat.shape)
    for ax in range(3):
        b = ((np.roll(y, -1, axis=ax) - y) ** 2).sum(-1)
        g += 0.5 * (b + np.roll(b, 1, axis=ax))
    r = np.linalg.norm(uu, axis=1)
    return (0.5 * (vv * vv).sum(1) + 0.5 * M.SQ5 * (uu * uu).sum(1) + r ** 3 / 3 + 0.5 * M.C * g.reshape(-1))


def run(job):
    t0 = time.time()
    lat = M.Lattice(n=NC, N=SIDE ** 3, q=3, shape=SIDE, well="node")
    u, v = P.incoherent_upper(lat, A_U, job["seed"])
    c = np.indices(lat.shape).astype(float)
    r2 = sum(((c[a] - SIDE // 2 + SIDE / 2) % SIDE - SIDE / 2) ** 2 for a in range(3))
    psi = (0.05 * np.exp(-0.5 * r2 / 9.0)).astype(complex)
    dps = np.fft.ifftn(-1j * lat.branch_omega() * np.fft.fftn(psi))
    u[:, 0] += psi.real.ravel(); u[:, 1] += psi.imag.ravel(); v[:, 0] += dps.real.ravel(); v[:, 1] += dps.imag.ravel()
    C, K, Gam = 1000.0, 1.0, job["Gamma"]
    eps = np.full(lat.N, C * 1e-5)
    Em0 = lat.energy(u, v); E0 = Em0 + eps.sum()
    acc = np.zeros(lat.N); nacc = 0
    t, T = 0.0, 33.0
    steps = int(round(1.0 / DT))
    for k in range(int(T)):
        for _ in range(steps):
            u, v = P.rk4(lat, u, v, None)
            if Gam:
                gam = Gam * e3(lat, u, v, LOW)
                a = np.exp(-gam * DT)
                dE = np.zeros(lat.N)
                for cc in TOWER:
                    uc, vc = u[:, 2 * cc:2 * cc + 2], v[:, 2 * cc:2 * cc + 2]
                    rr = np.linalg.norm(uc, axis=1); ok = rr > 0
                    rh = np.zeros_like(uc); rh[ok] = uc[ok] / rr[ok, None]
                    w = (vc * rh).sum(1); w2 = a * w
                    vc += (w2 - w)[:, None] * rh
                    dE += 0.5 * (w * w - w2 * w2)
                eps += dE
                Tt = (eps / C).reshape(lat.shape)
                lap = sum(np.roll(Tt, 1, axis=ax) + np.roll(Tt, -1, axis=ax) - 2 * Tt for ax in range(3))
                eps += DT * K * lap.reshape(-1)
        t += 1.0
        if t >= 10 - 1e-9:
            acc += e3(lat, u, v, UP); nacc += 1
    E1 = lat.energy(u, v) + eps.sum()
    return dict(job=job, acc=acc / nacc, absorbed=float(eps.sum() - lat.N * C * 1e-5), dE=abs(E1 - E0) / abs(Em0),
                secs=time.time() - t0)


def main():
    jobs = [dict(seed=s, Gamma=g) for s in (0, 1) for g in (10.0, 0.0)]
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=1)
    get = {(r["job"]["seed"], r["job"]["Gamma"]): r for r in res}
    c = np.indices((SIDE,) * 3)
    d = np.minimum(np.abs(c - SIDE // 2), SIDE - np.abs(c - SIDE // 2))
    rad = np.sqrt((d ** 2).sum(0)).ravel()
    print("=" * 88)
    print("IRREVERSIBILITY PILOT, q = 3 depletion law (predictions: IRREV_HYPOTHESES.md, 1c9e5bf)")
    print("=" * 88)
    print(f"validation: energy drift {max(r['dE'] for r in res):.1e} of the mechanical energy; absorbed by T = 33: "
          + ", ".join(f"seed {s} {get[(s, 10.0)]['absorbed']:.3e}" for s in (0, 1)))
    defs = [get[(s, 10.0)]["acc"] - get[(s, 0.0)]["acc"] for s in (0, 1)]
    dm = np.mean(defs, axis=0)
    mean = dm.mean()
    rs = np.arange(1, 17)
    prof = np.array([dm[(rad >= r - 0.5) & (rad < r + 0.5)].mean() for r in rs])
    print(f"    box-mean deficit {mean:+.3e}")
    print("    r    deficit d(r)     d(r) - mean     seeds (d - mean)")
    for i, r in enumerate(rs):
        sd = [defs[k][(rad >= r - 0.5) & (rad < r + 0.5)].mean() - defs[k].mean() for k in (0, 1)]
        print(f"    {r:2d}   {prof[i]:+.3e}      {prof[i]-mean:+.3e}      {sd[0]:+.2e} {sd[1]:+.2e}")
    sel = (rs >= 2) & (rs <= 8)
    y = prof[sel] - mean
    if np.all(y < 0):
        p = np.polyfit(np.log(rs[sel]), np.log(-y), 1)[0]
        print(f"    fitted exponent (r = 2..8): {p:.3f}  (predicted in [-2.6, -1.4]; inverse-square force would need -1)")
    else:
        print("    deficit not negative at every r = 2..8; no power-law fit")
    print("run time %.0f s (sum)" % sum(r["secs"] for r in res))


if __name__ == "__main__":
    main()
