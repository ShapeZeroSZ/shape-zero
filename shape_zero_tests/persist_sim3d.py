#!/usr/bin/env python3
"""
persist_sim3d.py -- the q = 3 upper bound on kappa, by direct simulation.

persist_q3_scan.py finds that at c = 1, q = 3 the three-wave channel a -> aa opens
above kappa ~ 7.5, and persist_resonance's optimiser puts the maximum of
G = w_a(K) - w_a(k1) - w_a(K - k1) on the body diagonal with K = 2 k1: the
SECOND HARMONIC OF AN a-WAVE ALONG (1,1,1) LANDING ON THE a-BRANCH,
2 w_a(k) = w_a(2k), Q(k,k,k) = sqrt5 + 6c(1 - cos k).

Prediction (same slowly-varying calculation as persist_sim.py Part 1, now for the
a-branch at 2k driven by the -(1-i)/4 psi^2 term):
    -i (2 w_a(2k) + kappa) dB/dt = -(1-i)/4 A^2 e^{i delta t},  delta = 2 w_a(k) - w_a(2k)
    resonant: |B| = r A^2 t,  r = sqrt2 / (4 (2 w_a(2k) + kappa));
    detuned:  |B| <= 2 r A^2 / |delta|.
Runs on a real 3-D lattice (model.Lattice, q = 3, L^3 sites, one dimer), plane wave
psi = A e^{i k (x + y + z)} launched on its own a-branch frequency, A = 0.01:
  S1 kappa = kappa_res(k) > 7.5, chosen so that the grid wavenumber k is exactly
     resonant: |B| grows linearly, slope within 10% of r A^2
  S2 kappa = 6.0 (inside the persistence window [4.9, 7.5]), same k: bounded by
     1.2 x 2 r A^2 / |delta|

usage:  python3 persist_sim3d.py predict | run | report
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as MS

K = MS.SQ5
C = MS.C
L, M = 11, 2                     # k = 2 pi M / L = 1.1424 along (1,1,1)
A, T, DTR = 0.01, 300.0, 2.0
OUT = os.path.join(HERE, "persist_sim3d_runs.json")


def wa_diag(k, kap):
    return 0.5 * (-kap + np.sqrt(kap * kap + 4 * (K + 6 * C * (1 - np.cos(k)))))


def kappa_res(k):
    """kappa at which the grid wavenumber k is exactly resonant (bracketed on [7.6, 20])."""
    f = lambda kap: 2 * wa_diag(k, kap) - wa_diag(2 * k, kap)
    lo, hi = 7.6, 20.0
    flo = f(lo)
    assert flo * f(hi) < 0, (flo, f(hi))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) * flo > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def cases():
    k = 2 * math.pi * M / L
    kr = kappa_res(k)
    return k, {"S1": kr, "S2": 6.0}


def predict():
    k, cs = cases()
    print("=" * 88)
    print("PREDICTIONS (committed before any run) -- q = 3, c = 1, second harmonic within the a-branch")
    print("=" * 88)
    print(f"  L = {L}, k = 2 pi {M}/{L} = {k:.5f} along (1,1,1); 2k = {2 * k:.5f}; A = {A}, T = {T}")
    pred = {}
    for name, kap in cs.items():
        d = 2 * wa_diag(k, kap) - wa_diag(2 * k, kap)
        r = math.sqrt(2) / (4 * (2 * wa_diag(2 * k, kap) + kap))
        if abs(d) * T < 0.1:
            B, law = r * A * A * T, "linear growth, slope r A^2"
        else:
            B, law = 2 * r * A * A * abs(math.sin(d * T / 2)) / abs(d), f"bounded, max {2 * r * A / abs(d):.3e} A"
        pred[name] = dict(kap=kap, delta=d, r=r, B_T=B)
        print(f"  {name}: kappa = {kap:.6f}  delta = {d:+.3e}  r = {r:.5f}  |B|/A at T = {B / A:.3e}  ({law})")
    print(f"  S1's kappa ({cs['S1']:.3f}) lies above the predicted q = 3 threshold 7.5; S2 inside the window.")
    print("  Criterion: S1 slope within 10% of r A^2; S2 max below 1.2 x its bound.")
    return k, pred


def run():
    k, pred = predict()
    res = {}
    for name, p in pred.items():
        kap = p["kap"]
        lat = MS.Lattice(n=1, N=L ** 3, q=3, kappa=kap)
        x, y, z = np.indices((L, L, L))
        ph = (k * (x + y + z)).reshape(-1)
        psi = A * np.exp(1j * ph)
        dpsi = -1j * wa_diag(k, kap) * psi
        u = np.stack([psi.real, psi.imag], 1)
        v = np.stack([dpsi.real, dpsi.imag], 1)
        ts, Bs = [], []
        t = 0.0
        e2 = np.exp(-1j * 2 * ph)
        w2 = wa_diag(2 * k, kap)
        wb2 = w2 + kap
        while t < T - 1e-9:
            u, v, _ = lat.run(u, v, DTR)
            t += DTR
            ps = u[:, 0] + 1j * u[:, 1]
            dp = v[:, 0] + 1j * v[:, 1]
            P2, D2 = (ps * e2).mean(), (dp * e2).mean()           # component at wavevector 2k
            a2 = (wb2 * P2 + 1j * D2) / (w2 + wb2)                # a-branch part
            ts.append(t); Bs.append(float(abs(a2)))
        res[name] = dict(t=ts, B=Bs)
        print(f"  {name} done", flush=True)
    json.dump(dict(k=k, pred=pred, res=res), open(OUT, "w"))


def report():
    d = json.load(open(OUT))
    print("MEASURED")
    for name, p in d["pred"].items():
        t, B = np.array(d["res"][name]["t"]), np.array(d["res"][name]["B"])
        if abs(p["delta"]) * T < 0.1:
            s = np.polyfit(t[: len(t) // 2], B[: len(t) // 2], 1)[0] / (p["r"] * A * A)
            print(f"  {name}: |B|/A at T {B[-1] / A:.3e} (pred {p['B_T'] / A:.3e}); slope / (r A^2) = {s:.4f} "
                  f"-> {'PASS' if abs(s - 1) < 0.1 else 'FAIL'}")
        else:
            bound = 2 * p["r"] * A * A / abs(p["delta"])
            print(f"  {name}: max |B|/A {B.max() / A:.3e} (bound x1.2 {1.2 * bound / A:.3e}); at T {B[-1] / A:.3e} "
                  f"(pred {p['B_T'] / A:.3e}) -> {'PASS' if B.max() <= 1.2 * bound else 'FAIL'}")


if __name__ == "__main__":
    {"predict": predict, "run": run, "report": report}[sys.argv[1]]()
