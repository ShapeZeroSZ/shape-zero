#!/usr/bin/env python3
"""
ampA_masked.py -- anomaly A, the common size factor ~0.6 (2026-09-29): exact first-order runs of
the certification geometry with the A' nonlinearity switched on only INSIDE the link segments
(ramp sites of both segments, every transverse site), only OUTSIDE them, and EVERYWHERE, each
against the linear run (nonlinearity off) of the same geometry and readout time. Plus (ours) a
q = 1 geometry with the segments further apart (60, 110), EVERYWHERE and linear only.
Predictions: ampA_MASKED_PREDICTIONS.md (committed before any run).

Slopes (deg per 1e-3) from single runs at A = 1e-3: per-order = angle(co_mask, co_lin),
split = split(mask) - split(lin), vector dco = co_mask - co_lin. Readout: q = 1 at the recorded
protocol time T (nodewell_1d.json); q = 3 at the recorded pair readout time t (runs nodeA1), stepping
in CHECK = 1 chunks as q3_gate.run_pair does. With the mask 1 everywhere the force is identical to
A''s, so EVERYWHERE reproduces the recorded 1e-3 runs.
usage: python3 ampA_masked.py -> ampA_masked_runs.json, ampA_masked_output.txt
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M
import q3_gate as Q

AMP = 1e-3
L = len(M.RAMP)


def pangle(a, b):
    a = np.asarray(a) / np.linalg.norm(a); b = np.asarray(b) / np.linalg.norm(b)
    return float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(a - b) / 2))))


class Masked1(M.Lattice):
    def __init__(self, n, mask):
        super().__init__(n=n, N=1200, well="node")
        self.mask = mask[:, None]

    def _onsite_nl(self, u):
        return self.mask * np.linalg.norm(u, axis=1, keepdims=True) * u

    def _onsite_cubic(self, u):
        return (self.mask[:, 0] * np.linalg.norm(u, axis=1) ** 3).sum() / 3


class Masked3(Q.Slab):
    def __init__(self, n, kappa, mask):
        super().__init__(n, 260, 8, kappa)
        self.well = "node"
        self.mask = mask[:, None]

    def _onsite_nl(self, u):
        return self.mask * np.linalg.norm(u, axis=1, keepdims=True) * u

    def _onsite_cubic(self, u):
        return (self.mask[:, 0] * np.linalg.norm(u, axis=1) ** 3).sum() / 3


def mask_for(kind, shape, segs):
    x = np.indices(shape)[0].reshape(-1)
    inside = np.zeros(x.shape, bool)
    for st in segs:
        inside |= (x >= st) & (x < st + L)
    return {"lin": np.zeros(x.shape), "IN": inside.astype(float), "OUT": (~inside).astype(float),
            "ALL": np.ones(x.shape)}[kind]


def job(a):
    q, n, order, kind, T, segs, tag = a
    if q == 1:
        gA, gB = {2: (0.12, 0.08), 3: (0.15, 0.10)}[n]
        spec = {"AB": [(segs[0], 0, gA), (segs[1], 1, gB)], "BA": [(segs[0], 1, gB), (segs[1], 0, gA)]}[order]
        lat = Masked1(n, mask_for(kind, (1200,), segs))
        W, Wm = M.make_links(lat, spec)
        u, v = lat.packet(width=8.0, n0=20, amp=AMP, per_mode=True)
        u, v, drift = lat.run(u, v, T, W, Wm)
    else:
        lat = Masked3(n, float(M.KAPPA), mask_for(kind, (260, 8, 8), Q.SEGS))
        W, Wm = M.make_links(lat, Q.spec_for(n, order))
        u, v = lat.packet3(AMP)
        E0 = lat.energy(u, v)
        t = 0.0
        while t < T - 1e-9:
            u, v, _ = lat.run(u, v, Q.CHECK, W, Wm); t += Q.CHECK
        drift = abs(lat.energy(u, v) - E0) / abs(E0)
    co, _ = lat.readout(u, v)
    return dict(q=q, n=n, order=order, kind=kind, tag=tag, T=T, co=co.tolist(), drift=float(drift))


def jobs():
    rows = json.load(open(os.path.join(HERE, "nodewell_1d.json")))
    T1 = {r["n"]: r["T"] for r in rows if r["amp"] == 1e-3 and r["axes"] == [0, 1]}
    d = json.load(open(os.path.join(HERE, "q3_gate_runs_260x8_nodeA1.json")))
    t3 = {r["n"]: r["t"] for r in d["runs"] if r["job"] == "AB"}
    J = []
    for n in (2, 3):
        for o in ("AB", "BA"):
            for k in ("lin", "IN", "OUT", "ALL"):
                J.append((3, n, o, k, t3[n], Q.SEGS, "cert"))
                J.append((1, n, o, k, T1[n], (60, 80), "cert"))
            for k in ("lin", "ALL"):
                J.append((1, n, o, k, T1[n] + 110.0, (60, 110), "sep"))
    return J


def main():
    J = jobs()
    with Pool(4) as p:
        res = p.map(job, J, chunksize=1)
    json.dump(res, open(os.path.join(HERE, "ampA_masked_runs.json"), "w"), indent=1)
    by = {(r["q"], r["tag"], r["n"], r["order"], r["kind"]): r for r in res}
    lines = ["MASKED FIRST-ORDER RUNS (A = 1e-3; deg per 1e-3; against ampA_MASKED_PREDICTIONS.md)"]
    for q, tag in ((1, "cert"), (3, "cert"), (1, "sep")):
        for n in (2, 3):
            lin = {o: by[(q, tag, n, o, "lin")]["co"] for o in ("AB", "BA")}
            for k in (("IN", "OUT", "ALL") if tag == "cert" else ("ALL",)):
                c = {o: by[(q, tag, n, o, k)]["co"] for o in ("AB", "BA")}
                sp = pangle(c["AB"], c["BA"]) - pangle(lin["AB"], lin["BA"])
                lines.append(f"  q={q} {tag:<4} u({n}) {k:<3}  split {sp:+.5f}  per-order AB {pangle(c['AB'], lin['AB']):.5f}"
                             f"  BA {pangle(c['BA'], lin['BA']):.5f}   drift {max(by[(q, tag, n, o, k)]['drift'] for o in ('AB', 'BA')):.1e}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "ampA_masked_output.txt"), "w").write(out + "\n")


if __name__ == "__main__":
    main()
