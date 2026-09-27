#!/usr/bin/env python3
"""nodewell_test.py -- per-dimer radial well (A) vs whole-node radial well (A').
Predictions: nodewell_predictions.txt (committed first, 8d5a507).

  sym     N1: V_A = sum_j |psi_j|^3/3 vs V_A' = |psi|^3/3 under gate 7's segment unitaries
          (applied to the launched state e_0) and under random U(n), n = 2, 3.
  p3      N3: P-3 spinor precession rate under A and A' (radialA_tests.p3_rates's protocol).
  run1d   N4/N5 at q = 1: gate 7's ordering tests under A' at A = 1e-3, 5e-4, 2.5e-4
          (amp_scaling's job, SZ_J_WELL=node) -> nodewell_1d.json
  report  certification evaluation (certify_gates.evaluate_rows) of nodewell_1d.json and of
          q3_gate_runs_260x8_nodeA{1,2,4}.json (q3_gate.py --amp ... --tag nodeA*, SZ_J_WELL=node)
usage:  python3 nodewell_test.py sym | p3 | run1d | report
"""
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import math
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M

OUT1 = os.path.join(HERE, "nodewell_1d.json")
TAG = {1e-3: "nodeA1", 5e-4: "nodeA2", 2.5e-4: "nodeA4"}


def sym():
    VA = lambda p: (np.abs(p) ** 3).sum() / 3
    VN = lambda p: np.linalg.norm(p) ** 3 / 3
    rng = np.random.default_rng(1)
    for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10)):
        lat = M.Lattice(n=n, N=200)
        e0 = np.zeros(n, complex); e0[0] = 1
        for lab, U in (("segment A (axis 0)", M.U_segment(lat, 0, gA)),
                       ("segment B (axis 1)", M.U_segment(lat, 1, gB)),
                       ("both, AB", M.U_segment(lat, 1, gB) @ M.U_segment(lat, 0, gA))):
            p = U @ e0
            print(f"  n = {n}, {lab:20s}: V_A {VA(e0):.4f} -> {VA(p):.4f} (rel {VA(p)/VA(e0)-1:+.3f});"
                  f"  V_A' {VN(e0):.4f} -> {VN(p):.4f} (rel {VN(p)/VN(e0)-1:+.1e})")
        Z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); Q, _ = np.linalg.qr(Z)
        x = rng.normal(size=n) + 1j * rng.normal(size=n)
        print(f"  n = {n}, random U(n)         : V_A rel {VA(Q @ x)/VA(x)-1:+.3f};  V_A' rel {VN(Q @ x)/VN(x)-1:+.1e}")


def p3_rate(well, A, chi_deg, T=300.0, N=200):
    KS = float(M.KAPPA); k = math.pi / 2
    lat = M.Lattice(n=2, N=N, well=well)
    w = lat.omega; chi = math.radians(chi_deg)
    e = np.exp(1j * k * np.arange(N))
    u = np.zeros((N, 4)); v = np.zeros((N, 4))
    for j, a in enumerate((A * math.cos(chi), A * math.sin(chi))):
        psi = a * e; dpsi = -1j * w * psi
        u[:, 2 * j], u[:, 2 * j + 1] = psi.real, psi.imag
        v[:, 2 * j], v[:, 2 * j + 1] = dpsi.real, dpsi.imag
    ts, ph, t = [], [], 0.0
    for _ in range(30):
        u, v, _ = lat.run(u, v, T / 30); t += T / 30
        psi = u[:, 0::2] + 1j * u[:, 1::2]; dps = v[:, 0::2] + 1j * v[:, 1::2]
        S = ((psi + (1j / w) * dps) * np.conj(e)[:, None]).sum(axis=0)
        ts.append(t); ph.append(np.angle(np.conj(S[0]) * S[1]))
    rate = np.polyfit(ts, np.unwrap(ph), 1)[0]
    lin = A * (math.cos(chi) - math.sin(chi)) / (2 * w + KS)
    return rate, lin


def p3():
    print("   A     chi   rate under A    rate under A'   A-law (linear)   A'/A-law")
    for A, c in [(0.05, 25), (0.1, 25), (0.2, 25), (0.15, 10), (0.15, 65), (0.15, 80)]:
        ra, lin = p3_rate("radial", A, c); rn, _ = p3_rate("node", A, c)
        print(f"  {A:5.2f}  {c:3d}   {ra:+.6f}      {rn:+.3e}      {lin:+.6f}       {rn / lin:+.1e}")


def run1d():
    assert M.J_WELL == "node", "run with SZ_J_WELL=node"
    import amp_scaling as S
    jobs = [(n, gA, gB, axes, amp) for amp in S.AMPS for n, gA, gB in S.CASES for axes in ((0, 1), (0, 0))]
    with Pool(3) as p:
        res = p.map(S.job, jobs)
    json.dump(res, open(OUT1, "w"))


def q3_rows():
    import q3_gate as Q
    rows = []
    for amp, tag in TAG.items():
        f = os.path.join(HERE, f"q3_gate_runs_260x8_{tag}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f)); Q.KAPPA_RUN[0] = d["kappa"]
        runs = {(r["n"], r["job"]): r for r in d["runs"]}
        cache = {n: Q.spectrum(n, d["L0"], d["S"], d["kappa"]) for n in (2, 3)}
        for n in (2, 3):
            gA, gB = Q.G[n]; fA, fB = Q.GFLOOR[n]
            pr = {"AB": Q.predict_averaged(cache, n, [(0, gA), (1, gB)]), "BA": Q.predict_averaged(cache, n, [(1, gB), (0, gA)])}
            rows.append(dict(n=n, axes=[0, 1], amp=amp, co={"AB": runs[(n, "AB")]["co"], "BA": runs[(n, "BA")]["co"]},
                             lin={k: v.tolist() for k, v in pr.items()}))
            pf = {"AB": Q.predict_averaged(cache, n, [(0, fA), (0, fB)]), "BA": Q.predict_averaged(cache, n, [(0, fB), (0, fA)])}
            rows.append(dict(n=n, axes=[0, 0], amp=amp, co={"AB": runs[(n, "fAB")]["co"], "BA": runs[(n, "fBA")]["co"]},
                             lin={k: v.tolist() for k, v in pf.items()}))
    return rows


def report():
    import certify_gates as CG
    import amp_scaling as S
    for q, rows in ((1, json.load(open(OUT1)) if os.path.exists(OUT1) else []), (3, q3_rows())):
        if not rows:
            continue
        print(f"\n  q = {q} under A' -- deviations from the LINEAR prediction, per amplitude")
        for line in S.analyse(rows, f"q = {q}"):
            print(line)
        ok, lines, slopes = CG.evaluate_rows(rows, q)
        print(f"  certification under A': {'CERTIFIED' if ok else 'NOT CERTIFIED'}")
        print("\n".join(lines))


if __name__ == "__main__":
    {"sym": sym, "p3": p3, "run1d": run1d, "report": report}[sys.argv[1]]()
