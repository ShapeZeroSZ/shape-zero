#!/usr/bin/env python3
"""Evaluate ampA_masked_runs.json against ampA_MASKED_PREDICTIONS.md (e890e2a): IN size, additivity,
IN/OUT cosine, ALL vs recorded vector-fit slopes, and the factor (projection ratio onto the model)."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SZ_J_WELL"] = "node"
import amp_scaling as S
R = json.load(open(os.path.join(HERE, "ampA_masked_runs.json")))
P = json.load(open(os.path.join(HERE, "ampA_mechanisms_predictions.json")))
SEP = None  # separated model: vectors not stored; recompute projection from ampA_mechanisms
by = {(r["q"], r["tag"], r["n"], r["order"], r["kind"]): np.array(r["co"]) for r in R}


def co1(rows, n, o):
    rr = sorted([r for r in rows if r["n"] == n and r["axes"] == [0, 1]], key=lambda r: -r["amp"])
    amps = np.array([r["amp"] for r in rr]); C = np.array([r["co"][o] for r in rr])
    return np.array([S.fit(amps, C[:, j])[1] for j in range(C.shape[1])]) * 1e-3


lines = []
rec = {1: json.load(open(S.OUT1)), 3: S.q3_rows()}
for q in (1, 3):
    for n in (2, 3):
        key = f"q={q} u({n})"
        for o in ("AB", "BA"):
            lin = by[(q, "cert", n, o, "lin")]
            d = {k: by[(q, "cert", n, o, k)] - lin for k in ("IN", "OUT", "ALL")}
            nA = np.linalg.norm(d["ALL"])
            add = np.linalg.norm(d["IN"] + d["OUT"] - d["ALL"]) / nA
            cos_io = d["IN"] @ d["OUT"] / (np.linalg.norm(d["IN"]) * np.linalg.norm(d["OUT"]))
            m = np.array(P[key]["total"]["dco"][o])
            fac = lambda v: v @ m / (m @ m)
            c1 = co1(rec[q], n, o)
            lines.append(f"  {key} {o}: |IN|/|ALL| {np.linalg.norm(d['IN']) / nA:.3f}; additivity {add:.3f}; "
                         f"cos(IN,OUT) {cos_io:+.3f}; ALL vs recorded co1: |ALL|/|co1| {nA / np.linalg.norm(c1):.3f}, "
                         f"cos {d['ALL'] @ c1 / (nA * np.linalg.norm(c1)):+.4f}; factor OUT {fac(d['OUT']):.3f}, ALL {fac(d['ALL']):.3f}, IN {fac(d['IN']):+.3f}")
# separated geometry: factor from the model's separated vectors
import ampA_mechanisms as X
rows = json.load(open(os.path.join(HERE, "nodewell_1d.json")))
T1 = {r["n"]: r["T"] for r in rows if r["amp"] == 1e-3 and r["axes"] == [0, 1]}
for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10)):
    sp = {"AB": [(0, gA), (1, gB)], "BA": [(1, gB), (0, gA)]}
    for o, s in sp.items():
        mdl = X.Model(1, n, s, T1[n] + 110.0, X.AMP_LIN, segs=(60, 110)); A, _ = X.aeff_sites(mdl)
        lin = mdl.run_lin()
        m = X.SCALE * (mdl.coords(mdl.run_angle(A) + mdl.run_born()) - mdl.coords(lin))
        dv = by[(1, "sep", n, o, "ALL")] - by[(1, "sep", n, o, "lin")]
        dc = by[(1, "cert", n, o, "ALL")] - by[(1, "cert", n, o, "lin")]
        mc = np.array(P[f"q=1 u({n})"]["total"]["dco"][o])
        lines.append(f"  q=1 separated u({n}) {o}: factor ALL {dv @ m / (m @ m):.3f} (certification geometry {dc @ mc / (mc @ mc):.3f}); "
                     f"cos {dv @ m / (np.linalg.norm(dv) * np.linalg.norm(m)):+.3f}; |sep|/|cert| measured {np.linalg.norm(dv) / np.linalg.norm(dc):.3f}, model {np.linalg.norm(m) / np.linalg.norm(mc):.3f}")
out = "\n".join(lines)
print(out)
open(os.path.join(HERE, "ampA_masked_compare_output.txt"), "w").write(out + "\n")
