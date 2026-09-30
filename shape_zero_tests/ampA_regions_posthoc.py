#!/usr/bin/env python3
"""POST HOC (2026-09-30, after the evaluation of 000deb4): directions of the AFTER and IN contributions
relative to the BEFORE contribution and to the model vectors; region sums."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "ampA_regions_predictions.json")))
R = json.load(open(os.path.join(HERE, "ampA_regions_runs.json")))
Mk = json.load(open(os.path.join(HERE, "ampA_masked_runs.json")))
co = {(r["tag"], r["n"], r["order"], r["kind"]): np.array(r["co"]) for r in R}
for r in Mk:
    if r["tag"] == "cert":
        co[({1: "q1", 3: "q3"}[r["q"]], r["n"], r["order"], r["kind"])] = np.array(r["co"])
nrm = np.linalg.norm
proj = lambda v, m: v @ m / (m @ m)
cos = lambda a, b: a @ b / (nrm(a) * nrm(b))
for tag in ("q1", "q3"):
    for n in (2, 3):
        for o in ("AB", "BA"):
            lin = co[(tag, n, o, "lin")]
            d = {k: co[(tag, n, o, k)] - lin for k in ("BEFORE", "BETWEEN", "AFTER", "IN", "ALL")}
            m = {k: np.array(v) for k, v in P[f"{tag} u({n}) {o}"].items()}
            mA = m["ALL"] + m["K4_pre"] + m["K4_mid"]
            s = d["BEFORE"] + d["BETWEEN"] + d["AFTER"] + d["IN"]
            print(f"{tag} u({n}) {o}: cos(AFTER,BEFORE) {cos(d['AFTER'], d['BEFORE']):+.3f}; AFTER on model-ALL {proj(d['AFTER'], mA):+.3f}; "
                  f"BEFORE on model-ALL {proj(d['BEFORE'], mA):+.3f}; BETWEEN on model-ALL {proj(d['BETWEEN'], mA):+.3f}; "
                  f"IN on model-ALL {proj(d['IN'], mA):+.3f}; ALL on model-ALL {proj(d['ALL'], mA):.3f}; "
                  f"4-region sum vs ALL {nrm(s - d['ALL']) / nrm(d['ALL']):.4f}")
