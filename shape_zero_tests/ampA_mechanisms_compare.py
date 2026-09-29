#!/usr/bin/env python3
"""Compare ampA_mechanisms_predictions.json (committed 8b61310) with the recorded A' certification
(nodewell_1d.json, q3 nodeA* runs): slopes, patterns and the first-order Bloch-vector direction."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["SZ_J_WELL"] = "node"
import amp_scaling as S
P = json.load(open(os.path.join(HERE, "ampA_mechanisms_predictions.json")))
meas_sl = json.load(open(os.path.join(HERE, "certify_gates_slopes_nodewell.json")))


def co1(rows, n, o):
    rr = sorted([r for r in rows if r["n"] == n and r["axes"] == [0, 1]], key=lambda r: -r["amp"])
    amps = np.array([r["amp"] for r in rr]); C = np.array([r["co"][o] for r in rr])
    return np.array([S.fit(amps, C[:, j])[1] for j in range(C.shape[1])]) * 1e-3


ok = lambda p, m: (np.sign(p) == np.sign(m)) and abs(p - m) <= 0.3 * abs(m) + 0.002
lines = []
for q, rows in ((1, json.load(open(S.OUT1))), (3, S.q3_rows())):
    ms = meas_sl[str(q)]
    for n in (2, 3):
        key = f"q={q} u({n})"
        lines.append(f"\n  {key}")
        for qty, mk in (("split", f"u({n}) split"), ("per-order AB", f"u({n}) per-order AB"),
                        ("per-order BA", f"u({n}) per-order BA")):
            m = ms[mk]
            row = f"    {qty:<13} measured {m:+.5f} |"
            for var in ("angle", "total"):
                p = P[key][var][qty]
                row += f" {var} {p:+.5f} ({'ok' if ok(p, m) else 'no'}, meas/pred {m / p:.2f}) |"
            lines.append(row)
        for o in ("AB", "BA"):
            c = co1(rows, n, o)
            row = f"    vector {o}: |co1| {np.linalg.norm(c):.2e}"
            for var in ("angle", "total"):
                d = np.array(P[key][var]["dco"][o])
                cos = c @ d / (np.linalg.norm(c) * np.linalg.norm(d))
                row += f" | {var}: cos {cos:+.3f}, |meas|/|pred| {np.linalg.norm(c) / np.linalg.norm(d):.2f}, projection ratio {c @ d / (d @ d):.2f}"
            lines.append(row)
out = "\n".join(lines)
print(out)
open(os.path.join(HERE, "ampA_mechanisms_compare_output.txt"), "w").write(out + "\n")
