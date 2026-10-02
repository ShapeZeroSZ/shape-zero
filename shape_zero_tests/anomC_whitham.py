#!/usr/bin/env python3
"""
anomC_whitham.py -- anomaly C (the beam's fourth-order cross kernel) in light of the Whitham result
(2026-10-02). Predictions: anomC_WHITHAM_PREDICTIONS.md (committed before any run). Design choices ours.

  predict  -> anomC_whitham_predictions.json/.txt
  run      -> anomC_whitham_runs.jsonl / .json  (L3 launch of kappa4_orbit3_launch.py, L = 4, T = 900)
  evaluate -> anomC_whitham_compare_output.txt

Fraction f = (r - r_S2) / (r_S1 - r_S2), r the beam's fourth-order growth as a fraction of the plane
wave's (kappa4_predict.py normalisation, r = fill X / F2):
  W-multi   Whitham multi-phase average (every component's phase independent): X = the dephased
            sextic moment, i.e. S1, f = 1.
  W-locked  one common phase (a phase-locked stationary beam), envelope weighting: X = <e^6>/<e^2>^3.
  W-totals  E/I at every order: factor 2/p per order (1/2 at second, 1/3 at fourth) -- excluded by
            the recorded second order (F2 from frequency PT matches the beams); no prediction.
  K-transfer (ours, not from the rule): the second-order cross-term survival
            f2 = (F2 - P2)/(2 - 2 P2) carried to fourth order, f = f2.
  0.9-fill  the post-hoc pattern of the recorded reading, f = 0.9 fill (extrapolated).
"""
import os
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kappa_cross_pt as X
import kappa4_orbit_launch as O

L = 4
WIDTHS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
AMPS = (0.10, 0.30, 0.40)
T = 900.0


def geom(w):
    e = X.env2d(L, w)
    c = np.fft.fft2(e) / (L * L)
    p = np.abs(c) ** 2
    fill = float(p.sum()); pn = p / fill
    s = float(pn[0, 0]); P2 = float((pn ** 2).sum())
    F2 = float(X.F_triads(L, w))
    rS1 = fill * (6 - 3 * P2 - 6 * s + 4 * s * s) / F2
    rS2 = fill * s * s / F2
    rcoh = fill * float(np.mean(e ** 6) / np.mean(e ** 2) ** 3) / F2
    f2 = (F2 - P2) / (2 - 2 * P2)
    return dict(fill=fill, s=s, P2=P2, F2=F2, rS1=rS1, rS2=rS2,
                f={"W-multi": 1.0, "W-locked": (rcoh - rS2) / (rS1 - rS2), "K-transfer": f2, "0.9-fill": 0.9 * fill})


def predict():
    res, lines = {}, ["PREDICTIONS (committed before any run): fraction f between S2 (0) and S1 (1)",
                      "     w   fill    r_S2    r_S1  |  W-multi  W-locked  K-transfer  0.9-fill"]
    for w in WIDTHS:
        g = geom(w)
        res[str(w)] = g
        lines.append(f"   {w:3.1f}  {g['fill']:.4f}  {g['rS2']:.4f}  {g['rS1']:.4f}  |  {g['f']['W-multi']:6.3f}  "
                     f"{g['f']['W-locked']:7.3f}  {g['f']['K-transfer']:9.3f}  {g['f']['0.9-fill']:8.3f}")
    print("\n".join(lines))
    json.dump(res, open(os.path.join(HERE, "anomC_whitham_predictions.json"), "w"), indent=1)
    open(os.path.join(HERE, "anomC_whitham_predictions.txt"), "w").write("\n".join(lines) + "\n")


def job(a):
    import kappa4_orbit3_launch as K
    w, A = a
    k, e = K.run_case(L, w, A, T)
    return dict(w=w, A=A, kappa=float(k), err=float(e))


def run():
    part = os.path.join(HERE, "anomC_whitham_runs.jsonl")
    done = {(r["w"], r["A"]) for r in map(json.loads, open(part))} if os.path.exists(part) else set()
    todo = [(w, A) for w in WIDTHS for A in AMPS if (w, A) not in done]
    with Pool(4) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, todo, chunksize=1):
            f.write(json.dumps(r) + "\n"); f.flush()
    json.dump([json.loads(l) for l in open(part)], open(os.path.join(HERE, "anomC_whitham_runs.json"), "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] in ("predict", "run"):
    {"predict": predict, "run": run}[sys.argv[1]]()


# ------------------------------------------------------------------ evaluation (written after the runs; criteria fixed in 202ca3f)
def evaluate():
    P = json.load(open(os.path.join(HERE, "anomC_whitham_predictions.json")))
    R = {(r["w"], r["A"]): r for r in json.load(open(os.path.join(HERE, "anomC_whitham_runs.json")))}
    g = {A: O.growth(A) for A in AMPS}
    lines = ["EVALUATION against anomC_WHITHAM_PREDICTIONS.md (202ca3f); L = 4, T = 900",
             "     w     A   ratio             r_meas           f_meas            fits (|df| <= 2 sigma)"]
    for w in WIDTHS:
        p = P[str(w)]
        b = R[(w, 0.10)]
        for A in (0.30, 0.40):
            r_ = R[(w, A)]
            rat = r_["kappa"] / b["kappa"]
            re = abs(rat) * math.hypot(r_["err"] / r_["kappa"], b["err"] / b["kappa"])
            rm = (rat - 1) / (g[A] - rat * g[0.10])
            drm = re * abs((g[A] - g[0.10]) / (g[A] - rat * g[0.10]) ** 2)
            fm = (rm - p["rS2"]) / (p["rS1"] - p["rS2"]); dfm = drm / (p["rS1"] - p["rS2"])
            fits = [k for k, v in p["f"].items() if abs(v - fm) <= 2 * dfm]
            lines.append(f"   {w:3.1f}  {A:4.2f}   {rat:.5f}+-{re:.5f}   {rm:.4f}+-{drm:.4f}   {fm:+.3f}+-{dfm:.3f}   {', '.join(fits) or 'none'}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "anomC_whitham_compare_output.txt"), "w").write(out + "\n")


if __name__ == "__main__" and sys.argv[1] == "evaluate":
    evaluate()
