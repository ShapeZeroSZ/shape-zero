#!/usr/bin/env python3
"""
persist_sim_posthoc.py -- POST-HOC reading of persist_sim_runs.json (written after
the runs; the committed predictions are in persist_sim_predictions.txt).

1. R6's committed criterion evaluated as stated: the largest growth among a-modes
   with |D(k, p)| > 0.05 (the report printed only the overall largest, which were
   small-p modes that the criterion's caveat excluded).
2. The b-branch criterion as committed ("no b-mode grows more than 10x above its
   initial noise") is ILL-POSED: the launch put noise on the a-branch only, so the
   initial b content is rounding-level (~1e-20) and any driven, non-resonant b
   response exceeds it by >1e10. Replaced here, post hoc, by the absolute b levels
   and whether they grow between snapshots.

usage:  python3 persist_sim_posthoc.py
"""
import json

import numpy as np

import persist_sim as P


def main():
    d = json.load(open(P.OUT))
    pred, res = d["pred"], d["res"]
    q = 2 * np.pi * np.fft.fftfreq(P.N)
    print("POST HOC (written after the runs)")
    for name in ("R5", "R6"):
        m, k = pred[name]["m"], pred[name]["k"]
        a0 = np.array(res[name]["a0"])
        last = res[name]["snaps"][-1]
        a = np.array(last["a"])
        harm = {(j * m) % P.N for j in range(-6, 7)}
        p = q - k
        D = 2 * P.wa(k, P.KS) - P.wa(k + p, P.KS) - P.wa(k - p, P.KS)
        sel = [i for i in range(P.N) if i not in harm and a0[i] > 0 and abs(D[i]) > 0.05]
        g = {i: a[i] / a0[i] for i in sel}
        i = max(g, key=g.get)
        print(f"\n  {name}: pump k = {k:.4f}")
        print(f"     largest a-growth among modes with |D| > 0.05: x{g[i]:.2f} at q = {q[i]:+.4f} "
              f"(p = {p[i]:+.4f}, D = {D[i]:+.4f})"
              + ("   [R6 criterion: < 10x -> PASS]" if name == "R6" and g[i] < 10 else
                 ("   [R6 criterion: FAIL]" if name == "R6" else "")))
        small = [i for i in range(P.N) if i not in harm and a0[i] > 0 and abs(D[i]) <= 0.05]
        gs = {i: a[i] / a0[i] for i in small}
        j = max(gs, key=gs.get)
        print(f"     largest a-growth among modes with |D| <= 0.05: x{gs[j]:.2f} at p = {p[j]:+.4f} "
              f"(D = {D[j]:+.2e})")
        print("     b-branch, absolute: max |b| outside the pump's harmonic set, by snapshot:")
        for s in res[name]["snaps"]:
            b = np.array(s["b"])
            mask = [ii for ii in range(P.N) if ii not in harm and (-ii) % P.N not in harm]
            ib = max(mask, key=lambda ii: b[ii])
            print(f"        T = {s['T']:.0f}: max |b|/A = {b[ib] / 0.2:.2e} at b-wavevector {-q[ib]:+.4f}")


if __name__ == "__main__":
    main()
