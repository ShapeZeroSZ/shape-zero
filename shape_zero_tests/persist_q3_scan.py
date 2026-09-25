#!/usr/bin/env python3
"""
persist_q3_scan.py -- the (kappa, c) scan of persist_resonance.py at q = 3.

Same channel functions and open/closed rule (a channel is open iff m kappa lies in
the range of its function), with 3-D ranges from multistart local optimisation
(persist_resonance.ranges_3d) -- numerical optima, not proved extrema. The
J-compatibility floor for the model's slab segments is kappa* = 2c/sqrt(K + 2c)
(transverse momentum conserved; MODEL_SPEC §3, "ADOPTED").

Requirements scanned: three-wave b->aa closed; three-wave a->aa (and the other
m >= 0 three-wave combinations) closed; four-wave aa->ab and aa->bb closed;
1->3 closed. The same-branch four-wave aa->aa is open everywhere and is reported,
not required.

usage:  python3 persist_q3_scan.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from multiprocessing import Pool

import numpy as np

import persist_resonance as P

KGRID = np.round(np.arange(0.1, 12.01, 0.1), 2)
CS = (0.1, 0.25, 0.5, 1.0, 2.0)


def one(args):
    kap, c = args
    r = P.ranges_3d(kap, c)
    return kap, c, P.channel_table(r, kap)


def spans(ks, mask):
    out, s, prev = [], None, None
    for k, m in zip(ks, mask):
        if m and s is None:
            s = k
        if not m and s is not None:
            out.append(f"[{s:.1f},{prev:.1f}]"); s = None
        prev = k
    if s is not None:
        out.append(f"[{s:.1f},{ks[-1]:.1f}+]")
    return " ".join(out) or "none"


def main():
    jobs = [(k, c) for c in CS for k in KGRID]
    with Pool(4) as p:
        res = p.map(one, jobs)
    print("=" * 96)
    print("q = 3: kappa ranges where each requirement holds (numerical optima)")
    print("=" * 96)
    print("   c     c/sqrt5  kappa*   3w b->aa closed        3w a->aa closed   4w aa->ab closed   1->3 closed")
    for c in CS:
        rows = [r for r in res if r[1] == c]
        ks = [r[0] for r in rows]
        t = [r[2] for r in rows]
        a = [not x["3w b->aa (= a+a->b)"] for x in t]
        b = [not (x["3w a->aa, b->ab"] or x["3w a->ab, b->bb"] or x["3w a->bb"]) for x in t]
        d = [not (x["4w aa->ab, bb->ab"] or x["4w aa->bb, bb->aa"]) for x in t]
        e = [not any(v for kk, v in x.items() if kk.startswith("1->3")) for x in t]
        s4 = all(x["4w aa->aa, bb->bb (p!=0)"] for x in t)
        kst = P.kappa_star(c)
        allm = [aa and bb and dd and ee and k >= kst for aa, bb, dd, ee, k in zip(a, b, d, e, ks)]
        print(f"   {c:4.2f}  {c / P.K:.3f}    {kst:.4f}   {spans(ks, a):22s} {spans(ks, b):17s} "
              f"{spans(ks, d):18s} {spans(ks, e)}")
        print(f"         ALL (with kappa >= kappa*): {spans(ks, allm)};  4w aa->aa open at every kappa: {s4}")


if __name__ == "__main__":
    main()
