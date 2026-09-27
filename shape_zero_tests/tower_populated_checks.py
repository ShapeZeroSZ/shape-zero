#!/usr/bin/env python3
"""tower_populated_checks.py -- POST HOC instrument checks on tower_populated_test.py, written after
its output was seen (predictions bc7c827). Not part of the pre-registered test.
 1. Upper-part charge: its relative drift reached 5.8e-2 (D64 seed 2). The both-branch incoherent
    state carries nearly cancelling charge, so the relative drift has a near-zero denominator. Report
    Q_u(0), the absolute drift, and the drift in units of the D<=8 charge.
 2. Time step: the D<=8 energy gain (f_late ~ +1.8e-3 at D16 seed 0) rerun at DT = 0.01 (was 0.02).
usage: python3 tower_populated_checks.py"""
import sys
from multiprocessing import Pool

import numpy as np

import tower_populated_test as TP


def job(args):
    n, seed, dt = args
    TP.M.DT = dt
    r = TP.run((n, TP.A_U, seed))
    ts = r["t"]; f = (r["El"] - r["El"][0]) / r["El"][0]
    q4 = ts >= 3 * TP.T / 4
    dQu = np.abs(r["Qu"] - r["Qu"][0]).max()
    return (f"D{2 * n} seed {seed} DT {dt}: Q_l(0) {r['Ql'][0]:+.4e}, Q_u(0) {r['Qu'][0]:+.4e}; max |Q_u - Q_u(0)| "
            f"{dQu:.2e} = {dQu / abs(r['Ql'][0]):.1e} of |Q_l(0)|; max |Q_l - Q_l(0)| "
            f"{np.abs(r['Ql'] - r['Ql'][0]).max():.2e}; f_late {f[q4].mean():+.3e}, slope {np.polyfit(ts, f, 1)[0]:+.2e}/t, "
            f"E_tot drift {r['dE_total']:.1e}")


if __name__ == "__main__":
    with Pool(3) as p:
        for line in p.map(job, [(32, 2, 0.02), (8, 0, 0.02), (8, 0, 0.01)]):
            print(line, flush=True)
