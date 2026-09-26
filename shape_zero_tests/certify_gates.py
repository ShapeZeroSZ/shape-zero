#!/usr/bin/env python3
"""
certify_gates.py -- the amplitude-scaling CERTIFICATION TEST of the linear gauge claim
(adopted 2026-09-26; MODEL_SPEC sec 4d). Run when the model changes, not routinely.

Runs gate 7 (q = 1, model.ordering_test, clearing readout) and q3_gate.py (q = 3) under
the working model at amplitudes A, A/2, A/4 (A = 1e-3), and fits every deviation from the
LINEAR prediction against amplitude (amp_scaling.py does the runs and the fits).

CERTIFIED iff, for every quantity:
  1. small-amplitude limit --
     q3_gate: every intercept (Abelian floor, signed split error, per-order vector-fit
       intercept) within max(2 sigma, 0.01 deg) of zero;
     gate 7: the floor intercept within max(2 sigma, 0.01 deg); the split and per-order
       intercepts within 1 deg (the single-carrier prediction's own error is expected there);
  2. linearity -- straight-line residuals < 5% of the largest deviation (floor, split);
     for per-order errors, of each Bloch-vector component's range;
  3. slopes -- reported (the first-order self-precession; the target for its derivation).

usage:  python3 certify_gates.py run        (the runs: ~30 min + ~1 h on 4 cores)
        python3 certify_gates.py evaluate   (from the saved runs)
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import amp_scaling as S

M = S.M


def run():
    subprocess.run([sys.executable, "-u", os.path.join(HERE, "amp_scaling.py"), "run1d"], check=True)
    for amp in S.AMPS:
        subprocess.run([sys.executable, "-u", os.path.join(HERE, "q3_gate.py"), "--amp", repr(amp),
                        "--tag", S.TAG[amp]], check=False)


def evaluate_rows(rows, q):
    ok_all, lines, slopes = True, [], {}
    ang = lambda a, b: M.angle(np.array(a), np.array(b))
    for n in (2, 3):
        for kind, axes in (("split", [0, 1]), ("floor", [0, 0])):
            rr = sorted([r for r in rows if r["n"] == n and r["axes"] == axes], key=lambda r: -r["amp"])
            amps = np.array([r["amp"] for r in rr])
            if kind == "floor":
                y = [ang(r["co"]["AB"], r["co"]["BA"]) for r in rr]
                a, b, sa, res = S.fit(amps, y)
                zero = abs(a) < max(2 * sa, 0.01)
                lin = res < 0.05 * max(y)
                slopes[f"u({n}) floor"] = b * 1e-3
                ok = zero and lin
                lines.append(f"   u({n}) floor: intercept {a:+.5f} +- {sa:.5f} (zero: {zero}); linear: {lin}; "
                             f"slope {b * 1e-3:.5f} deg per 1e-3 -> {'ok' if ok else 'FAIL'}")
                ok_all &= ok
                continue
            ms = [ang(r["co"]["AB"], r["co"]["BA"]) for r in rr]
            ps = [ang(r["lin"]["AB"], r["lin"]["BA"]) for r in rr]
            y = [m - p for m, p in zip(ms, ps)]
            a, b, sa, res = S.fit(amps, y)
            tol = max(2 * sa, 0.01) if q == 3 else 1.0
            lin = res < 0.05 * max(abs(v) for v in y)
            ok = abs(a) < tol and lin
            slopes[f"u({n}) split"] = b * 1e-3
            lines.append(f"   u({n}) split error: intercept {a:+.5f} +- {sa:.5f} (tolerance {tol:.3f}); linear: {lin}; "
                         f"slope {b * 1e-3:+.5f} per 1e-3 -> {'ok' if ok else 'FAIL'}")
            ok_all &= ok
            for o in ("AB", "BA"):
                C = np.array([r["co"][o] for r in rr])
                comps = [S.fit(amps, C[:, j]) for j in range(C.shape[1])]
                co0 = np.array([c[0] for c in comps])
                co1 = np.array([c[1] for c in comps])
                lin0 = np.array(rr[0]["lin"][o])
                sig0 = np.degrees(np.linalg.norm([c[2] for c in comps]) / max(np.linalg.norm(co0), 1e-12))
                i0 = ang(co0, lin0)
                tol = max(2 * sig0, 0.01) if q == 3 else 1.0
                linc = all(c[3] < 0.05 * max(np.ptp(C[:, j]), 1e-12) or np.ptp(C[:, j]) < 1e-9
                           for j, c in enumerate(comps))
                slope = ang(co0 + 1e-3 * co1, co0)
                slopes[f"u({n}) per-order {o}"] = slope
                ok = i0 < tol and linc
                lines.append(f"      per-order {o}: intercept {i0:.4f} +- {sig0:.4f} (tolerance {tol:.3f}); "
                             f"components linear: {linc}; slope {slope:.4f} deg per 1e-3 -> {'ok' if ok else 'FAIL'}")
                ok_all &= ok
    return ok_all, lines, slopes


def evaluate():
    print("CERTIFICATION TEST -- amplitude scaling of the deviations from the linear gauge prediction")
    import json
    res = {}
    for q, rows in ((1, json.load(open(S.OUT1))), (3, S.q3_rows())):
        ok, lines, slopes = evaluate_rows(rows, q)
        res[q] = slopes
        print(f"\n  q = {q} ({'gate 7' if q == 1 else 'q3_gate'}): {'CERTIFIED' if ok else 'NOT CERTIFIED'}")
        print("\n".join(lines))
    json.dump(res, open(os.path.join(HERE, "certify_gates_slopes.json"), "w"), indent=1)
    print("\n  slopes (deg per 1e-3 of amplitude) -> certify_gates_slopes.json")


if __name__ == "__main__":
    {"run": run, "evaluate": evaluate}[sys.argv[1]]()
