#!/usr/bin/env python3
"""
amp_scaling.py -- does the radial well's deviation from the LINEAR gauge prediction
scale linearly with amplitude and vanish at A = 0?

Runs (radial well, model.py default since 2026-09-26), amplitudes A = 1e-3, 5e-4, 2.5e-4:
  gate 7 (q = 1): model.ordering_test(readout="clear"), u(2) (0.12/0.08) and u(3)
    (0.15/0.10), axes (0,1) (split) and (0,0) (Abelian floor), N = 1200, one common
    readout time per pair after every window < 1e-6.
  q3_gate (q = 3): q3_gate.py --amp A (260 x 8 x 8, per-mode launch, one readout time per
    pair); runs saved as q3_gate_runs_260x8_ampA*.json.
Deviations are measured against the LINEAR prediction (no self-precession): at q = 1 the
single-carrier product U_B U_A (model.precession_prediction with the elementwise lattice,
which returns the plain product); at q = 3 q3_gate's spectrum-averaged predictor.

Quantities, for each amplitude:
  floor           angle(AB, BA) with commuting axes (linear theory: exactly 0)
  split error     measured split - linear predicted split (signed)
  per-order       angle(measured, linear prediction), AB and BA
Fits: straight line through the three amplitudes, dev(A) = a + b A, for the floor and the
signed split error; for the per-order errors, a straight-line fit to each Bloch-vector
COMPONENT, co(A) = co0 + A co1, then the intercept deviation angle(co0, linear prediction)
(the magnitude of a fixed error plus a term linear in A is not itself linear in A).

usage:  python3 amp_scaling.py predict | run1d | report
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

AMPS = (1e-3, 5e-4, 2.5e-4)
CASES = ((2, 0.12, 0.08), (3, 0.15, 0.10))
OUT1 = os.path.join(HERE, "amp_scaling_1d.json")
TAG = {1e-3: "ampA1", 5e-4: "ampA2", 2.5e-4: "ampA4"}


def predict():
    print("=" * 92)
    print("PREDICTIONS (committed before any run) -- amplitude scaling under the radial well")
    print("=" * 92)
    print("""  TOLERANCE, fixed now: an intercept is ZERO if |a| < max(2 sigma_a, 0.01 deg), sigma_a
  from the straight-line fit (3 points, 1 dof); 0.01 deg is a numerical floor well above
  the measured elementwise clearing floor (4e-5 deg). LINEAR if the 3-point fit residuals
  are < 5% of the largest deviation (for vector fits: of each component's range).

  HYPOTHESIS. The radial well adds a first-order self-precession on top of the LINEAR
  dynamics. So every deviation from the linear DYNAMICS is linear in A and vanishes at
  A = 0. But the linear PREDICTION is not the linear dynamics everywhere:
  - q = 1 (gate 7): the single-carrier product has its own error, measured with the
    elementwise well (second-order nonlinearity, ~ A -> 0) under the clearing readout:
    per-order 0.12 / 0.27 deg (u(2)), 0.40 / 0.26 deg (u(3)); split error +0.26 deg (u(2)),
    -0.02 deg (u(3)); floor 4e-5 deg. PREDICTED intercepts: the floor -> ZERO (u(2),
    u(3)); the signed split error -> +0.26 (u(2), NOT zero) and -0.02 (u(3), at the
    tolerance); the per-order vector-fit intercepts -> 0.12 / 0.27 and 0.40 / 0.26 deg (NOT
    zero). So at q = 1 the claim "the linear gauge prediction is exact at A -> 0" is
    PREDICTED TO FAIL for the split and per-order errors -- by the single-carrier
    approximation, not by anything nonlinear.
  - q = 3 (q3_gate): the spectrum-averaged predictor matched the linear dynamics to
    0.001-0.005 deg (elementwise, 2026-09-25). PREDICTED: every intercept (floor, split
    error, per-order) is ZERO within tolerance.
  - Slopes: all LINEAR. Expected size, deg per 1e-3 of amplitude: 1-D floors 0.160 (u(2)),
    0.096 (u(3)) (radialA_tests.py); q = 3 floors ~0.10 (u(2)), ~0.05 (u(3)); per-order
    precession ~3-4 deg per 1e-3 at q = 1, ~0.15-0.45 at q = 3.""")


def job(args):
    n, gA, gB, axes, amp = args
    r = M.ordering_test(n, gA, gB, axes=axes, readout="clear", amp=amp)
    lat_lin = M.Lattice(n=n, N=1200, well="elementwise")
    psi0 = np.zeros(n, complex); psi0[0] = 1
    a0, a1 = axes
    specs = {"AB": [(60, a0, gA), (80, a1, gB)], "BA": [(60, a1, gB), (80, a0, gA)]}
    lin = {k: M.precession_prediction(lat_lin, sp, r["T"], 8.0, 20, amp, psi0).tolist()
           for k, sp in specs.items()}
    return dict(n=n, axes=list(axes), amp=amp, T=r["T"], co={k: r[k][0].tolist() for k in ("AB", "BA")},
                lin=lin, drift=max(r["AB"][3], r["BA"][3]))


def run1d():
    jobs = [(n, gA, gB, axes, amp) for amp in AMPS for n, gA, gB in CASES for axes in ((0, 1), (0, 0))]
    with Pool(4) as p:
        res = p.map(job, jobs)
    json.dump(res, open(OUT1, "w"))


def fit(x, y):
    x, y = np.asarray(x), np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    dof = len(x) - 2
    s2 = (r @ r) / dof if dof > 0 else 0.0
    cov = s2 * np.linalg.inv(A.T @ A)
    return coef[0], coef[1], np.sqrt(cov[0, 0]), np.abs(r).max()


def verdicts(a, sa, resid, scale):
    zero = abs(a) < max(2 * sa, 0.01)
    lin = resid < 0.05 * scale if scale > 0 else True
    return zero, lin


def analyse(rows, label):
    """rows: list of dicts with amp, co{AB,BA} (split/floor pair), lin{AB,BA}."""
    out = []
    for n in (2, 3):
        for kind, axes in (("split", [0, 1]), ("floor", [0, 0])):
            rr = sorted([r for r in rows if r["n"] == n and r["axes"] == axes], key=lambda r: -r["amp"])
            if not rr:
                continue
            amps = np.array([r["amp"] for r in rr])
            ang = lambda a, b: M.angle(np.array(a), np.array(b))
            if kind == "floor":
                y = [ang(r["co"]["AB"], r["co"]["BA"]) for r in rr]
                a, b, sa, res = fit(amps, y)
                z, l = verdicts(a, sa, res, max(y))
                out.append(f"  {label} u({n}) floor: " + ", ".join(f"A={A:.2e}: {v:.5f}" for A, v in zip(amps, y))
                           + f" | fit a = {a:+.5f} +- {sa:.5f} deg, slope {b * 1e-3:.5f} deg per 1e-3, max resid {res:.1e}"
                           + f" | intercept ZERO: {'yes' if z else 'NO'}; linear: {'yes' if l else 'NO'}")
            else:
                ms = [ang(r["co"]["AB"], r["co"]["BA"]) for r in rr]
                ps = [ang(r["lin"]["AB"], r["lin"]["BA"]) for r in rr]
                y = [m - p for m, p in zip(ms, ps)]
                a, b, sa, res = fit(amps, y)
                z, l = verdicts(a, sa, res, max(abs(v) for v in y))
                out.append(f"  {label} u({n}) split error (signed): " + ", ".join(f"A={A:.2e}: {v:+.5f}" for A, v in zip(amps, y))
                           + f" | fit a = {a:+.5f} +- {sa:.5f}, slope {b * 1e-3:+.5f} per 1e-3, max resid {res:.1e}"
                           + f" | ZERO: {'yes' if z else 'NO'}; linear: {'yes' if l else 'NO'}")
                for o in ("AB", "BA"):
                    mags = [ang(r["co"][o], r["lin"][o]) for r in rr]
                    C = np.array([r["co"][o] for r in rr])
                    comps = [fit(amps, C[:, j]) for j in range(C.shape[1])]
                    co0 = np.array([c[0] for c in comps])
                    lin0 = np.array(rr[0]["lin"][o])
                    lin_ok = all(c[3] < 0.05 * max(np.ptp(C[:, j]), 1e-12) or np.ptp(C[:, j]) < 1e-9
                                 for j, c in enumerate(comps))
                    # intercept uncertainty in degrees: propagate component sigmas
                    sig0 = np.degrees(np.linalg.norm([c[2] for c in comps]) / max(np.linalg.norm(co0), 1e-12))
                    i0 = ang(co0, lin0)
                    out.append(f"      per-order {o}: " + ", ".join(f"{v:.4f}" for v in mags)
                               + f" deg | vector-fit intercept angle {i0:.4f} +- {sig0:.4f} deg"
                               + f" | ZERO: {'yes' if i0 < max(2 * sig0, 0.01) else 'NO'};"
                               + f" components linear: {'yes' if lin_ok else 'NO'}")
    return out


def q3_rows():
    import q3_gate as Q
    rows = []
    for amp in AMPS:
        f = os.path.join(HERE, f"q3_gate_runs_260x8_{TAG[amp]}.json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        Q.KAPPA_RUN[0] = d["kappa"]
        runs = {(r["n"], r["job"]): r for r in d["runs"]}
        cache = {n: Q.spectrum(n, d["L0"], d["S"], d["kappa"]) for n in (2, 3)}
        for n in (2, 3):
            gA, gB = Q.G[n]
            fA, fB = Q.GFLOOR[n]
            pr = {"AB": Q.predict_averaged(cache, n, [(0, gA), (1, gB)]),
                  "BA": Q.predict_averaged(cache, n, [(1, gB), (0, gA)])}
            rows.append(dict(n=n, axes=[0, 1], amp=amp, co={"AB": runs[(n, "AB")]["co"], "BA": runs[(n, "BA")]["co"]},
                             lin={k: v.tolist() for k, v in pr.items()}))
            pf = {"AB": Q.predict_averaged(cache, n, [(0, fA), (0, fB)]),
                  "BA": Q.predict_averaged(cache, n, [(0, fB), (0, fA)])}
            rows.append(dict(n=n, axes=[0, 0], amp=amp, co={"AB": runs[(n, "fAB")]["co"], "BA": runs[(n, "fBA")]["co"]},
                             lin={k: v.tolist() for k, v in pf.items()}))
    return rows


def report():
    print("MEASURED (radial well; deviations from the LINEAR prediction)")
    if os.path.exists(OUT1):
        for line in analyse(json.load(open(OUT1)), "q = 1 gate 7"):
            print(line)
    for line in analyse(q3_rows(), "q = 3 q3_gate"):
        print(line)


if __name__ == "__main__":
    {"predict": predict, "run1d": run1d, "report": report}[sys.argv[1]]()
