#!/usr/bin/env python3
"""
ampA_whitham.py -- anomaly A, 2026-10-01: the 2/p law from the conserved quantities (Whitham), tested on
a third power. Predictions: ampA_WHITHAM_PREDICTIONS.md (committed before any run). Design choices ours.

  predict  -> ampA_whitham_predictions.json/.txt   (model numbers, plus a numerical check of E/I)
  run      -> ampA_whitham_runs.jsonl / .json        (incremental)
  evaluate -> ampA_whitham_compare_output.txt

Diagnostic well (p = 6; a diagnostic only, NOT a change of premise, defined here, not in model.py):
F = -(sqrt5 + |u|^4) u on the whole node's radius, energy |u|^6/6.
Runs:
  packet: single segment (u(2), axis 0, g = 0.12), the width-scan geometry of ampA_regions (N = 2000,
          n0 = 150, start 300, T = 700), widths 16 and 32, A = 0.05 and 0.035; linear runs reused.
  stationary: the flat-top geometry of ampA_power (window readout), A = 0.05, plus its linear run
          reused from ampA_power_runs.json.
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
import ampA_mechanisms as X
import ampA_regions as RG
import ampA_power as PW

P6_WIDTHS = (16.0, 32.0)
P6_AMPS = (0.05, 0.035)
P6_STAT_AMP = 0.05


class Sextic(M.Lattice):
    """diagnostic only: -(sqrt5 + |u|^4) u, energy |u|^6/6."""
    def __init__(self, n, N):
        super().__init__(n=n, N=N, well="node")
        self.well = "sextic"

    def _onsite_nl(self, u):
        return ((u * u).sum(axis=1, keepdims=True) ** 2) * u

    def _onsite_cubic(self, u):
        return (((u * u).sum(axis=1)) ** 3).sum() / 6


# ------------------------------------------------------------------ model side
def scan_model(w, amp, p):
    """eikonal angle law with dw = |psi|^(p-2)/D, weight <F^p>/<F^2> at site crossings (factor 1)."""
    s = RG.make_setup(1, (RG.WS["N"],), RG.WS["n0"], w, (RG.WS["start"],))
    mdl = RG.RModel(1, 2, [(0, RG.WS["g"])], RG.WS["T"], amp, s)
    ph = amp * s["phi_hat"] * mdl.mask
    V = []
    for i in range(len(M.RAMP)):
        t = (RG.WS["start"] + i - s["n0"]) / X.V0
        F = np.abs(np.fft.ifft(ph * np.exp(-1j * s["om"] * t)))
        V.append((F ** p).sum() / (F ** 2).sum())
    lin = mdl.coords(mdl.run_lin())
    return (mdl.coords(mdl.run_angle([V])) - lin).tolist()


def ei_check(p, amp=0.05, width=16.0):
    """numerical check of the Whitham identity on a lattice packet: (E_nl - E_lin)/I vs (2/p) <dOmega>.
    E from model.Lattice.energy with the p-well; I = sum (2 w + kappa)|a|^2/2 from the per-mode a-branch
    amplitude; <dOmega> = <F^p>/<F^2>/D. Launch at amplitude amp, measured after relaxation (t = 50)."""
    cls = {3: lambda: M.Lattice(n=2, N=400, well="node"), 4: lambda: M.Lattice(n=2, N=400, well="smooth"),
           6: lambda: Sextic(2, 400)}[p]
    lat = cls()
    linl = M.Lattice(n=2, N=400, well="elementwise")
    u, v = lat.packet(amp=amp, n0=200, width=width, per_mode=True)
    u, v, _ = lat.run(u, v, 50.0)
    wa = lat.branch_omega(); wb = wa + lat.kappa
    psi = np.fft.fft(u[:, 0] + 1j * u[:, 1]); dps = np.fft.fft(v[:, 0] + 1j * v[:, 1])
    a = (wb * psi + 1j * dps) / (wa + wb)
    I = ((2 * wa + lat.kappa) * np.abs(a) ** 2).sum() / 2 / lat.N
    E_tot = lat.energy(u, v)
    E_nl = {3: lambda F: (F ** 3).sum() / 3, 4: lambda F: (F ** 4).sum() / 4, 6: lambda F: (F ** 6).sum() / 6}[p]
    F = np.sqrt((u * u).sum(1))
    E_lin = E_tot - E_nl(F)
    # linear-part energy per action of the same field = action-weighted linear frequency
    w_lin = ((2 * wa + lat.kappa) * wa * np.abs(a) ** 2).sum() / ((2 * wa + lat.kappa) * np.abs(a) ** 2).sum()
    EI = E_tot / I
    dOm = (F ** p).sum() / (F ** 2).sum() / X.D0
    return dict(E_over_I=EI, w_lin=w_lin, ratio=(EI - w_lin) / dOm)


def predict():
    res, lines = {}, ["PREDICTIONS (committed before any run)"]
    for w in P6_WIDTHS:
        for amp in P6_AMPS:
            d = scan_model(w, amp, 6)
            res[f"p6 w{w} A{amp}"] = {"dco": d}
            dd = np.degrees(np.linalg.norm(d))
            lines.append(f"  p = 6 packet width {w:>4} A = {amp}: eikonal {dd:.7f} deg; Whitham E/I (x 1/3) {dd / 3:.7f}; "
                         f"x 1/2 {dd / 2:.7f}; x 2/3 {2 * dd / 3:.7f}")
    V = P6_STAT_AMP ** 4
    ex, wk = PW.stationary_model(V)
    res["p6 stationary"] = {"deg_exact": float(np.degrees(ex)), "V": V}
    lines.append(f"  p = 6 stationary A = {P6_STAT_AMP}: V = {V:.3e}, flux-ratio (factor 1) {np.degrees(ex):.7f} deg; "
                 f"totals (x 1/3) {np.degrees(ex) / 3:.7f}")
    for p in (3, 4, 6):
        c = ei_check(p)
        res[f"EI check p{p}"] = c
        lines.append(f"  Whitham identity on a lattice packet, p = {p}: (E/I - w_lin)/<dOmega> = {c['ratio']:.4f} "
                     f"(derived 2/p = {2 / p:.4f})")
    print("\n".join(lines))
    json.dump(res, open(os.path.join(HERE, "ampA_whitham_predictions.json"), "w"), indent=1)
    open(os.path.join(HERE, "ampA_whitham_predictions.txt"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ lattice side
def job(a):
    kind, w, amp = a
    if kind == "scan":
        lat = Sextic(2, RG.WS["N"])
        W, Wm = M.make_links(lat, [(RG.WS["start"], 0, RG.WS["g"])])
        u, v = lat.packet(width=w, n0=RG.WS["n0"], amp=amp, per_mode=True)
        u, v, drift = lat.run(u, v, RG.WS["T"], W, Wm)
        co, _ = lat.readout(u, v)
        return dict(kind=kind, w=w, amp=amp, co=co.tolist(), drift=float(drift))
    lat = Sextic(2, PW.ST["N"])
    W, Wm = M.make_links(lat, [(PW.ST["start"], 0, PW.ST["g"])])
    u, v = PW.flat_packet(lat, amp)
    u, v, drift = lat.run(u, v, PW.ST["T"], W, Wm)
    lo, hi = PW.ST["win"]; mid = (lo + hi) // 2
    co = {"win": PW.window_coords(lat, u, v, lo, hi), "sub1": PW.window_coords(lat, u, v, lo, mid),
          "sub2": PW.window_coords(lat, u, v, mid, hi)}
    return dict(kind=kind, w=None, amp=amp, co=co, drift=float(drift))


def run():
    part = os.path.join(HERE, "ampA_whitham_runs.jsonl")
    J = [("scan", w, a) for w in P6_WIDTHS for a in P6_AMPS] + [("stat", None, P6_STAT_AMP)]
    done = {(r["kind"], r["w"], r["amp"]) for r in map(json.loads, open(part))} if os.path.exists(part) else set()
    todo = [a for a in J if a not in done]
    with Pool(4) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, todo, chunksize=1):
            f.write(json.dumps(r) + "\n"); f.flush()
    json.dump([json.loads(l) for l in open(part)], open(os.path.join(HERE, "ampA_whitham_runs.json"), "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] in ("predict", "run"):
    {"predict": predict, "run": run}[sys.argv[1]]()


# ------------------------------------------------------------------ evaluation (written after the runs; criteria fixed in e928ab1)
def evaluate():
    P = json.load(open(os.path.join(HERE, "ampA_whitham_predictions.json")))
    R = json.load(open(os.path.join(HERE, "ampA_whitham_runs.json")))
    RR = json.load(open(os.path.join(HERE, "ampA_regions_runs.json")))
    PR = json.load(open(os.path.join(HERE, "ampA_power_runs.json")))
    nrm = np.linalg.norm
    deg = lambda v: float(np.degrees(nrm(v)))
    L_ = ["EVALUATION against ampA_WHITHAM_PREDICTIONS.md (e928ab1)", "\n(2) p = 6 packet, single segment"]
    lin = {float(r["tag"].split("_")[1]): np.array(r["co"]) for r in RR if r["tag"].startswith("w_") and r["kind"] == "lin"}
    sc = {(r["w"], r["amp"]): np.array(r["co"]) for r in R if r["kind"] == "scan"}
    for w in P6_WIDTHS:
        dd = {}
        for amp in P6_AMPS:
            d = sc[(w, amp)] - lin[w]; m = np.array(P[f"p6 w{w} A{amp}"]["dco"]); dd[amp] = d
            L_.append(f"  width {w:>4} A = {amp}: measured {deg(d):.7f} deg, eikonal {deg(m):.7f}; factor {d @ m / (m @ m):.4f}; "
                      f"cos {d @ m / (nrm(d) * nrm(m)):+.4f}")
        L_.append(f"    fourth-order check: ratio {nrm(dd[0.05]) / nrm(dd[0.035]):.4f} (expected {(0.05 / 0.035) ** 4:.4f})")
    st = [r for r in R if r["kind"] == "stat"][0]
    sl = [r for r in PR if r["kind"] == "stat" and r["well"] == "lin"][0]
    pr = P["p6 stationary"]["deg_exact"]
    vals = [deg(np.array(st["co"][k]) - np.array(sl["co"][k])) for k in ("win", "sub1", "sub2")]
    L_.append(f"\n  p = 6 stationary: rotation change {vals[0]:.7f} deg (halves {vals[1]:.7f} / {vals[2]:.7f}); "
              f"flux-ratio prediction {pr:.7f}; factor {vals[0] / pr:.4f}")
    L_.append(f"  drifts: max {max(r['drift'] for r in R):.1e}")
    out = "\n".join(L_)
    print(out)
    open(os.path.join(HERE, "ampA_whitham_compare_output.txt"), "w").write(out + "\n")


if __name__ == "__main__" and sys.argv[1] == "evaluate":
    evaluate()
