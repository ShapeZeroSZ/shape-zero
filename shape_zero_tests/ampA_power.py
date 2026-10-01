#!/usr/bin/env python3
"""
ampA_power.py -- anomaly A, 2026-10-01: (1) a STATIONARY nonlinear wave through one segment, and
(2) the well-power test of the candidate "energy per action vs frequency shift" (ratio 2/p).
Predictions: ampA_POWER_PREDICTIONS.md (committed before any run). Design choices ours.

  predict  -> ampA_power_predictions.json
  run      -> ampA_power_runs.jsonl / .json (incremental)
  evaluate -> ampA_power_compare_output.txt

(1) Stationary: q = 1, N = 1200, one segment (u(2), axis 0, g = 0.12, start 520). A flat-top wave,
    envelope 0.5 [tanh((x - 100)/15) - tanh((x - 400)/15)], amplitude A, per-mode launch at k0 = pi/2.
    Read at T = 770 in the window 560 <= x < 660 (the steady transmitted plateau; the back edge is still
    upstream of the segment), with sub-windows [560, 610) and [610, 660) as a steadiness check.
    Wells: A' (A = 1e-3) and the smooth diagnostic (A = 1e-2), each against the linear run.
(2) Smooth well (p = 4) in the width-scan geometry of ampA_regions (single segment, widths 8, 16, 32;
    N = 2000, n0 = 150, start 300, T = 700), A = 1e-2 and 5e-3; the linear runs are reused from
    ampA_regions_runs.json (linear dynamics: Bloch coordinates independent of amplitude).
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

ST = dict(N=1200, start=520, a=100.0, b=400.0, ell=15.0, T=770.0, win=(560, 660), g=0.12)
PW_WIDTHS = (8.0, 16.0, 32.0)
PW_AMPS = (1e-2, 5e-3)


# ------------------------------------------------------------------ model side
def stationary_model(V):
    """exact stationary transfer-matrix rotation change (rad) for uniform potential V (K3b)."""
    ratio, exact, wkb = X.k3b_ratio(ST["g"], V=1e-4)
    return exact * V, wkb * V


def smooth_scan_model(w, amp):
    """eikonal angle law with the smooth well: dw = |psi|^2 / D, weight <F^4>/<F^2> at site crossings."""
    s = RG.make_setup(1, (RG.WS["N"],), RG.WS["n0"], w, (RG.WS["start"],))
    mdl = RG.RModel(1, 2, [(0, RG.WS["g"])], RG.WS["T"], amp, s)
    ax = (0,)
    ph = amp * s["phi_hat"] * mdl.mask
    V = []
    for i in range(len(M.RAMP)):
        t = (RG.WS["start"] + i - s["n0"]) / X.V0
        F = np.abs(np.fft.ifftn(ph * np.exp(-1j * s["om"] * t), axes=ax))
        V.append((F ** 4).sum() / (F ** 2).sum())
    lin = mdl.coords(mdl.run_lin())
    return (mdl.coords(mdl.run_angle([V])) - lin).tolist(), V[6]


def predict():
    res, lines = {}, ["PREDICTIONS (committed before any run)"]
    for well, A, V in (("node", 1e-3, 1e-3), ("smooth", 1e-2, 1e-4)):
        ex, wk = stationary_model(V)
        res[f"stationary {well}"] = {"rad_exact": ex, "rad_wkb": wk, "deg_exact": float(np.degrees(ex))}
        lines.append(f"  stationary {well:<6} A = {A:.0e}: V = {V:.0e}, rotation change {np.degrees(ex):.6f} deg "
                     f"(exact transfer matrix; WKB {np.degrees(wk):.6f})")
    for w in PW_WIDTHS:
        for amp in PW_AMPS:
            d, Vc = smooth_scan_model(w, amp)
            res[f"smooth w{w} A{amp}"] = {"dco": d, "V_centre": Vc}
            lines.append(f"  smooth width {w:>4} A = {amp:.0e}: eikonal rotation change {np.degrees(np.linalg.norm(d)):.6f} deg; "
                         f"candidate (x 1/2) {0.5 * np.degrees(np.linalg.norm(d)):.6f}; universal 2/3 {np.degrees(np.linalg.norm(d)) * 2 / 3:.6f}")
    print("\n".join(lines))
    json.dump(res, open(os.path.join(HERE, "ampA_power_predictions.json"), "w"), indent=1)
    open(os.path.join(HERE, "ampA_power_predictions.txt"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ lattice side
def flat_packet(lat, amp):
    x = np.arange(lat.N, dtype=float)
    env = 0.5 * (np.tanh((x - ST["a"]) / ST["ell"]) - np.tanh((x - ST["b"]) / ST["ell"]))
    psi = amp * env * np.exp(1j * M.K0 * x)
    dpsi = np.fft.ifft(-1j * lat.branch_omega() * np.fft.fft(psi))
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    u[:, 0], u[:, 1], v[:, 0], v[:, 1] = psi.real, psi.imag, dpsi.real, dpsi.imag
    return u, v


def window_coords(lat, u, v, lo, hi):
    psi = u[lo:hi, 0::2] + 1j * u[lo:hi, 1::2]
    dps = v[lo:hi, 0::2] + 1j * v[lo:hi, 1::2]
    chi = psi + (1j / lat.omega) * dps
    rs = chi.T @ chi.conj()
    tr = np.real(np.trace(rs))
    return [float(np.real(np.trace(S @ rs)) / tr) for S in lat.G]


def job(a):
    kind, well, w, amp = a
    if kind == "stat":
        lat = M.Lattice(n=2, N=ST["N"], well=("elementwise" if well == "lin" else well))
        if well == "lin":
            lat = RG.MaskedW(2, ST["N"], np.zeros(ST["N"]))
        W, Wm = M.make_links(lat, [(ST["start"], 0, ST["g"])])
        u, v = flat_packet(lat, amp)
        u, v, drift = lat.run(u, v, ST["T"], W, Wm)
        lo, hi = ST["win"]; mid = (lo + hi) // 2
        co = {"win": window_coords(lat, u, v, lo, hi), "sub1": window_coords(lat, u, v, lo, mid),
              "sub2": window_coords(lat, u, v, mid, hi)}
        amp_win = float(np.sqrt((u[lo:hi, :2] ** 2).sum(1)).mean())
        return dict(kind=kind, well=well, amp=amp, co=co, amp_win=amp_win, drift=float(drift))
    lat = M.Lattice(n=2, N=RG.WS["N"], well="smooth")
    W, Wm = M.make_links(lat, [(RG.WS["start"], 0, RG.WS["g"])])
    u, v = lat.packet(width=w, n0=RG.WS["n0"], amp=amp, per_mode=True)
    u, v, drift = lat.run(u, v, RG.WS["T"], W, Wm)
    co, _ = lat.readout(u, v)
    return dict(kind=kind, well=well, w=w, amp=amp, co=co.tolist(), drift=float(drift))


def jobs():
    J = [("scan", "smooth", w, a) for w in PW_WIDTHS for a in PW_AMPS]
    J += [("stat", "lin", None, 1e-3), ("stat", "node", None, 1e-3), ("stat", "smooth", None, 1e-2)]
    return J


def run():
    part = os.path.join(HERE, "ampA_power_runs.jsonl")
    key = lambda r: (r["kind"], r["well"], r.get("w"), r["amp"])
    done = {key(json.loads(l)) for l in open(part)} if os.path.exists(part) else set()
    todo = [a for a in jobs() if (a[0], a[1], a[2], a[3]) not in done]
    with Pool(4) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, todo, chunksize=1):
            f.write(json.dumps(r) + "\n"); f.flush()
    json.dump([json.loads(l) for l in open(part)], open(os.path.join(HERE, "ampA_power_runs.json"), "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] in ("predict", "run"):
    {"predict": predict, "run": run}[sys.argv[1]]()
