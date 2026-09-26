#!/usr/bin/env python3
"""
radialA_tests.py -- two tests of variant (A), the radial well (model_radial.py).

TEST 1 -- gates 7 and 8 with a CLEARING readout. model.py's gate 7/8 read at a fixed
T = 180 on N = 200 (MODEL_SPEC §4d.1 trap 6: the packet's tail is still in the
second window). Under (A) the Abelian floor rose from 0.050 to 0.161 deg. Two
hypotheses:
  H-trap  : the rise is the fixed-time readout -- under clearing it goes to ~0.
  H-prec  : (A) makes the spinor self-precession FIRST order in amplitude
            (Omega = A (cos chi - sin chi)/(2w + kappa), P-3 under (A)). Between the
            two segments the spinor precesses about z by an amount that depends on
            which segment came first, so commuting x-rotations no longer commute in
            effect. Estimate: ~2.4e-4 rad/unit time x ~43 time units between segments
            x a population difference ~0.3 -> ~0.17 deg. This survives clearing and
            scales LINEARLY with amplitude.
Prediction (H-prec): under clearing, the (A) u(2) floor stays >= 0.05 deg at amplitude
1e-3 and falls to 0.5 +- 0.1 of that at amplitude 5e-4; the elementwise floor under
clearing is < 1e-3 deg. The fixed-time rows reproduce gate 7's floors (0.161 / 0.050).
Protocol: gate 7's own geometry (segments at 60, 80, RAMP, width 8, n0 = 20), per-mode
launch (as model.py's gate 7), N = 1200, both orders of a pair read at ONE common time
1.1 x the later clearing time (every window < 1e-6), as gate7_readout.py does.

TEST 2 -- P-3 under (A). A uniform circular spinor wave, dimer amplitudes A cos chi and
A sin chi, has |psi_j| constant, so each dimer sees stiffness Q + A_j and turns at the
exact root of w^2 + kappa w = Q + A_j. Spinor precession rate (phase of conj(S0) S1):
    Omega = w_0 - w_1 = A (cos chi - sin chi)/(2w + kappa) + O(A^2)
-- LINEAR in A, and with angular factor cos chi - sin chi, not the elementwise P-3 law
C A^2 n_z. Prediction: measured Omega within 2% of the exact root difference and of the
linear formula, at A = 0.05, 0.1, 0.2 and chi = 10, 25, 65, 80 deg; Omega/(A^2 n_z) not
constant (spread > 20% across A).

usage:  python3 radialA_tests.py predict | run | report
"""
import os
# model.py's J sector defaults to the radial well since 2026-09-26; this script's
# "model.py" results are the ELEMENTWISE form, so pin it (before model is imported).
os.environ["SZ_J_WELL"] = "elementwise"
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import math
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_radial as MRAD
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as MS

KS = float(MS.KAPPA)
SEG, WIDTH, N0, THR = (60, 80), 8.0, 20, 1e-6
OUT = os.path.join(HERE, "radialA_runs.json")
MODS = {"elementwise": MS, "radial": MRAD}


# ------------------------------------------------------------------ test 1
def windows(lat, u, v):
    w = (u * u + (v * v) / lat.omega ** 2).sum(axis=1)
    idx = np.arange(lat.N)
    return [float(w[(idx >= s - 10) & (idx < s + len(MS.RAMP) + 10)].sum() / w.sum()) for s in SEG]


def specs(gA, gB, axes):
    a0, a1 = axes
    return (("AB", [(SEG[0], a0, gA), (SEG[1], a1, gB)], (1, 0)),
            ("BA", [(SEG[0], a1, gB), (SEG[1], a0, gA)], (0, 1)))


def ordering(job):
    model, n, gA, gB, axes, N, T, amp = job
    M = MODS[model]
    lat = M.Lattice(n=n, N=N, kappa=KS)
    psi0 = np.zeros(n, complex); psi0[0] = 1
    if T == "clear":
        tc = []
        for _, sp, _ in specs(gA, gB, axes):
            W, Wm = M.make_links(lat, sp)
            u, v = lat.packet(width=WIDTH, n0=N0, amp=amp, per_mode=True)
            t, seen = 0.0, False
            while t < 4000:
                u, v, _ = lat.run(u, v, 5.0, W, Wm); t += 5.0
                fr = windows(lat, u, v)
                seen = seen or max(fr) > 1e-3
                if seen and max(fr) < THR:
                    break
            tc.append(t)
        T = 1.1 * max(tc)
    res = {}
    for name, sp, order in specs(gA, gB, axes):
        W, Wm = M.make_links(lat, sp)
        u, v = lat.packet(width=WIDTH, n0=N0, amp=amp, per_mode=True)
        u, v, drift = lat.run(u, v, T, W, Wm)
        co, _ = lat.readout(u, v)
        gs = (gA, gB) if order == (1, 0) else (gB, gA)
        ax = axes if order == (1, 0) else (axes[1], axes[0])
        Up = MS.U_segment(lat, ax[1], gs[1]) @ MS.U_segment(lat, ax[0], gs[0])
        res[name] = dict(co=co.tolist(), pred=MS.coords_of_state(lat, Up @ psi0).tolist(),
                         win=windows(lat, u, v), drift=drift)
    return dict(job=[model, n, gA, gB, list(axes), N, T if isinstance(T, float) else T, amp], T=T, res=res)


JOBS1 = []
for model in ("elementwise", "radial"):
    for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10)):
        for axes in ((0, 1), (0, 0)):
            JOBS1.append((model, n, gA, gB, axes, 200, 180.0, 1e-3))
            JOBS1.append((model, n, gA, gB, axes, 1200, "clear", 1e-3))
JOBS1.append(("radial", 2, 0.12, 0.08, (0, 0), 1200, "clear", 5e-4))
JOBS1.append(("elementwise", 2, 0.12, 0.08, (0, 0), 1200, "clear", 5e-4))


# ------------------------------------------------------------------ test 2
def p3_rates(A, chi_deg, T=300.0, N=200):
    k = math.pi / 2
    lat = MRAD.Lattice(n=2, N=N, kappa=KS)
    w = lat.omega
    chi = math.radians(chi_deg)
    n = np.arange(N)
    e = np.exp(1j * k * n)
    u = np.zeros((N, 4)); v = np.zeros((N, 4))
    for j, a in enumerate((A * math.cos(chi), A * math.sin(chi))):
        psi = a * e; dpsi = -1j * w * psi
        u[:, 2 * j], u[:, 2 * j + 1] = psi.real, psi.imag
        v[:, 2 * j], v[:, 2 * j + 1] = dpsi.real, dpsi.imag
    ts, ph = [], []
    t = 0.0
    for _ in range(30):
        u, v, _ = lat.run(u, v, T / 30); t += T / 30
        psi = u[:, 0::2] + 1j * u[:, 1::2]
        dps = v[:, 0::2] + 1j * v[:, 1::2]
        chi_f = psi + (1j / w) * dps
        S = (chi_f * np.conj(e)[:, None]).sum(axis=0)
        ts.append(t); ph.append(np.angle(np.conj(S[0]) * S[1]))
    rate = np.polyfit(ts, np.unwrap(ph), 1)[0]
    Q = MS.SQ5 + 2 * MS.C * (1 - math.cos(k))
    root = lambda a: 0.5 * (-KS + math.sqrt(KS * KS + 4 * (Q + a)))
    exact = root(A * math.cos(chi)) - root(A * math.sin(chi))
    lin = A * (math.cos(chi) - math.sin(chi)) / (2 * w + KS)
    return dict(A=A, chi=chi_deg, rate=rate, exact=exact, lin=lin, nz=math.cos(2 * chi))


CASES2 = [(A, 25) for A in (0.05, 0.1, 0.2)] + [(0.15, c) for c in (10, 25, 65, 80)]


def predict():
    print(__doc__.split("usage:")[0])


def run():
    with Pool(4) as p:
        r1 = p.map(ordering, JOBS1)
    r2 = [p3_rates(A, c) for A, c in CASES2]
    json.dump(dict(t1=r1, t2=r2), open(OUT, "w"))


def report():
    d = json.load(open(OUT))
    print("TEST 1 -- gates 7 and 8, fixed-time (N = 200, T = 180) vs clearing (N = 1200, one common T)")
    print("   model        n  axes    readout      amp      T      split/floor (deg)  pred split  per-order AB/BA  max window")
    for r in d["t1"]:
        model, n, gA, gB, axes, N, T, amp = r["job"]
        a, b = r["res"]["AB"], r["res"]["BA"]
        split = MS.angle(np.array(a["co"]), np.array(b["co"]))
        pred = MS.angle(np.array(a["pred"]), np.array(b["pred"]))
        eA = MS.angle(np.array(a["co"]), np.array(a["pred"]))
        eB = MS.angle(np.array(b["co"]), np.array(b["pred"]))
        kind = "floor" if axes == [0, 0] else "split"
        print(f"   {model:11s}  {n}  {str(tuple(axes)):7s} {'fixed T' if N == 200 else 'clearing':10s} {amp:.0e}  "
              f"{r['T']:6.0f}  {kind} {split:9.5f}      {pred:8.3f}    {eA:.3f} / {eB:.3f}     "
              f"{max(a['win'] + b['win']):.1e}")
    fl = {(r["job"][0], r["job"][7]): MS.angle(np.array(r["res"]["AB"]["co"]), np.array(r["res"]["BA"]["co"]))
          for r in d["t1"] if r["job"][1] == 2 and r["job"][4] == [0, 0] and r["job"][5] == 1200}
    print(f"   u(2) floor under clearing, amplitude 5e-4 / 1e-3: radial {fl[('radial', 5e-4)] / fl[('radial', 1e-3)]:.3f}, "
          f"elementwise {fl[('elementwise', 5e-4)]:.2e} / {fl[('elementwise', 1e-3)]:.2e} deg")
    print("\nTEST 2 -- P-3 under (A): spinor precession rate")
    print("      A     chi   measured     exact roots   linear formula   meas/exact  meas/linear   rate/(A^2 n_z)")
    for r in d["t2"]:
        print(f"   {r['A']:5.2f}   {r['chi']:3d}   {r['rate']:+.6f}   {r['exact']:+.6f}     {r['lin']:+.6f}        "
              f"{r['rate'] / r['exact']:.4f}      {r['rate'] / r['lin']:.4f}       {r['rate'] / (r['A'] ** 2 * r['nz']):+.4f}")


if __name__ == "__main__":
    {"predict": predict, "run": run, "report": report}[sys.argv[1]]()
