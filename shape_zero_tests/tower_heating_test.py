#!/usr/bin/env python3
"""tower_heating_test.py -- tests the parametric-heating reading of the populated-tower gain
(PROVENANCE 6s, PREMISE_LEDGER C47). Predictions in tower_heating_predictions.txt, committed with this
script before any run. Same model and instruments as tower_populated_test.py: A' unchanged, C_r = 0,
no new coupling or parameter; q = 1 ring, N = 128, kappa*, packet 0.05 (width 8, k0 = pi/2, per-mode)
in the D <= 8 part (components 0-7); s(x) = |u_upper(x)|^2, the upper levels' combined size per site.

Runs (D16 = n 8 unless stated; upper complex components 4..n-1):
  A  amplitude scan: all 4 upper components incoherent (as tower_populated_test.py), rms A_U per
     component, A_U = 0.0025, 0.005, 0.01, 0.02; seeds 0, 1; T = 2000.
  V  variance scan at FIXED MEAN s = 4 * 0.005^2: components 4, 5 incoherent carrying a fraction x of
     the mean, components 6, 7 coherent and spatially uniform (one a-branch k = 0 mode each, so
     |psi_c|^2 is constant in space and time) carrying 1 - x. x = 1, 0.5, 0.25 (seeds 0, 1), x = 0
     (no site-to-site variation at t = 0; one run). T = 2000.
  U2 uniform but time-modulated: all 4 upper components one k = 0 mode on BOTH branches, equal
     amplitude (|psi_c|^2 uniform in space, oscillating in time at w_a(0) + w_b(0)), same mean s.
     T = 2000. Secondary.
  L  long runs: incoherent, A_U = 0.005: D16 seeds 0, 1 and D64 seed 0, T = 20000.
Measured per run: f(t) = dE_l,self / E_l,self(0); gain rate = slope of f fitted over [T/4, T];
mean_x s and var_x s averaged over the run; phase and |overlap| with the isolated packet at T; the
D <= 8 part's linear energy split between the a- and b-branches of its populated component
(diagnostic of pair creation).
usage: python3 tower_heating_test.py
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import tower_populated_test as TP  # noqa: E402

M = TP.M
N, AMP = TP.N, TP.AMP
A0 = 0.005
S_MEAN = 4 * A0 ** 2


def set_comp(u, v, c, psi, dps):
    u[:, 2 * c], u[:, 2 * c + 1] = psi.real, psi.imag
    v[:, 2 * c], v[:, 2 * c + 1] = dps.real, dps.imag


def incoherent(lat, u, v, comps, amp, seed):
    rng = np.random.default_rng(1000 + seed)
    wa = lat.branch_omega(); wb = wa + lat.kappa
    for c in comps:
        a = rng.normal(size=lat.N) + 1j * rng.normal(size=lat.N)
        b = rng.normal(size=lat.N) + 1j * rng.normal(size=lat.N)
        psi = np.fft.ifft(a + b); dps = np.fft.ifft(-1j * wa * a + 1j * wb * b)
        s = amp / np.sqrt(np.mean(np.abs(psi) ** 2))
        set_comp(u, v, c, s * psi, s * dps)


def coherent_uniform(lat, u, v, comps, amp):
    wa0 = lat.branch_omega()[0]
    for c in comps:
        psi = amp * np.exp(1j * 2.39996 * c) * np.ones(lat.N)       # fixed phases (golden angle)
        set_comp(u, v, c, psi, -1j * wa0 * psi)


def uniform_modulated(lat, u, v, comps, amp):
    wa0 = lat.branch_omega()[0]; wb0 = wa0 + lat.kappa
    for c in comps:
        al = amp / np.sqrt(2) * np.exp(1j * 2.39996 * c) * np.ones(lat.N)
        be = amp / np.sqrt(2) * np.exp(-1j * 2.39996 * c) * np.ones(lat.N)
        set_comp(u, v, c, al + be, -1j * wa0 * al + 1j * wb0 * be)


def branch_energy(lat, u, v):
    """Linear energy of the D<=8 part's populated component 0, split into a- and b-branches."""
    wa = lat.branch_omega(); wb = wa + lat.kappa
    P = np.fft.fft(u[:, 0] + 1j * u[:, 1]); D = np.fft.fft(v[:, 0] + 1j * v[:, 1])
    a = (wb * P + 1j * D) / (wa + wb); b = (wa * P - 1j * D) / (wa + wb)
    Q = M.SQ5 + 2 * M.C * (1 - np.cos(2 * np.pi * np.fft.fftfreq(lat.N)))
    Ea = 0.5 * ((wa ** 2 + Q) * np.abs(a) ** 2).sum() / lat.N
    Eb = 0.5 * ((wb ** 2 + Q) * np.abs(b) ** 2).sum() / lat.N
    return float(Ea), float(Eb)


def run(job):
    label, n, kind, par, seed, T, dts = job
    t0 = time.time()
    lat = M.Lattice(n=n, N=N, well="node")
    u, v = lat.packet(amp=AMP, n0=N // 2, width=8.0, per_mode=True)
    up = list(range(4, n))
    if kind == "incoh":
        incoherent(lat, u, v, up, par, seed)
    elif kind == "mix":                                   # fixed-mean variance scan, x = par
        x = par
        if x > 0:
            incoherent(lat, u, v, [4, 5], np.sqrt(2 * x) * A0, seed)
        if x < 1:
            coherent_uniform(lat, u, v, [6, 7], np.sqrt(2 * (1 - x)) * A0)
    elif kind == "umod":
        uniform_modulated(lat, u, v, up, A0)
    lo, hi = slice(0, 8), slice(8, lat.D)
    E0 = lat.energy(u, v)
    rec = dict(t=[], El=[], Ql=[], Qu=[], ms=[], vs=[], Ea=[], Eb=[], chi=[])
    every = max(1, int(round(10.0 / dts)))

    def sample(t, u, v, k):
        rec["t"].append(t); rec["El"].append(TP.self_energy(lat, u, v, lo))
        rec["Ql"].append(TP.charge(lat, u, v, lo))
        rec["Qu"].append(TP.charge(lat, u, v, hi) if n > 4 else 0.0)
        s = (u[:, 8:] ** 2).sum(axis=1) if n > 4 else np.zeros(lat.N)
        rec["ms"].append(s.mean()); rec["vs"].append(s.var())
        Ea, Eb = branch_energy(lat, u, v); rec["Ea"].append(Ea); rec["Eb"].append(Eb)
        if k % every == 0:
            rec["chi"].append(TP.chi_low(lat, u, v)[0].copy())

    sample(0.0, u, v, 0)
    t, k = 0.0, 0
    while t < T - 1e-9:
        u, v, _ = lat.run(u, v, dts); t += dts; k += 1
        sample(t, u, v, k)
    out = {kk: np.array(vv) for kk, vv in rec.items()}
    out.update(label=label, n=n, kind=kind, par=par, seed=seed, T=T,
               dE_total=abs(lat.energy(u, v) - E0) / abs(E0), secs=time.time() - t0)
    return out


def rate(ts, f, a, b):
    m = (ts >= a) & (ts <= b)
    p, cov = np.polyfit(ts[m], f[m], 1, cov=True)
    return p[0], np.sqrt(cov[0, 0])


def summary(r, ref):
    ts, T = r["t"], r["T"]
    f = (r["El"] - r["El"][0]) / r["El"][0]
    g, ge = rate(ts, f, T / 4, T)
    O = np.array([TP.overlap(a, b) for a, b in zip(ref["chi"], r["chi"])])
    ph = np.unwrap(np.angle(O))
    return dict(f=f, g=g, ge=ge, fT=f[-1], fl=f[ts >= 3 * T / 4].mean(), fmin=f.min(), ms=r["ms"].mean(), vs=r["vs"].mean(),
                vs0=r["vs"][0], ph=ph[-1], OT=abs(O[-1]),
                dEa=(r["Ea"][-1] - r["Ea"][0]) / r["El"][0], dEb=(r["Eb"][-1] - r["Eb"][0]) / r["El"][0],
                dQl=np.abs(r["Ql"] / r["Ql"][0] - 1).max(),
                dQu=np.abs(r["Qu"] - r["Qu"][0]).max() / abs(r["Ql"][0]))


def line(r, s):
    return (f"  {r['label']:<22s} seed {r['seed']}: gain rate {s['g']:+.3e} (+- {s['ge']:.1e}) /t; f(T) {s['fT']:+.3e}, late mean {s['fl']:+.3e}, "
            f"min f {s['fmin']:+.2e}; <s> {s['ms']:.3e}, <var s> {s['vs']:.3e} (t=0 {s['vs0']:.3e}); phase at T "
            f"{s['ph']:+.3f} rad, |O_ref| {s['OT']:.4f}; dE_a {s['dEa']:+.2e}, dE_b {s['dEb']:+.2e}; charge D<=8 "
            f"{s['dQl']:.1e}, upper {s['dQu']:.1e} of |Q_l|; E_tot {r['dE_total']:.1e} ({r['secs']:.0f} s)")


def main():
    jobs = [("L D64", 32, "incoh", A0, 0, 20000.0, 10.0)]            # longest first
    jobs += [("L D16", 8, "incoh", A0, s, 20000.0, 10.0) for s in (0, 1)]
    jobs += [("ref T20000", 4, "none", 0, 0, 20000.0, 10.0), ("ref T2000", 4, "none", 0, 0, 2000.0, 2.0)]
    jobs += [(f"A A_U={a}", 8, "incoh", a, s, 2000.0, 2.0) for a in (0.0025, 0.005, 0.01, 0.02) for s in (0, 1)]
    jobs += [(f"V x={x}", 8, "mix", x, s, 2000.0, 2.0) for x in (1.0, 0.5, 0.25) for s in (0, 1)]
    jobs += [("V x=0", 8, "mix", 0.0, 0, 2000.0, 2.0), ("U2 uniform modulated", 8, "umod", A0, 0, 2000.0, 2.0)]
    with Pool(4) as p:
        res = p.map(run, jobs, chunksize=1)
    ref = {r["T"]: r for r in res if r["kind"] == "none"}
    S = {}
    print(f"TOWER HEATING TEST (A'), N = {N}, packet {AMP}; mean-s target {S_MEAN:.1e}")
    for rf in ref.values():
        print(f"  reference T = {rf['T']:g}: D<=8 self-energy drift {abs(rf['El'][-1] / rf['El'][0] - 1):.1e}")
    for head, pre in (("(1) AMPLITUDE SCAN, D16, 4 incoherent components", "A "),
                      ("(1)/(3)/(4) VARIANCE SCAN AT FIXED MEAN, D16", "V "),
                      ("(3) UNIFORM, TIME-MODULATED (secondary)", "U2"),
                      ("(2) LONG RUNS, T = 20000", "L ")):
        print(); print(head)
        for r in res:
            if r["label"].startswith(pre):
                s = summary(r, ref[r["T"]]); S[(r["label"], r["seed"])] = (r, s)
                print(line(r, s))
    print(); print("FITS AND RATIOS")
    amps = (0.0025, 0.005, 0.01, 0.02)
    g = np.array([np.mean([S[(f"A A_U={a}", s)][1]["g"] for s in (0, 1)]) for a in amps])
    ph = np.array([np.mean([S[(f"A A_U={a}", s)][1]["ph"] for s in (0, 1)]) for a in amps])
    ms = np.array([np.mean([S[(f"A A_U={a}", s)][1]["ms"] for s in (0, 1)]) for a in amps])
    vs = np.array([np.mean([S[(f"A A_U={a}", s)][1]["vs"] for s in (0, 1)]) for a in amps])
    ok = g > 0
    print(f"  amplitude scan: seed-mean rates {', '.join(f'{x:.3e}' for x in g)}")
    if ok.sum() >= 2:
        print(f"    rate ~ A_U^p: p = {np.polyfit(np.log(np.array(amps)[ok]), np.log(g[ok]), 1)[0]:.2f}; "
              f"rate ~ <var s>^a: a = {np.polyfit(np.log(vs[ok]), np.log(g[ok]), 1)[0]:.2f}")
    print(f"    phase at T {', '.join(f'{x:+.3f}' for x in ph)}; |phase| ~ <s>^gamma: gamma = "
          f"{np.polyfit(np.log(ms), np.log(np.abs(ph)), 1)[0]:.2f}")
    xs = (1.0, 0.5, 0.25)
    gv = np.array([np.mean([S[(f"V x={x}", s)][1]["g"] for s in (0, 1)]) for x in xs])
    vv = np.array([np.mean([S[(f"V x={x}", s)][1]["vs"] for s in (0, 1)]) for x in xs])
    pv = np.array([np.mean([S[(f"V x={x}", s)][1]["ph"] for s in (0, 1)]) for x in xs])
    g0, p0, v0 = S[("V x=0", 0)][1]["g"], S[("V x=0", 0)][1]["ph"], S[("V x=0", 0)][1]["vs"]
    gu, pu = S[("U2 uniform modulated", 0)][1]["g"], S[("U2 uniform modulated", 0)][1]["ph"]
    print(f"  fixed-mean scan x = 1, 0.5, 0.25, 0: rates {', '.join(f'{x:.3e}' for x in gv)}, {g0:.3e}; "
          f"relative to x = 1: {', '.join(f'{x / gv[0]:.3f}' for x in gv)}, {g0 / gv[0]:.3f}")
    ok = gv > 0
    if ok.sum() >= 2:
        print(f"    rate ~ <var s>^a at fixed mean: a = {np.polyfit(np.log(vv[ok]), np.log(gv[ok]), 1)[0]:.2f} "
              f"(<var s> {', '.join(f'{x:.2e}' for x in vv)}, x=0: {v0:.2e})")
    allp = np.concatenate([pv, [p0]])
    print(f"    phase at T {', '.join(f'{x:+.3f}' for x in pv)}, {p0:+.3f}; spread (max-min)/mean "
          f"{(allp.max() - allp.min()) / abs(allp.mean()):.3f}")
    ga = np.mean([S[("A A_U=0.005", s)][1]["g"] for s in (0, 1)])
    pa = np.mean([S[("A A_U=0.005", s)][1]["ph"] for s in (0, 1)])
    print(f"  U2: rate {gu:.3e} = {gu / ga:.3f} of the incoherent rate at the same mean; phase {pu:+.3f} "
          f"= {pu / pa:.3f} of its phase")
    for lab, seeds in (("L D16", (0, 1)), ("L D64", (0,))):
        for sd in seeds:
            r, s = S[(lab, sd)]
            g1, _ = rate(r["t"], s["f"], 2000, 7000); g2, _ = rate(r["t"], s["f"], 15000, 20000)
            print(f"  {lab} seed {sd}: rate [2000,7000] {g1:.3e}, [15000,20000] {g2:.3e}, ratio {g2 / g1:.3f}; "
                  f"f(20000) {s['fT']:+.4f}; f at 5000/10000/15000 "
                  + ", ".join(f"{s['f'][np.argmin(np.abs(r['t'] - tt))]:+.4f}" for tt in (5000, 10000, 15000)))


if __name__ == "__main__":
    main()
