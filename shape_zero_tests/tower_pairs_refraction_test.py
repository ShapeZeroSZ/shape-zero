#!/usr/bin/env python3
"""tower_pairs_refraction_test.py -- (1) pair creation behind the populated-tower gain; (2) universal
refraction by the populated tower. Predictions in tower_pairs_refraction_predictions.txt, committed with
this script before any run. Model as tower_heating_test.py: A' unchanged, C_r = 0, no new coupling or
parameter; q = 1 ring, N = 128, kappa*; D <= 8 part = complex components 0-3; upper = 4..n-1;
s(x) = |u_upper(x)|^2.

CHIRALITY RESOLUTION. Per D <= 8 complex component and Fourier mode, psi = a e^{-i w_a t} + b e^{+i w_b t},
w_a^2 + kappa w_a = Q(k), w_b = w_a + kappa:
    a = (w_b P + i D)/(w_a + w_b),  b = (w_a P - i D)/(w_a + w_b)   (P, D the transforms of psi, dpsi).
Populations N_a = sum (w_a + kappa/2)|a_k|^2 / N, N_b likewise; energies E_a = sum w_a N_a(k),
E_b = sum w_b N_b(k). The charge Q = sum v.JJ u - (kappa/2)|u|^2 equals N_b - N_a EXACTLY (a change of
variables of an exact quadratic form; checked at t = 0), so with Q conserved, dN_a = dN_b is an
identity: what is tested is that the pair population grows, how, and how much of the gain it carries.
Pair part of the gain: dE_b + w0 dN_b (each b quantum's a-partner joins the packet, at w0 = the
packet's initial population-weighted a-frequency); a-redistribution part: sum (w_a - w0) dN_a(k);
all reference-subtracted (the isolated packet's own nonlinear b-content removed).

PART 1 (D16, T = 2000, packet 0.05, k0 = pi/2): incoherent, all four upper components, A_U = 0.005
  seeds 0-3 and A_U = 0.01 seeds 0, 1; fixed-mean scan (tower_heating_test.py's 'mix') x = 1, 0.5, 0.25
  seeds 0, 1 and x = 0.
PART 2a (refraction, T = 1000): weak probes (amplitude 1e-4 unless stated) in a coherent, spatially
  uniform upper background of mean s = 1e-4 spread equally over every upper component (static medium).
  Kinds: a- and b-branch at k0 = pi/4, pi/2, 3pi/4 (D16, colour 0); a-branch in colour 2 (D16); a- and
  b-branch at pi/2 at D32 and D64; a- and b-branch at pi/2 in the INCOHERENT background (A_U = 0.005,
  seed 0, D16); a-branch pi/2 at amplitudes 1e-3, 1e-2, 5e-2 (D16). Each against its own isolated run
  (D8, same probe). Frequency shift dw from the slope of arg<psi_ref, psi> over [100, T] (a: dw =
  -slope; b: +slope); stiffness shift sensed dK_eff = dw (2 w_a(k0) + kappa).
PART 2b (deflection, T = 80): weak probes (1e-4) at x0 = N/2 in a coherent background with a gradient,
  s(x) = s0 (1 + 0.5 sin(2 pi (x - x0)/N)), s0 = 1e-4 (density rising toward +x at the packet). Kinds:
  a- and b-branch at pi/4, pi/2, 3pi/4 (D16); a- and b-branch pi/2 at D32, D64. Physical wavenumber
  p = k (a-branch), -k (b-branch), population-weighted; dp = p - p_ref. Prediction (ray optics with
  dK = sqrt(s)): dp/dt = F / (2 w_a(k0) + kappa), F = -<d_x sqrt(s)> weighted by |psi|^2, from the run.
usage: python3 tower_pairs_refraction_test.py
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import tower_heating_test as TH  # noqa: E402
import tower_populated_test as TP  # noqa: E402

M = TP.M
N = TP.N
A0 = TH.A0


def branches(lat, u, v, comps=range(4)):
    """Per-mode a/b amplitudes of the D<=8 components; returns wa, wb, a, b (shape (ncomp, N))."""
    wa = lat.branch_omega(); wb = wa + lat.kappa
    A, B = [], []
    for c in comps:
        P = np.fft.fft(u[:, 2 * c] + 1j * u[:, 2 * c + 1]); D = np.fft.fft(v[:, 2 * c] + 1j * v[:, 2 * c + 1])
        A.append((wb * P + 1j * D) / (wa + wb)); B.append((wa * P - 1j * D) / (wa + wb))
    return wa, wb, np.array(A), np.array(B)


def populations(lat, u, v):
    wa, wb, a, b = branches(lat, u, v)
    w = (wa + lat.kappa / 2) / lat.N
    Na_k = (w * np.abs(a) ** 2).sum(axis=0); Nb_k = (w * np.abs(b) ** 2).sum(axis=0)
    return wa, wb, Na_k, Nb_k


# ------------------------------------------------------------------ part 1
def run_pairs(job):
    label, kind, par, seed = job
    t0 = time.time()
    n = 4 if kind == "ref" else 8
    lat = M.Lattice(n=n, N=N, well="node")
    u, v = lat.packet(amp=TP.AMP, n0=N // 2, width=8.0, per_mode=True)
    if kind == "incoh":
        TH.incoherent(lat, u, v, range(4, 8), par, seed)
    elif kind == "mix":
        if par > 0:
            TH.incoherent(lat, u, v, [4, 5], np.sqrt(2 * par) * A0, seed)
        if par < 1:
            TH.coherent_uniform(lat, u, v, [6, 7], np.sqrt(2 * (1 - par)) * A0)
    T, dts = 2000.0, 2.0
    lo = slice(0, 8)
    Q0 = TP.charge(lat, u, v, lo)
    wa, wb, Na0, Nb0 = populations(lat, u, v)
    ident = abs((Nb0.sum() - Na0.sum()) - Q0) / abs(Q0)
    w0 = float((wa * Na0).sum() / Na0.sum())
    rec = dict(t=[], El=[], Na=[], Nb=[], Ea=[], Eb=[], Ra=[], vs=[], ms=[], Ql=[])

    def sample(t, u, v):
        _, _, Na, Nb = populations(lat, u, v)
        rec["t"].append(t); rec["El"].append(TP.self_energy(lat, u, v, lo))
        rec["Na"].append(Na.sum()); rec["Nb"].append(Nb.sum())
        rec["Ea"].append((wa * Na).sum()); rec["Eb"].append((wb * Nb).sum())
        rec["Ra"].append(((wa - w0) * Na).sum())
        s = (u[:, 8:] ** 2).sum(axis=1) if n > 4 else np.zeros(N)
        rec["ms"].append(s.mean()); rec["vs"].append(s.var()); rec["Ql"].append(TP.charge(lat, u, v, lo))

    sample(0.0, u, v)
    t = 0.0
    while t < T - 1e-9:
        u, v, _ = lat.run(u, v, dts); t += dts
        sample(t, u, v)
    out = {k: np.array(x) for k, x in rec.items()}
    out.update(label=label, kind=kind, par=par, seed=seed, ident=ident, w0=w0, secs=time.time() - t0)
    return out


def slope(t, y, a, b):
    m = (t >= a) & (t <= b)
    return np.polyfit(t[m], y[m], 1)[0]


def pair_summary(r, ref):
    t = r["t"]; E0 = r["El"][0]; Na0 = r["Na"][0]
    d = {k: (r[k] - r[k][0]) - (ref[k] - ref[k][0]) for k in ("El", "Na", "Nb", "Ea", "Eb", "Ra")}
    gain = d["El"][-1]
    pair = d["Eb"][-1] + r["w0"] * d["Nb"][-1]
    return dict(fT=gain / E0, g=slope(t, d["El"] / E0, 500, 2000), nb=d["Nb"][-1] / Na0,
                pg=slope(t, d["Nb"] / Na0, 500, 2000), dna_dnb=(d["Na"][-1] - d["Nb"][-1]) / Na0,
                pair_frac=pair / gain if gain else np.nan, redis_frac=d["Ra"][-1] / gain if gain else np.nan,
                lin_frac=(d["Ea"][-1] + d["Eb"][-1]) / gain if gain else np.nan,
                dEb=d["Eb"][-1] / E0, vs=r["vs"].mean(), ms=r["ms"].mean(),
                dQ=np.abs(r["Ql"] / r["Ql"][0] - 1).max())


# ------------------------------------------------------------------ part 2
def launch(lat, amp, k0, branch, colour, n0):
    x = np.arange(lat.N)
    psi = amp * np.exp(-0.5 * ((x - n0) / 8.0) ** 2) * np.exp(1j * k0 * (x - n0))
    wa = lat.branch_omega(); wb = wa + lat.kappa
    F = np.fft.fft(psi)
    dps = np.fft.ifft((-1j * wa if branch == "a" else 1j * wb) * F)
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    TH.set_comp(u, v, colour, psi, dps)
    return u, v


def background(lat, u, v, smean, profile=None, incoh_seed=None):
    up = list(range(4, lat.n))
    if incoh_seed is not None:
        TH.incoherent(lat, u, v, up, np.sqrt(smean / len(up)), incoh_seed)
        return
    wa0 = lat.branch_omega()[0]
    prof = np.ones(lat.N) if profile is None else profile
    for c in up:
        psi = np.sqrt(smean * prof / len(up)) * np.exp(1j * 2.39996 * c)
        TH.set_comp(u, v, c, psi + 0j, -1j * wa0 * psi)


def w_a(k):
    Q = M.SQ5 + 2 * M.C * (1 - np.cos(k))
    return 0.5 * (-M.KAPPA + np.sqrt(M.KAPPA ** 2 + 4 * Q))


def run_probe(job):
    label, mode, n, amp, k0, br, col, bg, T, dts = job
    t0 = time.time()
    x0 = N // 2
    out = {}
    for which in ("tower", "ref"):
        lat = M.Lattice(n=(n if which == "tower" else 4), N=N, well="node")
        u, v = launch(lat, amp, k0, br, col, x0)
        prof = None
        if mode == "grad":
            prof = 1 + 0.5 * np.sin(2 * np.pi * (np.arange(N) - x0) / N)
        if which == "tower":
            background(lat, u, v, 1e-4, prof, incoh_seed=(0 if bg == "incoh" else None))
        rec = dict(t=[], psi=[], F=[], sq=[], s=[])
        kk = 2 * np.pi * np.fft.fftfreq(N)
        t = 0.0
        while True:
            psi = u[:, 2 * col] + 1j * u[:, 2 * col + 1]
            s = (u[:, 8:] ** 2).sum(axis=1) if lat.n > 4 else np.zeros(N)
            sq = np.sqrt(s)
            dsq = 0.5 * (np.roll(sq, -1) - np.roll(sq, 1))
            w = np.abs(psi) ** 2
            rec["t"].append(t); rec["psi"].append(psi.copy()); rec["F"].append(-(w * dsq).sum() / w.sum())
            rec["sq"].append(sq.mean()); rec["s"].append(s.mean())
            if t >= T - 1e-9:
                break
            u, v, _ = lat.run(u, v, dts); t += dts
        out[which] = {k: np.array(x) for k, x in rec.items()}
    tw, rf = out["tower"], out["ref"]
    t = tw["t"]
    sgn = 1.0 if br == "a" else -1.0
    res = dict(label=label, mode=mode, n=n, amp=amp, k0=k0, br=br, col=col, bg=bg, secs=time.time() - t0)
    if mode == "phase":
        ph = np.unwrap(np.array([np.angle(np.vdot(a, b)) for a, b in zip(rf["psi"], tw["psi"])]))
        m = t >= 100
        dw = -sgn * np.polyfit(t[m], ph[m], 1)[0]
        res.update(dw=dw, dK=dw * (2 * w_a(k0) + M.KAPPA), sq_mean=tw["sq"].mean(), s_mean=tw["s"].mean(),
                   O=abs(np.vdot(rf["psi"][-1], tw["psi"][-1])) / (np.linalg.norm(rf["psi"][-1]) * np.linalg.norm(tw["psi"][-1])))
    else:
        kk = 2 * np.pi * np.fft.fftfreq(N)

        def pmean(psi):
            P = np.abs(np.fft.fft(psi)) ** 2
            return sgn * (kk * P).sum() / P.sum()

        def xc(psi):
            w = np.abs(psi) ** 2; th = 2 * np.pi * np.arange(N) / N
            return np.angle((w * np.exp(1j * th)).sum()) * N / (2 * np.pi)

        dp = np.array([pmean(b) - pmean(a) for a, b in zip(rf["psi"], tw["psi"])])
        dx = np.array([((xc(b) - xc(a) + N / 2) % N) - N / 2 for a, b in zip(rf["psi"], tw["psi"])])
        pred = np.concatenate([[0.0], np.cumsum(0.5 * (tw["F"][1:] + tw["F"][:-1]) * np.diff(t))]) / (2 * w_a(k0) + M.KAPPA)
        # predicted displacement: the group velocity v(p) = 2c sin p / (2 w_a(p) + kappa), the same function of
        # the physical wavenumber for both branches, evaluated at p0 + dp_pred(t) against p0
        p0 = sgn * k0

        def vg(pp):
            return 2 * M.C * np.sin(pp) / (2 * w_a(pp) + M.KAPPA)
        dxp = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(t) * ((vg(p0 + pred[1:]) - vg(p0)) + (vg(p0 + pred[:-1]) - vg(p0))))])
        res.update(dp=dp[-1], dp_pred=pred[-1], ratio=dp[-1] / pred[-1], dx=dx[-1], dx_pred=dxp[-1], F0=tw["F"][0])
    return res


def main():
    pjobs = [("ref", "ref", 0, 0)]
    pjobs += [(f"incoh A_U={a}", "incoh", a, s) for a, ss in ((A0, (0, 1, 2, 3)), (0.01, (0, 1))) for s in ss]
    pjobs += [(f"mix x={x}", "mix", x, s) for x in (1.0, 0.5, 0.25) for s in (0, 1)] + [("mix x=0", "mix", 0.0, 0)]
    q = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    rjobs = [(f"phase D16 {b} k0={k:.3f}", "phase", 8, 1e-4, k, b, 0, "coh", 1000.0, 5.0) for b in "ab" for k in q]
    rjobs += [("phase D16 a colour2", "phase", 8, 1e-4, q[1], "a", 2, "coh", 1000.0, 5.0)]
    rjobs += [(f"phase D{2 * n} {b}", "phase", n, 1e-4, q[1], b, 0, "coh", 1000.0, 5.0) for n in (16, 32) for b in "ab"]
    rjobs += [(f"phase D16 {b} incoh", "phase", 8, 1e-4, q[1], b, 0, "incoh", 1000.0, 5.0) for b in "ab"]
    rjobs += [(f"phase D16 a amp={a}", "phase", 8, a, q[1], "a", 0, "coh", 1000.0, 5.0) for a in (1e-3, 1e-2, 5e-2)]
    rjobs += [(f"grad D16 {b} k0={k:.3f}", "grad", 8, 1e-4, k, b, 0, "coh", 80.0, 1.0) for b in "ab" for k in q]
    rjobs += [(f"grad D{2 * n} {b}", "grad", n, 1e-4, q[1], b, 0, "coh", 80.0, 1.0) for n in (16, 32) for b in "ab"]
    with Pool(4) as pool:
        pres = pool.map(run_pairs, pjobs, chunksize=1)
        rres = pool.map(run_probe, rjobs, chunksize=1)
    ref = pres[0]
    print(f"PAIRS AND REFRACTION (A'), N = {N}, kappa* = {M.KAPPA:.6f}")
    print(f"PART 1 -- chirality-resolved gain, D16, packet {TP.AMP}, T = 2000 (reference-subtracted)")
    print(f"  identity Q = N_b - N_a at t = 0: max relative error {max(r['ident'] for r in pres):.1e}; "
          f"packet w0 = {ref['w0']:.4f}; reference: dN_b/N_a(0) at T {(ref['Nb'][-1] - ref['Nb'][0]) / ref['Na'][0]:+.2e}")
    S = {}
    for r in pres[1:]:
        s = pair_summary(r, ref); S[(r["label"], r["seed"])] = s
        print(f"  {r['label']:<14s} seed {r['seed']}: gain f(T) {s['fT']:+.3e} (rate {s['g']:+.2e}); dN_b/N_a(0) "
              f"{s['nb']:+.3e} (rate {s['pg']:+.2e}); (dN_a - dN_b)/N_a(0) {s['dna_dnb']:+.1e}; of the gain: pair "
              f"{s['pair_frac']:.3f}, a-redistribution {s['redis_frac']:+.3f}, linear total {s['lin_frac']:.3f}; "
              f"<var s> {s['vs']:.2e}; charge drift {s['dQ']:.1e} ({r['secs']:.0f} s)")
    xs = (1.0, 0.5, 0.25)
    gv = np.array([np.mean([S[(f"mix x={x}", s)]["g"] for s in (0, 1)]) for x in xs])
    pv = np.array([np.mean([S[(f"mix x={x}", s)]["pg"] for s in (0, 1)]) for x in xs])
    vv = np.array([np.mean([S[(f"mix x={x}", s)]["vs"] for s in (0, 1)]) for x in xs])
    print(f"  fixed-mean scan x = 1, 0.5, 0.25, 0: pair rates {', '.join(f'{p:.3e}' for p in pv)}, "
          f"{S[('mix x=0', 0)]['pg']:.3e} (x=0 / x=1 = {S[('mix x=0', 0)]['pg'] / pv[0]:+.3f}); energy rates "
          f"{', '.join(f'{p:.3e}' for p in gv)}")
    if np.all(pv > 0):
        print(f"    exponents vs <var s>: pair {np.polyfit(np.log(vv), np.log(pv), 1)[0]:.2f}, energy "
              f"{np.polyfit(np.log(vv), np.log(gv), 1)[0]:.2f}")
    pf = [S[(f"incoh A_U={A0}", s)]["pair_frac"] for s in range(4)]
    print(f"  pair fraction of the gain, incoherent A_U = {A0}, seeds 0-3: {', '.join(f'{p:.3f}' for p in pf)}; "
          f"mean {np.mean(pf):.3f}")
    print()
    print("PART 2a -- refraction: stiffness shift sensed dK_eff = dw (2 w_a(k0) + kappa)")
    for r in rres:
        if r["mode"] == "phase":
            print(f"  {r['label']:<24s}: dw {r['dw']:+.4e}; dK_eff {r['dK']:.5e}; sqrt<s> {np.sqrt(r['s_mean']):.5e}, "
                  f"<sqrt s> {r['sq_mean']:.5e}; dK_eff/sqrt<s> {r['dK'] / np.sqrt(r['s_mean']):.4f}; "
                  f"dK_eff/<sqrt s> {r['dK'] / r['sq_mean']:.4f}; |O| at T {r['O']:.5f} ({r['secs']:.0f} s)")
    weak = [r for r in rres if r["mode"] == "phase" and r["amp"] == 1e-4 and r["bg"] == "coh"]
    dks = np.array([r["dK"] for r in weak])
    print(f"  weak probes, coherent background: dK_eff spread (max-min)/mean {(dks.max() - dks.min()) / dks.mean():.4f}, "
          f"mean/sqrt(1e-4) {dks.mean() / 1e-2:.4f}")
    print()
    print("PART 2b -- deflection in a gradient (density rising toward +x at the packet); p = physical wavenumber")
    for r in rres:
        if r["mode"] == "grad":
            print(f"  {r['label']:<24s}: dp(T) {r['dp']:+.4e}, predicted {r['dp_pred']:+.4e}, ratio {r['ratio']:.4f}; "
                  f"dx(T) {r['dx']:+.4f} sites (predicted {r['dx_pred']:+.4f}); F(0) {r['F0']:+.3e}; dp (2w_a + kappa) {r['dp'] * (2 * w_a(r['k0']) + M.KAPPA):+.4e} "
                  f"({r['secs']:.0f} s)")


if __name__ == "__main__":
    main()
