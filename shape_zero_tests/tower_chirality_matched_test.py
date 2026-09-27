#!/usr/bin/env python3
"""tower_chirality_matched_test.py -- the chirality comparison of universal refraction, rerun as a fresh
pre-registered test with MATCHED TRAJECTORIES (PROVENANCE 6v, PREMISE_LEDGER C50). Predictions in
tower_chirality_matched_predictions.txt, committed with this script before any run.

Model as tower_pairs_refraction_test.py: A' unchanged, C_r = 0, no new coupling or parameter; q = 1 ring,
N = 128, kappa*; weak probes (amplitude 1e-4, width 8) in the D <= 8 part, colour 0; background in the
upper components 4..n-1 with s(x) = |u_upper(x)|^2.

MATCHED TRAJECTORIES. The a-branch probe has carrier e^{+i k0 x}; the b-branch probe has carrier
e^{-i k0 x}. Both have physical wavenumber p = +k0 and travel toward +x at the same group velocity
v(p) = 2c sin p / (2 w_a(p) + kappa) (the same function of p for both branches), so they sample the same
background along the same path.

Kinds: a and b; k0 = pi/4, pi/2, 3pi/4; node sizes D16, D32, D64, D128 (n = 8, 16, 32, 64).
 P  phase (T = 1000): coherent, spatially uniform background, <s> = 1e-4 spread equally over every upper
    component. dw from the slope of arg<psi_ref, psi> over [100, T] (a: -slope, b: +slope);
    dK_eff = dw (2 w_a(k0) + kappa). Every kind at every node size.
 I  phase in an INCOHERENT background (T = 1000; rms per component sqrt(1e-4 / M), seed 0), k0 = pi/2,
    a and b, D16, D32, D64 (a and b see the same realisation at each node size).
 G  gradient (T = 80): coherent background s(x) = s0 (1 + 0.5 sin(2 pi (x - x0)/N)), s0 = 1e-4, density
    rising toward +x at the packet (x0 = N/2). dp = p - p_ref (population-weighted physical wavenumber),
    dx = centroid displacement against the isolated run; ray predictions dp_ray = integral of F/(2 w_a +
    kappa), F = -<d_x sqrt(s)> weighted by |psi|^2, and dx_ray from v(p0 + dp_ray, dK) - v(p0, 0) with dK
    the weighted sqrt(s) (the direct refractive term included). Every kind at every node size.
 GM mirrored gradient (T = 80): s(x) = s0 (1 - 0.5 sin(...)), density FALLING toward +x; k0 = pi/2, a and
    b, D16 and D64. "Away from the denser region" is then dp > 0.
Each run has its own isolated reference (D8, same probe).
usage: python3 tower_chirality_matched_test.py
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import tower_pairs_refraction_test as PR  # noqa: E402

M, N = PR.M, PR.N
Q3 = (np.pi / 4, np.pi / 2, 3 * np.pi / 4)
NODES = (8, 16, 32, 64)


def wq(p, dK):
    Q = M.SQ5 + 2 * M.C * (1 - np.cos(p)) + dK
    return 0.5 * (-M.KAPPA + np.sqrt(M.KAPPA ** 2 + 4 * Q))


def vg(p, dK):
    return 2 * M.C * np.sin(p) / (2 * wq(p, dK) + M.KAPPA)


def run(job):
    mode, n, br, k0 = job
    t0 = time.time()
    x0 = N // 2
    carrier = k0 if br == "a" else -k0
    sgn = 1.0 if br == "a" else -1.0
    T, dts = (1000.0, 5.0) if mode in ("P", "I") else (80.0, 1.0)
    prof = None
    if mode == "G":
        prof = 1 + 0.5 * np.sin(2 * np.pi * (np.arange(N) - x0) / N)
    elif mode == "GM":
        prof = 1 - 0.5 * np.sin(2 * np.pi * (np.arange(N) - x0) / N)
    out = {}
    for which in ("tower", "ref"):
        lat = M.Lattice(n=(n if which == "tower" else 4), N=N, well="node")
        u, v = PR.launch(lat, 1e-4, carrier, br, 0, x0)
        if which == "tower":
            PR.background(lat, u, v, 1e-4, prof, incoh_seed=(0 if mode == "I" else None))
        rec = dict(t=[], psi=[], F=[], dK=[], s=[], sq=[])
        t = 0.0
        while True:
            psi = u[:, 0] + 1j * u[:, 1]
            s = (u[:, 8:] ** 2).sum(axis=1) if lat.n > 4 else np.zeros(N)
            sq = np.sqrt(s); w = np.abs(psi) ** 2
            rec["t"].append(t); rec["psi"].append(psi.copy())
            rec["F"].append(-(w * 0.5 * (np.roll(sq, -1) - np.roll(sq, 1))).sum() / w.sum())
            rec["dK"].append((w * sq).sum() / w.sum()); rec["s"].append(s.mean()); rec["sq"].append(sq.mean())
            if t >= T - 1e-9:
                break
            u, v, _ = lat.run(u, v, dts); t += dts
        out[which] = {k: np.array(x) for k, x in rec.items()}
    tw, rf = out["tower"], out["ref"]
    t = tw["t"]
    res = dict(mode=mode, n=n, br=br, k0=k0, secs=time.time() - t0)
    fac = 2 * PR.w_a(k0) + M.KAPPA
    if mode in ("P", "I"):
        ph = np.unwrap(np.array([np.angle(np.vdot(a, b)) for a, b in zip(rf["psi"], tw["psi"])]))
        m = t >= 100
        dw = -sgn * np.polyfit(t[m], ph[m], 1)[0]
        res.update(dw=dw, dK=dw * fac, rs=np.sqrt(tw["s"].mean()), msq=tw["sq"].mean())
    else:
        kk = 2 * np.pi * np.fft.fftfreq(N)

        def pmean(psi):
            P = np.abs(np.fft.fft(psi)) ** 2
            return sgn * (kk * P).sum() / P.sum()

        def xc(psi):
            w = np.abs(psi) ** 2; th = 2 * np.pi * np.arange(N) / N
            return np.angle((w * np.exp(1j * th)).sum()) * N / (2 * np.pi)

        dp = pmean(tw["psi"][-1]) - pmean(rf["psi"][-1])
        dx = ((xc(tw["psi"][-1]) - xc(rf["psi"][-1]) + N / 2) % N) - N / 2
        pred = np.concatenate([[0.0], np.cumsum(0.5 * (tw["F"][1:] + tw["F"][:-1]) * np.diff(t))]) / fac
        dv = vg(k0 + pred, tw["dK"]) - vg(k0, 0.0)
        dxr = np.sum(0.5 * (dv[1:] + dv[:-1]) * np.diff(t))
        res.update(dp=dp, dp_ray=pred[-1], dx=dx, dx_ray=dxr)
    return res


def main():
    jobs = [("P", n, b, k) for n in NODES for k in Q3 for b in "ab"]
    jobs += [("I", n, b, Q3[1]) for n in (8, 16, 32) for b in "ab"]
    jobs += [("G", n, b, k) for n in NODES for k in Q3 for b in "ab"]
    jobs += [("GM", n, b, Q3[1]) for n in (8, 32) for b in "ab"]
    jobs.sort(key=lambda j: -(j[1] * (12 if j[0] in "PI" else 1)))
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=1)
    R = {(r["mode"], r["n"], r["br"], round(r["k0"], 6)): r for r in res}
    K = lambda k: round(k, 6)  # noqa: E731
    print(f"CHIRALITY, MATCHED TRAJECTORIES (A'), N = {N}, kappa* = {M.KAPPA:.6f}; weak probes 1e-4, p = +k0 for both")
    print("P  uniform coherent background, <s> = 1e-4: dK_eff = dw (2 w_a + kappa)")
    dev_ba, dev_n, vals = [], [], []
    for k in Q3:
        for n in NODES:
            a, b = R[("P", n, "a", K(k))], R[("P", n, "b", K(k))]
            dev_ba.append(abs(b["dK"] / a["dK"] - 1)); vals += [a["dK"] / a["rs"], b["dK"] / b["rs"]]
            print(f"  k0={k:.3f} D{2 * n:<3d}: dK_eff a {a['dK']:.7e}, b {b['dK']:.7e}, b/a {b['dK'] / a['dK']:.6f}; "
                  f"dK_eff/sqrt<s> {a['dK'] / a['rs']:.5f}; dw a {a['dw']:.6e}")
        for br in "ab":
            base = R[("P", 8, br, K(k))]["dK"]
            dev_n += [abs(R[("P", n, br, K(k))]["dK"] / base - 1) for n in NODES[1:]]
    print(f"  max |b/a - 1| {max(dev_ba):.1e}; max |D/D16 - 1| {max(dev_n):.1e}; dK_eff/sqrt<s> range "
          f"{min(vals):.5f} - {max(vals):.5f}")
    print("I  incoherent background, k0 = pi/2")
    for n in (8, 16, 32):
        a, b = R[("I", n, "a", K(Q3[1]))], R[("I", n, "b", K(Q3[1]))]
        print(f"  D{2 * n:<3d}: dK_eff a {a['dK']:.6e}, b {b['dK']:.6e}, b/a {b['dK'] / a['dK']:.5f}; dK_eff/<sqrt s> "
              f"a {a['dK'] / a['msq']:.4f}, b {b['dK'] / b['msq']:.4f}")
    print("G  gradient, density rising toward +x (away = dp < 0)")
    gba, gn, grat = [], [], []
    for k in Q3:
        for n in NODES:
            a, b = R[("G", n, "a", K(k))], R[("G", n, "b", K(k))]
            gba += [abs(b["dp"] / a["dp"] - 1), abs(b["dx"] / a["dx"] - 1)]
            grat += [a["dp"] / a["dp_ray"], b["dp"] / b["dp_ray"]]
            print(f"  k0={k:.3f} D{2 * n:<3d}: dp a {a['dp']:+.6e}, b {b['dp']:+.6e}, b/a {b['dp'] / a['dp']:.6f}, "
                  f"ray ratio a {a['dp'] / a['dp_ray']:.4f}; dx a {a['dx']:+.5f}, b {b['dx']:+.5f}, b/a "
                  f"{b['dx'] / a['dx']:.5f}, ray {a['dx_ray']:+.5f}")
        for br in "ab":
            base = R[("G", 8, br, K(k))]
            gn += [abs(R[("G", n, br, K(k))]["dp"] / base["dp"] - 1) for n in NODES[1:]]
    print(f"  max |b/a - 1| (dp, dx) {max(gba):.1e}; max |D/D16 - 1| (dp) {max(gn):.1e}; dp/ray {min(grat):.4f} - "
          f"{max(grat):.4f}; all dp < 0: {all(R[('G', n, b, K(k))]['dp'] < 0 for n in NODES for b in 'ab' for k in Q3)}")
    print("GM mirrored gradient, density falling toward +x (away = dp > 0), k0 = pi/2")
    for n in (8, 32):
        a, b = R[("GM", n, "a", K(Q3[1]))], R[("GM", n, "b", K(Q3[1]))]
        print(f"  D{2 * n:<3d}: dp a {a['dp']:+.6e}, b {b['dp']:+.6e}, b/a {b['dp'] / a['dp']:.6f}; ray ratio a "
              f"{a['dp'] / a['dp_ray']:.4f}, b {b['dp'] / b['dp_ray']:.4f}; dx a {a['dx']:+.5f} (ray {a['dx_ray']:+.5f}), "
              f"b {b['dx']:+.5f}")


if __name__ == "__main__":
    main()
