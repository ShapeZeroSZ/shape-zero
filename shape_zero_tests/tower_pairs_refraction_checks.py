#!/usr/bin/env python3
"""tower_pairs_refraction_checks.py -- POST HOC checks on tower_pairs_refraction_test.py, written after its
output was seen (predictions 3063afd). Not part of the pre-registered test.
 1. Same-trajectory chirality comparison. In the test the b-branch probes (carrier e^{+i k0 x}) travelled
    toward -x while the a-branch probes travelled toward +x; sqrt(s) = sqrt(s0 (1 + 0.5 sin)) is not
    symmetric about x0, so the two sampled different gradients. Here the b-branch probe has carrier
    e^{-i k0 x}, so it travels toward +x on the a-probe's trajectory (physical wavenumber +k0).
 2. Displacement with the direct refractive term. The test's dx prediction varied only the wavenumber;
    here v(p, dK) = 2c sin p / (2 w(p, dK) + kappa), w^2 + kappa w = Q(p) + dK, with dK = the packet-
    weighted sqrt(s) at each time, against the isolated reference's v(p0, 0).
usage: python3 tower_pairs_refraction_checks.py"""
from multiprocessing import Pool

import numpy as np

import tower_pairs_refraction_test as PR

M, N = PR.M, PR.N


def wq(p, dK):
    Q = M.SQ5 + 2 * M.C * (1 - np.cos(p)) + dK
    return 0.5 * (-M.KAPPA + np.sqrt(M.KAPPA ** 2 + 4 * Q))


def vg(p, dK):
    return 2 * M.C * np.sin(p) / (2 * wq(p, dK) + M.KAPPA)


def job(args):
    br, k0 = args
    x0, T, dts = N // 2, 80.0, 1.0
    carrier = k0 if br == "a" else -k0          # both travel toward +x; physical p = +k0
    sgn = 1.0 if br == "a" else -1.0
    out = {}
    for which in ("tower", "ref"):
        lat = M.Lattice(n=(8 if which == "tower" else 4), N=N, well="node")
        u, v = PR.launch(lat, 1e-4, carrier, br, 0, x0)
        if which == "tower":
            PR.background(lat, u, v, 1e-4, 1 + 0.5 * np.sin(2 * np.pi * (np.arange(N) - x0) / N))
        rec = dict(t=[], psi=[], F=[], dK=[])
        t = 0.0
        while True:
            psi = u[:, 0] + 1j * u[:, 1]
            sq = np.sqrt((u[:, 8:] ** 2).sum(axis=1)) if lat.n > 4 else np.zeros(N)
            w = np.abs(psi) ** 2
            rec["t"].append(t); rec["psi"].append(psi.copy())
            rec["F"].append(-(w * 0.5 * (np.roll(sq, -1) - np.roll(sq, 1))).sum() / w.sum())
            rec["dK"].append((w * sq).sum() / w.sum())
            if t >= T - 1e-9:
                break
            u, v, _ = lat.run(u, v, dts); t += dts
        out[which] = {k: np.array(x) for k, x in rec.items()}
    tw, rf = out["tower"], out["ref"]
    t = tw["t"]; kk = 2 * np.pi * np.fft.fftfreq(N)

    def pmean(psi):
        P = np.abs(np.fft.fft(psi)) ** 2
        return sgn * (kk * P).sum() / P.sum()

    def xc(psi):
        w = np.abs(psi) ** 2; th = 2 * np.pi * np.arange(N) / N
        return np.angle((w * np.exp(1j * th)).sum()) * N / (2 * np.pi)

    dp = pmean(tw["psi"][-1]) - pmean(rf["psi"][-1])
    dx = ((xc(tw["psi"][-1]) - xc(rf["psi"][-1]) + N / 2) % N) - N / 2
    fac = 2 * PR.w_a(k0) + M.KAPPA
    pred = np.concatenate([[0.0], np.cumsum(0.5 * (tw["F"][1:] + tw["F"][:-1]) * np.diff(t))]) / fac
    dv = vg(k0 + pred, tw["dK"]) - vg(k0, 0.0)
    dxp = np.sum(0.5 * (dv[1:] + dv[:-1]) * np.diff(t))
    dv0 = vg(k0, tw["dK"]) - vg(k0, 0.0)                 # the direct term alone
    dx0 = np.sum(0.5 * (dv0[1:] + dv0[:-1]) * np.diff(t))
    return (br, k0, dp, pred[-1], dx, dxp, dx0)


if __name__ == "__main__":
    q = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    with Pool(4) as p:
        res = p.map(job, [(b, k) for b in "ab" for k in q])
    print("POST HOC. Both branches launched toward +x (physical wavenumber +k0), D16, gradient as in the test")
    for br, k0, dp, pr, dx, dxp, dx0 in res:
        print(f"  {br} k0={k0:.3f}: dp {dp:+.4e} (ray {pr:+.4e}, ratio {dp / pr:.4f}); dx {dx:+.4f} sites; "
              f"full ray dx {dxp:+.4f} (direct term alone {dx0:+.4f})")
    for k0 in q:
        a = [r for r in res if r[0] == "a" and r[1] == k0][0]; b = [r for r in res if r[0] == "b" and r[1] == k0][0]
        print(f"  k0={k0:.3f}: dp b/a {b[2] / a[2]:.4f}; dx b/a {b[4] / a[4]:.4f}")
