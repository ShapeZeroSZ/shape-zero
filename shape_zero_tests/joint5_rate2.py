#!/usr/bin/env python3
"""
joint5_rate2.py -- (A) q = 3 rate convergence at large sides; (B) is the nonreciprocity fraction
nu = (Gamma(+k) - Gamma(-k)) / Gamma universal across bump width and shape?
Predictions: joint5_rate2_predictions.txt (committed first, db25603). Physics and normalisation as
joint5_rate.py (fixed bump, peak 1; s = sqrt5, c = 1, beta = 0.05, k = pi/2).
Shapes (all separable per axis unless noted): gauss(sigma); sech2(w); twobump(sigma, a) = two Gaussians
at x0 = +-a along the probe axis. usage: python3 joint5_rate2.py A | B
"""
import sys
import time

import numpy as np

S0, C, B, K0 = np.sqrt(5.0), 1.0, 0.05, np.pi / 2


def phi_axis(shape, q, axis0):
    """1-D lattice transform of the shape's profile along one axis."""
    q = np.atleast_1d(np.asarray(q, float))
    kind, par = shape
    if kind == "gauss":
        sig = par[0]; nmax = int(12 * sig) + 2
        n = np.arange(-nmax, nmax + 1); prof = np.exp(-0.5 * n ** 2 / sig ** 2)
    elif kind == "sech2":
        w = par[0]; nmax = int(20 * w) + 2
        n = np.arange(-nmax, nmax + 1); prof = 1 / np.cosh(n / w) ** 2
    elif kind == "twobump":
        sig, a = par; nmax = int(12 * sig + a) + 2
        n = np.arange(-nmax, nmax + 1)
        prof = np.exp(-0.5 * (n - a) ** 2 / sig ** 2) + np.exp(-0.5 * (n + a) ** 2 / sig ** 2) if axis0 \
            else np.exp(-0.5 * n ** 2 / sig ** 2)
    return (prof[None, :] * np.exp(-1j * np.outer(q, n))).sum(1)


def probe(sign):
    w = sign * B * C * np.sin(K0) + np.sqrt(B ** 2 * C ** 2 * np.sin(K0) ** 2 + S0 + 2 * C * (1 - np.cos(K0)))
    return w, 2 * B * C * np.sin(sign * K0) - 2 * w


def continuum(q, sign, shape, ng):
    w, dpr = probe(sign); kx = sign * K0
    A = 2 * C * np.hypot(1.0, B * w); psi = np.arctan2(B * w, 1.0)
    t = -np.pi + (np.arange(ng) + 0.5) * 2 * np.pi / ng
    if q == 2:
        grids = [t]; wt = 2 * np.pi / ng
    else:
        ty, tz = np.meshgrid(t, t, indexing="ij"); grids = [ty.ravel(), tz.ravel()]; wt = (2 * np.pi / ng) ** 2
    trans = sum(2 * C * (1 - np.cos(g)) for g in grids)
    x = (2 * C - (w * w - S0 - trans)) / A
    ok = np.abs(x) < 1
    pt = np.ones_like(trans)
    for g in grids:
        pt = pt * np.abs(phi_axis(shape, g, False)) ** 2
    tot = 0.0
    for rs in (+1, -1):
        p = rs * np.arccos(np.clip(x, -1, 1)) - psi
        jac = np.abs(A * np.sin(p + psi))
        tot += np.where(ok, np.abs(phi_axis(shape, p - kx, True)) ** 2 * pt / np.where(ok, jac, 1), 0).sum() * wt
    return np.pi * S0 ** 2 / ((2 * np.pi) ** q * abs(dpr)) * tot


def box(q, L, sign, shape, eps_list):
    w, dpr = probe(sign)
    ks = 2 * np.pi * np.fft.fftfreq(L)
    ax = [ks.reshape((L,) + (1,) * (q - 1))] + [ks.reshape(tuple(L if b == a else 1 for b in range(q))) for a in range(1, q)]
    d = S0 - w * w + 2 * C * (1 - np.cos(ax[0])) + 2 * B * C * w * np.sin(ax[0])
    amp = np.abs(phi_axis(shape, ks - sign * K0, True)).reshape(ax[0].shape) ** 2
    for a in range(1, q):
        d = d + 2 * C * (1 - np.cos(ax[a]))
        amp = amp * (np.abs(phi_axis(shape, ks, False)) ** 2).reshape(ax[a].shape)
    amp = np.broadcast_to(amp, d.shape).copy()
    amp[tuple([int(round(sign * K0 * L / (2 * np.pi))) % L] + [0] * (q - 1))] = 0.0
    V = L ** q
    return [np.pi * S0 ** 2 / (V * abs(dpr)) * float((amp * (e / np.pi) / (d * d + e * e)).sum()) for e in eps_list]


def part_A():
    g = ("gauss", (2.0,))
    for ng in (600, 1200, 2000):
        cp, cm = continuum(3, 1, g, ng), continuum(3, -1, g, ng)
        print(f"continuum q = 3 (transverse grid {ng}^2): +k {cp:.5f}  -k {cm:.5f}  diff {cp - cm:+.5f}  nu {(cp - cm) / (0.5 * (cp + cm)):+.5f}", flush=True)
    eps = [0.05, 0.025, 0.0125]
    for L in (96, 128, 160, 192, 256):
        t0 = time.time()
        rp = box(3, L, 1, g, eps); rm = box(3, L, -1, g, eps)
        xp, xm = 2 * rp[-1] - rp[-2], 2 * rm[-1] - rm[-2]
        print(f"  L={L:3d}: +k eps {', '.join(f'{r:.4f}' for r in rp)} -> {xp:.4f};  -k -> {xm:.4f};  diff {xp - xm:+.4f}"
              f"  ({time.time() - t0:.0f} s)", flush=True)


def part_B():
    for q in (2, 3):
        ng = 4000 if q == 2 else 1200
        print(f"\nq = {q}: nu = (Gamma+ - Gamma-)/mean, continuum")
        rows = [("gauss", (s,)) for s in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)] + \
               [("sech2", (w,)) for w in (1.1, 2.2)] + [("twobump", (1.5, 3.0)), ("twobump", (1.5, 5.0))]
        for sh in rows:
            gp, gm = continuum(q, 1, sh, ng), continuum(q, -1, sh, ng)
            print(f"   {sh[0]:8s} {str(sh[1]):12s}: Gamma+ {gp:10.4f}  Gamma- {gm:10.4f}  nu {(gp - gm) / (0.5 * (gp + gm)):+.5f}", flush=True)


if __name__ == "__main__":
    {"A": part_A, "B": part_B}[sys.argv[1]]()
