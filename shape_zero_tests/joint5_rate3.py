#!/usr/bin/env python3
"""
joint5_rate3.py -- census row 9: is the wide-bump limit nu -> 0 at k = pi/2 kinematic, and is the
leading correction lim nu * sigma^2 universal across bump shapes?
Predictions: joint5_rate3_predictions.txt (committed first). Physics and normalisation as
joint5_rate2.py (scalar beta sector, s = sqrt5, c = 1, beta = 0.05, fixed bump, golden-rule rate);
this file generalises its continuum shell integral to any probe wavenumber k and focuses the
transverse grid on the forward region |t| <= min(pi, TF / sigma_rms).

Shapes at rms width sigma (rms of the profile read as a distribution, per axis):
  gauss   exp(-x^2 / 2 sigma^2)
  sech2   sech^2(x / w), w = sigma sqrt(12) / pi
  twobump Gaussians of width sigma at +-a, a = sigma, along the probe axis; transverse profile a
          single Gaussian of width sigma (as joint5_rate2.py). Probe-axis rms sqrt2 sigma.

usage:  python3 joint5_rate3.py validate | h1 | h2

[After the predictions commit: the lattice transforms are evaluated on the 1-D transverse grid and in
chunks -- memory only, same arithmetic; checked to reproduce the committed version's values.]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint5_rate2 as R2

S0, C, B = R2.S0, R2.C, R2.B
TF = 14.0


def shape_at(kind, sig):
    if kind == "gauss":
        return ("gauss", (sig,))
    if kind == "sech2":
        return ("sech2", (sig * np.sqrt(12) / np.pi,))
    return ("twobump", (sig, sig))


def probe(k, sign):
    kk = sign * k
    w = B * C * np.sin(kk) + np.sqrt(B ** 2 * C ** 2 * np.sin(kk) ** 2 + S0 + 2 * C * (1 - np.cos(kk)))
    return w, 2 * B * C * np.sin(kk) - 2 * w


def rate(q, k, sign, shape, sig, ng, focus=True):
    """Continuum golden-rule rate, as joint5_rate2.continuum, probe at sign*k."""
    w, dpr = probe(k, sign); kx = sign * k
    A = 2 * C * np.hypot(1.0, B * w); psi = np.arctan2(B * w, 1.0)
    T = min(np.pi, TF / sig) if focus else np.pi
    t = -T + (np.arange(ng) + 0.5) * 2 * T / ng; dt = 2 * T / ng
    if q == 2:
        grids = [t]; wt = dt
    else:
        ty, tz = np.meshgrid(t, t, indexing="ij"); grids = [ty.ravel(), tz.ravel()]; wt = dt * dt
    trans = sum(2 * C * (1 - np.cos(g)) for g in grids)
    x = (2 * C - (w * w - S0 - trans)) / A
    ok = np.abs(x) < 1
    pt1 = np.abs(R2.phi_axis(shape, t, False)) ** 2          # transverse factor on the 1-D grid
    pt = pt1 if q == 2 else np.outer(pt1, pt1).ravel()
    tot = 0.0
    for rs in (+1, -1):
        p = rs * np.arccos(np.clip(x, -1, 1)) - psi
        jac = np.abs(A * np.sin(p + psi))
        amp = np.zeros_like(p)
        if ok.any():
            idx = np.flatnonzero(ok); arg = (p - kx)[idx]
            ph = np.concatenate([np.abs(R2.phi_axis(shape, arg[i:i + 4000], True)) ** 2
                                 for i in range(0, len(arg), 4000)])   # chunked: memory only
            amp[idx] = ph * pt[idx] / jac[idx]
        tot += amp.sum() * wt
    return np.pi * S0 ** 2 / ((2 * np.pi) ** q * abs(dpr)) * tot


def nu(q, k, kind, sig, ng):
    sh = shape_at(kind, sig)
    gp, gm = rate(q, k, 1, sh, sig, ng), rate(q, k, -1, sh, sig, ng)
    return (gp - gm) / (0.5 * (gp + gm))


def dwell_nu(k):
    bb = 2 * B * C * np.sin(k); Q = S0 + 2 * C * (1 - np.cos(k)); r = np.sqrt(bb * bb + 4 * Q)
    wp, wm = (bb + r) / 2, (-bb + r) / 2
    vp = (2 * C * np.sin(k) + 2 * B * C * wp * np.cos(k)) / (2 * wp - bb)
    vm = (2 * C * np.sin(k) - 2 * B * C * wm * np.cos(k)) / (2 * wm + bb)
    return 2 * (vm - vp) / (vp + vm)


def fit_limit(sigs, vals):
    """nu sigma^2 = A + B/sigma^2 + C/sigma^4 (least squares); returns A and the 2-term A."""
    s = np.asarray(sigs, float); y = np.asarray(vals)
    M3 = np.vstack([np.ones_like(s), s ** -2, s ** -4]).T
    M2 = M3[:, :2]
    return np.linalg.lstsq(M3, y, rcond=None)[0][0], np.linalg.lstsq(M2, y, rcond=None)[0][0]


def validate():
    print("VALIDATION: focused rate vs joint5_rate2.continuum (k = pi/2, Gaussian)")
    for q, ng in ((2, 4000), (3, 600)):
        for sig in (2.0, 3.0):
            sh = ("gauss", (sig,))
            for sign in (1, -1):
                ref = R2.continuum(q, sign, sh, ng)
                new = rate(q, np.pi / 2, sign, sh, sig, ng)
                print(f"   q={q} sigma={sig} {'+' if sign > 0 else '-'}k: joint5_rate2 {ref:.6f}  focused {new:.6f}"
                      f"  rel {(new - ref) / ref:+.1e}", flush=True)
    print("   grid convergence (q = 3, sigma = 12, gauss, nu):",
          "  ".join(f"ng {ng}: {nu(3, np.pi / 2, 'gauss', 12.0, ng):+.7f}" for ng in (200, 300, 400)), flush=True)


def h1():
    print("H1: wide-bump limit at other k (q = 2, Gaussian) vs the dwell-time value")
    sigs = [6, 8, 12, 16, 24, 32]
    for k, lab in ((np.pi / 4, "pi/4"), (np.pi / 3, "pi/3"), (2 * np.pi / 3, "2pi/3"), (np.pi / 2, "pi/2")):
        vals = [nu(2, k, "gauss", s, 3000) for s in sigs]
        # nu = nu_inf + b/sigma^2 + ...: fit in 1/sigma^2
        x = np.array(sigs, float) ** -2
        M = np.vstack([np.ones_like(x), x, x * x]).T
        lim = np.linalg.lstsq(M, np.array(vals), rcond=None)[0][0]
        pred = dwell_nu(k)
        rel = (lim - pred) / pred if abs(pred) > 1e-12 else float("nan")
        print(f"   k={lab:6s}: nu(sigma) " + " ".join(f"{v:+.5f}" for v in vals) +
              f"  -> limit {lim:+.5f}   dwell-time {pred:+.5f}   rel {rel:+.2e}", flush=True)


def h2():
    print("H2: lim nu * sigma_rms^2 at k = pi/2 (fit A + B/s^2 + C/s^4; 2-term fit in brackets)")
    pred = {2: {"gauss": -0.05147, "sech2": -0.06773, "twobump": -0.05147},
            3: {"gauss": -0.10294, "sech2": -0.13546, "twobump": -0.10294}}
    lims = {}
    for q, sigs, ng in ((2, [6, 8, 12, 16, 24, 32], 3000), (3, [6, 8, 10, 12, 16, 20], 400)):
        lims[q] = {}
        for kind in ("gauss", "sech2", "twobump"):
            vals = [nu(q, np.pi / 2, kind, s, ng) * s * s for s in sigs]
            a3, a2 = fit_limit(sigs, vals)
            lims[q][kind] = a3
            p = pred[q][kind]
            print(f"   q={q} {kind:8s}: nu s^2 " + " ".join(f"{v:+.5f}" for v in vals) +
                  f"  -> {a3:+.5f} [{a2:+.5f}]   predicted {p:+.5f}  rel {(a3 - p) / p:+.2e}", flush=True)
        v = np.array(list(lims[q].values()))
        print(f"   q={q}: spread across shapes at equal rms width (transverse rms): max/min - 1 = "
              f"{np.abs(v).max() / np.abs(v).min() - 1:.3f};  two-bump by probe-axis rms: "
              f"{2 * lims[q]['twobump']:+.5f}", flush=True)


if __name__ == "__main__":
    {"validate": validate, "h1": h1, "h2": h2}[sys.argv[1]]()
