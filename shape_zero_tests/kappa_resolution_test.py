#!/usr/bin/env python3
"""
kappa_resolution_test.py -- is the small-box gap a narrow-beam lattice effect?

SELF-CONTAINED. Runs itself when pasted. PyTorch float64 on a GPU if present,
otherwise NumPy (about 5 minutes on one CPU core -- all boxes are small).

At w/L = 1/4 (beam width a quarter of the box), the factor F = kappa /
(kappa_pw * fill) for w = 2 sits 9% below the model 2 - s, while w = 3 and 4
sit on it. HYPOTHESIS: the gap is a lattice effect of NARROW beams -- a narrow
beam carries sideways ripples on the scale of single sites, where lattice waves
differ from smooth space, which the model (smooth) does not include. The model's
inputs fill and s are computed on the actual grid, so this is NOT coarse
sampling of the beam's shape.

PREDICTION, stated before running: at fixed w/L = 1/4, the gap GROWS as the
beam narrows (w = 1 further below the model than w = 2) and VANISHES for wide
beams (w = 5, 6 on the model within error).

VALIDATION: w = 2, 3, 4 must reproduce the Colab values (-0.00447, -0.00478,
-0.00482), and the plane wave -0.01869, or it stops.

Physics and readout identical to kappa_widthscan_gpu.py.
"""

import math
import time

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

PHI = (1.0 + math.sqrt(5.0)) / 2.0
SQ5 = math.sqrt(5.0)
C = 1.0
K = math.pi / 2
BETA = 0.05
DT = 0.01
REC_EVERY = 10
A_LIN = 0.02
A_NL = 0.30
T_RUN = 300.0
TH = abs(2 * C * BETA * math.sin(K))
KAPPA_PW = -0.01869

if HAVE_TORCH:
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)


def to_backend(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV) if HAVE_TORCH else a


def to_numpy(a):
    return a.detach().cpu().numpy() if HAVE_TORCH else a


def roll(a, s, ax):
    return torch.roll(a, shifts=s, dims=ax) if HAVE_TORCH else np.roll(a, s, axis=ax)


def mode_series(x, side):
    m = side // 4
    if HAVE_TORCH:
        return torch.fft.fft(x.mean(dim=(2, 3)), dim=1)[:, m] / side
    return np.fft.fft(x.mean(axis=(2, 3)), axis=1)[:, m] / side


def force(x, v):
    lap = (roll(x, 1, 1) + roll(x, -1, 1) + roll(x, 1, 2) + roll(x, -1, 2)
           + roll(x, 1, 3) + roll(x, -1, 3) - 6.0 * x)
    return -(x * x - x - 1.0) + C * lap + BETA * C * (roll(v, 1, 1) - roll(v, -1, 1))


def energy(x, v):
    e = 0.5 * v * v + x ** 3 / 3.0 - x ** 2 / 2.0 - x
    for ax in (1, 2, 3):
        d = roll(x, -1, ax) - x
        e = e + 0.5 * C * d * d
    return to_numpy(e.sum(dim=(1, 2, 3))) if HAVE_TORCH else e.sum(axis=(1, 2, 3))


def envelope(side, width):
    """width None -> plane wave."""
    if width is None:
        return np.ones((side,) * 3)
    idx = np.indices((side,) * 3).astype(float)
    env = np.ones((side,) * 3)
    for a in (1, 2):
        d = idx[a] - side / 2.0
        d = (d + side / 2) % side - side / 2
        env = env * np.exp(-0.5 * (d / width) ** 2)
    return env


def seed(side, width, amp, sign):
    idx0 = np.indices((side,) * 3)[0].astype(float)
    psi = amp * envelope(side, width) * np.exp(1j * sign * K * idx0)
    k = 2 * np.pi * np.fft.fftfreq(side)
    k0, k1, k2 = np.meshgrid(k, k, k, indexing="ij")
    b = BETA * C * np.sin(k0)
    om = b + np.sqrt(b * b + SQ5 + 2 * C * ((1 - np.cos(k0)) + (1 - np.cos(k1))
                                          + (1 - np.cos(k2))))
    v0 = np.fft.ifftn(-1j * om * np.fft.fftn(psi)).real
    return PHI + psi.real, v0


def run(side, width, label):
    """Returns (t, S, drift); S columns: lin+, lin-, nl+, nl-."""
    xs, vs = [], []
    for a in (A_LIN, A_NL):
        for s in (+1, -1):
            x0, v0 = seed(side, width, a, s)
            xs.append(x0)
            vs.append(v0)
    x = to_backend(np.stack(xs))
    v = to_backend(np.stack(vs))
    e0 = energy(x, v)
    n = int(round(T_RUN / DT))
    t_rec = [0.0]
    rec = [to_numpy(mode_series(x, side))]
    t0 = time.time()
    h = DT
    for step in range(n):
        k1v = force(x, v); k1x = v
        x2 = x + 0.5 * h * k1x; v2 = v + 0.5 * h * k1v
        k2v = force(x2, v2); k2x = v2
        x3 = x + 0.5 * h * k2x; v3 = v + 0.5 * h * k2v
        k3v = force(x3, v3); k3x = v3
        x4 = x + h * k3x; v4 = v + h * k3v
        k4v = force(x4, v4); k4x = v4
        x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        if (step + 1) % REC_EVERY == 0:
            t_rec.append((step + 1) * h)
            rec.append(to_numpy(mode_series(x, side)))
    drift = float(np.max(np.abs(energy(x, v) - e0) / np.abs(e0)))
    wl = "plane" if width is None else f"w={width:g}"
    print(f"    {label:10s} {wl:6s} side {side:2d}   {time.time()-t0:6.0f} s   "
          f"drift {drift:.1e}", flush=True)
    return np.array(t_rec), np.array(rec), drift


def fit(series, t, weighted=True):
    ph = np.unwrap(np.angle(series))
    w = np.abs(series)
    good = w > 0.05 * w.max()
    tt, pp = t[good], ph[good]
    ww = w[good] if weighted else np.ones(good.sum())
    Am = np.vstack([tt, np.ones_like(tt)]).T
    Wm = np.diag(ww)
    sol, *_ = np.linalg.lstsq(Wm @ Am, Wm @ pp, rcond=None)
    r = Am @ sol - pp
    n = len(tt)
    sig2 = float(np.sum((ww * r) ** 2) / max(n - 2, 1))
    cov = sig2 * np.linalg.inv((Wm @ Am).T @ (Wm @ Am))
    return abs(sol[0]), float(np.sqrt(np.mean(r ** 2))), float(np.sqrt(cov[0, 0]))


def kappa_of(t, S):
    """kappa over [0, T_RUN] with error bar. Returns (kappa, lin ratio, resid, err)."""
    def dw(c1, c2, weighted):
        a = fit(S[:, c1], t, weighted)
        b = fit(S[:, c2], t, weighted)
        return a[0] - b[0], max(a[1], b[1]), math.hypot(a[2], b[2])
    dl, _, _ = dw(0, 1, True)
    dn, res, se = dw(2, 3, True)
    dn_u, _, _ = dw(2, 3, False)
    ratio = abs(dl) / TH
    kap = (abs(dn) / TH - 1.0) / A_NL ** 2
    kap_u = (abs(dn_u) / TH - 1.0) / A_NL ** 2
    err = max(abs(kap - kap_u), se / (TH * A_NL ** 2))
    return kap, ratio, res, err


def geometry(side, width):
    env = envelope(side, width)[0]
    fill = float(np.mean(env ** 2))
    s = float(np.mean(env) ** 2 / np.mean(env ** 2))
    return fill, s


def close(val, ref, rel=0.04, absol=8e-5):
    return abs(val - ref) <= max(absol, rel * abs(ref))



KNOWN = {(2.0, 8): -0.00447, (3.0, 12): -0.00478, (4.0, 16): -0.00482}
CASES = [(1.0, 4), (2.0, 8), (3.0, 12), (4.0, 16), (5.0, 20), (6.0, 24)]


def main():
    print("=" * 78)
    print("RESOLUTION TEST -- w/L = 1/4, beam widths 1 to 6")
    print("=" * 78)
    if HAVE_TORCH:
        dev = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"
        print(f"  backend: PyTorch float64 on {dev}")
    else:
        print("  backend: NumPy on CPU")
    print("  prediction: the gap GROWS as the beam narrows and VANISHES for wide beams\n")
    print("Runs (T = 300):")
    tP, SP, dP = run(8, None, "plane wave")
    res = {}
    for w, L in CASES:
        t, S, d = run(L, w, "localised")
        k, r, rs, e = kappa_of(t, S)
        res[(w, L)] = (k, r, e, d)
    print()
    print("VALIDATION")
    kp, rp, _, _ = kappa_of(tP, SP)
    ok = close(kp, KAPPA_PW, rel=0.02) and abs(rp - 1) < 2e-5
    print(f"  plane wave     kappa {kp:+.5f}  (ref {KAPPA_PW:+.5f})   {'PASS' if ok else 'FAIL'}")
    for key, ref in sorted(KNOWN.items()):
        k, r = res[key][0], res[key][1]
        g = close(k, ref) and abs(r - 1) < 2e-5
        print(f"  w = {key[0]:g}, L = {key[1]:2d}  kappa {k:+.5f}  (ref {ref:+.5f})   {'PASS' if g else 'FAIL'}")
        ok &= g
    worst = max([dP] + [v[3] for v in res.values()])
    print(f"  energy drift   worst {worst:.1e}   {'PASS' if worst < 1e-6 else 'FAIL'}")
    ok &= worst < 1e-6
    if not ok:
        print("\n  VALIDATION FAILED -- stopping.")
        return
    print()
    print("THE TABLE -- F and its gap from the model, in the gap's own error bars")
    print("   w     L     s       kappa               F               2 - s    gap")
    gaps = {}
    for w, L in CASES:
        k, _, e, _ = res[(w, L)]
        fill, s = geometry(L, w)
        F = k / (KAPPA_PW * fill)
        Fe = e / abs(KAPPA_PW * fill)
        gaps[w] = ((F - (2 - s)) / (2 - s), (F - (2 - s)) / Fe)
        print(f"   {w:3.1f}  {L:3d}   {s:.3f}   {k:+.5f}+-{e:.5f}   {F:5.2f} +- {Fe:4.2f}    "
              f"{2 - s:.2f}   {gaps[w][0]*100:+5.1f}%  ({gaps[w][1]:+5.1f} err bars)")
    print()
    print("FACTS")
    narrowing = gaps[1.0][0] < gaps[2.0][0] < 0
    wide_ok = all(abs(gaps[w][1]) < 2 for w in (5.0, 6.0))
    print(f"  w = 1 further below the model than w = 2: {'yes' if narrowing else 'NO'}")
    print(f"  w = 5 and 6 within 2 error bars of the model: {'yes' if wide_ok else 'NO'}")
    print(f"  prediction {'HELD' if (narrowing and wide_ok) else 'FAILED'}")


if __name__ == "__main__":
    main()
