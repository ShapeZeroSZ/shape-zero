#!/usr/bin/env python3
"""
kappa_widthscan_gpu.py -- is the box-size mechanism general across beam widths?

SELF-CONTAINED. Paste into Colab and run; it runs itself. PyTorch float64 on
the GPU if present, otherwise NumPy on the CPU (slow).

================================================================================
BACKGROUND
================================================================================
For a localised beam of width w = 2 on a periodic cubic lattice of side L, the
box-averaged kappa is kappa_pw * fill * F, with kappa_pw = -0.01869 (plane
wave) and fill = mean(env^2). F rises from 1.23 at L = 8 toward ~2 and above.
The model F = 2 - s (s = share of the beam's power in its box-wide component)
gets the shape right, but all eleven points at L >= 12 lie ABOVE it -- the true
cross factor for this lattice exceeds 2. That result used ONE width. The w = 3
value in the documents, and a table across geometries, were never re-measured
with correct seeding.

================================================================================
THE TEST
================================================================================
A Gaussian beam's fill and self-share depend on w/L, not on w and L separately.
So a width-3 beam in a 12-box has (nearly) the same geometry as a width-2 beam
in an 8-box. This script measures four MATCHED GROUPS -- same w/L, different
widths:

    w/L = 1/4   : w=2 L=8,   w=3 L=12,  w=4 L=16
    w/L = 1/8   : w=2 L=16,  w=3 L=24,  w=4 L=32,  w=1.5 L=12
    w/L = 1/12  : w=2 L=24,  w=3 L=36,  w=4 L=48
    w/L = 1/16  : w=2 L=32,  w=3 L=48,  w=4 L=64,  w=1.5 L=24

IF THE MECHANISM IS GENERAL, beams of different widths in the same group give
the same F. If they differ, the difference measures how the cross coupling
varies across the beam's sideways components -- narrower beams have wider
sideways spectra -- which is the refinement still open.

PREDICTION, stated before running: F AGREES across widths within each group,
to within two combined error bars.

Also produced along the way: the w = 3 kappa at four box sizes, replacing the
unverified w = 3 value.

The summary reports FACTS. The table is the result.

VALIDATION: the plane wave and w = 2 at L = 8, 16, 24, 32 must reproduce the
known values, or it stops.

Physics and readout identical to kappa_extended_gpu.py (validated there):
force -(x^2 - x - 1) + c*lap(x) + beta*c*(v[n-1]-v[n+1]); Fourier-space seed;
uniform readout of mode L/4; kappa = (|dw(A)|/|2 c beta sin k| - 1)/A^2; error
bar = max(|weighted - unweighted fit|, fit standard error). RK4, dt = 0.01,
float64, T = 300.

Runtime: roughly 15-20 minutes on a T4 (the w = 4, L = 64 box dominates).
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

KNOWN_W2 = {8: -0.00447, 16: -0.00204, 24: -0.00115, 32: -0.00047}
GROUPS = [
    ("1/4",  [(2.0, 8), (3.0, 12), (4.0, 16)]),
    ("1/8",  [(2.0, 16), (3.0, 24), (4.0, 32), (1.5, 12)]),
    ("1/12", [(2.0, 24), (3.0, 36), (4.0, 48)]),
    ("1/16", [(2.0, 32), (3.0, 48), (4.0, 64), (1.5, 24)]),
]

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


def main():
    print("=" * 78)
    print("KAPPA ACROSS BEAM WIDTHS -- is the mechanism general?")
    print("=" * 78)
    if HAVE_TORCH:
        dev = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"
        print(f"  backend: PyTorch float64 on {dev}")
    else:
        print("  backend: NumPy on CPU (no PyTorch -- slow)")
    print("  prediction, stated before running: F AGREES across widths in each group\n")

    cases = sorted({c for _, g in GROUPS for c in g}, key=lambda c: (c[1], c[0]))
    print("Runs (T = 300):")
    tP, SP, dP = run(8, None, "plane wave")
    res = {}
    for w, L in cases:
        t, S, d = run(L, w, "localised")
        k, r, rs, e = kappa_of(t, S)
        res[(w, L)] = (k, r, rs, e, d)
    print()

    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    kp, rp, _, _ = kappa_of(tP, SP)
    ok = close(kp, KAPPA_PW, rel=0.02) and abs(rp - 1) < 2e-5
    print(f"  plane wave       kappa {kp:+.5f}  (ref {KAPPA_PW:+.5f})   {'PASS' if ok else 'FAIL'}")
    for L, ref in sorted(KNOWN_W2.items()):
        k, r = res[(2.0, L)][0], res[(2.0, L)][1]
        g = close(k, ref) and abs(r - 1) < 2e-5
        print(f"  w = 2, L = {L:2d}    kappa {k:+.5f}  (ref {ref:+.5f})   {'PASS' if g else 'FAIL'}")
        ok &= g
    worst = max([dP] + [v[4] for v in res.values()])
    print(f"  energy drift     worst {worst:.1e}   {'PASS' if worst < 1e-6 else 'FAIL'}")
    ok &= worst < 1e-6
    if not ok:
        print("\n  VALIDATION FAILED -- stopping. Paste this output back.")
        return
    print("  all validation checks pass\n")

    print("=" * 78)
    print("THE TABLE -- F = kappa / (kappa_pw * fill), with its error bar")
    print("=" * 78)
    print("   w/L    w     L     fill      s       kappa               F            2 - s")
    F = {}
    for lbl, group in GROUPS:
        for w, L in group:
            k, _, rs, e, _ = res[(w, L)]
            fill, s = geometry(L, w)
            f = k / (KAPPA_PW * fill)
            fe = e / abs(KAPPA_PW * fill)
            F[(w, L)] = (f, fe, s)
            print(f"   {lbl:5s} {w:4.1f}  {L:3d}   {fill:.5f}  {s:.3f}   "
                  f"{k:+.5f}±{e:.5f}   {f:5.2f} ± {fe:4.2f}   {2 - s:.2f}")
        print()

    print("=" * 78)
    print("FACTS -- summaries of the table above; the table is the result")
    print("=" * 78)
    print("  Within each group (same w/L), the largest difference in F between two")
    print("  widths, in units of their combined error bar:")
    agree_all = True
    for lbl, group in GROUPS:
        worst_sig, pair = 0.0, None
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = F[group[i]], F[group[j]]
                sig = abs(a[0] - b[0]) / math.hypot(a[1], b[1])
                if sig > worst_sig:
                    worst_sig, pair = sig, (group[i][0], group[j][0])
        agree = worst_sig < 2.0
        agree_all &= agree
        print(f"    w/L = {lbl:4s}: largest difference {worst_sig:4.1f} combined error bars "
              f"(w = {pair[0]:g} vs {pair[1]:g})   {'within 2' if agree else 'OUTSIDE 2'}")
    above = sum(1 for (f, fe, s) in F.values() if f > 2 - s)
    print(f"\n  Points with F above the model 2 - s: {above} of {len(F)}")
    print(f"\n  The prediction (F agrees across widths in every group) "
          f"{'HELD' if agree_all else 'FAILED'}.")
    print("\n  Paste the whole output back for the record.")


if __name__ == "__main__":
    main()
