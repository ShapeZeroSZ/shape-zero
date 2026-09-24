#!/usr/bin/env python3
"""
kappa_extended_gpu.py -- where does kappa go, and is its wobble real?

SELF-CONTAINED. Paste into Colab and run; it runs itself. PyTorch float64 on
the GPU if present, otherwise NumPy on the CPU (very slow at large boxes).

================================================================================
BACKGROUND
================================================================================
For a localised beam (Gaussian, width 2) on a periodic cubic lattice of side L,
the box-averaged kappa shrinks with L while keeping its sign. The ratio
kappa / (kappa_pw * fill) stays between about 1.2 and 2.8 -- roughly the factor
2 of cross- versus self-modulation -- but appeared to wobble, peaking at L = 24
and dipping at L = 32 (each ~13% off its neighbours), and did not settle by
L = 48. (kappa_pw = -0.01869, the plane-wave value; fill = mean(env^2).)

================================================================================
THREE QUESTIONS
================================================================================
Q1  WHERE IS ZERO? Out to L = 80, does the ratio stay positive and clear of
    zero by more than its error bar? If so, kappa reaches zero only as the
    limit of an infinite box (an isolated beam), not at a crossing.
Q2  IS THE WOBBLE REAL? Are the peak and trough larger than their error bars?
    A LOCAL TEST BEFORE RELEASE found the error bar grows fast with box size
    (2% at L = 8, 10% at L = 16, 21% at L = 20), so a +-13% wobble at L = 24
    and 32 may well lie inside it.
    The simulation is exact, so the error bar here is not noise: it is how far
    kappa moves between two reasonable ways of reading the frequency (weighted
    vs unweighted phase fit), or the fit's own standard error, whichever is
    larger.
Q3  DOES IT MOVE WITH OBSERVATION TIME? Every box is read over [0,150] and
    [0,300]. If the peak or trough shifts between windows, the cause is waves
    wrapping around the box during the run (a finite-time effect). If they
    stay put, it is a property of the box sizes themselves.

This script reports FACTS in its summary, not verdicts. Read the tables.

Physics and readout identical to kappa_boxscan_gpu.py (validated there):
force -(x^2 - x - 1) + c*lap(x) + beta*c*(v[n-1]-v[n+1]); Fourier-space seed;
uniform readout of mode L/4; kappa = (|dw(A)|/|2 c beta sin k| - 1)/A^2.
RK4, dt = 0.01, float64, T = 300.

VALIDATION: the plane wave and the ten box sizes already measured must
reproduce over [0,300], or it stops.

Runtime: roughly 20-30 minutes on a T4 (the L = 80 box dominates).
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
W_TRANS = 2.0
DT = 0.01
REC_EVERY = 10
A_LIN = 0.02
A_NL = 0.30
T_RUN = 300.0
TH = abs(2 * C * BETA * math.sin(K))
KAPPA_PW = -0.01869

KNOWN = {8: -0.00447, 12: -0.00289, 16: -0.00204, 20: -0.00147, 24: -0.00115,
         28: -0.00075, 32: -0.00047, 36: -0.00040, 40: -0.00033, 48: -0.00025}
NEW = [64, 80]
SIDES = sorted(set(KNOWN) | set(NEW))

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


def envelope(side, plane):
    if plane:
        return np.ones((side,) * 3)
    idx = np.indices((side,) * 3).astype(float)
    env = np.ones((side,) * 3)
    for a in (1, 2):
        d = idx[a] - side / 2.0
        d = (d + side / 2) % side - side / 2
        env = env * np.exp(-0.5 * (d / W_TRANS) ** 2)
    return env


def seed(side, plane, amp, sign):
    idx0 = np.indices((side,) * 3)[0].astype(float)
    psi = amp * envelope(side, plane) * np.exp(1j * sign * K * idx0)
    k = 2 * np.pi * np.fft.fftfreq(side)
    k0, k1, k2 = np.meshgrid(k, k, k, indexing="ij")
    b = BETA * C * np.sin(k0)
    om = b + np.sqrt(b * b + SQ5 + 2 * C * ((1 - np.cos(k0)) + (1 - np.cos(k1))
                                          + (1 - np.cos(k2))))
    v0 = np.fft.ifftn(-1j * om * np.fft.fftn(psi)).real
    return PHI + psi.real, v0


def run(side, plane, label):
    """Returns (t, S, drift); S columns: lin+, lin-, nl+, nl-."""
    xs, vs = [], []
    for a in (A_LIN, A_NL):
        for s in (+1, -1):
            x0, v0 = seed(side, plane, a, s)
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
    print(f"    {label:10s} side {side:2d}   {time.time()-t0:6.0f} s   "
          f"drift {drift:.1e}", flush=True)
    return np.array(t_rec), np.array(rec), drift


def fit(series, t, weighted=True):
    """Phase regression. Returns (|slope|, resid, standard error of slope)."""
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


def kappa_window(t, S, ta, tb):
    """kappa over [ta, tb] with an error bar. Returns (kappa, lin ratio,
    resid, err)."""
    sel = (t >= ta - 1e-9) & (t <= tb + 1e-9)
    tt = t[sel]

    def dw(c1, c2, weighted):
        a = fit(S[sel, c1], tt, weighted)
        b = fit(S[sel, c2], tt, weighted)
        return a[0] - b[0], max(a[1], b[1]), math.hypot(a[2], b[2])

    dl, _, _ = dw(0, 1, True)
    dn, res, se = dw(2, 3, True)
    dn_u, _, _ = dw(2, 3, False)
    ratio = abs(dl) / TH
    kap = (abs(dn) / TH - 1.0) / A_NL ** 2
    kap_u = (abs(dn_u) / TH - 1.0) / A_NL ** 2
    err = max(abs(kap - kap_u), se / (TH * A_NL ** 2))
    return kap, ratio, res, err


def fill_of(side):
    env = envelope(side, False)[0]
    return float(np.mean(env ** 2))


def close(val, ref, rel=0.04, absol=8e-5):
    return abs(val - ref) <= max(absol, rel * abs(ref))


def extrema(Ls, R):
    """Each interior point's departure from the mean of its neighbours."""
    out = {}
    for i in range(1, len(Ls) - 1):
        nb = 0.5 * (R[Ls[i - 1]] + R[Ls[i + 1]])
        out[Ls[i]] = (R[Ls[i]] - nb) / nb
    return out


def resolved(d, r, e):
    """A departure counts only if it is over 10% AND over twice its error bar.
    An earlier version reported a 10% departure carrying a 32% error bar as a
    'peak' -- measurement scatter called structure."""
    return abs(d) > 0.10 and abs(d) > 2 * e / abs(r)


def main():
    print("=" * 78)
    print("KAPPA TO LARGE BOXES -- where is zero, and is the wobble real?")
    print("=" * 78)
    if HAVE_TORCH:
        dev = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"
        print(f"  backend: PyTorch float64 on {dev}\n")
    else:
        print("  backend: NumPy on CPU (no PyTorch -- very slow at large boxes)\n")

    print("Runs (T = 300):")
    tP, SP, dP = run(8, True, "plane wave")
    data = {L: run(L, False, "localised") for L in SIDES}
    print()

    print("=" * 78)
    print("VALIDATION -- [0,300] must reproduce the known values")
    print("=" * 78)
    kp, rp, _, _ = kappa_window(tP, SP, 0, 300)
    ok = close(kp, KAPPA_PW, rel=0.02) and abs(rp - 1) < 2e-5
    print(f"  plane wave  kappa {kp:+.5f}  (ref {KAPPA_PW:+.5f})   {'PASS' if ok else 'FAIL'}")
    for L, ref in sorted(KNOWN.items()):
        t, S, _ = data[L]
        k, r, _, _ = kappa_window(t, S, 0, 300)
        g = close(k, ref) and abs(r - 1) < 2e-5
        print(f"  L = {L:2d}      kappa {k:+.5f}  (ref {ref:+.5f})   {'PASS' if g else 'FAIL'}")
        ok &= g
    worst = max([dP] + [d[2] for d in data.values()])
    print(f"  energy drift worst {worst:.1e}   {'PASS' if worst < 1e-6 else 'FAIL'}")
    ok &= worst < 1e-6
    if not ok:
        print("\n  VALIDATION FAILED -- stopping. Paste this output back.")
        return
    print("  all validation checks pass\n")

    print("=" * 78)
    print("THE TABLE -- ratio = kappa / (kappa_pw * fill); err is its error bar")
    print("=" * 78)
    print("   L      fill      kappa[0,300]      ratio[0,300]     ratio[0,150]   resid")
    R300, R150, E300, E150 = {}, {}, {}, {}
    for L in SIDES:
        t, S, _ = data[L]
        f = fill_of(L)
        k3, _, res, e3 = kappa_window(t, S, 0, 300)
        k1, _, _, e1 = kappa_window(t, S, 0, 150)
        R300[L], E300[L] = k3 / (KAPPA_PW * f), e3 / abs(KAPPA_PW * f)
        R150[L], E150[L] = k1 / (KAPPA_PW * f), e1 / abs(KAPPA_PW * f)
        new = "  new" if L in NEW else ""
        print(f"   {L:2d}   {f:.5f}   {k3:+.5f}±{e3:.5f}   {R300[L]:5.2f} ± {E300[L]:4.2f}"
              f"     {R150[L]:5.2f} ± {E150[L]:4.2f}   {res:.3f}{new}")
    print()

    print("=" * 78)
    print("FACTS -- summaries of the table above; the table is the result")
    print("=" * 78)
    print("  Q1  where is zero:")
    for L in NEW:
        clear = R300[L] - 3 * E300[L]
        print(f"      L = {L}: ratio {R300[L]:.2f} ± {E300[L]:.2f}; "
              f"{'positive by more than 3 error bars' if clear > 0 else 'NOT clear of zero by 3 error bars'}")
    print("  Q2  size of each point's departure from its neighbours, vs its error bar:")
    ex = extrema(SIDES, R300)
    for L, d in ex.items():
        eb = E300[L] / abs(R300[L])
        flag = "RESOLVED" if resolved(d, R300[L], E300[L]) else "within error"
        print(f"      L = {L:2d}: departure {d*100:+5.0f}%   error bar {eb*100:4.0f}%   {flag}")
    print("  Q3  departures that are RESOLVED (over 10% and over 2 error bars):")
    for lbl, R, E in (("[0,150]", R150, E150), ("[0,300]", R300, E300)):
        e = extrema(SIDES, R)
        pk = [L for L in e if e[L] > 0 and resolved(e[L], R[L], E[L])]
        tr = [L for L in e if e[L] < 0 and resolved(e[L], R[L], E[L])]
        print(f"      {lbl}: peaks at {pk or 'none'}, troughs at {tr or 'none'}")
    print("\n  Paste the whole output back for the record.")


if __name__ == "__main__":
    main()
