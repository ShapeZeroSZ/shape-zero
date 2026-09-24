#!/usr/bin/env python3
"""
kappa_boxscan_gpu.py -- the shape of kappa against box size

SELF-CONTAINED. Paste into Colab and run; it runs itself. Uses PyTorch on the
GPU if present (float64), otherwise NumPy on the CPU (slow).

================================================================================
BACKGROUND
================================================================================
For a transversely localised beam (Gaussian, width w = 2) on a periodic cubic
lattice of side L, the measured kappa depends on L:

    L       8         12        16        24        32
    kappa  -0.00447  -0.00289  -0.00204  -0.00115  -0.00047

kappa_side_gpu.py established this is STATIC: present from t = 0 and a clean
A^2 coefficient. A candidate mechanism, with nothing fitted:

    kappa_model(L) = kappa_pw * fill(L) * (2 - s(L))

where kappa_pw = -0.01869 (plane wave), fill = mean(env^2), and s is the share
of the beam's power in its box-wide component. The box-wide component is
shifted by itself (weight 1) and by the beam's sideways components (weight 2,
the standard cross-versus-self ratio). Against the five measurements it gives
measured/model = 0.91, 1.07, 1.23, 1.47, 1.05 -- right in kind, not exact,
with side 24 standing out.

================================================================================
WHAT THIS SCRIPT DECIDES
================================================================================
Q1  Is side 24 a real bump in the curve, or a stray point?  Adds L = 20 and 28
    either side of it. (SUPERSEDED: this script has no error bars, so it cannot
    answer Q1. kappa_extended_gpu.py measured them: no departure is resolved.)
Q2  Does kappa / (kappa_pw * fill) settle to a constant at large L?  Adds
    L = 36, 40, 48.

PREDICTION, stated before running: Q2 -- YES, it settles. Reason: a static
effect from the sideways components is a sum over the beam's spectrum on the
box's grid; for a smooth coupling that sum converges as the grid refines, so
kappa becomes proportional to fill at large L. If it keeps wandering, the
coupling has sharp structure (e.g. near-resonances on particular grids).
No prediction for Q1.

================================================================================
VALIDATION (must pass, or nothing after it is trusted)
================================================================================
  plane wave:         kappa -0.01869
  localised L = 8, 12, 16, 24, 32: the five known values above
  energy drift small in every run
Side 24 has never been measured by this code before, so its check is also an
independent re-measurement of the outlier.

Physics and readout are identical to kappa_side_gpu.py and
kappa_readout_test.py: force -(x^2 - x - 1) + c*lap(x) + beta*c*(v[n-1]-v[n+1]),
Fourier-space seed (each wavevector at its own branch frequency), uniform
readout of mode L/4, amplitude-weighted phase regression, kappa over [0, 300],
kappa = (|dw(A)|/|2 c beta sin k| - 1)/A^2. RK4, dt = 0.01, float64.

Runtime: roughly 5-15 minutes on a GPU.
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
KAPPA_PW_REF = -0.01869

KNOWN = {8: -0.00447, 12: -0.00289, 16: -0.00204, 24: -0.00115, 32: -0.00047}
NEW = [20, 28, 36, 40, 48]
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
    """Fourier-space seed: every wavevector at its own branch frequency."""
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
    t = np.array(t_rec)
    S = np.array(rec)
    kap, ratio, resid = kappa_of(t, S)
    print(f"    {label:10s} side {side:2d}   {time.time()-t0:5.0f} s   "
          f"kappa {kap:+.5f}   lin ratio {ratio:.6f}   resid {resid:.3f}   "
          f"drift {drift:.1e}", flush=True)
    return kap, ratio, resid, drift


def omega_of(series, t):
    ph = np.unwrap(np.angle(series))
    w = np.abs(series)
    good = w > 0.05 * w.max()
    tt, pp, ww = t[good], ph[good], w[good]
    Am = np.vstack([tt, np.ones_like(tt)]).T
    Wm = np.diag(ww)
    sol, *_ = np.linalg.lstsq(Wm @ Am, Wm @ pp, rcond=None)
    return abs(sol[0]), float(np.sqrt(np.mean((Am @ sol - pp) ** 2)))


def kappa_of(t, S):
    wl_p, r1 = omega_of(S[:, 0], t)
    wl_m, r2 = omega_of(S[:, 1], t)
    wn_p, r3 = omega_of(S[:, 2], t)
    wn_m, r4 = omega_of(S[:, 3], t)
    ratio = abs(wl_p - wl_m) / TH
    kap = (abs(wn_p - wn_m) / TH - 1.0) / A_NL ** 2
    return kap, ratio, max(r1, r2, r3, r4)


def close(val, ref, rel=0.04, absol=8e-5):
    return abs(val - ref) <= max(absol, rel * abs(ref))


def geometry(side):
    env = envelope(side, False)[0]            # one transverse slice
    fill = float(np.mean(env ** 2))
    s = float(np.mean(env) ** 2 / np.mean(env ** 2))
    return fill, s


def main():
    print("=" * 78)
    print("KAPPA vs BOX SIZE -- the shape of the curve")
    print("=" * 78)
    if HAVE_TORCH:
        dev = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"
        print(f"  backend: PyTorch float64 on {dev}")
    else:
        print("  backend: NumPy on CPU (no PyTorch -- this will be slow)")
    print("  prediction, stated before running: kappa/(kappa_pw*fill) SETTLES at large L\n")

    print("Runs (T = 300):")
    kpw, rpw, respw, dpw = run(8, True, "plane wave")
    res = {}
    for side in SIDES:
        res[side] = run(side, False, "localised")
    print()

    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    ok = close(kpw, KAPPA_PW_REF, rel=0.02) and abs(rpw - 1) < 2e-5
    print(f"  plane wave     kappa {kpw:+.5f}  (ref {KAPPA_PW_REF:+.5f})   {'PASS' if ok else 'FAIL'}")
    for side, ref in sorted(KNOWN.items()):
        kap, ratio, resid, drift = res[side]
        good = close(kap, ref) and abs(ratio - 1) < 2e-5
        tag = "  <- the outlier, first independent re-measurement" if side == 24 else ""
        print(f"  L = {side:2d}         kappa {kap:+.5f}  (ref {ref:+.5f})   "
              f"{'PASS' if good else 'FAIL'}{tag}")
        ok &= good
    worst = max([dpw] + [r[3] for r in res.values()])
    ok_d = worst < 1e-6
    print(f"  energy drift   worst {worst:.1e}   {'PASS' if ok_d else 'FAIL'}")
    ok &= ok_d
    if not ok:
        print("\n  VALIDATION FAILED -- stopping; the table below would not be trusted.")
        print("  Paste this output back so the discrepancy can be found.")
        return
    print("  all validation checks pass\n")

    print("=" * 78)
    print("THE CURVE")
    print("=" * 78)
    print("   L    fill     s     measured    model      meas/model   meas/(kpw*fill)")
    ratios = {}
    for side in SIDES:
        kap = res[side][0]
        fill, s = geometry(side)
        model = kpw * fill * (2 - s)
        ratios[side] = kap / (kpw * fill)
        new = "  new" if side in NEW else ""
        print(f"   {side:2d}  {fill:.4f}  {s:.3f}   {kap:+.5f}   {model:+.5f}     "
              f"{kap/model:5.2f}        {ratios[side]:5.2f}{new}")
    print()

    print("=" * 78)
    print("READING")
    print("=" * 78)
    # Q1 -- WITHDRAWN AS A TEST. This script computes no error bars, so it
    # cannot say whether a departure is real. Two earlier versions got this
    # wrong in opposite directions: one tested side 24 alone against a fixed
    # threshold ("no bump"); the next compared every point with its neighbours
    # and reported "NOT SMOOTH: peak at 24, trough at 32". kappa_extended_gpu.py
    # then measured error bars: the +13% at 24 carries +-32%, the -14% at 32
    # carries +-31% -- every departure is within error. Use that script.
    print("  Q1  departures from neighbours (NO error bars here -- not a test):")
    dev = {}
    for i2 in range(1, len(SIDES) - 1):
        nb = 0.5 * (ratios[SIDES[i2 - 1]] + ratios[SIDES[i2 + 1]])
        dev[SIDES[i2]] = (ratios[SIDES[i2]] - nb) / nb
    print("      " + "  ".join(f"L{L}:{dev[L]*100:+.0f}%" for L in dev))
    print("      Whether these are real needs error bars: see kappa_extended_gpu.py,")
    print("      which found every departure within its error bar.")
    big = [ratios[L] for L in (36, 40, 48)]
    spread = (max(big) - min(big)) / abs(np.mean(big))
    print(f"  Q2  kappa/(kappa_pw*fill) at L = 36, 40, 48: "
          f"{big[0]:.2f}, {big[1]:.2f}, {big[2]:.2f}  (spread {spread*100:.0f}%)")
    print("      NO error bars here -- not a test. kappa_extended_gpu.py measured them")
    print("      (+-0.86 to +-1.83 at these sizes): the values are consistent with a")
    print("      constant, so 'does not settle' is unresolved, not established.")
    print("\n  Paste the whole output back for the record.")


if __name__ == "__main__":
    main()
