#!/usr/bin/env python3
"""
kappa_cross_kernel.py -- measure the cross kernel K(q_perp) directly.

K(q_perp) is the direction-odd frequency shift of the box-wide component
(K, 0, 0) caused by ONE sideways component (K, q_perp), per unit of that
component's (real amplitude)^2. Two-wave runs:

    probe  A0 = 0.02  at (s K, 0, 0)
    pump   A1        at (s K, qy, qz)      (a single travelling wave, not a cos pair)

for s = +1 and s = -1, plus probe-only runs; the probe's odd shift with the pump
minus without it, over A1^2, is K(q). R(q) = K(q) / (kappa_pw * TH) with kappa_pw
the plane-wave value at the same amplitude (A1 = 0.30: -0.01869; A1 = 0.10:
-0.01756, the amplitude sweep of pinned_asymmetry_reference.py).

Physics, own-branch Fourier-space seeding, integrator (RK4, dt = 0.01, T = 300)
and phase-regression readout of the box-wide mode are those of
kappa_resolution_test.py, on an (Lx, Ly, Lz) box: Lx = 8 (holds K = pi/2 and
2K = pi), Ly = 16, Lz = 1 for q along y (transverse-uniform in z ON PURPOSE:
here the field is meant to depend on y only) and Ly = Lz = 16 for the diagonal.

PREDICTION, stated before running (kappa_cross_pt.py, A -> 0): R falls from 2 at
q -> 0 to 1.39 at (pi, 0) and 1.12 at (pi, pi); measured values should match to
within amplitude corrections of a few percent.

VALIDATION: the plane wave at A = 0.30 must give kappa_pw = -0.01869.
"""

import math
import sys
import time
import numpy as np

PHI = (1.0 + math.sqrt(5.0)) / 2.0
SQ5 = math.sqrt(5.0)
C = 1.0
K = math.pi / 2
BETA = 0.05
DT = 0.01
REC_EVERY = 10
T_RUN = 300.0
TH = abs(2 * C * BETA * math.sin(K))
A_PROBE = 0.02
KPW = {0.30: -0.01869, 0.10: -0.01756}


def force(x, v):
    lap = (np.roll(x, 1, 1) + np.roll(x, -1, 1) + np.roll(x, 1, 2) + np.roll(x, -1, 2)
           + np.roll(x, 1, 3) + np.roll(x, -1, 3) - 6.0 * x)
    return -(x * x - x - 1.0) + C * lap + BETA * C * (np.roll(v, 1, 1) - np.roll(v, -1, 1))


def energy(x, v):
    e = 0.5 * v * v + x ** 3 / 3.0 - x ** 2 / 2.0 - x
    for ax in (1, 2, 3):
        d = np.roll(x, -1, ax) - x
        e = e + 0.5 * C * d * d
    return e.sum(axis=(1, 2, 3))


def seed(shape, waves, s):
    """waves: list of (amp, my, mz) -> psi = sum amp e^{i(sKx + qy y + qz z)};
    each Fourier component gets its own branch frequency (as kappa_resolution_test.py)."""
    Lx, Ly, Lz = shape
    ix, iy, iz = np.indices(shape).astype(float)
    psi = np.zeros(shape, complex)
    for amp, my, mz in waves:
        psi += amp * np.exp(1j * (s * K * ix + 2 * np.pi * my / Ly * iy + 2 * np.pi * mz / Lz * iz))
    k0 = 2 * np.pi * np.fft.fftfreq(Lx)
    k1 = 2 * np.pi * np.fft.fftfreq(Ly)
    k2 = 2 * np.pi * np.fft.fftfreq(Lz)
    a, b_, c_ = np.meshgrid(k0, k1, k2, indexing="ij")
    b = BETA * C * np.sin(a)
    om = b + np.sqrt(b * b + SQ5 + 2 * C * ((1 - np.cos(a)) + (1 - np.cos(b_)) + (1 - np.cos(c_))))
    v0 = np.fft.ifftn(-1j * om * np.fft.fftn(psi)).real
    return PHI + psi.real, v0


def run(shape, configs):
    """configs: list of (waves, s). Returns t, series (n_t, B), worst drift."""
    xs, vs = zip(*[seed(shape, w, s) for w, s in configs])
    x, v = np.stack(xs), np.stack(vs)
    e0 = energy(x, v)
    m = shape[0] // 4

    def mode(x):
        return np.fft.fft(x.mean(axis=(2, 3)), axis=1)[:, m] / shape[0]

    n = int(round(T_RUN / DT))
    t_rec, rec = [0.0], [mode(x)]
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
            rec.append(mode(x))
    drift = float(np.max(np.abs(energy(x, v) - e0) / np.abs(e0)))
    return np.array(t_rec), np.array(rec), drift


def fit(series, t, weighted=True):
    """As kappa_resolution_test.py."""
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
    return abs(sol[0]), float(np.sqrt(cov[0, 0]))


def odd(t, S, ip, im, weighted=True):
    a = fit(S[:, ip], t, weighted)
    b = fit(S[:, im], t, weighted)
    return a[0] - b[0], math.hypot(a[1], b[1])


def validate():
    t, S, d = run((8, 1, 1), [([(0.30, 0, 0)], +1), ([(0.30, 0, 0)], -1)])
    dw, _ = odd(t, S, 0, 1)
    k = (abs(dw) / TH - 1) / 0.09
    ok = abs(k - KPW[0.30]) <= 0.02 * abs(KPW[0.30]) and d < 1e-6
    print(f"  VALIDATION plane wave A = 0.30: kappa {k:+.5f} (ref {KPW[0.30]:+.5f}), "
          f"drift {d:.1e}   {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def measure(shape, qlist, amps):
    """qlist: (my, mz) grid indices. One batched run."""
    configs, keys = [], []
    for s in (+1, -1):
        configs.append(([(A_PROBE, 0, 0)], s)); keys.append(("base", s))
    for A1 in amps:
        for my, mz in qlist:
            for s in (+1, -1):
                configs.append(([(A_PROBE, 0, 0), (A1, my, mz)], s))
                keys.append((A1, my, mz, s))
    t0 = time.time()
    t, S, d = run(shape, configs)
    print(f"  box {shape}: {len(configs)} runs, {time.time()-t0:.0f} s, drift {d:.1e}", flush=True)
    idx = {k: i for i, k in enumerate(keys)}
    base, base_e = odd(t, S, idx[("base", +1)], idx[("base", -1)])
    base_u, _ = odd(t, S, idx[("base", +1)], idx[("base", -1)], False)
    out = {}
    for A1 in amps:
        for my, mz in qlist:
            dw, e = odd(t, S, idx[(A1, my, mz, +1)], idx[(A1, my, mz, -1)])
            dwu, _ = odd(t, S, idx[(A1, my, mz, +1)], idx[(A1, my, mz, -1)], False)
            Kq = (dw - base) / A1 ** 2
            Ku = (dwu - base_u) / A1 ** 2
            err = max(abs(Kq - Ku), math.hypot(e, base_e) / A1 ** 2)
            ampmin = float(np.min(np.abs(S[:, idx[(A1, my, mz, +1)]])) / (A_PROBE / 2))
            out[(A1, my, mz)] = (Kq, err, ampmin)
    return out, d


def main():
    sys.path.insert(0, ".")
    import kappa_cross_pt as PT
    print("=" * 74)
    print("CROSS KERNEL, MEASURED -- two-wave runs (probe 0.02 + one sideways pump)")
    print("=" * 74)
    if not validate():
        print("  VALIDATION FAILED -- stopping."); return
    amps = (0.30, 0.10)
    axis = [(m, 0) for m in range(1, 9)]
    diag = [(m, m) for m in range(1, 9)]
    r_axis, d1 = measure((8, 16, 1), axis, amps)
    r_diag, d2 = measure((8, 16, 16), diag, amps)
    print(f"  energy drift worst {max(d1, d2):.1e}   {'PASS' if max(d1, d2) < 1e-6 else 'FAIL'}")
    print("\n  R(q) = K(q) / (kappa_pw TH);  PT = kappa_cross_pt.py (A -> 0)")
    print("  probe min |a|/a0 over the run shown as 'probe' (1 = undisturbed)")
    print("   q/pi (y, z)     R meas A1=0.30     R meas A1=0.10      PT      probe")
    for tag, res in (("axis", r_axis), ("diag", r_diag)):
        for m in range(1, 9):
            my, mz = (m, 0) if tag == "axis" else (m, m)
            qy, qz = 2 * math.pi * my / 16, 2 * math.pi * mz / 16
            pt = PT.R(qy, qz)
            cells = []
            for A1 in amps:
                Kq, e, pmin = res[(A1, my, mz)]
                k0 = KPW[A1] * TH
                cells.append((Kq / k0, e / abs(k0), pmin))
            print(f"   ({qy/math.pi:5.3f}, {qz/math.pi:5.3f})   "
                  f"{cells[0][0]:6.3f} +- {cells[0][1]:5.3f}    "
                  f"{cells[1][0]:6.3f} +- {cells[1][1]:5.3f}    {pt:6.3f}   {cells[0][2]:.3f}",
                  flush=True)


if __name__ == "__main__":
    main()
