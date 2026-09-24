#!/usr/bin/env python3
"""
kappa_cross_oos.py -- out-of-sample test of the derived F (P3a vs P3b).

Predictions recorded before this ran: kappa_cross_oos_predictions.txt
(commit 3b19bf7). Measures F at A = 0.10 for three never-measured beams:
    C1  w = 0.75, box 8 x 6 x 6
    C2  w = 1.25, box 8 x 10 x 10
    C3  w = 1.00, box 12 x 12 x 12
Physics, own-branch Fourier-space seed, RK4 (dt = 0.01, T = 300), readout of the
box-wide mode and the kappa error bar are those of kappa_resolution_test.py
(fit and kappa_of imported), generalised to an Lx x L x L box: the x axis holds
K = pi/2 (Lx a multiple of 4), the Gaussian envelope and the transverse average
are over the two L axes.

VALIDATION (same run, A = 0.10): plane wave kappa_pw = -0.01755 (the amplitude
sweep); cubic w = 1, L = 4 reproduces kappa_cross_amplitude.py (-0.004121); and an
8 x 4 x 4 box gives the same kappa -- the check that x length does not matter.
"""

import math
import time
import numpy as np
import kappa_resolution_test as RT
import kappa_cross_pt as PT

A = 0.10
RT.A_NL = A
KPW_REF = -0.01755
K4_REF = -0.004121


def envelope(Lx, L, width):
    if width is None:
        return np.ones((Lx, L, L))
    idx = np.indices((Lx, L, L)).astype(float)
    env = np.ones((Lx, L, L))
    for a in (1, 2):
        d = idx[a] - L / 2.0
        d = (d + L / 2) % L - L / 2
        env = env * np.exp(-0.5 * (d / width) ** 2)
    return env


def seed(Lx, L, width, amp, sign):
    idx0 = np.indices((Lx, L, L))[0].astype(float)
    psi = amp * envelope(Lx, L, width) * np.exp(1j * sign * RT.K * idx0)
    kx = 2 * np.pi * np.fft.fftfreq(Lx)
    kt = 2 * np.pi * np.fft.fftfreq(L)
    k0, k1, k2 = np.meshgrid(kx, kt, kt, indexing="ij")
    b = RT.BETA * RT.C * np.sin(k0)
    om = b + np.sqrt(b * b + RT.SQ5 + 2 * RT.C * ((1 - np.cos(k0)) + (1 - np.cos(k1))
                                                  + (1 - np.cos(k2))))
    v0 = np.fft.ifftn(-1j * om * np.fft.fftn(psi)).real
    return RT.PHI + psi.real, v0


def run(Lx, L, width, label):
    assert Lx % 4 == 0
    xs, vs = [], []
    for a in (RT.A_LIN, A):
        for s in (+1, -1):
            x0, v0 = seed(Lx, L, width, a, s)
            xs.append(x0); vs.append(v0)
    x, v = np.stack(xs), np.stack(vs)
    e0 = RT.energy(x, v)
    m = Lx // 4

    def mode(x):
        return np.fft.fft(x.mean(axis=(2, 3)), axis=1)[:, m] / Lx

    n = int(round(RT.T_RUN / RT.DT))
    t_rec, rec = [0.0], [mode(x)]
    h = RT.DT
    t0 = time.time()
    for step in range(n):
        k1v = RT.force(x, v); k1x = v
        x2 = x + 0.5 * h * k1x; v2 = v + 0.5 * h * k1v
        k2v = RT.force(x2, v2); k2x = v2
        x3 = x + 0.5 * h * k2x; v3 = v + 0.5 * h * k2v
        k3v = RT.force(x3, v3); k3x = v3
        x4 = x + h * k3x; v4 = v + h * k3v
        k4v = RT.force(x4, v4); k4x = v4
        x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        if (step + 1) % RT.REC_EVERY == 0:
            t_rec.append((step + 1) * h)
            rec.append(mode(x))
    drift = float(np.max(np.abs(RT.energy(x, v) - e0) / np.abs(e0)))
    print(f"    {label:22s} box {Lx:2d} x {L:2d} x {L:2d}   {time.time()-t0:5.0f} s   drift {drift:.1e}",
          flush=True)
    t, S = np.array(t_rec), np.array(rec)
    k, r, _, e = RT.kappa_of(t, S)
    return k, r, e, drift


def main():
    print("=" * 78)
    print("OUT-OF-SAMPLE TEST -- derived F, P3a vs P3b, A = 0.10")
    print("=" * 78)
    print("Runs (T = 300):")
    kp, rp, ep, dp = run(8, 8, None, "plane wave")
    k4c, r4c, e4c, d4c = run(4, 4, 1.0, "w = 1 cubic")
    k4x, r4x, e4x, d4x = run(8, 4, 1.0, "w = 1, x = 8")
    cases = [("C1", 0.75, 8, 6), ("C2", 1.25, 8, 10), ("C3", 1.0, 12, 12)]
    res = {c[0]: run(c[2], c[3], c[1], f"{c[0]} w = {c[1]}") for c in cases}
    print("\nVALIDATION")
    ok = abs(kp - KPW_REF) <= 0.02 * abs(KPW_REF) and abs(rp - 1) < 2e-5
    print(f"  plane wave        kappa {kp:+.5f} (ref {KPW_REF:+.5f})   {'PASS' if ok else 'FAIL'}")
    g = abs(k4c - K4_REF) <= max(8e-5, 0.04 * abs(K4_REF))
    print(f"  w = 1 cubic L = 4 kappa {k4c:+.6f} (ref {K4_REF:+.6f})  {'PASS' if g else 'FAIL'}")
    ok &= g
    g = abs(k4x - k4c) <= 2 * math.hypot(e4x, e4c)
    print(f"  8 x 4 x 4 box     kappa {k4x:+.6f} vs cubic {k4c:+.6f}   {'PASS' if g else 'FAIL'}"
          "   (x length does not matter)")
    ok &= g
    worst = max([dp, d4c, d4x] + [v[3] for v in res.values()])
    print(f"  energy drift      worst {worst:.1e}   {'PASS' if worst < 1e-6 else 'FAIL'}")
    ok &= worst < 1e-6
    if not ok:
        print("\n  VALIDATION FAILED -- stopping."); return
    print("\nRESULT  (predictions: kappa_cross_oos_predictions.txt, recorded before this run)")
    print("  config   w      box        measured F        P3a    pull     P3b    pull    P3a-P3b in err bars")
    for tag, w, Lx, L in cases:
        k, _, e, _ = res[tag]
        fill, s = PT.geometry(L, w)
        F = k / (kp * fill)
        Fe = abs(F) * math.hypot(e / abs(k), ep / abs(kp))
        a, b = PT.F_diag(L, w), PT.F_triads(L, w)
        print(f"   {tag}    {w:4.2f}  {Lx:2d}x{L:2d}x{L:2d}   {F:6.3f} +- {Fe:5.3f}    {a:6.3f} {(a-F)/Fe:+6.1f}   "
              f"{b:6.3f} {(b-F)/Fe:+6.1f}     {abs(a-b)/Fe:5.1f}", flush=True)


if __name__ == "__main__":
    main()
