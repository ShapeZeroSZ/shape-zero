#!/usr/bin/env python3
"""platform_error_budget.py -- error budget for the pendulum-ring platform of LAB_NOTE.md (branch
realisation-gyroscopic). Expectations committed first: platform_error_budget_predictions.txt.

Platform (design choices ours): N planar pendula on a ring, motors off (kappa = 0), nearest-neighbour springs,
one bond rotor per bond, per-site coil trim. Per unit I_eff:
    th_n'' = -K_n sin th_n - gamma th_n' + c_n (th_{n+1} - th_n) + c_{n-1} (th_{n-1} - th_n)
             - b_n th_{n+1}' + b_{n-1} th_{n-1}'
Linear analysis: eigenvalues of the first-order system; the +-k modes are the eigenvectors (Im lambda < 0,
w = -Im lambda) of largest overlap with e^{+-ikn}. Time domain: RK4 with sin th, ring-down from a launched
running wave, w from the weighted phase slope of z(t) = sum_n th_n e^{-ikn}.
Predictions tested: P1 Delta w(k)/Delta w(pi/2) = sin k; P2 d(Delta w) = -(1/4) b <dK^2>/c^2 at k = pi/2 (the
+-S even part); P3 its shape independence.
usage: python3 platform_error_budget.py
"""
import numpy as np

# ---- design (ours) --------------------------------------------------------------------------------------
I_EFF = 0.0471          # kg m^2
TAU = 1.5451            # N m / rad, gravity
K0 = TAU / I_EFF        # 32.81 s^-2
C0 = 0.25 * K0          # c-hat = 0.25
B0 = 0.10 * np.sqrt(K0) # beta-hat = 0.10 ; b = H / I_eff
N0 = 32
S = 0.05
DIS = dict(K=0.003, c=0.01, b=0.005)
NREAL = 20


def eta_shape(N, kind, p=None):
    x = np.arange(N, dtype=float)
    if kind == "gauss":
        e = np.exp(-0.5 * ((x - N / 2) / p) ** 2)
    elif kind == "sech2":
        e = 1.0 / np.cosh((x - N / 2) / p) ** 2
    elif kind == "two":
        e = np.exp(-0.5 * ((x - N / 4) / p) ** 2) + np.exp(-0.5 * ((x - 3 * N / 4) / p) ** 2)
    elif kind == "ramp":
        e = x.copy()
    e = e - e.mean()
    return e / np.sqrt((e ** 2).mean())


def system(Kn, cn, bn, gamma):
    N = len(Kn)
    Km = np.diag(Kn + cn + np.roll(cn, 1))
    G = np.zeros((N, N))
    for n in range(N):
        m = (n + 1) % N
        Km[n, m] -= cn[n]; Km[m, n] -= cn[n]
        G[n, m] -= bn[n]; G[m, n] += bn[n]
    A = np.zeros((2 * N, 2 * N))
    A[:N, N:] = np.eye(N)
    A[N:, :N] = -Km
    A[N:, N:] = G - gamma * np.eye(N)
    lam, V = np.linalg.eig(A)
    return lam, V[:N]


def mode_w(lam, V, m, sign):
    N = V.shape[0]
    k = 2 * np.pi * m / N
    pw = np.exp(1j * sign * k * np.arange(N)) / np.sqrt(N)
    sel = lam.imag < 0
    ov = np.abs(pw.conj() @ V[:, sel]) / np.linalg.norm(V[:, sel], axis=0)
    j = np.argmax(ov)
    return -lam[sel][j].imag, ov[j]


def dw(Kn, cn, bn, gamma, m):
    lam, V = system(Kn, cn, bn, gamma)
    return mode_w(lam, V, m, +1)[0] - mode_w(lam, V, m, -1)[0]


def calibrate(Kn, cn, bn, gamma):
    """As an experimenter would: rotors off -> band -> K, c; rotors on, S = 0 -> b from Delta w(pi/2)."""
    N = len(Kn)
    lam, V = system(Kn, cn, 0 * bn, gamma)
    ms = np.arange(0, N // 2 + 1)
    w = np.array([0.5 * (mode_w(lam, V, m, 1)[0] + mode_w(lam, V, m, -1)[0]) for m in ms])
    X = np.vstack([np.ones_like(w), 2 * (1 - np.cos(2 * np.pi * ms / N))]).T
    (Kc, cc), *_ = np.linalg.lstsq(X, w ** 2, rcond=None)
    bc = dw(Kn, cn, bn, gamma, N // 4) / 2
    return Kc, cc, bc


def C_ratio(Kn, cn, bn, gamma, eta, S=S, one_sided=False):
    N = len(Kn)
    Kc, cc, bc = calibrate(Kn, cn, bn, gamma)
    dK = Kc * S * eta
    m = N // 4
    d0 = dw(Kn, cn, bn, gamma, m)
    dp = dw(Kn + dK, cn, bn, gamma, m)
    if one_sided:
        shift = dp - d0
    else:
        dm = dw(Kn - dK, cn, bn, gamma, m)
        shift = 0.5 * (dp + dm) - d0
    pred = -0.25 * bc * np.mean(dK ** 2) / cc ** 2
    return shift / pred, shift, pred


def realisation(N, rng):
    return (K0 * (1 + DIS["K"] * rng.standard_normal(N)), C0 * (1 + DIS["c"] * rng.standard_normal(N)),
            B0 * (1 + DIS["b"] * rng.standard_normal(N)))


def p1_residual(Kn, cn, bn, gamma):
    N = len(Kn)
    ref = dw(Kn, cn, bn, gamma, N // 4)
    return max(abs(dw(Kn, cn, bn, gamma, m) / ref - np.sin(2 * np.pi * m / N)) for m in range(1, N // 2))


# ---- time domain ---------------------------------------------------------------------------------------
def ringdown_w(Kn, cn, bn, gamma, m, sign, A, T=300.0, dt=0.01, every=5):
    N = len(Kn)
    k = 2 * np.pi * m / N
    x = np.arange(N)
    Qk = K0 + 2 * C0 * (1 - np.cos(k))
    w0 = B0 * np.sin(k) * sign + np.sqrt(B0 ** 2 * np.sin(k) ** 2 + Qk)
    th = A * np.cos(k * x)
    v = sign * A * w0 * np.sin(k * x)

    def acc(th, v):
        return (-Kn * np.sin(th) - gamma * v + cn * (np.roll(th, -1) - th) + np.roll(cn, 1) * (np.roll(th, 1) - th)
                - bn * np.roll(v, -1) + np.roll(bn, 1) * np.roll(v, 1))

    ph = np.exp(-1j * k * x)
    ts, zs = [], []
    nsteps = int(round(T / dt))
    for i in range(nsteps + 1):
        if i % every == 0:
            ts.append(i * dt); zs.append(ph @ th)
        k1v = acc(th, v); k1x = v
        k2v = acc(th + .5 * dt * k1x, v + .5 * dt * k1v); k2x = v + .5 * dt * k1v
        k3v = acc(th + .5 * dt * k2x, v + .5 * dt * k2v); k3x = v + .5 * dt * k2v
        k4v = acc(th + dt * k3x, v + dt * k3v); k4x = v + dt * k3v
        th = th + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    ts, zs = np.array(ts), np.array(zs)
    phase = np.unwrap(np.angle(zs))
    wts = np.abs(zs) ** 2
    slope = np.polyfit(ts, phase, 1, w=np.sqrt(wts))[0]
    return -slope if sign > 0 else slope


def dw_time(Kn, cn, bn, gamma, m, A):
    return ringdown_w(Kn, cn, bn, gamma, m, +1, A) - ringdown_w(Kn, cn, bn, gamma, m, -1, A)


def main():
    N = N0
    gamma500 = np.sqrt(K0 + 2 * C0) / 500
    ones = np.ones(N)
    ideal = (K0 * ones, C0 * ones, B0 * ones)
    g4 = eta_shape(N, "gauss", 4)
    print(f"PLATFORM ERROR BUDGET -- N = {N}, K = {K0:.3f} s^-2, c = {C0:.3f} s^-2 (c-hat 0.25), b = {B0:.4f} s^-1 "
          f"(beta-hat 0.10), S = {S}")
    print(f"  ideal Delta w(pi/2) = {dw(*ideal, 0.0, N // 4):.6f} rad/s (2b = {2 * B0:.6f}); predicted d(Delta w) = "
          f"{-0.25 * B0 * (K0 * S) ** 2 / C0 ** 2:.4e} rad/s, relative {-S ** 2 / (8 * 0.25 ** 2):.4e}")
    print("E1 finite ring, ideal (Gaussian sigma = N/8, +-S even part)")
    print(f"  P1 max residual (N = 32) {p1_residual(*ideal, 0.0):.1e}")
    for NN in (32, 64, 128):
        o = np.ones(NN)
        for SS in (0.05, 0.02):
            r, sh, pr = C_ratio(K0 * o, C0 * o, B0 * o, 0.0, eta_shape(NN, "gauss", NN / 8), S=SS)
            print(f"  N = {NN:<3d} S = {SS}: d(Delta w) {sh:+.5e}, 1/4-law {pr:+.5e}, ratio {r:.4f}")
    print("E2 damping")
    for Q in (500, 200):
        gm = np.sqrt(K0 + 2 * C0) / Q
        d0 = dw(*ideal, 0.0, N // 4); dg = dw(*ideal, gm, N // 4)
        r0 = C_ratio(*ideal, 0.0, g4)[0]; rg = C_ratio(*ideal, gm, g4)[0]
        print(f"  Q = {Q}: Delta w(pi/2) change {dg / d0 - 1:+.2e}; C ratio {rg:.5f} vs {r0:.5f} (change {rg / r0 - 1:+.2e})")
    print(f"E3 disorder (K {DIS['K']:.1%}, c {DIS['c']:.0%}, b {DIS['b']:.1%} rms), {NREAL} realisations, Q = 500")
    rng = np.random.default_rng(2026)
    P1, R, R1, SH = [], [], [], []
    shapes = [("gauss", 3), ("gauss", 5), ("sech2", 4), ("two", 2.5), ("ramp", None)]
    for i in range(NREAL):
        Kn, cn, bn = realisation(N, rng)
        P1.append(p1_residual(Kn, cn, bn, gamma500))
        R.append(C_ratio(Kn, cn, bn, gamma500, g4)[0])
        R1.append(C_ratio(Kn, cn, bn, gamma500, g4, one_sided=True)[0])
        SH.append([C_ratio(Kn, cn, bn, gamma500, eta_shape(N, s, p))[0] for s, p in shapes])
    P1, R, R1, SH = map(np.array, (P1, R, R1, SH))
    print(f"  P1 max residual: mean {P1.mean():.1e}, max {P1.max():.1e}, sd {P1.std():.1e}")
    print(f"  P2 +-S protocol: C ratio mean {R.mean():.4f}, sd {R.std():.4f}, range {R.min():.4f} - {R.max():.4f}")
    print(f"  P2 one-sided:    C ratio mean {R1.mean():.4f}, sd {R1.std():.4f}, range {R1.min():.4f} - {R1.max():.4f}")
    loc = SH[:, :4]
    spread = (loc.max(axis=1) - loc.min(axis=1)) / np.abs(loc.mean(axis=1))
    print(f"  P3 shapes (gauss 3, gauss 5, sech2 4, two bumps) mean ratios {', '.join(f'{x:.4f}' for x in loc.mean(0))}; "
          f"ramp {SH[:, 4].mean():.4f}")
    print(f"     spread among the four localised shapes per realisation: mean {spread.mean():.3f}, max {spread.max():.3f}; "
          f"ramp / localised mean {np.mean(SH[:, 4] / loc.mean(axis=1)):.3f}")
    print(f"  ideal-ring shapes: {', '.join(f'{C_ratio(*ideal, 0.0, eta_shape(N, s, p))[0]:.4f}' for s, p in shapes)}")
    print("E4/E5 time domain (RK4 with sin th, Q = 500, T = 300 s, 20 Hz sampling), disorder realisation 0")
    rng = np.random.default_rng(2026)
    Kn, cn, bn = realisation(N, rng)
    Kc, cc, bc = calibrate(Kn, cn, bn, gamma500)
    dK = Kc * S * g4
    m = N // 4
    pred = -0.25 * bc * np.mean(dK ** 2) / cc ** 2
    e0 = dw(Kn, cn, bn, gamma500, m)
    ee = 0.5 * (dw(Kn + dK, cn, bn, gamma500, m) + dw(Kn - dK, cn, bn, gamma500, m)) - e0
    print(f"  eigen: Delta w(pi/2) {e0:.7f}; d(Delta w) {ee:+.5e}; ratio to 1/4-law {ee / pred:.4f}")
    for A in (0.02, 0.1):
        t0 = dw_time(Kn, cn, bn, gamma500, m, A)
        tp = dw_time(Kn + dK, cn, bn, gamma500, m, A); tm = dw_time(Kn - dK, cn, bn, gamma500, m, A)
        tt = 0.5 * (tp + tm) - t0
        print(f"  A = {A} rad: Delta w(pi/2) {t0:.7f} (vs eigen {t0 / e0 - 1:+.2e}); d(Delta w) {tt:+.5e} "
              f"(vs eigen {tt / ee - 1:+.2e}); ratio to 1/4-law {tt / pred:.4f}")
    print("  P1 in the time domain (A = 0.02): Delta w(m)/Delta w(8) - sin k at m = 2, 4, 12, 14:",
          ", ".join(f"{dw_time(Kn, cn, bn, gamma500, mm, 0.02) / dw_time(Kn, cn, bn, gamma500, m, 0.02) - np.sin(2 * np.pi * mm / N):+.1e}"
                    for mm in (2, 4, 12, 14)))


if __name__ == "__main__":
    main()
