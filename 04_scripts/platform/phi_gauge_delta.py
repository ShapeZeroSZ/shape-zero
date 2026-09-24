#!/usr/bin/env python3
"""
phi_gauge_delta.py — A2: analytic derivation of the asymmetry correction delta
(Shape Zero action log item A2; refines v5.2 Result 2)

Second-order (Lindstedt) perturbation theory for the traveling wave with the
gyroscopic gauge term gives a closed-form nonlinear dispersion:

    L(k, w) = (A^2/4) * (4/sqrt5 + 2 / L(2k, 2w)),
    L(k, w) = -w^2 + sqrt5 + 2c(1 - cos k) - 2c*beta*w*sin k.

(Reduces exactly to the textbook quadratic-oscillator shift -5a^2/(12 w0^3)
at c = 0.) This predicts, at k = pi/2, c = 1:
    self-softening  -0.0973 A^2      (v5.2 measured: -0.101 A^2)
    asymmetry corr. +0.0349 beta A^2 (v5.2 measured: -0.0598 beta A^2)

Hypothesis under test: the v5.2 delta was fitted as pure A^2 over
A in [0.05, 0.2], where the near-resonant virtual (0, pi) pair channel — the
same channel that produces the Result 3 parametric decay, detuned by only
Delta = Delta0 - 2*beta (= 0.024 at beta = 0.05) — contributes a large
direction-selective O(A^4) term that contaminated the fit. Test: refined ICs,
two-term fit  dev = d2*A^2 + d4*A^4.  PT is confirmed if d2 -> +0.0349*beta.
"""

import numpy as np

SQ5 = np.sqrt(5)
PHI = (1 + SQ5) / 2
N, C, DT, T, TLO = 64, 1.0, 0.02, 600.0, 50.0
M = N // 4
K = 2 * np.pi * M / N
Q0 = SQ5 + 2 * C * (1 - np.cos(K))

def w_lin(kk, beta):
    """Exact linear root of  w^2 + 2c*beta*sin(kk)*w - Q0(kk) = 0  (w > 0)."""
    b = C * beta * np.sin(kk)
    q = SQ5 + 2 * C * (1 - np.cos(kk))
    return -b + np.sqrt(b * b + q)

def Lfun(kk, w, beta):
    return -w*w + SQ5 + 2*C*(1 - np.cos(kk)) - 2*C*beta*w*np.sin(kk)

def w_pt(kk, beta, A):
    """Self-consistent second-order PT frequency."""
    w = w_lin(kk, beta)
    for _ in range(60):
        L2 = Lfun(2*kk, 2*w, beta)
        rhs = (A*A/4) * (4/SQ5 + 2/L2)
        # solve L(kk, w) = rhs  ->  w^2 + 2c*beta*sin(kk)*w - (Q(kk) - rhs) = 0
        b = C * beta * np.sin(kk)
        q = SQ5 + 2*C*(1 - np.cos(kk)) - rhs
        w = -b + np.sqrt(b*b + q)
    return w

def force(x, v, beta):
    xp, xm = np.roll(x, -1), np.roll(x, 1)
    vp, vm = np.roll(v, -1), np.roll(v, 1)
    return -(x*x - x - 1.0) + C*(xp + xm - 2.0*x) + C*beta*(vp - vm)

def run_wave(amp, beta, direction):
    """Traveling wave with EXACT beta-dependent linear frequency in the IC."""
    n = np.arange(N)
    w0 = w_lin(direction * K, beta)
    x = PHI + amp * np.cos(K * n)
    v = direction * amp * w0 * np.sin(K * n)
    steps = int(T / DT)
    rec = np.empty((steps, N))
    for s in range(steps):
        k1v = force(x, v, beta);                           k1x = v
        k2v = force(x + 0.5*DT*k1x, v + 0.5*DT*k1v, beta); k2x = v + 0.5*DT*k1v
        k3v = force(x + 0.5*DT*k2x, v + 0.5*DT*k2v, beta); k3x = v + 0.5*DT*k2v
        k4v = force(x + DT*k3x, v + DT*k3v, beta);         k4x = v + DT*k3v
        x = x + DT/6*(k1x + 2*k2x + 2*k3x + k4x)
        v = v + DT/6*(k1v + 2*k2v + 2*k3v + k4v)
        rec[s] = x
    return rec - PHI

def mode_freq(rec, direction):
    uk = np.fft.fft(rec, axis=1)[:, M]
    i0 = int(TLO / DT)
    t = np.arange(len(uk)) * DT
    slope = np.polyfit(t[i0:], np.unwrap(np.angle(uk[i0:])), 1)[0]
    return -direction * slope

if __name__ == '__main__':
    AMPS = np.array([0.04, 0.06, 0.09, 0.13, 0.17, 0.20])
    A2, A4 = AMPS**2, AMPS**4

    # --- self-shift validation at beta = 0 (two-term fit) ---
    ws = np.array([mode_freq(run_wave(A, 0.0, +1), +1) for A in AMPS])
    dw = ws - w_lin(K, 0.0)
    X = np.vstack([A2, A4]).T
    s2, s4 = np.linalg.lstsq(X, dw, rcond=None)[0]
    s_pt = (w_pt(K, 0.0, 1e-3) - w_lin(K, 0.0)) / 1e-6
    print(f'self-shift:   two-term fit  {s2:+.4f} A^2 {s4:+.3f} A^4   '
          f'| PT {s_pt:+.4f} A^2   | v5.2 single-term fit -0.101')

    # --- asymmetry correction ---
    print(f'\n{"beta":>6} {"d2 (2-term)":>12} {"d4 (2-term)":>12} '
          f'{"d2/beta":>9} {"PT d2/beta":>11} {"1-term fit (v5.2 style)":>24}')
    for beta in (0.025, 0.05):
        dev = []
        for A in AMPS:
            wp = mode_freq(run_wave(A, beta, +1), +1)
            wm = mode_freq(run_wave(A, beta, -1), -1)
            dev.append((wp - wm) + 2*C*beta*np.sin(K))
        dev = np.array(dev)
        d2, d4 = np.linalg.lstsq(X, dev, rcond=None)[0]
        one = float(np.sum(A2*dev) / np.sum(A2*A2))          # v5.2-style fit
        pt = (w_pt(K, beta, 1e-3) - w_pt(-K, beta, 1e-3)
              + 2*C*beta*np.sin(K)) / 1e-6
        print(f'{beta:>6.3f} {d2:>12.5f} {d4:>12.4f} {d2/beta:>9.4f} '
              f'{pt/beta:>11.4f} {one:>17.5f} ({one/beta:+.4f}/beta)')

    # pair-channel detunings for reference
    w0k, w00, w0p = w_lin(K, 0), np.sqrt(SQ5), np.sqrt(SQ5 + 4*C)
    D0 = 2*w0k - w00 - w0p
    print(f'\n(0,pi) pair detuning: Delta0 = {D0:.4f}; at beta: '
          f'Delta(+k) = Delta0 - 2*beta, Delta(-k) = Delta0 + 2*beta')

# ---------------------------------------------------------------------------
# Part 2 — reproduction of the v5.2 protocol bias (record of the correction).
# The v5.2 runs set the IC velocity with the beta = 0 frequency. The O(beta)
# mismatch seeds a counter-rotating admixture in the measurement bin whose
# nonlinear frequency pulling is odd and linear in beta — so it passed the
# v5.2 parity and linearity checks while biasing delta from +0.0349*beta to
# about -0.060*beta. Same integrator, same estimator; only the IC differs.
# ---------------------------------------------------------------------------
def run_wave_v52(amp, beta, direction):
    n = np.arange(N)
    x = PHI + amp * np.cos(K * n)
    v = direction * amp * w_lin(K, 0.0) * np.sin(K * n)   # beta=0 frequency
    steps = int(T / DT)
    rec = np.empty((steps, N))
    for s in range(steps):
        k1v = force(x, v, beta);                           k1x = v
        k2v = force(x + 0.5*DT*k1x, v + 0.5*DT*k1v, beta); k2x = v + 0.5*DT*k1v
        k3v = force(x + 0.5*DT*k2x, v + 0.5*DT*k2v, beta); k3x = v + 0.5*DT*k2v
        k4v = force(x + DT*k3x, v + DT*k3v, beta);         k4x = v + DT*k3v
        x = x + DT/6*(k1x + 2*k2x + 2*k3x + k4x)
        v = v + DT/6*(k1v + 2*k2v + 2*k3v + k4v)
        rec[s] = x
    return rec - PHI

if __name__ == '__main__':
    beta = 0.05
    AMPS52 = np.array([0.05, 0.1, 0.15, 0.2])
    dev52 = []
    for A in AMPS52:
        wp = mode_freq(run_wave_v52(A, beta, +1), +1)
        wm = mode_freq(run_wave_v52(A, beta, -1), -1)
        dev52.append((wp - wm) + 2*C*beta*np.sin(K))
    dev52 = np.array(dev52)
    one = float(np.sum(AMPS52**2 * dev52) / np.sum(AMPS52**4))
    print(f'\nv5.2-protocol reproduction (beta=0.05): delta = {one:+.5f} '
          f'= {one/beta:+.4f}*beta   (v5.2 reported -0.0598*beta)')
