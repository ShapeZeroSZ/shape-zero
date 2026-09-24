#!/usr/bin/env python3
"""
s2_universality.py — S2: universality inversion (phi-necessity stress test)
(Shape Zero action log, Phase S)

Control lattice: on-site force -(K u + alpha u^2), K = 2.0, alpha = 0.7
(vs the phi-lattice's K = sqrt5, alpha = 1). Generalized closed form:

  L(k,w) = (A^2/4) * alpha^2 * (4/K + 2/L(2k,2w)),
  L(k,w) = -w^2 + K + 2c(1-cos k) - 2c*beta*w*sin k.

Predictions STATED BEFORE MEASUREMENT (K=2, alpha=0.7, c=1, k=pi/2, w0=2):
  self-softening      -0.0551 A^2
  asym correction     +0.0196 beta A^2   (general law: 4*alpha^2/L2^2)
  linear gauge law    dw = -2c*beta*sin k   (STIFFNESS-FREE: identical to phi)
  decay               Delta0 = 0.1363; window near the cross-softened center

If confirmed: every measured law is universal; the phi-lattice value enters
only through rescaled constants. The phi-claim then lives in the origin
derivation and the endurance-selection principle, not in the potential.
"""

import numpy as np

K_S, ALPHA, C = 2.0, 0.7, 1.0
N, DT, T, TLO = 64, 0.02, 600.0, 50.0
M = N // 4
K = 2 * np.pi * M / N

def w_lin(kk, beta):
    b = C * beta * np.sin(kk)
    q = K_S + 2 * C * (1 - np.cos(kk))
    return -b + np.sqrt(b * b + q)

def Lfun(kk, w, beta):
    return -w*w + K_S + 2*C*(1 - np.cos(kk)) - 2*C*beta*w*np.sin(kk)

def w_pt(kk, beta, A):
    w = w_lin(kk, beta)
    for _ in range(60):
        rhs = (A*A/4) * ALPHA**2 * (4/K_S + 2/Lfun(2*kk, 2*w, beta))
        b = C * beta * np.sin(kk)
        q = K_S + 2*C*(1 - np.cos(kk)) - rhs
        w = -b + np.sqrt(b*b + q)
    return w

def force(u, v, beta):
    up, um = np.roll(u, -1), np.roll(u, 1)
    vp, vm = np.roll(v, -1), np.roll(v, 1)
    return -(K_S*u + ALPHA*u*u) + C*(up + um - 2*u) + C*beta*(vp - vm)

def run_wave(amp, beta, direction, t_end=T, seed=None):
    n = np.arange(N)
    w0 = w_lin(direction * K, beta)
    u = amp * np.cos(K * n)
    v = direction * amp * w0 * np.sin(K * n)
    if seed is not None:
        rng = np.random.default_rng(seed)
        u = u + 1e-6*rng.standard_normal(N)
        v = v + 1e-6*rng.standard_normal(N)
    steps = int(t_end / DT)
    rec = np.empty((steps, N))
    for s in range(steps):
        k1v = force(u, v, beta);                        k1u = v
        k2v = force(u+0.5*DT*k1u, v+0.5*DT*k1v, beta);  k2u = v+0.5*DT*k1v
        k3v = force(u+0.5*DT*k2u, v+0.5*DT*k2v, beta);  k3u = v+0.5*DT*k2v
        k4v = force(u+DT*k3u, v+DT*k3v, beta);          k4u = v+DT*k3v
        u = u + DT/6*(k1u+2*k2u+2*k3u+k4u)
        v = v + DT/6*(k1v+2*k2v+2*k3v+k4v)
        rec[s] = u
    return rec

def mode_freq(rec, direction):
    uk = np.fft.fft(rec, axis=1)[:, M]
    i0 = int(TLO / DT)
    t = np.arange(len(uk)) * DT
    return -direction * np.polyfit(t[i0:], np.unwrap(np.angle(uk[i0:])), 1)[0]

if __name__ == '__main__':
    AMPS = np.array([0.04, 0.06, 0.09, 0.13, 0.17, 0.20])
    A2, A4 = AMPS**2, AMPS**4
    X = np.vstack([A2, A4]).T

    # linear gauge law (stiffness-free prediction)
    beta = 0.05
    wp = mode_freq(run_wave(0.01, beta, +1), +1)
    wm = mode_freq(run_wave(0.01, beta, -1), -1)
    print(f'linear asym: {wp-wm:+.5f}   predicted -2c*beta*sin k = {-2*C*beta:+.5f}')

    # self-shift
    ws = np.array([mode_freq(run_wave(A, 0.0, +1), +1) for A in AMPS])
    s2, s4 = np.linalg.lstsq(X, ws - w_lin(K, 0.0), rcond=None)[0]
    s_pt = (w_pt(K, 0, 1e-3) - w_lin(K, 0)) / 1e-6
    print(f'self-shift : {s2:+.4f} A^2   PT {s_pt:+.4f}   (phi-lattice: -0.0973)')

    # asymmetry correction
    dev = []
    for A in AMPS:
        wp = mode_freq(run_wave(A, beta, +1), +1)
        wm = mode_freq(run_wave(A, beta, -1), -1)
        dev.append((wp - wm) + 2*C*beta*np.sin(K))
    d2, d4 = np.linalg.lstsq(X, np.array(dev), rcond=None)[0]
    pt = (w_pt(K, beta, 1e-3) - w_pt(-K, beta, 1e-3) + 2*C*beta) / 1e-6
    print(f'asym corr  : {d2/beta:+.4f} beta A^2   PT {pt/beta:+.4f}'
          f'   general law 4a^2/L2^2 = {4*ALPHA**2/Lfun(2*K, 2*w_lin(K,0), 0)**2:+.4f}'
          f'   (phi-lattice: +0.0349)')

    # decay window spot checks near the cross-softened resonance
    W_SUM = np.sqrt(K_S) + np.sqrt(K_S + 4*C)
    print(f'\ndecay: Delta0 = {2*w_lin(K,0) - W_SUM:.4f}  (phi-lattice: 0.1238)')
    for b in (0.06, 0.07, 0.08):
        rec = run_wave(0.35, b, +1, seed=0)
        r = abs(np.fft.fft(rec[-1])[M]) / abs(np.fft.fft(rec[0])[M])
        rm = run_wave(0.35, b, -1, seed=0)
        r2 = abs(np.fft.fft(rm[-1])[M]) / abs(np.fft.fft(rm[0])[M])
        print(f'  beta={b:.2f}, A=0.35: +k retention {r:.3f}   -k retention {r2:.3f}')
