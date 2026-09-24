#!/usr/bin/env python3
"""
s2b_winding.py — S2b: what organizes endurance — resonance geometry or
Diophantine winding? (Shape Zero action log, Phase S)

Two carriers (k1 = pi/2 fixed, k2 swept over the ring) coexist on the
phi-lattice at beta = 0. Joint survival S = r1 * r2 over T = 600 is measured
against two candidate predictors:

  M3(k2): three-wave mismatch, min over the momentum-exact sum/difference
          channels |w(k1 +- k2) - (w1 +- w2)|   [dispersion geometry]
  B(rho): Diophantine badness of rho = w2/w1, min_{q<=10} q^2 |rho - p/q|
          [noble-winding selection; smaller = more rational = worse]

Prediction hierarchy STATED BEFORE RUNNING: in a quadratic-nonlinearity chain
the leading interactions are three-wave, so M3 should dominate survival.
Whether any residual Diophantine signal exists at this order is the open
question — an honest null is the expected outcome, correctly scoping the
noble-selection principle to torus/phase-locking dynamics.
"""

import numpy as np

SQ5 = np.sqrt(5)
N, C, DT, T = 128, 1.0, 0.02, 600.0
M1 = N // 4
K1 = 2 * np.pi * M1 / N
AMP = 0.22

def w_of(k):
    return np.sqrt(SQ5 + 2 * C * (1 - np.cos(k)))

def force(u, v):
    up, um = np.roll(u, -1), np.roll(u, 1)
    return -(SQ5*u + u*u) + C*(up + um - 2*u)

def run_two(m2, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    k2 = 2*np.pi*m2/N
    u = AMP*np.cos(K1*n) + AMP*np.cos(k2*n)
    v = AMP*w_of(K1)*np.sin(K1*n) + AMP*w_of(k2)*np.sin(k2*n)
    u += 1e-6*rng.standard_normal(N); v += 1e-6*rng.standard_normal(N)
    a0 = np.abs(np.fft.fft(u)[[M1, m2]])
    for _ in range(int(T/DT)):
        k1v = force(u, v);                      k1u = v
        k2v = force(u+0.5*DT*k1u, v+0.5*DT*k1v); k2u = v+0.5*DT*k1v
        k3v = force(u+0.5*DT*k2u, v+0.5*DT*k2v); k3u = v+0.5*DT*k2v
        k4v = force(u+DT*k3u, v+DT*k3v);        k4u = v+DT*k3v
        u = u + DT/6*(k1u+2*k2u+2*k3u+k4u)
        v = v + DT/6*(k1v+2*k2v+2*k3v+k4v)
    aT = np.abs(np.fft.fft(u)[[M1, m2]])
    return float(aT[0]/a0[0]) * float(aT[1]/a0[1])

def M3(m2):
    k2 = 2*np.pi*m2/N
    w1, w2 = w_of(K1), w_of(k2)
    s = abs(w_of(K1 + k2) - (w1 + w2))
    d = abs(w_of(K1 - k2) - abs(w1 - w2))
    return min(s, d)

def dioph(rho, qmax=10):
    return min(q*q*abs(rho - round(q*rho)/q) for q in range(1, qmax+1))

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx@ry)/np.sqrt((rx@rx)*(ry@ry)))

if __name__ == '__main__':
    m2s = [m for m in range(20, 53) if abs(m - M1) > 2]
    S, m3s, bs, rhos = [], [], [], []
    print(f'{"m2":>4} {"rho=w2/w1":>10} {"M3":>8} {"B(rho)":>8} {"survival":>9}')
    for m2 in m2s:
        s = run_two(m2)
        rho = w_of(2*np.pi*m2/N)/w_of(K1)
        S.append(s); m3s.append(M3(m2)); bs.append(dioph(rho)); rhos.append(rho)
        print(f'{m2:>4} {rho:>10.5f} {M3(m2):>8.4f} {dioph(rho):>8.4f} {s:>9.3f}')
    S, m3s, bs = map(np.array, (S, m3s, bs))
    print(f'\nSpearman(survival, M3 three-wave mismatch): {spearman(S, m3s):+.3f}')
    print(f'Spearman(survival, Diophantine badness B) : {spearman(S, bs):+.3f}')
    worst = np.argsort(S)[:4]
    print('\nfour deepest dips:')
    for i in worst:
        print(f'  m2={m2s[i]}: survival {S[i]:.3f}, M3 {m3s[i]:.4f}, rho {rhos[i]:.4f}')
    np.save('/home/claude/s2b_data.npy', np.vstack([m2s, S, m3s, bs, rhos]))
