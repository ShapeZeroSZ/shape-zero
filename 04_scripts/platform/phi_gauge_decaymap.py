#!/usr/bin/env python3
"""
phi_gauge_decaymap.py — A5: map of the nonreciprocal decay window
(Shape Zero action log item A5; extends v5.2 Result 3, feeds paper R1)

The +k pump decays via the four-wave channel k0 + k0 -> 0 + pi. Using the
A2-derived nonlinear dispersion (w_pt), the resonance condition
    2 * w_pt(k0, beta, A) = w(0) + w(pi)
gives a predicted resonance curve beta_res(A) ~ 0.0619 - 0.0973 A^2 that this
script overlays on a measured retention map over the (beta, A) plane.
Directional protection is spot-checked at -k, where the detuning
Delta(-k) = Delta0 + 2*beta cannot reach resonance in this amplitude range.
"""

import numpy as np
import phi_gauge_delta as D

SQ5 = np.sqrt(5)
T_RUN = 400.0
W_SUM = np.sqrt(SQ5) + np.sqrt(SQ5 + 4 * D.C)      # w(0) + w(pi) = 3.99256

def run_retention(amp, beta, direction, seed=0):
    rng = np.random.default_rng(seed)
    n = np.arange(D.N)
    w0 = D.w_lin(direction * D.K, beta)
    x = D.PHI + amp * np.cos(D.K * n) + 1e-6 * rng.standard_normal(D.N)
    v = direction * amp * w0 * np.sin(D.K * n) + 1e-6 * rng.standard_normal(D.N)
    a0 = abs(np.fft.fft(x - D.PHI)[D.M])
    steps = int(T_RUN / D.DT)
    for _ in range(steps):
        k1v = D.force(x, v, beta);                             k1x = v
        k2v = D.force(x+0.5*D.DT*k1x, v+0.5*D.DT*k1v, beta);   k2x = v+0.5*D.DT*k1v
        k3v = D.force(x+0.5*D.DT*k2x, v+0.5*D.DT*k2v, beta);   k3x = v+0.5*D.DT*k2v
        k4v = D.force(x+D.DT*k3x, v+D.DT*k3v, beta);           k4x = v+D.DT*k3v
        x = x + D.DT/6*(k1x+2*k2x+2*k3x+k4x)
        v = v + D.DT/6*(k1v+2*k2v+2*k3v+k4v)
    u = np.fft.fft(x - D.PHI)
    return abs(u[D.M]) / a0, abs(u[D.N // 2]) / a0

def beta_res(A):
    """Solve 2*w_pt(K, beta, A) = W_SUM for beta (bisection)."""
    lo, hi = -0.05, 0.20
    f = lambda b: 2 * D.w_pt(D.K, b, A) - W_SUM
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

if __name__ == '__main__':
    BETAS = np.arange(0.03, 0.105, 0.01)
    AMPS = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    R = np.zeros((len(AMPS), len(BETAS)))
    print('retention map (+k pump), rows A, cols beta:')
    print('        ' + ' '.join(f'{b:5.2f}' for b in BETAS))
    for i, A in enumerate(AMPS):
        for j, b in enumerate(BETAS):
            R[i, j], _ = run_retention(A, b, +1)
        print(f'A={A:.2f} ' + ' '.join(f'{r:5.3f}' for r in R[i]))
    np.save('/home/claude/decaymap_R.npy', R)

    print('\npredicted resonance curve beta_res(A) from the A2-derived dispersion:')
    for A in AMPS:
        print(f'  A={A:.2f}: beta_res = {beta_res(A):.4f}')

    print('\ndirectional protection spot checks (-k pump):')
    for A, b in ((0.30, 0.05), (0.40, 0.05), (0.30, 0.08), (0.40, 0.08)):
        r, _ = run_retention(A, b, -1)
        print(f'  A={A:.2f}, beta={b:.2f}: retention = {r:.4f}')
