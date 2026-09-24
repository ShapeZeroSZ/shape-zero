#!/usr/bin/env python3
"""
phi_gauge_nonlinear.py — follow-up to phi_gauge_test.py (Shape Zero v5.1 §5.2.7)

Question: is the synthetic gauge signature protected against the network's own
nonlinearity? The on-site restoring force about phi is sqrt(5)*u + u^2. The u^2
term is direction-blind, so linear theory predicts:

  * band centre  (omega(k)+omega(-k))/2  SOFTENS with amplitude (~ A^2)
  * band asymmetry  d_omega = omega(k)-omega(-k)  stays PINNED at -2*c*beta*sin(k)

The exact-root argument: omega^2 + 2*c*beta*sin(k)*omega - W^2 = 0 gives
d_omega = -2*c*beta*sin(k) for ANY W^2, so a symmetric nonlinear shift of W^2
cancels in the difference. But omega(k) != omega(-k) when beta != 0, so the
nonlinear shift itself can differ between the two movers — a possible beta*A^2
correction. This script measures whether one appears.

Method: excite a single traveling wave at k = pi/2 (right-mover, then
left-mover), gamma = 0 (gyroscopic coupling is conservative), and read each
mover's frequency from the phase slope of its spatial-FFT bin — precision far
below the FFT bin width.
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
N = 64
C = 1.0
DT = 0.02
T = 300.0
M = N // 4                      # k = pi/2
K = 2 * np.pi * M / N
OMEGA0 = np.sqrt(np.sqrt(5) + 2 * C * (1 - np.cos(K)))   # linear band at k, beta=0


def force(x, v, beta):
    xp, xm = np.roll(x, -1), np.roll(x, 1)
    vp, vm = np.roll(v, -1), np.roll(v, 1)
    return -(x * x - x - 1.0) + C * (xp + xm - 2.0 * x) + C * beta * (vp - vm)


def run_wave(amp, beta, direction):
    """Evolve a single traveling wave; return u_n(t). direction=+1 right, -1 left."""
    n = np.arange(N)
    x = PHI + amp * np.cos(K * n)
    v = direction * amp * OMEGA0 * np.sin(K * n)
    steps = int(T / DT)
    rec = np.empty((steps, N))
    for s in range(steps):
        k1v = force(x, v, beta);                          k1x = v
        k2v = force(x + 0.5*DT*k1x, v + 0.5*DT*k1v, beta); k2x = v + 0.5*DT*k1v
        k3v = force(x + 0.5*DT*k2x, v + 0.5*DT*k2v, beta); k3x = v + 0.5*DT*k2v
        k4v = force(x + DT*k3x, v + DT*k3v, beta);         k4x = v + DT*k3v
        x = x + DT/6 * (k1x + 2*k2x + 2*k3x + k4x)
        v = v + DT/6 * (k1v + 2*k2v + 2*k3v + k4v)
        rec[s] = x
    return rec - PHI


def mode_freq(rec, direction, t_lo=20.0):
    """Frequency of the excited mover via phase-slope of spatial-FFT bin K.

    Right-mover ~ e^{-i w t} in bin K (slope -w); left-mover ~ e^{+i w t}
    (slope +w). Returns positive frequency, plus mode-energy retention.
    """
    uk = np.fft.fft(rec, axis=1)[:, M]
    i0 = int(t_lo / DT)
    t = np.arange(len(uk)) * DT
    phase = np.unwrap(np.angle(uk[i0:]))
    slope = np.polyfit(t[i0:], phase, 1)[0]
    retention = np.abs(uk[-len(uk)//10:]).mean() / np.abs(uk[i0:i0+len(uk)//10]).mean()
    return -direction * slope, retention


if __name__ == '__main__':
    amps = [0.001, 0.05, 0.1, 0.2, 0.3, 0.4]
    print(f'k=pi/2, gamma=0, N={N}, c={C}; theory d_omega(beta=0.05) = {-2*C*0.05*np.sin(K):.4f}')
    print(f'{"beta":>5} {"A":>6} {"omega(+k)":>10} {"omega(-k)":>10} '
          f'{"mean":>8} {"d_omega":>9} {"retain":>7}')
    curves = {}
    for beta in (0.0, 0.05):
        rows = []
        for A in amps:
            wp, rp = mode_freq(run_wave(A, beta, +1), +1)
            wm, rm = mode_freq(run_wave(A, beta, -1), -1)
            rows.append((A, wp, wm, 0.5*(wp+wm), wp-wm, min(rp, rm)))
            print(f'{beta:>5.2f} {A:>6.3f} {wp:>10.5f} {wm:>10.5f} '
                  f'{0.5*(wp+wm):>8.5f} {wp-wm:>9.5f} {min(rp,rm):>7.3f}')
        curves[beta] = np.array(rows)
