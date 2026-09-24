#!/usr/bin/env python3
"""
phi_gauge_precession.py — A5: finite-amplitude spinor self-precession
(Shape Zero action log item A5; the O(2)-breaking measurement anticipated in v5.2)

The per-component u^2 nonlinearity is not U(2)-invariant: it acts as
independent self-phase modulation on each dimer. Harmonic balance for a
circular dimer wave (channels: DC, co-rotating (2k,2w), counter-rotating
(-2k,-2w); the kappa term splits the last two) gives

  dw_circ = a^2 * [ -1/sqrt5 - 1/(4 L(2k,2w)) - 1/(4 L(-2k,-2w)) ] / (2w + kappa),
  L(K,W)  = -W^2 - kappa*W + sqrt5 + 2c(1 - cos K).

Prediction at k = pi/2, c = 1, kappa = 0.5 (STATED BEFORE MEASUREMENT):
spinor Bloch precession about the z (dimer-population) axis at rate
  Omega = C * A^2 * n_z,   C = -0.0896,   with n_z conserved.
"""

import numpy as np
import phi_gauge_chiral as P

SQ5 = np.sqrt(5)

def L(K, W):
    return -W*W - P.KAPPA*W + SQ5 + 2*P.C*(1 - np.cos(K))

def C_predicted():
    w = P.OMEGA
    bracket = -1/SQ5 - 1/(4*L(2*P.K0, 2*w)) - 1/(4*L(-2*P.K0, -2*w))
    return bracket / (2*w + P.KAPPA)

def run_uniform(A, chi, t_end=300.0, n_meas=30):
    """Uniform circular-carrier plane wave, spinor (cos chi, sin chi)."""
    n = np.arange(P.N)
    s0 = np.array([np.cos(chi), np.sin(chi)])
    u = np.zeros((P.N, 4)); v = np.zeros((P.N, 4))
    for j in (0, 1):
        comp = A * s0[j] * np.exp(1j * P.K0 * n)
        d = -1j * P.OMEGA * comp
        u[:, 2*j], u[:, 2*j+1] = comp.real, comp.imag
        v[:, 2*j], v[:, 2*j+1] = d.real, d.imag
    W = np.zeros((P.N, 4, 4)); Wm = W
    steps = int(t_end / P.DT)
    every = steps // n_meas
    ts, phase, nz = [], [], []
    for s in range(steps):
        if s % every == 0:
            psi = u[:, 0::2] + 1j*u[:, 1::2]
            dps = v[:, 0::2] + 1j*v[:, 1::2]
            chi_f = psi + (1j / P.OMEGA) * dps
            S = (chi_f * np.exp(-1j*P.K0*n)[:, None]).sum(axis=0)
            ts.append(s * P.DT)
            phase.append(np.angle(np.conj(S[0]) * S[1]))
            nrm = abs(S[0])**2 + abs(S[1])**2
            nz.append((abs(S[0])**2 - abs(S[1])**2) / nrm)
        k1v = P.force(u, v, W, Wm);                             k1u = u*0 + v
        k2v = P.force(u+0.5*P.DT*k1u, v+0.5*P.DT*k1v, W, Wm);   k2u = v+0.5*P.DT*k1v
        k3v = P.force(u+0.5*P.DT*k2u, v+0.5*P.DT*k2v, W, Wm);   k3u = v+0.5*P.DT*k2v
        k4v = P.force(u+P.DT*k3u, v+P.DT*k3v, W, Wm);           k4u = v+P.DT*k3v
        u = u + P.DT/6*(k1u + 2*k2u + 2*k3u + k4u)
        v = v + P.DT/6*(k1v + 2*k2v + 2*k3v + k4v)
    rate = np.polyfit(ts, np.unwrap(np.array(phase)), 1)[0]
    return rate, nz[0], nz[-1]

if __name__ == '__main__':
    C_pt = C_predicted()
    print(f'prediction: Omega = C * A^2 * n_z,  C = {C_pt:+.4f}  (omega = {P.OMEGA:.4f})\n')
    print(f'{"A":>5} {"chi":>5} {"n_z(0)":>7} {"n_z(T)":>7} {"rate":>11} '
          f'{"rate/(A^2 n_z)":>14}')
    xs, ys = [], []
    cases = ([(0.15, c) for c in (10, 25, 40, 50, 65, 80)]     # n_z sweep
             + [(a, 20) for a in (0.05, 0.10, 0.20)])          # A sweep
    for A, chid in cases:
        chi = np.radians(chid)
        rate, nz0, nzT = run_uniform(A, chi)
        denom = A*A*np.cos(2*chi)
        print(f'{A:>5.2f} {chid:>4}\u00b0 {nz0:>7.3f} {nzT:>7.3f} {rate:>11.3e} '
              f'{rate/denom:>14.4f}')
        xs.append(denom); ys.append(rate)
    xs, ys = np.array(xs), np.array(ys)
    C_meas = float(np.sum(xs*ys) / np.sum(xs*xs))
    resid = float(np.max(np.abs(ys - C_meas*xs)) / np.max(np.abs(ys)))
    print(f'\nglobal fit: C_measured = {C_meas:+.4f}   '
          f'(predicted {C_pt:+.4f}; max residual {100*resid:.1f}%)')
