#!/usr/bin/env python3
"""
phi_gauge_test.py — Minimal computational test protocol, Shape Zero spec v5.1 §5.2.7

Question: does the velocity coupling beta generate a synthetic U(1) gauge field
(Peierls phase) on a ring of phi-attractor oscillators?

Method: integrate the FULL NONLINEAR network
    x_i'' + gamma x_i' + (x_i^2 - x_i - 1) = c * sum_nn (x_j - x_i) + beta-term
at small amplitude about the phi attractor, then measure the dispersion
asymmetry  d_omega(k) = omega(k) - omega(-k)  by space-time FFT.

A genuine gauge phase shifts the band k -> k - beta*A, so:
  * antisymmetric (gyroscopic) coupling  c*beta*(v_{i+1} - v_{i-1})
      theory:  d_omega(k) = -2*c*beta*sin(k)          (nonzero, linear in beta)
  * symmetric coupling as written in the spec  c*beta*(v_{i+1} + v_{i-1})
      theory:  d_omega(k) = 0 exactly (k-dependent DAMPING only; modes near
      k=pi go UNSTABLE once 2*c*beta > gamma — also not gauge behaviour)

Falsification criteria (spec 5.2.7): if the antisymmetric form gives
d_omega = 0 for all beta, the U(1) emergence claim fails as stated.
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
N = 64          # ring sites
C = 1.0         # nearest-neighbour coupling
DT = 0.02
T = 600.0       # long record -> frequency resolution 2*pi/T ~ 0.0105
AMP = 1e-3      # perturbation amplitude (linear regime about phi)
SEED = 7


def force(x, v, beta, form, gamma):
    """Full nonlinear force. form in {'antisym', 'sym'}."""
    xp, xm = np.roll(x, -1), np.roll(x, 1)   # i+1, i-1
    vp, vm = np.roll(v, -1), np.roll(v, 1)
    f = -gamma * v - (x * x - x - 1.0) + C * (xp + xm - 2.0 * x)
    if form == 'antisym':
        f += C * beta * (vp - vm)            # gyroscopic — conservative
    else:
        f += C * beta * (vp + vm)            # spec form — dissipative/antidissipative
    return f


def run(beta, form, gamma):
    rng = np.random.default_rng(SEED)
    x = np.full(N, PHI)
    v = AMP * rng.standard_normal(N)         # excite all k, both directions
    steps = int(T / DT)
    rec = np.empty((steps, N))
    for s in range(steps):                   # RK4
        k1v = force(x, v, beta, form, gamma);            k1x = v
        k2v = force(x + 0.5*DT*k1x, v + 0.5*DT*k1v, beta, form, gamma); k2x = v + 0.5*DT*k1v
        k3v = force(x + 0.5*DT*k2x, v + 0.5*DT*k2v, beta, form, gamma); k3x = v + 0.5*DT*k2v
        k4v = force(x + DT*k3x, v + DT*k3v, beta, form, gamma);         k4x = v + DT*k3v
        x = x + DT/6 * (k1x + 2*k2x + 2*k3x + k4x)
        v = v + DT/6 * (k1v + 2*k2v + 2*k3v + k4v)
        rec[s] = x
    return rec - PHI                         # perturbation field u_n(t)


def peak(omega_axis, power, lo, hi):
    """Quadratically interpolated peak location of power in omega in [lo, hi)."""
    sel = np.where((omega_axis >= lo) & (omega_axis < hi))[0]
    p = sel[np.argmax(power[sel])]
    a, b, cc = power[p-1], power[p], power[p+1]
    denom = a - 2*b + cc
    frac = 0.5 * (a - cc) / denom if denom != 0 else 0.0
    dw = omega_axis[1] - omega_axis[0]
    return omega_axis[p] + frac * dw


def dispersion_asym(rec, m):
    """d_omega(k) for spatial mode m (k = 2*pi*m/N).

    For real u, temporal spectrum of the spatial-FFT bin k holds two peaks:
      negative frequency at -omega(k)   (right-mover)
      positive frequency at +omega(-k)  (conjugate of left-mover)
    """
    uk = np.fft.fft(rec, axis=1)[:, m]
    uk = uk * np.hanning(len(uk))
    spec = np.abs(np.fft.fft(uk)) ** 2
    om = 2 * np.pi * np.fft.fftfreq(len(uk), d=DT)
    order = np.argsort(om)
    om, spec = om[order], spec[order]
    w_pos = peak(om, spec, 0.5, 4.0)         # +omega(-k)
    w_neg = peak(om, spec, -4.0, -0.5)       # -omega(k)
    return (-w_neg) - w_pos                  # omega(k) - omega(-k)


if __name__ == '__main__':
    betas = [0.0, 0.02, 0.05, 0.10]
    m = N // 4                               # k = pi/2, where |sin k| = 1
    k = 2 * np.pi * m / N
    print(f'N={N}  c={C}  k=pi/2  record T={T}  freq resolution ~{2*np.pi/T:.4f}')
    print(f'{"form":>9} {"beta":>6} {"gamma":>6} {"d_omega meas":>13} {"d_omega theory":>15}')
    results = {}
    for form in ('antisym', 'sym'):
        for beta in betas:
            gamma = 0.005 if form == 'antisym' else 0.30   # sym needs gamma > 2*c*beta
            rec = run(beta, form, gamma)
            if not np.isfinite(rec).all() or np.abs(rec).max() > 1.0:
                print(f'{form:>9} {beta:>6.2f} {gamma:>6.3f}   *** UNSTABLE / left linear regime ***')
                continue
            d = dispersion_asym(rec, m)
            th = -2 * C * beta * np.sin(k) if form == 'antisym' else 0.0
            results[(form, beta)] = d
            print(f'{form:>9} {beta:>6.2f} {gamma:>6.3f} {d:>13.4f} {th:>15.4f}')

    # loop-phase estimate at largest beta (antisym): d_theta = d_omega * T_loop
    if ('antisym', 0.10) in results:
        Omega = np.sqrt(np.sqrt(5) + 2 * C * (1 - np.cos(k)))
        v_g = C * np.sin(k) / Omega
        d_theta = results[('antisym', 0.10)] * (N / v_g)
        print(f'\nLoop holonomy estimate at beta=0.10 (antisym): '
              f'd_theta = {d_theta:.1f} rad over one N-site loop')
