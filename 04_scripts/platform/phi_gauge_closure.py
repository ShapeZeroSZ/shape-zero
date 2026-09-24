#!/usr/bin/env python3
"""
phi_gauge_closure.py — closes the U(1) story (Shape Zero v5.1 §5.2.7)

Open item from the amplitude sweep: the gauge asymmetry deviates from exact
pinning by  delta(beta) * A^2  with delta(0.05) = +0.00179 (fit below).
[RETRACTED: delta(0.05) = -0.0031 and the fit delta = -0.0598 * beta, measured
when phi_gauge_nonlinear.py seeded both directions at the beta = 0 frequency.
With each direction at its own root (PROVENANCE §6o) this script emits
delta = +0.0357 * beta, max residual 1.5%, parity sum ~1e-14 -- the sign flips,
and it now agrees with the second-order PT value +0.0349 * beta
(phi_gauge_delta.py). kappa = -delta / (2 c beta sin k) = -0.0179.] Hypothesis: the
correction exists only because beta splits the two movers' frequencies, so
their (direction-blind) softenings no longer cancel. That mechanism predicts:

  1. delta(beta) linear in beta   (split omega(+k)-omega(-k) = -2*c*beta*sin k)
  2. delta(-beta) = -delta(beta)  (parity: reversing beta swaps the movers)

If instead delta had a beta-independent part or even symmetry, the correction
would be a distinct nonlinear-gauge effect and the story would not be closed.
"""

import numpy as np
from phi_gauge_nonlinear import run_wave, mode_freq, C, K

AMPS = np.array([0.05, 0.1, 0.15, 0.2])   # stay below the beta~0.06-0.08 parametric decay window
BETAS = np.array([-0.05, 0.025, 0.05, 0.075, 0.10, 0.15])


def asym_correction_coef(beta):
    """Fit delta in  (d_omega + 2*c*beta*sin k) = delta * A^2  through origin."""
    dev = []
    for A in AMPS:
        wp, _ = mode_freq(run_wave(A, beta, +1), +1)
        wm, _ = mode_freq(run_wave(A, beta, -1), -1)
        dev.append((wp - wm) + 2 * C * beta * np.sin(K))
    dev = np.array(dev)
    A2 = AMPS ** 2
    return float(np.sum(A2 * dev) / np.sum(A2 * A2)), dev


if __name__ == '__main__':
    print(f'{"beta":>7} {"delta (fit)":>12} {"delta/beta":>11}')
    coefs = []
    for b in BETAS:
        d, _ = asym_correction_coef(b)
        coefs.append(d)
        print(f'{b:>7.3f} {d:>12.5f} {d/b:>11.4f}')
    coefs = np.array(coefs)
    slope = float(np.sum(BETAS * coefs) / np.sum(BETAS * BETAS))
    resid = coefs - slope * BETAS
    print(f'\nlinear fit through origin: delta = {slope:.4f} * beta')
    print(f'max residual: {np.abs(resid).max():.2e}  '
          f'({100*np.abs(resid).max()/np.abs(coefs).max():.1f}% of largest delta)')
    print('parity check: delta(-0.05) + delta(+0.05) =',
          f'{coefs[0] + coefs[2]:.2e}  (0 if odd in beta)')
