#!/usr/bin/env python3
"""
hk_alignment_convexity.py — A6: convexity of the alignment term HK(nu, .)^2
(Shape Zero action log item A6)

Facts used (both elementary/citable, Liero-Mielke-Savare):
 1. The HK geodesic between Diracs at base distance < pi/2 is a single moving
    weighted Dirac: the projection of the straight cone-development segment
    z(t) = (1-t) z0 + t z1, with m_t = |z(t)|^2.
 2. HK^2 between Diracs m*delta_x, s*delta_y is the MINIMUM of two mechanism
    costs: transport chord  m + s - 2 sqrt(ms) cos d   (optimal for d < pi/2)
    and independent kill-create  m + s  (optimal beyond). Equivalently
    HK^2 = m + s - 2 sqrt(ms) cos(min(d, pi/2)).

Consequence tested here: a minimum of transversally-crossing smooth costs has
a CONCAVE KINK at the crossing, so F(t) = HK(nu, mu_t)^2 along a geodesic
whose base distance to nu crosses pi/2 is NOT lambda-convex for any lambda.
Predicted derivative jump at the crossing: dF' = -2 sqrt(m s) |d'| < 0.

Rescue regime also verified: while d stays below pi/2 (single-mechanism,
transport-connected phase), F'' = 2 L^2 exactly (flat cone development);
and for nu = 0 (the void), F = m_t with F'' = 2 L^2 always (the A3 lemma).
"""

import numpy as np

# geodesic: unit-mass Diracs, x: 0 -> 1  (base distance 1 < pi/2)
z0 = np.array([1.0, 0.0])
z1 = np.array([np.cos(1.0), np.sin(1.0)])
L2 = float(np.sum((z1 - z0)**2))                     # squared HK speed

def state(t):
    z = (1 - t)*z0 + t*z1
    r2 = float(z @ z)
    x = float(np.arctan2(z[1], z[0]))                # base position in [0, 1]
    return r2, x

def F(t, y, s):
    m, x = state(t)
    d = abs(x - y)
    return m + s - 2*np.sqrt(m*s)*np.cos(min(d, np.pi/2))

def second_diff(y, s, ts, h=1e-4):
    return [(F(t-h, y, s) - 2*F(t, y, s) + F(t+h, y, s))/h**2 for t in ts]

if __name__ == '__main__':
    s = 1.0
    ts_probe = np.linspace(0.1, 0.9, 9)

    # (c) nu = 0: the A3 lemma, F = m_t, F'' = 2 L^2
    d2 = [(state(t-1e-4)[0] - 2*state(t)[0] + state(t+1e-4)[0])/1e-8
          for t in ts_probe]
    print(f'nu = 0 (void):        F\u2033 = {np.mean(d2):.6f}   2L\u00b2 = {2*L2:.6f}   (A3 lemma)')

    # (b) inside the horizon: y = 0.5, d < pi/2 throughout -> exact 2L^2
    d2 = second_diff(0.5, s, ts_probe)
    print(f'inside horizon (y=0.5): F\u2033 = {np.mean(d2):.6f} \u00b1 {np.std(d2):.1e}'
          f'   2L\u00b2 = {2*L2:.6f}   (exact convexity)')

    # (a) crossing case: place y so d crosses pi/2 inside (0,1)
    tstar = 0.35
    m_st, x_st = state(tstar)
    y = x_st + np.pi/2
    h = 1e-5
    Fp_before = (F(tstar - h, y, s) - F(tstar - 2*h, y, s)) / h
    Fp_after  = (F(tstar + 2*h, y, s) - F(tstar + h, y, s)) / h
    jump = Fp_after - Fp_before
    xdot = (state(tstar + h)[1] - state(tstar - h)[1]) / (2*h)
    ddot = -xdot                                     # d = y - x_t here
    pred = 2*np.sqrt(m_st*s)*ddot                    # entering interaction
    print(f'\ncrossing case (y = x(t*) + pi/2, t* = {tstar}):')
    print(f'  measured F\u2032 jump at horizon: {jump:+.6f}')
    print(f'  predicted 2\u221a(ms)\u00b7\u1e0b        : {pred:+.6f}')
    print(f'  concave kink -> distributional F\u2033 contains {jump:+.4f}\u00b7\u03b4(t\u2212t*):')
    print('  F is not \u03bb-convex along this HK geodesic for ANY \u03bb.')
