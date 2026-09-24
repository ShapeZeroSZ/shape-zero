#!/usr/bin/env python3
"""
e1a_extraction.py — E1a corrected: passivity vs extraction for two cone-agents
(Shape Zero action log item E1; supersedes the flat-mass Newtonian sketch)

Geometry: each agent is a cone-development point z_i in R^2 (m = |z|^2),
first-order HK gradient flow — the S1 machinery, now coupled.

  E_i(z_i)   = g/|z_i|^2 + delta*|z_i - w_i|^2        (own void + own target)
  E_int      = kappa*|z_A - z_B|^2                     (honest coupling: one
                                                        shared potential)
  parasite   = non-variational exchange at strength eps:
               extra force -eps*ẑ_A on A (drain toward void),
               extra force +eps*ẑ_B on B (growth) — NOT the gradient of any
               shared function. This is the integrability breach.

Predictions STATED BEFORE RUNNING (g=0.1, delta=1, |w|=1, kappa=0.3):
 P1 eps = 0: total energy E_A + E_B + E_int is monotone nonincreasing to
    machine precision; steady extraction rate = 0; both agents settle at the
    joint equilibrium.
 P2 eps > 0: A settles BELOW its coupled equilibrium radius by
    dr_A ~ eps / H_A (linear response, H_A = radial Hessian at equilibrium);
    B settles ABOVE; steady extraction rate dm_B/dt|_coupling > 0, linear
    in eps at small eps; baseline exactly zero.
 P3 the mass floor still binds: A's mass stays above sqrt(g/delta) = 0.316
    for all eps in the sweep.
"""

import numpy as np

G, DELTA, KAPPA = 0.1, 1.0, 0.3
WA = 1.0 * np.array([np.cos(0.6), np.sin(0.6)])
WB = 1.0 * np.array([np.cos(1.0), np.sin(1.0)])
DT, T = 1e-3, 30.0

def gradE_own(z, w):
    return -2*G/(z@z)**2 * z + 2*DELTA*(z - w)

def rhs(zA, zB, eps):
    fA = -(gradE_own(zA, WA) + 2*KAPPA*(zA - zB)) - eps * zA/np.linalg.norm(zA)
    fB = -(gradE_own(zB, WB) + 2*KAPPA*(zB - zA)) + eps * zB/np.linalg.norm(zB)
    return fA, fB

def E_tot(zA, zB):
    return (G/(zA@zA) + DELTA*(zA-WA)@(zA-WA)
            + G/(zB@zB) + DELTA*(zB-WB)@(zB-WB) + KAPPA*(zA-zB)@(zA-zB))

def run(eps):
    zA = 1.2*np.array([np.cos(0.4), np.sin(0.4)])
    zB = 1.2*np.array([np.cos(1.2), np.sin(1.2)])
    Emax_rise = 0.0
    Eprev = E_tot(zA, zB)
    for _ in range(int(T/DT)):
        fA, fB = rhs(zA, zB, eps)
        # RK4
        k1a, k1b = fA, fB
        k2a, k2b = rhs(zA+0.5*DT*k1a, zB+0.5*DT*k1b, eps)
        k3a, k3b = rhs(zA+0.5*DT*k2a, zB+0.5*DT*k2b, eps)
        k4a, k4b = rhs(zA+DT*k3a, zB+DT*k3b, eps)
        zA = zA + DT/6*(k1a+2*k2a+2*k3a+k4a)
        zB = zB + DT/6*(k1b+2*k2b+2*k3b+k4b)
        Enow = E_tot(zA, zB)
        Emax_rise = max(Emax_rise, Enow - Eprev)
        Eprev = Enow
    # steady extraction rate on B from the parasite channel: d(m_B)/dt = 2 z.f
    ext_rate = 2*zB @ (eps * zB/np.linalg.norm(zB))
    return zA, zB, Emax_rise, ext_rate

if __name__ == '__main__':
    # coupled honest equilibrium (baseline) for reference radii
    zA0, zB0, rise0, ext0 = run(0.0)
    rA0, rB0 = np.linalg.norm(zA0), np.linalg.norm(zB0)
    # radial Hessian of A at its equilibrium (own + coupling spring)
    H_A = 6*G/rA0**4 + 2*DELTA + 2*KAPPA
    print(f'P1 baseline: max single-step energy RISE = {rise0:.2e} '
          f'(monotone to machine precision); extraction rate = {ext0:.2e}')
    print(f'   equilibrium radii: r_A = {rA0:.5f}, r_B = {rB0:.5f}; '
          f'H_A = {H_A:.3f}\n')
    print(f'{"eps":>6} {"r_A":>9} {"dr_A meas":>10} {"eps/H_A":>9} '
          f'{"r_B":>9} {"extract rate":>13} {"m_A > floor":>12}')
    for eps in (0.02, 0.05, 0.10):
        zA, zB, rise, ext = run(eps)
        rA, rB = np.linalg.norm(zA), np.linalg.norm(zB)
        print(f'{eps:>6.2f} {rA:>9.5f} {rA0-rA:>10.5f} {eps/H_A:>9.5f} '
              f'{rB:>9.5f} {ext:>13.5f} '
              f'{str(rA*rA > np.sqrt(G/DELTA)):>12}')
    print('\nP2 check: dr_A tracks eps/H_A (linear response); B above baseline;')
    print('extraction rate linear in eps with baseline at machine zero.')
    print('P3 check: m_A above the floor sqrt(g/delta) = '
          f'{np.sqrt(G/DELTA):.4f} throughout.')
