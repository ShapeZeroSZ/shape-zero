#!/usr/bin/env python3
"""
s1_gradient_flow.py — S1: dynamical endurance test of the corrected Item 1
(Shape Zero action log, Phase S)

HK gradient flow of a weighted Dirac reduces, via the cone development, to a
plane ODE: metric ds^2 = dr^2 + r^2 dx^2 is flat polar, so z = sqrt(m)*e^{ix}
and z' = -grad E(z) with

  E(z) = g/|z|^2  +  delta * ( |z - w|^2        if z.w > 0   [inside horizon]
                               |z|^2 + s        otherwise )  [beyond horizon]

with w = sqrt(s) * (cos y, sin y). The dot-product sign IS the pi/2 horizon.

Predictions stated before running (g = 0.1, delta = 1, s = 1):
 P1  in-phase asymptotic convergence rate = lambda_pred = 2*delta - 2g/m_eq^2
     = 1.8535 (tangential Hessian eigenvalue; the derived bound is SHARP).
 P2  beyond-horizon flow relaxes to the marginal mass m* = sqrt(g/delta)
     = 0.3162 and stays there forever: E trapped at 2*sqrt(g*delta) + delta*s
     = 1.6325 vs global minimum 0.0922. Permanent metastability.
 P3  the basin boundary is the horizon itself: initial angle pi/2 from the
     target, sharp under bisection.
 P4  energy dissipation identity dE/dt = -|grad E|^2 (integrator check).
"""

import numpy as np

G, DELTA, S = 0.1, 1.0, 1.0
Y = 0.8
W = np.sqrt(S) * np.array([np.cos(Y), np.sin(Y)])

def E(z):
    r2 = z @ z
    if z @ W > 0:
        d = z - W
        return G / r2 + DELTA * (d @ d)
    return G / r2 + DELTA * (r2 + S)

def gradE(z):
    r2 = z @ z
    gv = -2 * G / r2**2 * z
    if z @ W > 0:
        return gv + 2 * DELTA * (z - W)
    return gv + 2 * DELTA * z

def flow(z0, T, dt=1e-3, record=False):
    z = np.array(z0, float)
    traj = []
    for s_ in range(int(T / dt)):
        if record and s_ % 50 == 0:
            traj.append((s_ * dt, z.copy(), E(z)))
        k1 = -gradE(z)
        k2 = -gradE(z + 0.5*dt*k1)
        k3 = -gradE(z + 0.5*dt*k2)
        k4 = -gradE(z + dt*k3)
        z = z + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    return (z, traj) if record else z

if __name__ == '__main__':
    # global minimizer on the target ray: rho^3(rho-1) = g
    rho = 1.0
    for _ in range(80):
        rho = 1 + G / rho**3
    zmin = rho * W / np.linalg.norm(W)
    m_eq = rho**2
    lam_pred = 2*DELTA - 2*G/m_eq**2
    print(f'minimizer: rho = {rho:.5f}, m_eq = {m_eq:.5f}, E_min = {E(zmin):.5f}')
    print(f'P1 predicted sharp rate lambda = {lam_pred:.4f}\n')

    # P1: in-phase convergence rate
    z0 = 1.3 * np.array([np.cos(Y - 1.2), np.sin(Y - 1.2)])   # inside horizon
    _, traj = flow(z0, 12.0, record=True)
    ts = np.array([t for t, _, _ in traj])
    ds = np.array([np.linalg.norm(z - zmin) for _, z, _ in traj])
    sel = (ds > 1e-9) & (ts > 4.0)                            # asymptotic window
    rate = -np.polyfit(ts[sel], np.log(ds[sel]), 1)[0]
    print(f'P1 measured asymptotic rate: {rate:.4f}   (predicted {lam_pred:.4f})')

    # P4: dissipation identity at a probe point
    zp = np.array([0.9, 0.4])
    h = 1e-6
    dEdt = (E(zp - h*gradE(zp)) - E(zp)) / h
    print(f'P4 dissipation: dE/dt = {dEdt:.6f}   -|gradE|^2 = {-(gradE(zp)@gradE(zp)):.6f}')

    # P2: beyond-horizon metastability
    z0 = 1.2 * np.array([np.cos(Y + 2.2), np.sin(Y + 2.2)])   # 2.2 rad > pi/2 away
    zT, traj = flow(z0, 40.0, record=True)
    mT = zT @ zT
    print(f'\nP2 beyond horizon: final m = {mT:.5f}  (m* = {np.sqrt(G/DELTA):.5f});'
          f'  final E = {E(zT):.5f}  (predicted plateau {2*np.sqrt(G*DELTA)+DELTA*S:.5f};'
          f'  global min {E(zmin):.5f})')
    ang_drift = abs(np.arctan2(zT[1], zT[0]) - np.arctan2(z0[1], z0[0]))
    print(f'   angular drift over T = 40: {ang_drift:.2e}  (no transport channel)')

    # P3: basin boundary via bisection on initial angle offset
    def converges(dtheta, T=60.0):
        z0 = 1.2 * np.array([np.cos(Y + dtheta), np.sin(Y + dtheta)])
        zT = flow(z0, T, dt=2e-3)
        return np.linalg.norm(zT - zmin) < 1e-3
    lo, hi = 1.0, 2.0                                          # converge / trapped
    for _ in range(40):
        mid = 0.5*(lo+hi)
        if converges(mid): lo = mid
        else: hi = mid
    print(f'\nP3 basin boundary at initial offset {0.5*(lo+hi):.8f} rad'
          f'   (pi/2 = {np.pi/2:.8f})')
