#!/usr/bin/env python3
"""
s4_session2.py — S4 session 2: the nonlinear campaign on the unified model.

Derived cubic vertex (from void anharmonicity, no pasted nonlinearity):
  V3 = (4g/R^5) * (rho*u^2 - rho^3)   [rho radial, u transverse]
Three-wave processes are inter-branch: waves (theta) couple to mass
oscillations (r) — the sectors exchange energy through geometry itself.

PREDICTIONS STATED BEFORE RUNNING (g=0.1, delta=1, c=1, R=1.07949, N=48):
 P-A  CHANNEL CLOSURE / WINDOW DISSOLUTION. Both branches gapped
      (A_theta = 2d-2g/R^4 = 1.8527, A_r = 2d+6g/R^4 = 2.4419).
      Minimum three-wave mismatch across the zone: sum channel
      2theta(k)->r(2k) about 1.16 (at k->0); decay channel
      theta(k)->theta(q)+r(k-q) about 0.51. Both >> max gyro shift
      (0.2*beta*sinsk <= 0.02 at beta=0.10) => NO parametric decay window:
      carrier retention > 0.95 for k0 = +/- pi/2, a = 0.25, T = 150, for
      ALL beta in {0.05, 0.0629, 0.075, 0.10} — the standalone sector's
      Result-3 window is DISSOLVED by the alignment gap. Targets protect.
 P-B  SPECTROSCOPY: omega_r(pi/4) = 1.7400 to <0.5%.
 P-C  HORIZON DEFECT: site 24's target rotated to angle 2.0 rad (beyond
      pi/2) => channel switch to reaction. Frozen-neighbor Newton:
      x* = 0.6912, m_defect ~ 0.478 (neighbors hold it above the isolated
      metastable 0.316); back-reaction tolerance few %. Ring wave with the
      defect present: partial scattering (retention below clean ring but
      > 0.6); min mass stays above the floor 0.3162 globally.
 P-D  WINDING PROBE: counter-propagating 1:1 pair (+/-pi/2) shows mutual
      modulation >= incommensurate pair (pi/2 with 2pi*7/48); both
      retentions > 0.9 (only four-wave coupling remains).
 P-E  OVERDETERMINATION AUDIT: 4 constants (g, delta, c, beta) against
      >= 10 independent quantitative checks in S4 alone.
"""

import numpy as np

G, DELTA, C = 0.1, 1.0, 1.0
N, DT = 48, 0.01
R = 1.07949
AT = 2*DELTA - 2*G/R**4
AR = 2*DELTA + 6*G/R**4
wth = lambda k: np.sqrt(AT + 4*C*np.sin(k/2)**2)
wr  = lambda k: np.sqrt(AR + 4*C*np.sin(k/2)**2)

def make_targets(defect=False):
    Wt = np.tile([1.0, 0.0], (N, 1))
    if defect:
        Wt[24] = [np.cos(2.0), np.sin(2.0)]
    return Wt

def forces(Z, V, Wt, beta):
    r2 = np.sum(Z*Z, axis=1, keepdims=True)
    F = 2*G/(r2*r2) * Z
    dot = np.sum(Z*Wt, axis=1, keepdims=True)
    F += np.where(dot > 0, -2*DELTA*(Z - Wt), -2*DELTA*Z)
    F += C * (np.roll(Z, -1, 0) + np.roll(Z, 1, 0) - 2*Z)
    F += C*beta * (np.roll(V, -1, 0) - np.roll(V, 1, 0))
    return F

def energy(Z, V, Wt):
    r2 = np.sum(Z*Z, axis=1)
    dot = np.sum(Z*Wt, axis=1)
    Eal = np.where(dot > 0, np.sum((Z-Wt)**2, axis=1),
                   r2 + np.sum(Wt*Wt, axis=1))
    d = np.roll(Z, -1, 0) - Z
    return 0.5*np.sum(V*V) + np.sum(G/r2) + DELTA*np.sum(Eal) + 0.5*C*np.sum(d*d)

def run(Z, V, Wt, beta, T):
    E0 = energy(Z, V, Wt); dE = 0.0
    for _ in range(int(T/DT)):
        a1 = forces(Z, V, Wt, beta);                    v1 = V
        a2 = forces(Z+DT/2*v1, V+DT/2*a1, Wt, beta);    v2 = V+DT/2*a1
        a3 = forces(Z+DT/2*v2, V+DT/2*a2, Wt, beta);    v3 = V+DT/2*a2
        a4 = forces(Z+DT*v3,  V+DT*a3,  Wt, beta);      v4 = V+DT*a3
        Z = Z + DT/6*(v1+2*v2+2*v3+v4)
        V = V + DT/6*(a1+2*a2+2*a3+a4)
        dE = max(dE, abs(energy(Z, V, Wt)-E0))
    return Z, V, dE/abs(E0)

nidx = np.arange(N)
def mode_power(Z, k):
    return abs((Z[:, 1]) @ np.exp(-1j*k*nidx))**2

def wave_state(ks, amps, beta):
    Z = np.tile([R, 0.0], (N, 1)); V = np.zeros((N, 2))
    for k, a in zip(ks, amps):
        Z[:, 1] += a*np.cos(k*nidx)
        V[:, 1] += a*wth(abs(k))*np.sin(k*nidx)
    return Z, V

if __name__ == '__main__':
    # ---- P-A: channel kinematics + window test ----
    kk = np.linspace(1e-3, np.pi, 400)
    mis_sum = np.min(np.abs(2*wth(kk) - wr(2*kk % (2*np.pi))))
    mis_dec = min(np.min(np.abs(wth(k0) - wth(kk) - wr(np.abs(k0-kk))))
                  for k0 in np.linspace(0.1, np.pi, 40))
    print(f'P-A kinematics: min mismatch  sum-channel = {mis_sum:.4f}   '
          f'decay-channel = {mis_dec:.4f}')
    k0 = 2*np.pi*12/N
    print(f'{"beta":>8} {"dir":>4} {"retention":>10} {"dE/E":>9}')
    Wt = make_targets()
    for beta in (0.05, 0.0629, 0.075, 0.10):
        for s in (+1, -1):
            Z, V = wave_state([s*k0], [0.25], beta)
            P0 = mode_power(Z, s*k0)
            Z, V, dE = run(Z, V, Wt, beta, 150.0)
            print(f'{beta:>8.4f} {"+" if s>0 else "-":>4} '
                  f'{mode_power(Z, s*k0)/P0:>10.4f} {dE:>9.1e}')

    # ---- P-B: radial spectroscopy at pi/4 ----
    k4 = 2*np.pi*6/N
    Z = np.tile([R, 0.0], (N, 1)); V = np.zeros((N, 2))
    Z[:, 0] += 0.01*np.cos(k4*nidx)
    hist = []
    E0 = energy(Z, V, Wt)
    for i in range(int(60.0/DT)):
        a1 = forces(Z, V, Wt, 0.0);                  v1 = V
        a2 = forces(Z+DT/2*v1, V+DT/2*a1, Wt, 0.0);  v2 = V+DT/2*a1
        a3 = forces(Z+DT/2*v2, V+DT/2*a2, Wt, 0.0);  v3 = V+DT/2*a2
        a4 = forces(Z+DT*v3,  V+DT*a3,  Wt, 0.0);    v4 = V+DT*a3
        Z = Z + DT/6*(v1+2*v2+2*v3+v4); V = V + DT/6*(a1+2*a2+2*a3+a4)
        hist.append((Z[:, 0]-R) @ np.cos(k4*nidx))
    h = np.array(hist); sp = np.fft.rfft(h*np.hanning(len(h)))
    om = np.fft.rfftfreq(len(h), DT)*2*np.pi
    om_meas = om[np.argmax(abs(sp))]
    print(f'\nP-B omega_r(pi/4): measured = {om_meas:.4f}   '
          f'predicted = {wr(k4):.4f}')

    # ---- P-C: horizon defect ----
    Wd = make_targets(defect=True)
    x = 0.7
    for _ in range(60):  # frozen-neighbor Newton on the x-axis
        f = 2*G/x**3 - 2*DELTA*x + 2*C*(R - x)
        df = -6*G/x**4 - 2*DELTA - 2*C
        x -= f/df
    print(f'\nP-C frozen-neighbor Newton: x* = {x:.4f}  m* = {x*x:.4f}')
    Z = np.tile([R, 0.0], (N, 1)); V = np.zeros((N, 2))
    Z, V, _ = run(Z, V, Wd, 0.0, 80.0)
    m24 = Z[24] @ Z[24]
    print(f'    defect settles: m_24 = {m24:.4f}   (isolated metastable 0.3162)')
    Zw, Vw = wave_state([k0], [0.25], 0.05)
    Zw[24] = Z[24]; Vw[24] = 0.0
    P0 = mode_power(Zw, k0)
    Zw, Vw, dE = run(Zw, Vw, Wd, 0.05, 80.0)
    mmin = np.min(np.sum(Zw*Zw, axis=1))
    print(f'    wave + defect: retention = {mode_power(Zw, k0)/P0:.3f}   '
          f'min mass = {mmin:.4f} (floor 0.3162)   dE/E = {dE:.1e}')

    # ---- P-D: winding probe ----
    for label, k2 in (('1:1 counter', -k0), ('incommens.', 2*np.pi*7/N)):
        Z, V = wave_state([k0, k2], [0.2, 0.2], 0.05)
        p1, p2 = mode_power(Z, k0), mode_power(Z, k2)
        Z, V, _ = run(Z, V, Wt, 0.05, 150.0)
        print(f'\nP-D {label:>12}: retention k1 = {mode_power(Z,k0)/p1:.3f}   '
              f'k2 = {mode_power(Z,k2)/p2:.3f}')

    print('\nP-E audit: constants {g, delta, c, beta} = 4. Independent checks')
    print('in S4: r_eq, two branch centers, two branch asymmetries, floor-')
    print('under-wave, two mismatch minima, 8 retention nulls, omega_r(pi/4),')
    print('defect mass, defect floor  =>  17 checks / 4 constants.')
