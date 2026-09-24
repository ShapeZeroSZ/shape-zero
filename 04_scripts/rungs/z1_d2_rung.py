#!/usr/bin/env python3
"""
z1_d2_rung.py — Z1 ladder, rung D2: is the cone forced? Is the void chosen?
(Shape Zero action log item Z1; predictions stated before running)

DERIVATION (tags in log): at D2, isotropy is minimal [SELECTED-minimal];
isotropy conserves L [FORCED — Noether]; the reduced radial potential is
V_eff(r) = V(r) + L^2/(2 r^2): THE CENTRIFUGAL TERM IS THE VOID TERM,
with g = L^2/2 fixed by the state's own motion [FORCED for rotating
states]. In cone coordinates L = m*xdot: g = p^2/2 — the void repels
whatever carries momentum. Flat kinetic energy on R^2 IS the cone
[FORCED given minimal metric].

PREDICTIONS (delta = 1, minimal isotropic confinement V = delta*|z|^2/2):
 P1 circular radius r* = (L^2/delta)^(1/4) = sqrt(L): for L = 0.3, 0.6,
    1.0 predict r* = 0.54772, 0.77460, 1.00000 (to <0.1%).
 P2 UNIVERSAL 2:1 — radial oscillation / angular revolution frequency
    ratio = 2 exactly, independent of L and delta (omega_r = 2*sqrt(delta),
    Omega = sqrt(delta)). Rationality reappears at D2 as structure.
 P3 phase protection: launched from r0 = 1.4 with v_r = -0.5 inward and
    tangential v_t = L/r0, the minimum radius over the run equals the
    barrier root r_min^2 = E - sqrt(E^2 - L^2)  (delta = 1), to 4+ digits;
    for L = 0 the trajectory reaches the origin (r_min -> integrator
    floor). Only the phaseless can die into the void.
"""

import numpy as np

DELTA, DT = 1.0, 2e-4

def accel(z):
    return -DELTA * z          # isotropic minimal confinement

def rk4(z, v, steps, record_r=False):
    rs = []
    for _ in range(steps):
        a1 = accel(z);              v1 = v
        a2 = accel(z + DT/2*v1);    v2 = v + DT/2*a1
        a3 = accel(z + DT/2*v2);    v3 = v + DT/2*a2
        a4 = accel(z + DT*v3);      v4 = v + DT/2*a3*2  # v + DT*a3
        z = z + DT/6*(v1 + 2*v2 + 2*v3 + v4)
        v = v + DT/6*(a1 + 2*a2 + 2*a3 + a4)
        if record_r:
            rs.append(np.hypot(z[0], z[1]))
    return z, v, np.array(rs)

if __name__ == '__main__':
    # ---- P1 + P2 ----
    print(f'{"L":>5} {"r* meas":>9} {"r* pred":>9} {"w_r/Omega":>10}')
    for L in (0.3, 0.6, 1.0):
        rstar = np.sqrt(L)
        z = np.array([rstar + 0.02, 0.0])
        v = np.array([0.0, L / z[0]])          # slight radial offset
        T = 40.0
        _, _, rs = rk4(z, v, int(T/DT), record_r=True)
        r_mean = rs.mean()
        # radial frequency from FFT of r(t)
        sp = np.abs(np.fft.rfft((rs - rs.mean()) * np.hanning(len(rs))))
        fr = np.fft.rfftfreq(len(rs), DT) * 2*np.pi
        w_r = fr[np.argmax(sp)]
        # angular frequency: Omega = L / <r^2>
        Omega = L / (rs**2).mean()
        print(f'{L:>5.1f} {r_mean:>9.5f} {rstar:>9.5f} {w_r/Omega:>10.5f}')

    # ---- P3 ----
    print(f'\n{"L":>5} {"r_min meas":>11} {"r_min pred":>11}')
    for L in (0.0, 0.2, 0.5, 0.9):
        r0, vr = 1.4, -0.5
        z = np.array([r0, 0.0]); v = np.array([vr, L/r0])
        E = 0.5*(v @ v) + 0.5*DELTA*(z @ z)
        pred = 0.0 if L == 0 else np.sqrt(E - np.sqrt(max(E*E - L*L, 0)))
        _, _, rs = rk4(z, v, int(30.0/DT), record_r=True)
        print(f'{L:>5.1f} {rs.min():>11.6f} {pred:>11.6f}')
    print('\nP3 reading: L != 0 -> barrier (phase protects existence);')
    print('L == 0 -> the origin is reachable (the phaseless die into the void).')
