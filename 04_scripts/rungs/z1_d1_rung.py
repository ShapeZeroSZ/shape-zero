#!/usr/bin/env python3
"""
z1_d1_rung.py — Z1 ladder, rung D1: is rationality born here?
(Shape Zero action log item Z1; predictions stated before running)

CLAIM CHAIN (tags in log): minimal nontrivial conservative D1 dynamics is
Newtonian [FORCED]; persistent bounded motion is periodic [FORCED]; periodic
spectrum is the integer lattice {n*omega} [FORCED — Fourier]; plurality is
the rung's one purchase [CHOSEN]; given two oscillators + generic coupling,
resonance makes the rationals dynamically operative [FORCED]; persistence
grades them [FORCED — KAM-type; measured in-house as S2b].

PREDICTIONS:
 P1 single anharmonic oscillator (K=2, alpha=0.7, A=0.2): spectral lines at
    integer multiples of the fundamental (2.000 to <0.5%); second-harmonic
    amplitude ratio |c2/A| = alpha*A/(6K) = 0.0117 within ~15%.
 P2 two oscillators, linear spring eps=0.05, osc-1 excited (A=0.25), osc-2
    at rest, frequency ratio r = omega2/omega1 in {2.0, 3.0, phi}:
    peak energy transferred to osc-2 strictly ordered E(2:1) > E(3:1) >
    E(phi), with E(2:1)/E(phi) >= 5. Mechanism: osc-1's n-th integer line
    drives osc-2 linearly-resonantly iff r = n; phi has no line to ride.
"""

import numpy as np

K1, ALPHA, DT = 2.0, 0.7, 1e-3
PHI = (1 + np.sqrt(5)) / 2

def run_single(A=0.2, T=200.0):
    x, v = A, 0.0
    xs = []
    f = lambda x: -K1*x - ALPHA*x*x
    for i in range(int(T/DT)):
        k1v = f(x);            k1x = v
        k2v = f(x+DT/2*k1x);   k2x = v+DT/2*k1v
        k3v = f(x+DT/2*k2x);   k3x = v+DT/2*k2v
        k4v = f(x+DT*k3x);     k4x = v+DT*k3v
        x += DT/6*(k1x+2*k2x+2*k3x+k4x); v += DT/6*(k1v+2*k2v+2*k3v+k4v)
        xs.append(x)
    return np.array(xs)

def run_pair(r, eps=0.05, A=0.25, T=400.0):
    K2 = (r*r)*K1
    x1, v1, x2, v2 = A, 0.0, 1e-3, 0.0
    E2max = 0.0
    def acc(x1, x2):
        a1 = -K1*x1 - ALPHA*x1*x1 + eps*(x2 - x1)
        a2 = -K2*x2 - ALPHA*x2*x2 + eps*(x1 - x2)
        return a1, a2
    for i in range(int(T/DT)):
        a1, a2 = acc(x1, x2)
        k1 = (v1, v2, a1, a2)
        a1b, a2b = acc(x1+DT/2*k1[0], x2+DT/2*k1[1])
        k2 = (v1+DT/2*k1[2], v2+DT/2*k1[3], a1b, a2b)
        a1c, a2c = acc(x1+DT/2*k2[0], x2+DT/2*k2[1])
        k3 = (v1+DT/2*k2[2], v2+DT/2*k2[3], a1c, a2c)
        a1d, a2d = acc(x1+DT*k3[0], x2+DT*k3[1])
        k4 = (v1+DT*k3[2], v2+DT*k3[3], a1d, a2d)
        x1 += DT/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        x2 += DT/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
        v1 += DT/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
        v2 += DT/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
        E2max = max(E2max, 0.5*v2*v2 + 0.5*K2*x2*x2)
    E1_0 = 0.5*K1*A*A
    return E2max / E1_0

if __name__ == '__main__':
    # ---- P1: integer lattice + second-harmonic amplitude ----
    xs = run_single()
    sp = np.abs(np.fft.rfft((xs - xs.mean()) * np.hanning(len(xs))))
    fr = np.fft.rfftfreq(len(xs), DT) * 2*np.pi
    i1 = np.argmax(sp)
    w1 = fr[i1]
    # second line: search near 2*w1
    band = (fr > 1.6*w1) & (fr < 2.4*w1)
    i2 = np.where(band)[0][np.argmax(sp[band])]
    w2, ratio_amp = fr[i2], sp[i2]/sp[i1]
    print(f'P1 fundamental = {w1:.5f}   second line = {w2:.5f}   '
          f'line ratio = {w2/w1:.4f}  (integer to {abs(w2/w1-2)*100:.2f}%)')
    print(f'   |c2/A| measured = {ratio_amp:.5f}   '
          f'predicted alpha*A/(6K) = {ALPHA*0.2/(6*K1):.5f}\n')

    # ---- P2: resonance ordering ----
    print(f'{"ratio":>8} {"peak E2/E1(0)":>14}')
    res = {}
    for label, r in (('2:1', 2.0), ('3:1', 3.0), ('phi', PHI)):
        res[label] = run_pair(r)
        print(f'{label:>8} {res[label]:>14.5f}')
    print(f'\nordering 2:1 > 3:1 > phi : '
          f'{res["2:1"] > res["3:1"] > res["phi"]}')
    print(f'separation E(2:1)/E(phi) = {res["2:1"]/res["phi"]:.1f}x  '
          f'(predicted >= 5x)')
