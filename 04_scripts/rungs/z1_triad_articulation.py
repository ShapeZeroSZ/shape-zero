#!/usr/bin/env python3
"""
z1_triad_articulation.py — does persistence + conservativity + minimality
force the TRIAD as the first stable articulation of energetic distinction?
(Z1 program; predictions stated before running.)

CLAIM (Birkhoff-framed): (i) every bilinear coupling is removable by a
linear canonical transformation — binary distinctions are coordinate
choices, generating NO new spectral content and absorbing their own
coupling constant; pushed past sqrt(K1 K2) they destabilize instead:
binary = dissolve or explode. (ii) Cubic content is the first that no
linear transformation removes, and nonlinear normal forms remove it only
OFF resonance: the invariant kernel of interaction is the RESONANT TRIAD
(minimal instance 2w1 = w2 — two quanta alike, one distinct), announced
by combination frequencies. (iii) In the unified geometry the triadic
vertex is derived (S4: 4g/R^5; D2: g = L^2/2) — no free parameter enters.

PREDICTIONS (K1 = 2, K2 = 5):
 A  bilinear eps = 0.3: normal-mode frequencies 2.24270 / 1.40368; each
    normal-mode energy separately constant (~RK4 floor); quadrature
    projections of x2 at 2w1 and w1+w2 below 1e-8 — ZERO articulation.
 B  bilinear eps = 3.5 > sqrt(10): runaway at rate sqrt((sqrt(58)-7)/2)
    = 0.55488 from the indefinite quadratic form.
 C  triadic vertex V = gamma x1^2 x2, gamma = 0.5, A = 0.2: x2 acquires
    lines at 2w1 (amp 0.00333) and DC (amp 0.00200), within ~15%;
    energy conserved to integrator floor. The triad articulates,
    conservatively.
"""

import numpy as np

K1, K2, DT = 2.0, 5.0, 1e-3

def run(eps, gamma, x0, T, record=False):
    x = np.array(x0, float); v = np.zeros(2)
    def acc(x):
        return np.array([-K1*x[0] - eps*x[1] - 2*gamma*x[0]*x[1],
                         -K2*x[1] - eps*x[0] - gamma*x[0]*x[0]])
    E = lambda x, v: 0.5*(v@v) + 0.5*K1*x[0]**2 + 0.5*K2*x[1]**2 \
                     + eps*x[0]*x[1] + gamma*x[0]**2*x[1]
    E0 = E(x, v); dE = 0.0; tr = []
    for i in range(int(T/DT)):
        a1 = acc(x); v1 = v
        a2 = acc(x+DT/2*v1); v2 = v+DT/2*a1
        a3 = acc(x+DT/2*v2); v3 = v+DT/2*a2
        a4 = acc(x+DT*v3);  v4 = v+DT*a3
        x = x + DT/6*(v1+2*v2+2*v3+v4); v = v + DT/6*(a1+2*a2+2*a3+a4)
        dE = max(dE, abs(E(x, v) - E0))
        if record: tr.append(np.concatenate([x, v]))
    return (np.array(tr) if record else (x, v)), dE/max(abs(E0), 1e-12)

if __name__ == '__main__':
    # ---- A: bilinear dissolution ----
    P = np.array([[K1, 0.3], [0.3, K2]])
    lam, U = np.linalg.eigh(P)
    w_pred = np.sqrt(lam)
    print(f'A  normal-mode frequencies predicted: {w_pred[1]:.5f}, {w_pred[0]:.5f}')
    tr, dE = run(0.3, 0.0, [0.2, 0.15], 200.0, record=True)
    q = tr[:, :2] @ U; p = tr[:, 2:] @ U
    Em = 0.5*p**2 + 0.5*lam*q**2
    drift = np.max(np.abs(Em - Em[0]), axis=0) / Em[0]
    print(f'   per-normal-mode energy drift: {drift[0]:.1e}, {drift[1]:.1e} '
          f'(pred ~RK4 floor: modes are FREE — the coupling dissolved)')
    ts = np.arange(len(tr))*DT
    # measure w1 from mode-1 crossings (continuous readout)
    z = q[:, 1]; idx = np.where(np.diff(np.sign(z)) != 0)[0]
    tc = ts[idx] - z[idx]*(ts[idx+1]-ts[idx])/(z[idx+1]-z[idx])
    w1m = np.pi*(len(tc)-1)/(tc[-1]-tc[0])
    proj = lambda s, w: 2*np.sqrt(np.mean(s*np.cos(w*ts))**2
                                  + np.mean(s*np.sin(w*ts))**2)
    x2 = tr[:, 1] - tr[:, 1].mean()
    print(f'   combination projections in x2: at 2w = {proj(x2, 2*w1m):.1e}, '
          f'at w+ + w- = {proj(x2, w_pred.sum()):.1e}  (pred < 1e-8)')
    print(f'   measured mode frequency {w1m:.5f} vs {w_pred[1]:.5f}; dE/E = {dE:.1e}')

    # ---- B: bilinear destabilization ----
    lam2 = np.linalg.eigvalsh(np.array([[K1, 3.5], [3.5, K2]]))
    rate_pred = np.sqrt(-lam2[0])
    tr, _ = run(3.5, 0.0, [0.01, 0.01], 12.0, record=True)
    amp = np.linalg.norm(tr[:, :2], axis=1)
    ts = np.arange(len(tr))*DT
    sl = np.polyfit(ts[len(ts)//2:], np.log(amp[len(ts)//2:]), 1)[0]
    print(f'\nB  runaway rate measured {sl:.5f} vs predicted {rate_pred:.5f} '
          f'— binary past sqrt(K1K2): fails persistence outright')

    # ---- C: triadic articulation ----
    tr, dE = run(0.0, 0.5, [0.2, 0.0], 300.0, record=True)
    ts = np.arange(len(tr))*DT
    x1 = tr[:, 0]; idx = np.where(np.diff(np.sign(x1)) != 0)[0]
    tc = ts[idx] - x1[idx]*(ts[idx+1]-ts[idx])/(x1[idx+1]-x1[idx])
    w1m = np.pi*(len(tc)-1)/(tc[-1]-tc[0])
    x2 = tr[:, 1]
    a2w = proj(x2 - x2.mean(), 2*w1m)
    adc = abs(x2.mean())
    print(f'\nC  triad lines in x2: at 2w1 = {a2w:.5f} (pred 0.00333); '
          f'DC = {adc:.5f} (pred 0.00200); dE/E = {dE:.1e}')
    print('\nVerdict inputs: binary dissolves (A) or explodes (B), never')
    print('articulates; the triad articulates conservatively, and in the')
    print('unified geometry its coefficient is derived, not chosen.')
