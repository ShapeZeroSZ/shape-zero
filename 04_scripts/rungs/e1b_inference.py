#!/usr/bin/env python3
"""
e1b_inference.py — E1b: objective inference + the integrability residual
(Shape Zero action log item E1b)

An observer sees ONLY an agent's trajectory z(t). Questions:
 1. Can the agent's objective (its target w, and its constants g, delta) be
    reconstructed from motion alone?  [honest agents leak their objectives]
 2. Can non-gradient ("dangerous") influence be detected and classified,
    even when it changes nothing about the outcome?

Agents (all start identically, same g=0.1, delta=1, same hidden target):
  HONEST    ż = -∇E                      E = g/|z|² + δ|z-w|²   (transport side)
  CURL(c)   ż = -∇E + c·J∇E             J = 90° rotation; E still Lyapunov
                                         (ż·∇E = -|∇E|²) → SAME equilibrium
  PARASITE(ε) ż = -∇E - ε·ẑ             E1a's drained agent

Detector: fit the declared gradient family (g, δ, w) to the observed
velocities by Gauss–Newton; the INTEGRABILITY RESIDUAL is the RMS velocity
mismatch of the best fit. Classification: direction of the unexplained
force vs. the fitted gradient (perpendicular ⇒ curl; radial ⇒ drain).

PREDICTIONS STATED BEFORE RUNNING:
 P1 honest: w recovered to <1e-3; residual at integrator/sampling floor.
 P2 curl:   endpoint identical to honest (<1e-3) yet residual ∝ c
            (ratios 3.0 and 6.0 for c = 0.1/0.3/0.6), residual ratio vs
            honest > 1e3; unexplained force ⊥ fitted gradient (|cos| < 0.1).
 P3 parasite: residual ∝ ε; unexplained force anti-radial (cos ≈ -1.000).
"""

import numpy as np

G, DELTA = 0.1, 1.0
W_TRUE = np.array([np.cos(0.6), np.sin(0.6)])
Z0 = 1.4 * np.array([np.cos(0.1), np.sin(0.1)])
DT, T, EVERY = 1e-3, 6.0, 10
J = np.array([[0.0, -1.0], [1.0, 0.0]])

def gradE(z, g, d, w):
    r2 = z @ z
    return -2*g/(r2*r2) * z + 2*d*(z - w)

def rhs(z, kind, s):
    gE = gradE(z, G, DELTA, W_TRUE)
    if kind == 'honest':   return -gE
    if kind == 'curl':     return -gE + s * (J @ gE)
    if kind == 'parasite': return -gE - s * z/np.linalg.norm(z)

def trajectory(kind, s=0.0):
    z = Z0.copy(); zs, vs = [], []
    for i in range(int(T/DT)):
        k1 = rhs(z, kind, s); k2 = rhs(z + 0.5*DT*k1, kind, s)
        k3 = rhs(z + 0.5*DT*k2, kind, s); k4 = rhs(z + DT*k3, kind, s)
        z = z + DT/6*(k1 + 2*k2 + 2*k3 + k4)
        if i % EVERY == 0:
            zs.append(z.copy()); vs.append(rhs(z, kind, s))
    return np.array(zs), np.array(vs)

def fit_family(zs, vs, iters=60):
    """Gauss-Newton fit of (g, d, wx, wy) minimizing |v + gradE(z;θ)|²."""
    th = np.array([0.2, 0.8, 0.5, 0.5])
    def resid(th):
        g, d, wx, wy = th
        return (vs + np.array([gradE(z, g, d, np.array([wx, wy])) for z in zs])).ravel()
    for _ in range(iters):
        r0 = resid(th)
        Jm = np.zeros((r0.size, 4))
        for j in range(4):
            e = np.zeros(4); e[j] = 1e-6
            Jm[:, j] = (resid(th + e) - r0) / 1e-6
        step, *_ = np.linalg.lstsq(Jm, -r0, rcond=None)
        th = th + step
        if np.linalg.norm(step) < 1e-12: break
    r = resid(th).reshape(-1, 2)
    return th, np.sqrt(np.mean(np.sum(r*r, axis=1)))

def classify(zs, vs, th):
    """Mean |cos| of unexplained force vs fitted gradient, and vs -ẑ."""
    g, d, wx, wy = th; w = np.array([wx, wy])
    cos_g, cos_r = [], []
    for z, v in zip(zs, vs):
        u = v + gradE(z, g, d, w)              # unexplained component
        nu = np.linalg.norm(u)
        if nu < 1e-10: continue
        gE = gradE(z, g, d, w)
        cos_g.append(abs(u @ gE) / (nu*np.linalg.norm(gE)))
        cos_r.append(u @ (-z/np.linalg.norm(z)) / nu)
    return (np.mean(cos_g) if cos_g else 0.0), (np.mean(cos_r) if cos_r else 0.0)

if __name__ == '__main__':
    zsH, vsH = trajectory('honest')
    thH, resH = fit_family(zsH, vsH)
    wH = thH[2:]
    print('P1 HONEST: recovered w = ({:.5f}, {:.5f})  true = ({:.5f}, {:.5f})'
          .format(*wH, *W_TRUE))
    print(f'   |w error| = {np.linalg.norm(wH-W_TRUE):.2e}   '
          f'g,d = {thH[0]:.4f},{thH[1]:.4f}   residual = {resH:.2e}\n')

    print(f'{"agent":>12} {"s":>5} {"residual":>10} {"ratio/honest":>12} '
          f'{"|cos∠grad|":>10} {"cos∠(-ẑ)":>9} {"endpoint Δ":>11}')
    endH = zsH[-1]
    prev = None
    for c in (0.1, 0.3, 0.6):
        zs, vs = trajectory('curl', c)
        th, res = fit_family(zs, vs)
        cg, cr = classify(zs, vs, th)
        print(f'{"curl":>12} {c:>5.2f} {res:>10.2e} {res/resH:>12.1e} '
              f'{cg:>10.3f} {cr:>9.3f} {np.linalg.norm(zs[-1]-endH):>11.2e}')
    for e in (0.02, 0.05):
        zs, vs = trajectory('parasite', e)
        th, res = fit_family(zs, vs)
        cg, cr = classify(zs, vs, th)
        print(f'{"parasite":>12} {e:>5.2f} {res:>10.2e} {res/resH:>12.1e} '
              f'{cg:>10.3f} {cr:>9.3f} {np.linalg.norm(zs[-1]-endH):>11.2e}')
    print('\nP2 check: curl endpoints match honest while residual is orders above;')
    print('          residual linear in c; unexplained force ⊥ gradient.')
    print('P3 check: parasite unexplained force anti-radial, cos ≈ -1.')
