#!/usr/bin/env python3
"""g3_cone_lensing.py — cone-deficit lensing (log: G3). Constant deflection
pi(1/zeta - 1) after aperture correction 2*arcsin(b/r0); Newtonian varies.

(Cone deficit renamed beta -> zeta on 2026-09-25, to free beta for the lattice
gyroscopic coupling; the code variable keeps the name beta/BETA.)
"""
import numpy as np
DT = 2e-3
def deflect(beta, GM, b, r0=30.0):
    x,y = -r0,b; r = np.hypot(x,y); th = np.arctan2(y,x)
    rdot = np.cos(th); thdot = -np.sin(th)/r
    J = beta*beta*r*r*thdot; th_un = th
    acc = lambda r: J*J/(beta*beta*r**3) - GM/r**2
    while True:
        k1r=rdot; k1v=acc(r); k2r=rdot+DT/2*k1v; k2v=acc(r+DT/2*k1r)
        k3r=rdot+DT/2*k2v; k3v=acc(r+DT/2*k2r); k4r=rdot+DT*k3v; k4v=acc(r+DT*k3r)
        r += DT/6*(k1r+2*k2r+2*k3r+k4r); rdot += DT/6*(k1v+2*k2v+2*k3v+k4v)
        th_un += J/(beta*beta*r*r)*DT
        if r > r0 and rdot > 0: break
    return abs(th_un-th)-np.pi
beta = 0.95
print(f'pred: {np.pi*(1/beta-1):.6f}, constant in b (after +2*arcsin(b/30))')
for b in (0.3,0.6,1.2):
    d = deflect(beta,0.0,b)
    print(f'b={b}: raw {d:.6f}  corrected {d+2*np.arcsin(b/30):.6f}')
for b in (0.3,0.6,1.2):
    print(f'newtonian b={b}: {deflect(1.0,0.1,b):.4f}')
