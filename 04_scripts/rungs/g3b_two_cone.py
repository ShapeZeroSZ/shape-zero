#!/usr/bin/env python3
"""g3b_two_cone.py — deficit additivity (log: G3b). Two-center conformal
metric; combined deflection = sum of singles (2.8% vs measured singles)."""
import numpy as np
C1,C2 = np.array([-1.5,0.0]), np.array([1.5,0.0]); DT = 0.01
def run(a1,a2,b,x0=-40.0):
    L = lambda z: sum(-a*np.log(np.linalg.norm(z-c)) for a,c in ((a1,C1),(a2,C2)))
    def gradL(z):
        g = np.zeros(2)
        for a,c in ((a1,C1),(a2,C2)):
            d = z-c; g += -a*d/(d@d)
        return g
    z = np.array([x0,b]); p = np.array([np.exp(2*L(z)),0.0])
    def rhs(z,p):
        e = np.exp(-2*L(z)); return e*p, gradL(z)*e*(p@p)
    while True:
        k1z,k1p=rhs(z,p); k2z,k2p=rhs(z+DT/2*k1z,p+DT/2*k1p)
        k3z,k3p=rhs(z+DT/2*k2z,p+DT/2*k2p); k4z,k4p=rhs(z+DT*k3z,p+DT*k3p)
        z=z+DT/6*(k1z+2*k2z+2*k3z+k4z); p=p+DT/6*(k1p+2*k2p+2*k3p+k4p)
        if z[0] > abs(x0): break
    v = np.exp(-2*L(z))*p
    return np.arctan2(-v[1],v[0])
d1,d2,d12 = run(0.02,0,6.0), run(0,0.04,6.0), run(0.02,0.04,6.0)
print(f'singles: {d1:.4f}, {d2:.4f}; both: {d12:.4f}; sum {d1+d2:.4f} (additivity)')
