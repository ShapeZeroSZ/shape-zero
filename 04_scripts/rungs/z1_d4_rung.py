#!/usr/bin/env python3
"""
z1_d4_rung.py — Z1 ladder, rung D4: is non-commutativity forced?
(Shape Zero action log item Z1; predictions stated before running)

DERIVATION: on R^4 the compatible complex structures form a SPHERE
(so(4) = su(2) + su(2); quaternion left/right multiplications). The base
structure (readout convention) is a gauge choice; the su(2) COMMUTING with
it (right multiplications R_i, R_j, R_k) acts on the Bloch sphere of the
C^2 spinor. Any dynamics touching two axes of the sphere cannot commute
[FORCED — quaternion algebra]; zero-power velocity forces at node level
are exactly the antisymmetric ones [FORCED]; with structure compatibility
this is the u(2) envelope (A1's theorem, tagged).

PREDICTIONS (delta = 1, kappa = 0.3, carrier A = 0.2):
 P1 spinor precession about a pulse axis at rate EXACTLY kappa (Larmor
    splitting is exact, not perturbative): theta/t = 0.300000.
 P2 orthogonal-axis pi/2 + pi/2 pulses in opposite orders: final Bloch
    vectors 90.000000 deg apart; parallel-axis control 0.000000 deg.
 P3 passivity: energy conserved through all pulses (antisymmetric M does
    zero work); a SYMMETRIC velocity coupling pumps (dE/E grows).
"""

import numpy as np

DELTA, KAPPA, A, DT = 1.0, 0.3, 0.2, 2e-4
W0 = np.sqrt(DELTA)

# quaternion right-multiplication generators on (u1,u2,u3,u4) ~ u1+u2 i+u3 j+u4 k
R_i = np.array([[0,-1,0,0],[1,0,0,0],[0,0,0,1],[0,0,-1,0]], float)
R_j = np.array([[0,0,-1,0],[0,0,0,-1],[1,0,0,0],[0,1,0,0]], float)
assert np.allclose(R_i @ R_i, -np.eye(4)) and np.allclose(R_j @ R_j, -np.eye(4))
assert np.allclose(R_i @ R_j + R_j @ R_i, 0)          # anticommute: orthogonal axes

def bloch(u, v):
    p1, p2 = (u[0]+1j*u[1], u[2]+1j*u[3])
    d1, d2 = (v[0]+1j*v[1], v[2]+1j*v[3])
    c1, c2 = p1 + 1j*d1/W0, p2 + 1j*d2/W0
    N = abs(c1)**2 + abs(c2)**2
    return np.array([2*(np.conj(c1)*c2).real, 2*(np.conj(c1)*c2).imag,
                     abs(c1)**2 - abs(c2)**2]) / N

def evolve(u, v, M, kap, T):
    for _ in range(int(T/DT)):
        f = lambda u, v: -DELTA*u + kap*(M @ v)
        a1=f(u,v); v1=v
        a2=f(u+DT/2*v1, v+DT/2*a1); v2=v+DT/2*a1
        a3=f(u+DT/2*v2, v+DT/2*a2); v3=v+DT/2*a2
        a4=f(u+DT*v3, v+DT*a3); v4=v+DT*a3
        u=u+DT/6*(v1+2*v2+2*v3+v4); v=v+DT/6*(a1+2*a2+2*a3+a4)
    return u, v

def energy(u, v):
    return 0.5*(v @ v) + 0.5*DELTA*(u @ u)

def prep(chi):
    """Real-coordinate carrier for spinor chi at t=0 (positive-frequency)."""
    z1, z2 = A*chi[0], A*chi[1]
    u = np.array([z1.real, z1.imag, z2.real, z2.imag])
    v = np.array([(-1j*W0*z1).real, (-1j*W0*z1).imag,
                  (-1j*W0*z2).real, (-1j*W0*z2).imag])
    return u, v

if __name__ == '__main__':
    ang = lambda a, b: np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)
                                                    /np.linalg.norm(b), -1, 1)))
    # ---- P1: precession rate about R_j axis, starting from Bloch +z ----
    print('P1 precession rate (pred 0.300000):')
    for T in (2.0, 4.0):
        u, v = prep(np.array([1.0, 0.0]))
        n0 = bloch(u, v)
        u, v = evolve(u, v, R_j, KAPPA, T)
        th = np.radians(ang(n0, bloch(u, v)))
        print(f'   t = {T}: theta/t = {th/T:.6f}')

    # ---- P2: ordering splitting ----
    Tq = (np.pi/2)/KAPPA
    def pulse_seq(Ma, Mb):
        u, v = prep(np.array([1.0, 0.0]))
        u, v = evolve(u, v, Ma, KAPPA, Tq)
        u, v = evolve(u, v, Mb, KAPPA, Tq)
        e_dev = abs(energy(u, v) - energy(*prep(np.array([1.0,0.0])))) \
                / energy(*prep(np.array([1.0,0.0])))
        return bloch(u, v), e_dev
    nA, eA = pulse_seq(R_i, R_j)
    nB, eB = pulse_seq(R_j, R_i)
    nC, _  = pulse_seq(R_j, R_j)
    nD, _  = pulse_seq(R_j, R_j)   # identical parallel sequence (control)
    print(f'\nP2 ordering splitting (i then j vs j then i) = {ang(nA, nB):.6f} deg'
          f'   (pred 90.000000)')
    print(f'   parallel control = {ang(nC, nD):.6f} deg   (pred 0.000000)')
    print(f'   energy drift through pulses: {max(eA, eB):.1e}   (pred ~RK4 floor)')

    # ---- P3: symmetric coupling pumps ----
    S = np.zeros((4,4)); S[0,2] = S[2,0] = 1.0
    u, v = prep(np.array([1.0, 1.0])/np.sqrt(2))
    E0 = energy(u, v)
    u, v = evolve(u, v, S, KAPPA, 10.0)
    print(f'\nP3 symmetric velocity coupling, T=10: dE/E = '
          f'{(energy(u,v)-E0)/E0:+.3f}   (pred: pumps, |dE/E| >> conservative)')
