#!/usr/bin/env python3
"""
phi_gauge_wilson.py — non-Abelian test, Shape Zero v5.1 §5.2.7 Step 4

Dimerized ring: each node carries a 2-component internal space u_i in R^2
(two phi-attractor oscillators). Links carry orthogonal transport matrices
O[i] (lattice-gauge link variables); coupling energy (c/2)*|u_{i+1} - O[i]u_i|^2.
In the linear regime the on-site dynamics is O(2)-invariant, every link is
locally pure gauge (perfect transmission), and the ONLY invariant is the
path-ordered holonomy: a right-moving packet crossing links 1..n acquires
polarization  p -> O[n]...O[1] p.

Test: two non-identity links W_A (rotation by 45 deg) and W_B (reflection
diag(1,-1)) placed on the ring in both orders. Same link multiset; only
path-ordering differs.

  non-Abelian pair:  W_B W_A p0  vs  W_A W_B p0   -> predicted 90 deg apart
  Abelian control :  R(45), R(60) in both orders  -> predicted identical

If the two orderings of the non-Abelian pair give the same final polarization,
the non-commutative transport claim fails as stated.
"""

import numpy as np

SQ5 = np.sqrt(5)
N = 128
C = 1.0
DT = 0.02
K = np.pi / 2
OMEGA = np.sqrt(SQ5 + 2 * C * (1 - np.cos(K)))
VG = C * np.sin(K) / OMEGA
AMP = 1e-3                       # linear regime: internal O(2) symmetry exact
W_PACKET = 6.0
I0 = 5
LINK_SITES = (30, 90)            # packet crosses 30 first, then 90
T_MEAS = 210.0                   # past link 90, before wrap-around


def rot(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


REFL = np.array([[1.0, 0.0], [0.0, -1.0]])


def links(first, second):
    O = np.tile(np.eye(2), (N, 1, 1))
    O[LINK_SITES[0]] = first
    O[LINK_SITES[1]] = second
    return O


def force(u, v, O, Ot, Om, OmT):
    up, um = np.roll(u, -1, axis=0), np.roll(u, 1, axis=0)
    onsite = -(SQ5 * u + u * u)                     # per-component, about phi
    hop = C * (np.einsum('nab,nb->na', Ot, up) - u) \
        + C * (np.einsum('nab,nb->na', Om, um) - u)
    return onsite + hop


def run(O, p0):
    Ot = np.transpose(O, (0, 2, 1))                 # O[i]^T
    Om = np.roll(O, 1, axis=0)                      # O[i-1]
    OmT = None
    n = np.arange(N)
    g = np.exp(-((n - I0) % N - 0) ** 2 / (2 * W_PACKET ** 2))
    g = np.exp(-(((n - I0 + N // 2) % N - N // 2) ** 2) / (2 * W_PACKET ** 2))
    u = AMP * (g * np.cos(K * (n - I0)))[:, None] * p0[None, :]
    v = AMP * (g * OMEGA * np.sin(K * (n - I0)))[:, None] * p0[None, :]
    for _ in range(int(T_MEAS / DT)):
        k1v = force(u, v, O, Ot, Om, OmT);                        k1u = v
        k2v = force(u + 0.5*DT*k1u, v + 0.5*DT*k1v, O, Ot, Om, OmT); k2u = v + 0.5*DT*k1v
        k3v = force(u + 0.5*DT*k2u, v + 0.5*DT*k2v, O, Ot, Om, OmT); k3u = v + 0.5*DT*k2v
        k4v = force(u + DT*k3u, v + DT*k3v, O, Ot, Om, OmT);         k4u = v + DT*k3v
        u = u + DT/6 * (k1u + 2*k2u + 2*k3u + k4u)
        v = v + DT/6 * (k1v + 2*k2v + 2*k3v + k4v)
    S = u.T @ u + (v.T @ v) / OMEGA ** 2            # polarization matrix
    ang = 0.5 * np.degrees(np.arctan2(2 * S[0, 1], S[0, 0] - S[1, 1]))
    evals = np.linalg.eigvalsh(S)
    purity = 1 - evals[0] / evals[1]                # 1 = fully polarized
    return ang, purity


def pred_angle(W2, W1, p0):
    p = W2 @ W1 @ p0
    return np.degrees(np.arctan2(p[1], p[0])) % 180.0


def norm(a):                                        # axis angle mod 180
    return a % 180.0


if __name__ == '__main__':
    p0 = np.array([1.0, 0.0])
    WA, WB = rot(45), REFL
    RA, RB = rot(45), rot(60)                       # Abelian control
    cases = [
        ('A then B (W_B W_A)', links(WA, WB), (WB, WA)),
        ('B then A (W_A W_B)', links(WB, WA), (WA, WB)),
        ('control: R45 then R60', links(RA, RB), (RB, RA)),
        ('control: R60 then R45', links(RB, RA), (RA, RB)),
    ]
    print(f'k=pi/2, v_g={VG:.3f}, links at sites {LINK_SITES}, measure at t={T_MEAS}')
    print(f'{"configuration":>24} {"measured":>9} {"predicted":>10} {"purity":>7}')
    meas = []
    for name, O, (W2, W1) in cases:
        ang, pur = run(O, p0)
        pred = pred_angle(W2, W1, p0)
        meas.append(norm(ang))
        print(f'{name:>24} {norm(ang):>8.2f}\u00b0 {pred:>9.2f}\u00b0 {pur:>7.4f}')
    print(f'\nnon-Abelian pair splitting : {abs(meas[0]-meas[1]):.2f}\u00b0  (predicted 90\u00b0)')
    print(f'Abelian control splitting  : {abs(meas[2]-meas[3]):.2f}\u00b0  (predicted 0\u00b0)')
