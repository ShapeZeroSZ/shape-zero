#!/usr/bin/env python3
"""
kk_rotor_check.py -- late hypothesis LK (Klein's step, the quantum circle); predictions in KK_HYPOTHESES.md,
committed at 7a8f588 (2026-09-28T04:53:57Z) before this script was written.

One node, n = 1, unit inertia, hbar in lattice units as the added input. From KC1 the canonical Hamiltonian is
    H = -hbar^2/2 lap + (kappa/2) L_z + (1/2)(sqrt5 + kappa^2/4) r^2 + c3 r^3/3,   L_z = -i hbar (x d_y - y d_x)
(A0 = -kappa/2; c3 = 1 for A', 0 for the harmonic limit).
Method A: 2-D Cartesian grid, 4th-order differences, full H including the kappa term; l read from <L_z>/hbar.
Method B: the A' cubic in the 2-D oscillator basis at fixed |l| (generalised Laguerre, exact quadrature); independent
          of the grid, used for the tower to high precision.
"""
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.special import roots_genlaguerre, eval_genlaguerre, gammaln

K = math.sqrt(5.0)
KA = 2.0 / math.sqrt(K + 2.0)
W = math.sqrt(K + KA * KA / 4)
WA = (-KA + math.sqrt(KA * KA + 4 * K)) / 2
WB = WA + KA


def d1(n, h):
    e = np.ones(n)
    return sp.diags([e[2:] / 12, -8 * e[1:] / 12, 8 * e[1:] / 12, -e[2:] / 12], [-2, -1, 1, 2]) / h


def d2(n, h):
    e = np.ones(n)
    return sp.diags([-e[2:] / 12, 16 * e[1:] / 12, -30 * e / 12, 16 * e[1:] / 12, -e[2:] / 12], [-2, -1, 0, 1, 2]) / h ** 2


def grid_levels(hbar, c3, nlev=12, npts=151, box=7.0):
    eps = math.sqrt(hbar / W)
    L = box * eps
    x = np.linspace(-L, L, npts); h = x[1] - x[0]
    I = sp.identity(npts)
    D1, D2 = d1(npts, h), d2(npts, h)
    X, Y = np.meshgrid(x, x, indexing="ij")
    Dx, Dy = sp.kron(D1, I), sp.kron(I, D1)
    lap = sp.kron(D2, I) + sp.kron(I, D2)
    Xd, Yd = sp.diags(X.ravel()), sp.diags(Y.ravel())
    Lz = -1j * hbar * (Xd @ Dy - Yd @ Dx)
    Lz = 0.5 * (Lz + Lz.conj().T)
    r = np.sqrt(X ** 2 + Y ** 2).ravel()
    V = 0.5 * W * W * r ** 2 + c3 * r ** 3 / 3
    H = (-0.5 * hbar ** 2 * lap + 0.5 * KA * Lz + sp.diags(V)).tocsc()
    H = 0.5 * (H + H.conj().T)
    vals, vecs = sla.eigsh(H, k=nlev, sigma=0.0, which="LM")
    o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]
    ls = [float(np.real(np.vdot(vecs[:, i], Lz @ vecs[:, i])) / hbar) for i in range(nlev)]
    return vals, ls


def basis_levels(hbar, l, c3, nb=60):
    """Eigenvalues of H_rot at fixed |l| (no kappa term), 2-D oscillator basis."""
    eps = math.sqrt(hbar / W)
    a = abs(l)
    xq, wq = roots_genlaguerre(200, a + 1.5)        # integrand x^{a} e^{-x} * x^{3/2} * poly
    nrm = np.array([math.exp(0.5 * (gammaln(n + a + 1) - gammaln(n + 1))) for n in range(nb)])
    Lm = np.array([eval_genlaguerre(n, a, xq) for n in range(nb)]) / nrm[:, None]
    M = (Lm * wq) @ Lm.T                             # <n| x^{3/2} |m>
    H = np.diag([hbar * W * (2 * n + a + 1) for n in range(nb)]) + c3 * eps ** 3 * M / 3
    return np.linalg.eigvalsh(H)


def first_order(hbar, l):
    eps = math.sqrt(hbar / W)
    return hbar * W * (abs(l) + 1) + eps ** 3 * math.exp(gammaln(abs(l) + 2.5) - gammaln(abs(l) + 1)) / 3


print("=" * 88)
print("KK LATE HYPOTHESIS LK -- the quantum circle (predictions: KK_HYPOTHESES.md, 7a8f588)")
print("=" * 88)
print(f"w_rot = {W:.6f}, w_a = {WA:.6f}, w_b = {WB:.6f}, kappa* = {KA:.6f}")

print("\nLK1  harmonic limit (cubic off), grid, hbar = 0.1: E / hbar against Fock-Darwin w_rot(2 n_r + |l| + 1) + (kappa/2) l")
vals, ls = grid_levels(0.1, 0.0)
worst = 0.0
for E, l in zip(vals, ls):
    lr = int(round(l))
    cands = [W * (2 * nr + abs(lr) + 1) + KA / 2 * lr for nr in range(6)]
    fd = min(cands, key=lambda c: abs(c - E / 0.1))
    worst = max(worst, abs(E / 0.1 - fd) / fd)
    print(f"    E/hbar = {E/0.1:.6f}  <L_z>/hbar = {l:+.4f}  Fock-Darwin {fd:.6f}")
print(f"    max relative deviation {worst:.1e} (predicted <= 1e-5)")
E0 = vals[0]
q = {int(round(l)): E for E, l in zip(vals, ls) if abs(E - E0 - 0.1 * W) < 0.1 * 1.5 and int(round(l)) != 0}
print(f"    one-quantum states: l = -1: (E - E0)/hbar = {(q[-1]-E0)/0.1:.6f} (w_a = {WA:.6f}); "
      f"l = +1: {(q[1]-E0)/0.1:.6f} (w_b = {WB:.6f})")

for hbar in (1e-3, 0.1):
    print(f"\nLK2-LK4  A' cubic on, hbar = {hbar} (eps = {math.sqrt(hbar/W):.5f})")
    Er = {l: basis_levels(hbar, l, 1.0) for l in range(0, 8)}
    E0 = Er[0][0]
    m = [(Er[l][0] - E0) / (hbar * W) for l in range(8)]
    sp_ = [(Er[l + 1][0] - Er[l][0]) / (hbar * W) for l in range(7)]
    fo = [(first_order(hbar, l + 1) - first_order(hbar, l)) / (hbar * W) for l in range(7)]
    print("    basis: gauge-invariant m_l / (hbar w_rot), l = 1..7: " + " ".join(f"{x:.5f}" for x in m[1:]))
    print("    spacings l -> l+1 (0..6):           " + " ".join(f"{x:.5f}" for x in sp_))
    print("    first-order prediction:             " + " ".join(f"{x:.5f}" for x in fo))
    print(f"    max |basis - first order| / first order = {max(abs(a-b)/b for a, b in zip(sp_, fo)):.1e}")
    e10 = (Er[0][1] - E0) / (hbar * W); e02 = (Er[2][0] - E0) / (hbar * W)
    print(f"    LK4: E(n_r = 1, l = 0) - E0 = {e10:.5f}, E(0, 2) - E0 = {e02:.5f} -> "
          f"{'(1,0) above (0,2)' if e10 > e02 else '(1,0) below (0,2)'}")
    print(f"    LK3: lightest charged |l| = {1 + int(np.argmin(m[1:]))}; superlinear: m_2 - 2 m_1 = {m[2]-2*m[1]:+.5f}")

print("\nLK3  grid with the kappa term and the cubic, hbar = 0.1: +-1 masses after removing (kappa/2) hbar l")
vals, ls = grid_levels(0.1, 1.0)
E0 = vals[0]
one = {}
for E, l in zip(vals, ls):
    lr = int(round(l))
    if lr in (-1, 1) and lr not in one:
        one[lr] = E
gm = (one[-1] - E0 + KA / 2 * 0.1, one[1] - E0 - KA / 2 * 0.1)
print(f"    lab: l = -1 {(one[-1]-E0)/0.1:.5f}, l = +1 {(one[1]-E0)/0.1:.5f} (units hbar); gauge-invariant "
      f"{gm[0]/(0.1*W):.5f}, {gm[1]/(0.1*W):.5f} (units hbar w_rot); basis m_1 = "
      f"{(basis_levels(0.1,1,1.0)[0]-basis_levels(0.1,0,1.0)[0])/(0.1*W):.5f}")
