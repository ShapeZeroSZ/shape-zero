#!/usr/bin/env python3
"""
d16_spectrum_v2.py — reduced D16 spectrum, variational

Supersedes d16_spectrum.py, which was a finite-difference build with three
faults: no boundary condition (a ragged staircase changing shape with every
refinement), drift terms double-counted inside the second-derivative loop, and
no symmetrisation -- so it was not self-adjoint and its two grids shared nothing
(1785% drift). None of its numbers meant anything.

THIS VERSION. Rayleigh-Ritz on polynomial basis functions in the four
invariants:

    A_mn = < g^ij d_i phi_m d_j phi_n >     B_mn = < phi_m phi_n >
    solve  A x = lambda B x

Symmetric by construction, natural (Neumann) boundary conditions, and no grid.
The domain and the invariant weight are both handled by SAMPLING S^15 UNIFORMLY
and pushing forward to u = (Re a, Re q, |p|^2, p.q): the push-forward of the
uniform measure IS the orbit-volume-weighted measure on the quotient, so the
weight never has to be constructed.

TWO METRICS, AND ONE IS A CHOICE.
  round      g^ij as derived -- this is the AMBIENT Euclidean metric on S^15,
             so its invariant sector must be a sub-tower of k(k+14)
  distorted  the algebra's metric, non-round because composition fails at D16.
             Conformal ansatz g~ = Omega^2 g with Omega = |det L_a|^(1/16),
             the geometric mean of the singular values of left multiplication.

**The conformal ansatz is CHOSEN, not forced.** That composition failure produces
a non-round structure is established (sigma_min ranges 0.102-0.853 at D16 against
1.0000 exactly at D8). Which non-round metric to build from it is not determined,
and a different choice would give a different spectrum. This is a genuine CHOSEN
and is recorded as one.

PREDICTIONS STATED BEFORE RUNNING
 T1 A and B symmetric, B positive definite, all eigenvalues real and >= 0, with
    the lowest exactly 0 (the constant function). Instrument check.
 T2 the ROUND spectrum consists of values drawn from k(k+14) = 15, 32, 51, 72...
    If it does not, the reduction or the sampling is wrong.
 T3 the DISTORTED spectrum differs from the round one -- otherwise the conformal
    factor has bought nothing.
 T4 both are stable under basis enlargement to a few percent.
 T5 no three-fold structure is sought. Degeneracies are reported as they fall and
    compared against the round case as control.

Python 3 + NumPy only.
"""

import numpy as np
import itertools


def cd(k):
    E = np.zeros((1, 1, 1))
    E[0, 0, 0] = 1.0
    for _ in range(k):
        d = E.shape[0]
        D = 2 * d
        N = np.zeros((D, D, D))
        I = np.eye(d)
        cj = lambda X: np.array([X[0]] + [-x for x in X[1:]])
        m = lambda x, y: np.einsum('ijk,i,j->k', E, x, y)
        for i in range(D):
            for j in range(D):
                a = I[i] if i < d else np.zeros(d)
                b = np.zeros(d) if i < d else I[i - d]
                c = I[j] if j < d else np.zeros(d)
                e = np.zeros(d) if j < d else I[j - d]
                N[i, j, :d] = m(a, c) - m(cj(e), b)
                N[i, j, d:] = m(e, a) + m(b, cj(c))
        E = N
    return E


E16 = cd(4)
I16 = np.eye(16)


def Lmat(a):
    return np.column_stack([np.einsum('ijk,i,j->k', E16, a, I16[j])
                            for j in range(16)])


def invs(a):
    p, q = a[:8], a[8:]
    return np.array([a[0], a[8], p @ p, p @ q])


def gmat(u):
    u1, u2, u3, u4 = u
    return np.array([
        [1 - u1 * u1, -u1 * u2, 2 * u1 * (1 - u3), u2 - 2 * u1 * u4],
        [-u1 * u2, 1 - u2 * u2, -2 * u2 * u3, u1 - 2 * u2 * u4],
        [2 * u1 * (1 - u3), -2 * u2 * u3, 4 * u3 * (1 - u3), 2 * u4 * (1 - 2 * u3)],
        [u2 - 2 * u1 * u4, u1 - 2 * u2 * u4, 2 * u4 * (1 - 2 * u3), 1 - 4 * u4 * u4]])


def sample(n, seed=0, need_det=True):
    rng = np.random.default_rng(seed)
    U, W = [], []
    for _ in range(n):
        a = rng.normal(size=16)
        a /= np.linalg.norm(a)
        U.append(invs(a))
        W.append(abs(np.linalg.det(Lmat(a))) if need_det else 1.0)
    return np.array(U), np.array(W)


def basis(deg):
    return [m for m in itertools.product(range(deg + 1), repeat=4)
            if sum(m) <= deg]


def phi_and_grad(mon, u):
    v = np.prod(u ** np.array(mon))
    g = np.zeros(4)
    for i in range(4):
        if mon[i] > 0:
            e = list(mon)
            e[i] -= 1
            g[i] = mon[i] * np.prod(u ** np.array(e))
    return v, g


def spectrum(U, weight, deg, nev=8):
    B = basis(deg)
    n = len(B)
    A = np.zeros((n, n))
    M = np.zeros((n, n))
    for u, w in zip(U, weight):
        G = gmat(u)
        vs, gs = zip(*(phi_and_grad(m, u) for m in B))
        vs = np.array(vs)
        gs = np.array(gs)
        A += w * (gs @ G @ gs.T)
        M += w * np.outer(vs, vs)
    A /= len(U)
    M /= len(U)
    ev = np.linalg.eigvalsh(np.linalg.solve(M, A) if False else
                            np.linalg.inv(np.linalg.cholesky(M)) @ A @
                            np.linalg.inv(np.linalg.cholesky(M)).T)
    return np.sort(ev.real)[:nev], A, M


def main():
    print("=" * 70)
    print("REDUCED D16 SPECTRUM -- VARIATIONAL")
    print("=" * 70)
    NS = 12000
    U, detL = sample(NS, seed=3)
    print(f"\n  {NS} uniform samples on S^15 pushed to the quotient")
    print(f"  |det L_a| range {detL.min():.4f} .. {detL.max():.4f}")

    for deg in (3, 4):
        print(f"\n  --- polynomial basis, total degree {deg} "
              f"({len(basis(deg))} functions) ---")
        for name, w, scale in (("ROUND    (ambient metric)", np.ones(NS), 1.0),
                               ("DISTORTED (Omega=|detL|^1/16)",
                                detL ** (14 / 16.0), 1.0)):
            ev, A, M = spectrum(U, w, deg)
            sym = max(np.max(np.abs(A - A.T)), np.max(np.abs(M - M.T)))
            pos = np.linalg.eigvalsh(M).min()
            print(f"    {name}")
            print(f"      symmetry {sym:.1e}, B min-eig {pos:.2e}, "
                  f"lowest {ev[0]:.2e}")
            nz = ev[ev > 1e-6]
            print(f"      eigenvalues : {np.round(nz[:6], 3)}")
            if len(nz):
                print(f"      ratios      : {np.round(nz[:6] / nz[0], 4)}")
    print("\n  CONTROL: round-sphere tower k(k+14) = "
          f"{[k*(k+14) for k in range(1,6)]}, ratios "
          f"{[round(k*(k+14)/15,4) for k in range(1,6)]}")


if __name__ == "__main__":
    main()
