#!/usr/bin/env python3
"""
cone_coupling_search.py — what would have to be added for zeta to couple?

sigma_topology.py closed the easy routes: the deficit angle drops out locally
(2D conformal invariance, cone flat away from the apex) and topologically
(pi_1 = pi_2 = 0). So the question is bounded: at what order in derivatives
does a constructible term first fail to be conformally invariant, and does one
exist there?

WEIGHT COUNTING. Under g -> e^{2s} g in two dimensions, sqrt(g) -> e^{2s}
sqrt(g) and g^{mu nu} -> e^{-2s} g^{mu nu}. A term with k inverse metrics
scales as e^{(2-2k)s}. So:
    quadratic (k=1) : invariant  -> cannot see the cone
    quartic   (k=2) : weight e^{-2s} -> CAN see the cone
Quartic is therefore the first candidate order, and the question is what
constructible quartic terms exist.

THE INTERESTING CANDIDATE. target_wz_check.py measured ||[m,m]_m|| = 6.48, so
the target has a genuine bracket. That normally supplies a Skyrme term, built
from the bracket of the two derivative directions:

    B^e = c_{abe} eps^{mu nu} J^a_mu J^b_nu ,     L_skyrme = |B|^2

which is exactly the octonionic cross product of the two derivative vectors.

PREDICTIONS STATED BEFORE RUNNING
 K1 the 7-dimensional cross product from c satisfies the composition identity
    |u x v|^2 = |u|^2 |v|^2 - (u.v)^2  to machine precision.
 K2 CONSEQUENCE, and the point of this script: that identity collapses the
    Skyrme term into the metric terms. Predict L_skyrme = 2*(L1 - L2)
    identically, where L1 = (tr M)^2, L2 = tr(M^2), M = J J^T. The bracket
    supplies NO independent quartic term.
 K3 the space of independent constructible quartic scalars has dimension
    exactly 2 -- spanned by L1 and L2, with every c-built candidate dependent.
 K4 the quadratic term is conformally invariant and the quartic terms are not,
    verified by explicit rescaling rather than by weight counting alone.
 K5 therefore quartic is the lowest order at which zeta can couple, and terms
    do exist there -- but they carry no new structure from the octonions.

Python 3 + NumPy only.

(Cone deficit renamed beta -> zeta on 2026-09-25, to free beta for the lattice
gyroscopic coupling; the code variable keeps the name beta/BETA.)
"""

import numpy as np
from itertools import permutations


def oriented_lines():
    LINES = [tuple(sorted(((i + s - 1) % 7) + 1 for s in (0, 1, 3)))
             for i in range(1, 8)]
    used = {p: set() for p in range(1, 8)}
    assign = {}

    def bt(li):
        if li == len(LINES):
            return True
        Ln = LINES[li]
        for perm in permutations(range(3)):
            if all(perm[j] not in used[Ln[j]] for j in range(3)):
                for j in range(3):
                    used[Ln[j]].add(perm[j])
                assign[Ln] = perm
                if bt(li + 1):
                    return True
                for j in range(3):
                    used[Ln[j]].discard(perm[j])
                del assign[Ln]
        return False

    bt(0)
    return [tuple(Ln[assign[Ln].index(r)] for r in (0, 1, 2)) for Ln in LINES]


def imaginary_c(oriented):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in oriented:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


def cross(u, v, c):
    return np.einsum('abe,a,b->e', c, u, v)


def quartics(J, c):
    """All candidate quartic scalars from J (shape 2 x 7)."""
    M = J @ J.T                                   # 2x2
    L1 = np.trace(M) ** 2
    L2 = np.trace(M @ M)
    B = 2.0 * cross(J[0], J[1], c)                # eps^{mu nu} contraction
    L3 = B @ B
    P = np.einsum('abe,cde->abcd', c, c)
    L4 = np.einsum('abcd,ma,nb,mc,nd->', P, J, J, J, J)
    L5 = np.linalg.det(M)
    return np.array([L1, L2, L3, L4, L5])


def main():
    c = imaginary_c(oriented_lines())
    rng = np.random.default_rng(23)
    print("=" * 70)
    print("CONE COUPLING SEARCH :: WHERE CAN zeta FIRST ENTER?")
    print("=" * 70)

    # ---- K1 composition identity for the cross product --------------
    worst = 0.0
    for _ in range(500):
        u, v = rng.normal(size=7), rng.normal(size=7)
        lhs = cross(u, v, c) @ cross(u, v, c)
        rhs = (u @ u) * (v @ v) - (u @ v) ** 2
        worst = max(worst, abs(lhs - rhs))
    print(f"\nK1  max | |uxv|^2 - (|u|^2|v|^2 - (u.v)^2) | = {worst:.3e}"
          f"   [predicted 0 -> {'PASS' if worst < 1e-9 else 'MISS'}]")

    # ---- K2 does the Skyrme term collapse? --------------------------
    dev = 0.0
    for _ in range(500):
        J = rng.normal(size=(2, 7))
        q = quartics(J, c)
        dev = max(dev, abs(q[2] - 2.0 * (q[0] - q[1])))
    print(f"\nK2  max | L_skyrme - 2(L1 - L2) | = {dev:.3e}"
          f"   [predicted 0 -> {'PASS' if dev < 1e-9 else 'MISS'}]")
    print("    -> the bracket term is NOT independent")

    # ---- K3 how many independent quartics? --------------------------
    rows = np.array([quartics(rng.normal(size=(2, 7)), c) for _ in range(400)])
    rank = np.linalg.matrix_rank(rows, tol=1e-8)
    print(f"\nK3  rank over {rows.shape[0]} random configurations = {rank}"
          f"   [predicted 2 -> {'PASS' if rank == 2 else 'MISS'}]")
    print("    candidates tested: (trM)^2, tr(M^2), |B|^2, P-contraction, detM")

    # ---- K4 conformal behaviour, checked not asserted ---------------
    print("\nK4  CONFORMAL RESCALING  g -> e^{2s} g   (s = ln 2)")
    print("-" * 70)
    s = np.log(2.0)
    J = rng.normal(size=(2, 7))
    g = np.eye(2)
    gs = np.exp(2 * s) * g

    def quad_density(J, g):
        gi = np.linalg.inv(g)
        return np.einsum('mn,ma,na->', gi, J, J) * np.sqrt(np.linalg.det(g))

    def quart_density(J, g):
        gi = np.linalg.inv(g)
        M = np.einsum('mn,ma,nb->ab', gi, J, J)
        return np.einsum('ab,ab->', M, M) * np.sqrt(np.linalg.det(g))

    q0, q1 = quad_density(J, g), quad_density(J, gs)
    r0, r1 = quart_density(J, g), quart_density(J, gs)
    print(f"    quadratic : {q0:.6f} -> {q1:.6f}   ratio {q1/q0:.6f}"
          f"   [predict 1]")
    print(f"    quartic   : {r0:.6f} -> {r1:.6f}   ratio {r1/r0:.6f}"
          f"   [predict e^-2s = {np.exp(-2*s):.6f}]")
    okq = abs(q1 / q0 - 1) < 1e-10
    okr = abs(r1 / r0 - np.exp(-2 * s)) < 1e-10
    print(f"    quadratic blind to the metric : {'PASS' if okq else 'MISS'}")
    print(f"    quartic sees it               : {'PASS' if okr else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  Quartic is the lowest order where zeta can enter, and terms do")
    print("  exist there -- two of them, L1 and L2, built from the forced")
    print("  target metric alone.")
    print()
    print("  But the octonions contribute NOTHING to them. The bracket that")
    print("  makes the target non-symmetric -- the same ||[m,m]_m|| = 6.48")
    print("  that removed the integrability route -- would normally supply an")
    print("  independent Skyrme term. The composition identity collapses it")
    print("  into the metric terms exactly.")
    print()
    print("  So a coupling to the cone is available, but it is a generic")
    print("  sigma-model quartic. It carries no octonionic content, and")
    print("  nothing about it is forced: its coefficient is a new CHOSEN.")


if __name__ == "__main__":
    main()
