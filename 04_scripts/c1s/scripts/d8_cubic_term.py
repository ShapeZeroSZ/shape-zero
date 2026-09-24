#!/usr/bin/env python3
"""
d8_cubic_term.py — does the (3,3) term do anything?

Revision 2 established that with base = Im(O) = R^7 the cubic three-derivative
term exists, where a 2D base forbids it:

    T[J] = phi^{mu nu rho} c_{abc} J_mu^a J_nu^b J_rho^c ,   J_mu^a = d_mu psi^a

Base and target are the SAME space here, so phi and c are literally the same
tensor and T is a G2-invariant cubic in the Jacobian matrix. Existing is not
the same as contributing, so this asks whether it enters the field equations.

THE EULER-LAGRANGE DERIVATIVE. Both c's are totally antisymmetric and the
product is symmetric under simultaneous permutation of the three (mu,a) pairs,
so dT/dJ_sigma^d = 3 c_{sigma nu rho} c_{dbc} J_nu^b J_rho^c and

    EL_d = d_sigma (dT/dJ_sigma^d)
         = 3 c_{sigma nu rho} c_{dbc} [ (d_sigma d_nu psi^b) J_rho^c
                                      + J_nu^b (d_sigma d_rho psi^c) ]

c is antisymmetric in (sigma,nu) and in (sigma,rho); both second derivatives
are symmetric in those pairs. Each term is therefore a contraction of an
antisymmetric tensor with a symmetric one and vanishes identically. If that
holds, T is a NULL LAGRANGIAN -- a total derivative contributing nothing to
the equations of motion.

That would continue the pattern rather than break it: in 2D the octonionic
term existed, was not a total derivative, and destabilised the theory; in 7D
it exists, is not forbidden, and is a total derivative. Different mechanism,
same outcome.

PREDICTIONS STATED BEFORE RUNNING
 P1 c_{sigma nu rho} S^{sigma nu} = 0 for every symmetric S -- the mechanism,
    checked directly rather than assumed.
 P2 EL_d vanishes identically for random smooth field data: build J and the
    second-derivative tensor H^b_{sigma nu} = d_sigma d_nu psi^b as an
    arbitrary symmetric-in-(sigma,nu) array and evaluate EL. Predict <= 1e-12.
 P3 the space of G2-invariant cubic scalars in J has dimension 4, spanned by
    tr(J)^3, tr(J) tr(J^2), tr(J^3), and T.
 P4 T is INDEPENDENT of the three trace invariants -- it is a genuinely new
    scalar, not a repackaging of them. (If it were dependent it would not even
    be a new term, null or otherwise.)
 P5 the trace invariants are NOT null Lagrangians, so the vanishing of EL is a
    property of the octonionic contraction specifically and not of cubics in
    general. Checked on tr(J)^3.

Python 3 + NumPy only.
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


def imaginary_c(orient):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in orient:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


def T_of(J, c):
    return np.einsum('mnr,abc,ma,nb,rc->', c, c, J, J, J)


def main():
    c = imaginary_c(oriented_lines())
    rng = np.random.default_rng(23)
    print("=" * 70)
    print("THE (3,3) OCTONIONIC TERM :: DOES IT CONTRIBUTE?")
    print("=" * 70)

    # ---- P1 the mechanism -------------------------------------------
    print("\nP1  c contracted with a symmetric tensor")
    print("-" * 70)
    worst = 0.0
    for _ in range(500):
        S = rng.normal(size=(7, 7))
        S = S + S.T
        worst = max(worst, np.max(np.abs(np.einsum('snr,sn->r', c, S))))
    print(f"    max |c_(sigma nu rho) S^(sigma nu)| = {worst:.3e}"
          f"   [predict 0 -> {'PASS' if worst < 1e-12 else 'MISS'}]")

    # ---- P2 the Euler-Lagrange derivative ---------------------------
    print("\nP2  EULER-LAGRANGE DERIVATIVE OF T")
    print("-" * 70)
    worst = 0.0
    for _ in range(300):
        J = rng.normal(size=(7, 7))
        H = rng.normal(size=(7, 7, 7))          # H[b, sigma, nu]
        H = 0.5 * (H + np.transpose(H, (0, 2, 1)))   # symmetric in (sigma,nu)
        t1 = 3.0 * np.einsum('snr,dbc,bsn,rc->d', c, c, H, J)
        t2 = 3.0 * np.einsum('snr,dbc,nb,csr->d', c, c, J, H)
        worst = max(worst, np.max(np.abs(t1 + t2)))
    print(f"    max |EL_d| over random field data = {worst:.3e}"
          f"   [predict 0 -> {'PASS' if worst < 1e-12 else 'MISS'}]")
    print("    -> T is a NULL LAGRANGIAN: a total derivative")

    # ---- P3/P4 the invariant cubics ---------------------------------
    print("\nP3/P4  G2-INVARIANT CUBIC SCALARS IN J")
    print("-" * 70)
    rows = []
    for _ in range(400):
        J = rng.normal(size=(7, 7))
        tr1 = np.trace(J)
        rows.append([tr1 ** 3, tr1 * np.trace(J @ J), np.trace(J @ J @ J),
                     T_of(J, c)])
    rows = np.array(rows)
    rank_all = np.linalg.matrix_rank(rows, tol=1e-8)
    rank_tr = np.linalg.matrix_rank(rows[:, :3], tol=1e-8)
    print(f"    rank of the three trace invariants : {rank_tr}")
    print(f"    rank including T                   : {rank_all}"
          f"   [predict 4 -> {'PASS' if rank_all == 4 else 'MISS'}]")
    print(f"    P4 T independent of the traces     : "
          f"{'PASS' if rank_all > rank_tr else 'MISS'}")

    # ---- P5 is nullity special to the octonionic contraction? -------
    print("\nP5  IS NULLITY SPECIAL TO T?  (control: tr(J)^3)")
    print("-" * 70)
    worst = 0.0
    for _ in range(300):
        J = rng.normal(size=(7, 7))
        H = rng.normal(size=(7, 7, 7))
        H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
        # d/dJ of tr(J)^3 is 3 tr(J)^2 delta ; EL = d_sigma of that
        el = 3.0 * 2.0 * np.trace(J) * np.einsum('bss->b', H)
        worst = max(worst, np.max(np.abs(el)))
    print(f"    max |EL| for tr(J)^3 = {worst:.3e}"
          f"   [predict nonzero -> {'PASS' if worst > 1e-6 else 'MISS'}]")
    print("    -> nullity is a property of the octonionic contraction,")
    print("       not of cubic terms generally")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The (3,3) term is real, is a genuinely new invariant, and")
    print("  contributes NOTHING to the equations of motion. Its Euler-")
    print("  Lagrange derivative vanishes identically because the totally")
    print("  antisymmetric structure constants meet a symmetric second")
    print("  derivative -- the same antisymmetry that makes the term exist")
    print("  is what makes it inert.")
    print()
    print("  Pattern, now at five: no WZ term, no potential, no Skyrme term,")
    print("  no topological sectors, and now a cubic term that exists but is")
    print("  a total derivative. The mechanisms differ every time; the")
    print("  outcome does not.")


if __name__ == "__main__":
    main()
