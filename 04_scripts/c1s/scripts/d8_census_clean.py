#!/usr/bin/env python3
"""
d8_census_clean.py — is every (3,4) octonionic term a total derivative?

Supersedes d8_term_census.py, which computed Euler-Lagrange derivatives by
einsum string surgery, carried dead branches, and hung. The criterion below
needs no jet bookkeeping.

CRITERION. A Lagrangian is a null Lagrangian exactly when it is a divergence.
So rather than varying, build the space of CURRENTS and push it forward:

    currents   V^mu  : cubic in the field, THREE derivatives, one free index
    divergence d_mu V^mu : cubic, FOUR derivatives -- lands in the (3,4) space

Then every (3,4) invariant is null iff the divergence image covers the whole
(3,4) space. Pure finite linear algebra.

INDEX BOOKKEEPING. Base and target are both Im(O), so all indices are R^7.
  J_{mu a}      = d_mu psi^a                two indices
  H_{mu nu a}   = d_mu d_nu psi^a           symmetric in (mu, nu)
  (3,4) terms   J J H : 7 indices = one c (3) + two deltas (2+2)
  currents      J J J with one free index : 5 contracted = one c + one delta
Differentiating a current sends one J to H, with the derivative index set equal
to the current's free index.

Note the parity consequence already recorded: 7 = 3 + 2 + 2 is the only split,
so EVERY (3,4) invariant carries exactly one c. There are no non-octonionic
terms at this order to compare against.

PREDICTIONS STATED BEFORE RUNNING
 U1 105 raw (3,4) contractions; the independent space is small, dimension <= 8.
    [MISSED: the space is 22-dimensional.]
 U2 60 raw currents (6 free-slot choices x C(5,3) placements of c).
 U3 the divergence image EQUALS the (3,4) space -- every invariant at this
    order is a total derivative, so no octonionic term contributes dynamics
    there. Predicted from the pattern; a MISS is the interesting outcome and
    would mean octonionic dynamics exists in the corrected arena.
    [MISSED, and the miss is the result. dim(3,4) = 22, divergence image = 8,
    so FOURTEEN independent invariants are not divergences and DO enter the
    field equations. Stable at tolerances 1e-8, 1e-10, 1e-12, and confirmed by
    explicit projection: rank 14 outside the divergence span with a residual
    singular value at 86% of the leading scale. Octonionic terms contribute
    dynamics in the corrected arena.]
 U4 control: the divergence map is not trivial -- its image has dimension > 0,
    confirming the test can distinguish.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations, combinations

L6 = 'abcdef'
L7 = 'abcdefg'


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


# ---------------------------------------------------------------- (3,4) space
def terms_34():
    """slots: J1=(0,1) J2=(2,3) H=(4,5,6), H symmetric in (4,5)."""
    out = []
    for tri in combinations(range(7), 3):
        rest = [i for i in range(7) if i not in tri]
        a = rest[0]
        for b in rest[1:]:
            others = tuple(r for r in rest[1:] if r != b)
            out.append((tri, (a, b), others))
    return out


def eval_34(spec, J, H, c, I7):
    tri, p1, p2 = spec
    sub = (f"{L7[0]}{L7[1]},{L7[2]}{L7[3]},{L7[4]}{L7[5]}{L7[6]},"
           + "".join(L7[i] for i in tri) + ","
           + f"{L7[p1[0]]}{L7[p1[1]]},{L7[p2[0]]}{L7[p2[1]]}->")
    return np.einsum(sub, J, J, H, c, I7, I7)


# ---------------------------------------------------------------- currents
def currents():
    """J J J, slots J1=(0,1) J2=(2,3) J3=(4,5); one free slot, one c, one delta."""
    out = []
    for free in range(6):
        rest = [i for i in range(6) if i != free]
        for tri in combinations(rest, 3):
            pair = tuple(i for i in rest if i not in tri)
            out.append((free, tri, pair))
    return out


def eval_div(spec, J, H, c, I7):
    """d_mu V^mu, with the derivative index identified with the free slot."""
    free, tri, pair = spec
    fl = L6[free]
    total = 0.0
    for which in range(3):
        subs = [f"{L6[0]}{L6[1]}", f"{L6[2]}{L6[3]}", f"{L6[4]}{L6[5]}"]
        ops = [J, J, J]
        subs[which] = fl + subs[which]          # J -> H, derivative index = free
        ops[which] = H
        sub = ",".join(subs) + "," + "".join(L6[i] for i in tri) \
            + "," + f"{L6[pair[0]]}{L6[pair[1]]}->"
        total = total + np.einsum(sub, *ops, c, I7)
    return total


def main():
    c = imaginary_c(oriented_lines())
    I7 = np.eye(7)
    rng = np.random.default_rng(53)
    T34 = terms_34()
    CUR = currents()
    print("=" * 70)
    print("(3,4) CENSUS :: IS EVERY OCTONIONIC TERM A DIVERGENCE?")
    print("=" * 70)
    print(f"\nU1  raw (3,4) contractions : {len(T34)}"
          f"   [C(7,3) x 3 = 105 -> {'PASS' if len(T34) == 105 else 'MISS'}]")
    print(f"U2  raw currents           : {len(CUR)}"
          f"   [6 x C(5,3) = 60 -> {'PASS' if len(CUR) == 60 else 'MISS'}]")

    def draw():
        J = rng.normal(size=(7, 7))
        H = rng.normal(size=(7, 7, 7))
        H = 0.5 * (H + np.transpose(H, (1, 0, 2)))
        return J, H

    NS = 260
    A = np.zeros((NS, len(T34)))
    B = np.zeros((NS, len(CUR)))
    for s in range(NS):
        J, H = draw()
        A[s] = [eval_34(t, J, H, c, I7) for t in T34]
        B[s] = [eval_div(v, J, H, c, I7) for v in CUR]

    def rank(M):
        if M.size == 0:
            return 0
        sv = np.linalg.svd(M, compute_uv=False)
        return int(np.sum(sv > 1e-8 * max(sv[0], 1e-30)))

    r34, rdiv, rboth = rank(A), rank(B), rank(np.hstack([A, B]))
    print(f"\n    dim of the (3,4) invariant space   : {r34}"
          f"   [predict <= 8 -> {'PASS' if r34 <= 8 else 'MISS'}]")
    print(f"    dim of the divergence image        : {rdiv}")
    print(f"    dim of their combined span         : {rboth}")
    # COVERAGE, corrected. rboth == r34 only says the divergences LIE IN the
    # (3,4) space, which holds by construction. Coverage requires the
    # divergence image to have the same dimension as the space itself. The
    # first version of this script tested the former and printed a confident
    # conclusion in the wrong direction.
    covered = (rdiv == r34)
    print(f"\nU3  divergence image covers the space : "
          f"{'YES' if covered else 'NO'}"
          f"   [predicted YES -> {'PASS' if covered else 'MISS'}]")
    if not covered:
        print(f"    NON-NULL directions remaining      : {rboth - rdiv}"
              f"  (out of {r34})")
    print(f"U4  divergence map nontrivial         : "
          f"{'PASS' if rdiv > 0 else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    if covered:
        print("  Every cubic four-derivative invariant in the corrected arena")
        print("  is a divergence. Combined with the (3,3) result, no octonionic")
        print("  term contributes dynamics at cubic order in 7D -- and by the")
        print("  parity count there are no non-octonionic terms at (3,4) either,")
        print("  so that order is empty of field equations entirely.")
    else:
        print("  NOT every invariant is a divergence. Some octonionic term at")
        print("  (3,4) DOES contribute to the field equations -- the first time")
        print("  in this program that the octonionic sector supplies dynamics.")


if __name__ == "__main__":
    main()
