#!/usr/bin/env python3
"""
coset_audit.py — is Spin(7)/G2 appearing twice one fact or two?

Last session ended on a flagged lead: the target space of the octonion-structure
field (SO(7)/G2, from z1_holonomy_orbit.py) is the same homogeneous space that
carries nearly-parallel G2 structures. Flagged as a lead precisely because this
conversation keeps showing that matching objects are not shared objects.

Two things could be true:

  MERELY ISOMORPHIC. Both are 7-dimensional cosets of the same groups, but the
  identification is abstract -- no canonical map, so nothing transfers.

  CANONICAL. There is a distinguished isomorphism, built from the structure
  already present, so the two appearances are one object seen twice.

The discriminator is whether a canonical map exists. There is an obvious
candidate: contract the structure constants with an imaginary octonion,

    v  |-->  L_v ,   (L_v)_{ab} = c_{v a b}

This is built from c alone. If it lands in g2-perp, is bijective, and is
G2-equivariant, the identification is canonical and forced.

PREDICTIONS STATED BEFORE RUNNING
 C1 g2-perp is an IRREDUCIBLE G2-representation, so by Schur the space of
    G2-invariant symmetric bilinear forms on it is exactly 1-dimensional. The
    target metric is therefore forced up to scale -- there is no squashing
    parameter to choose.
 C2 L_v lies entirely in g2-perp: its g2 component is zero to 1e-12, for every
    v. (If it had a g2 part the map would not be onto the moduli directions.)
 C3 the map v |--> L_v has rank 7 -- a bijection Im(O) -> g2-perp.
 C4 it is G2-equivariant, intertwining the action on Im(O) with the adjoint
    action on g2-perp, up to a single consistent sign.
 C5 therefore the identification is canonical: each direction the octonion
    structure can move is labelled by an imaginary octonion, by c itself.

 NOT PREDICTED, and reported as a non-result: whether anything about
 nearly-parallel GEOMETRY transfers. That geometry lives on a 7-manifold used
 as a BASE. Here the coset is a TARGET. Same space, different job.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations

TOL = 1e-9


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


def derivations(c):
    rows = []
    for i in range(7):
        for j in range(7):
            for k in range(7):
                M = np.zeros((7, 7))
                M[k, :] += c[i, j, :]
                M[:, i] -= c[:, j, k]
                M[:, j] -= c[i, :, k]
                rows.append(M.ravel())
    _, s, Vt = np.linalg.svd(np.array(rows))
    return [v.reshape(7, 7) for v in Vt[np.sum(s > TOL):]]


def so7_basis():
    out = []
    for b in range(7):
        for cc in range(b + 1, 7):
            M = np.zeros((7, 7))
            M[b, cc], M[cc, b] = 1.0, -1.0
            out.append(M / np.sqrt(2.0))
    return out


def main():
    oriented = oriented_lines()
    c = imaginary_c(oriented)
    D7 = derivations(c)
    so7 = so7_basis()

    Qg = np.linalg.qr(np.array([X.ravel() for X in D7]).T)[0]
    cols = [M.ravel() - Qg @ (Qg.T @ M.ravel()) for M in so7]
    U, s, _ = np.linalg.svd(np.array(cols).T)
    perp = [U[:, k].reshape(7, 7) for k in range(int(np.sum(s > TOL)))]
    Qp = np.linalg.qr(np.array([X.ravel() for X in perp]).T)[0]

    print("=" * 70)
    print("COSET AUDIT :: CANONICAL, OR MERELY ISOMORPHIC?")
    print("=" * 70)
    print(f"\n  dim g2 = {len(D7)}, dim g2-perp = {len(perp)}")

    # ---- C1 uniqueness of the invariant metric ---------------------
    # symmetric 7x7 forms S on g2-perp, invariant: S(ad_D x, y) + S(x, ad_D y) = 0
    rows = []
    for D in D7:
        A = np.zeros((7, 7))
        for i, P in enumerate(perp):
            img = D @ P - P @ D
            A[:, i] = Qp.T @ img.ravel()
        # invariance of S: A^T S + S A = 0
        for i in range(7):
            for j in range(7):
                r = np.zeros(49)
                for m in range(7):
                    r[m * 7 + j] += A[m, i]
                    r[i * 7 + m] += A[m, j]
                rows.append(r)
    # restrict to symmetric S
    sym = []
    for i in range(7):
        for j in range(i, 7):
            v = np.zeros(49)
            v[i * 7 + j] = 1.0
            v[j * 7 + i] = 1.0
            sym.append(v / np.linalg.norm(v))
    S = np.array(sym).T
    M = np.array(rows) @ S
    _, sv, _ = np.linalg.svd(M)
    ninv = S.shape[1] - int(np.sum(sv > 1e-8))
    print(f"\nC1  G2-invariant symmetric forms on g2-perp : {ninv}"
          f"   [predicted 1 -> {'PASS' if ninv == 1 else 'MISS'}]")
    print("    -> target metric forced up to scale; no squashing parameter")

    # ---- C2/C3 the candidate map ------------------------------------
    print("\nC2/C3  THE CANDIDATE MAP  v |--> L_v ,  (L_v)_ab = c_vab")
    print("-" * 70)
    Ls = [c[i] for i in range(7)]                # (L_{e_i})_{ab} = c[i,a,b]
    antis = max(np.linalg.norm(L + L.T) for L in Ls)
    g2parts = [np.linalg.norm(Qg @ (Qg.T @ L.ravel())) for L in Ls]
    print(f"    antisymmetry of L_v          : {antis:.3e}")
    print(f"    max g2 component of L_v      : {max(g2parts):.3e}"
          f"   [predicted 0 -> {'PASS' if max(g2parts) < 1e-12 else 'MISS'}]")
    rank = np.linalg.matrix_rank(np.array([L.ravel() for L in Ls]), tol=1e-8)
    print(f"    rank of v |--> L_v           : {rank}"
          f"   [predicted 7 -> {'PASS' if rank == 7 else 'MISS'}]")

    # ---- C4 equivariance --------------------------------------------
    print("\nC4  EQUIVARIANCE")
    print("-" * 70)
    res_minus, res_plus = 0.0, 0.0
    for D in D7:
        Dn = D / np.linalg.norm(D)
        for i in range(7):
            Dv = Dn[:, i]                        # (D e_i)_n = D_{n i}
            L_Dv = np.einsum('n,nab->ab', Dv, c)
            comm = Dn @ Ls[i] - Ls[i] @ Dn
            res_minus = max(res_minus, np.linalg.norm(L_Dv - comm))
            res_plus = max(res_plus, np.linalg.norm(L_Dv + comm))
    best = min(res_minus, res_plus)
    sign = '-' if res_minus < res_plus else '+'
    print(f"    max || L_(Dv) {sign} [D, L_v] || = {best:.3e}"
          f"   [predicted 0 -> {'PASS' if best < 1e-10 else 'MISS'}]")
    print(f"    (other sign: {max(res_minus, res_plus):.3e}, so the convention"
          f" is consistent)")

    # ---- C5 verdict --------------------------------------------------
    ok = (ninv == 1 and max(g2parts) < 1e-12 and rank == 7 and best < 1e-10)
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if ok:
        print("  CANONICAL. The moduli directions of the octonion structure are")
        print("  labelled by imaginary octonions, via c itself -- not by an")
        print("  arbitrary choice of isomorphism. Both appearances of the coset")
        print("  trace to the single fact that G2 = Aut(O) and the 7 is unique.")
    else:
        print("  NOT ESTABLISHED -- see the failing line above.")
    print("\n  NON-RESULT, stated as such: this identifies the TARGET, and")
    print("  fixes its metric. It does not make the base 7-dimensional. The")
    print("  nearly-parallel G2 geometry uses this coset as a BASE manifold;")
    print("  here it is the target of a field over a 2-dimensional cone.")
    print("  Same space, different job. Nothing about that geometry transfers")
    print("  on the strength of the coset alone.")


if __name__ == "__main__":
    main()
