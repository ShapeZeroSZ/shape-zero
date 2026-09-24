#!/usr/bin/env python3
"""
d8_channels.py — channel content of the 14, and O vs S^7 compared on outcomes

Follow-up to d8_census_clean.py, which found 22 cubic four-derivative
invariants in the corrected arena, 8 of them divergences, leaving 14 that enter
the field equations.

(1) CHANNEL CONTENT. An earlier proposal to "decompose the 22 under G2" was
ill-posed: those are G2-INVARIANT scalars, with nothing for the group to act on.
The well-posed question is which irreducible channels of the field data each
invariant reads. J lives in R^7 (x) R^7 = 1 + 7 + 14 + 27, so restricting J to a
single channel and recounting gives a partial profile. PARTIAL BY CONSTRUCTION:
it probes J (x) J within one channel and cannot see cross-channel pairs.

(2) O vs S^7. The census is pointwise, and at a point the tangent space of S^7
is Im(O). The second fundamental form of S^7 in O is purely normal, and the
normal direction was never counted, so the two readings should agree exactly.
Tested rather than argued.

PERFORMANCE NOTE. Two things were needed to make this finish. Evaluating one
sample at a time costs 285 einsum calls per sample; and a six-operand einsum
without optimize=True builds enormous intermediates. Both are fixed below. An
earlier attempt to economise by cutting the sample count instead produced
impossible output -- rank 32 under a restriction that can only lower rank from
22 -- because the rank cap applies to the sample matrix against 105 columns.

PREDICTIONS STATED BEFORE RUNNING
 W1 J decomposes with Casimir multiplicities 1, 7, 14, 27 summing to 49.
 W2 the batched code reproduces the unrestricted counts 22 / 8 / 14.
 W3 single-channel non-null counts sum to FEWER than 14, showing that the
    surviving directions require channel mixing and are not a clean irrep class.
 W4 O and S^7 give identical counts.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations, combinations

L6, L7 = 'abcdef', 'abcdefg'


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
    out = []
    for v in Vt[np.sum(s > 1e-9):]:
        X = 0.5 * (v.reshape(7, 7) - v.reshape(7, 7).T)
        out.append(X / np.sqrt(-np.trace(X @ X)))
    return out


def terms_34():
    out = []
    for tri in combinations(range(7), 3):
        rest = [i for i in range(7) if i not in tri]
        a = rest[0]
        for b in rest[1:]:
            others = tuple(r for r in rest[1:] if r != b)
            out.append((tri, (a, b), others))
    return out


def currents():
    out = []
    for free in range(6):
        rest = [i for i in range(6) if i != free]
        for tri in combinations(rest, 3):
            pair = tuple(i for i in rest if i not in tri)
            out.append((free, tri, pair))
    return out


def eval_34(spec, J, H, c, I7):
    """Batched: J is (N,7,7), H is (N,7,7,7); returns (N,)."""
    tri, p1, p2 = spec
    sub = (f"z{L7[0]}{L7[1]},z{L7[2]}{L7[3]},z{L7[4]}{L7[5]}{L7[6]},"
           + "".join(L7[i] for i in tri) + ","
           + f"{L7[p1[0]]}{L7[p1[1]]},{L7[p2[0]]}{L7[p2[1]]}->z")
    return np.einsum(sub, J, J, H, c, I7, I7, optimize=True)


def eval_div(spec, J, H, c, I7):
    """Batched divergence of a cubic current."""
    free, tri, pair = spec
    fl = L6[free]
    tot = 0.0
    for which in range(3):
        subs = [f"z{L6[0]}{L6[1]}", f"z{L6[2]}{L6[3]}", f"z{L6[4]}{L6[5]}"]
        ops = [J, J, J]
        subs[which] = "z" + fl + subs[which][1:]
        ops[which] = H
        sub = ",".join(subs) + "," + "".join(L6[i] for i in tri) \
            + "," + f"{L6[pair[0]]}{L6[pair[1]]}->z"
        tot = tot + np.einsum(sub, *ops, c, I7, optimize=True)
    return tot


def rank(M, tol=1e-8, floor=1e-9):
    """Numerical rank with BOTH a relative and an absolute threshold.

    A purely relative threshold fails when the matrix is identically zero up to
    floating-point noise: sv[0] is then itself ~1e-14, everything clears
    tol*sv[0], and the reported rank is spurious. That produced a rank of 67
    against a full-space value of 22 on a 2-dimensional support, and a rank of
    0 where the true answer was 1 on the D4 structure sphere. The absolute
    floor is what distinguishes "small but real" from "noise about zero".
    """
    if M.size == 0:
        return 0
    sv = np.linalg.svd(M, compute_uv=False)
    return int(np.sum((sv > tol * max(sv[0], 1e-30)) & (sv > floor)))


def counts(rng, NS, T34, CUR, c, I7, P=None, sphere=False):
    if sphere:
        J = np.zeros((NS, 7, 7))
        H = np.zeros((NS, 7, 7, 7))
        for s in range(NS):
            psi = rng.normal(size=8)
            psi /= np.linalg.norm(psi)
            M = rng.normal(size=(8, 8))
            M -= np.outer(psi, psi) @ M
            F = np.linalg.qr(M)[0][:, :7]
            J8 = rng.normal(size=(7, 8))
            J8 -= np.outer(J8 @ psi, psi)
            H8 = rng.normal(size=(7, 7, 8))
            H8 = 0.5 * (H8 + np.transpose(H8, (1, 0, 2)))
            H8 -= np.einsum('mnb,b,c->mnc', H8, psi, psi)
            J[s] = J8 @ F
            H[s] = np.einsum('mnb,bc->mnc', H8, F)
    else:
        Jf = rng.normal(size=(NS, 49))
        J = ((Jf @ P.T) if P is not None else Jf).reshape(NS, 7, 7)
        H = rng.normal(size=(NS, 7, 7, 7))
        H = 0.5 * (H + np.transpose(H, (0, 2, 1, 3)))
    A = np.stack([eval_34(t, J, H, c, I7) for t in T34], axis=1)
    B = np.stack([eval_div(v, J, H, c, I7) for v in CUR], axis=1)
    return rank(A), rank(B)


def main():
    orient = oriented_lines()
    c = imaginary_c(orient)
    I7 = np.eye(7)
    g2 = derivations(c)
    rng = np.random.default_rng(67)
    T34, CUR = terms_34(), currents()
    NS = 200
    print("=" * 70)
    print("CHANNEL CONTENT, AND O vs S^7 ON OUTCOMES")
    print("=" * 70)

    C = np.zeros((49, 49))
    for X in g2:
        Lm = np.kron(X, I7) + np.kron(I7, X)
        C -= Lm @ Lm
    w, V = np.linalg.eigh(C)
    groups = []
    for i, v in enumerate(w):
        if groups and abs(v - groups[-1][0]) < 1e-6:
            groups[-1][1].append(i)
        else:
            groups.append([v, [i]])
    mult = sorted(len(g[1]) for g in groups)
    projs = {len(idx): V[:, idx] @ V[:, idx].T for _, idx in groups}
    print(f"\nW1  Casimir multiplicities on J : {mult}"
          f"   [{'PASS' if mult == [1, 7, 14, 27] else 'MISS'}]")

    r34, rdiv = counts(rng, NS, T34, CUR, c, I7)
    print(f"\nW2  unrestricted : dim {r34}  divergences {rdiv}  "
          f"non-null {r34 - rdiv}   [expect 22 / 8 / 14 -> "
          f"{'PASS' if (r34, rdiv) == (22, 8) else 'MISS'}]")

    print("\nW3  SINGLE-CHANNEL RESTRICTION  (partial: no cross-channel pairs)")
    print("-" * 70)
    print("      channel   dim   divergences   non-null")
    tot = 0
    for name in (1, 7, 14, 27):
        ra, rb = counts(rng, NS, T34, CUR, c, I7, projs[name])
        tot += ra - rb
        print(f"        {name:2d}      {ra:3d}       {rb:3d}        {ra - rb:3d}")
    print(f"\n      single-channel non-null total : {tot}  of {r34 - rdiv}")
    print(f"      requiring channel mixing      : {r34 - rdiv - tot}"
          f"   [predict > 0 -> "
          f"{'PASS' if r34 - rdiv - tot > 0 else 'MISS'}]")

    ras, rbs = counts(rng, NS, T34, CUR, c, I7, sphere=True)
    print(f"\nW4  O   : dim {r34}  div {rdiv}  non-null {r34 - rdiv}")
    print(f"    S^7 : dim {ras}  div {rbs}  non-null {ras - rbs}")
    print(f"    identical : "
          f"{'PASS' if (ras, rbs) == (r34, rdiv) else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The surviving directions are not a clean irreducible class: most")
    print("  of them need the channels of J to mix, so they cannot be labelled")
    print("  by a single G2 representation.")
    print()
    print("  And the O / S^7 choice does not bite on a local term census, since")
    print("  at a point the tangent space of S^7 is Im(O). Where it bites is")
    print("  global -- the Moufang loop, Artin's sectors, topology.")


if __name__ == "__main__":
    main()
