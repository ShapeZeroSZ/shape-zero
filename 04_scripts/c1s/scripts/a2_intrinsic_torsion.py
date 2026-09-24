#!/usr/bin/env python3
"""
a2_intrinsic_torsion.py — the replacement for A-2 leg (ii)
(follows a2_leg_ii.py, which showed the leg as written is either false or
definitional; this computes the non-circular question it should have asked)

Leg (ii) was meant to constrain TRANSPORT, not couplings. The non-circular
form of that question is whether the octonion structure is background or
dynamical -- equivalently, whether the connection preserves the multiplication.
The obstruction has a standard name: the INTRINSIC TORSION of the G2-structure.

Nothing here needs a manifold. The whole classification is linear algebra over
the structure constants, and it answers a question the program can act on:
HOW MANY independent ways can transport fail to be G2-valued.

THE SPLIT. A general torsion tensor T^a_{bc} (antisymmetric in bc) lives in
V (x) Lambda^2 V, dimension 7 * 21 = 147. Changing to a different G2-compatible
connection shifts T by the image of

    delta : V* (x) g2  ->  V (x) Lambda^2 V ,    delta(A)^a_{bc} = A_b{}^a{}_c - A_c{}^a{}_b

whose domain has dimension 7 * 14 = 98. Whatever survives in the cokernel
CANNOT be removed by re-choosing the connection. That cokernel is the
intrinsic torsion.

PREDICTIONS STATED BEFORE RUNNING
 T1 dim g2 = 14 and dim g2-perp = 7 inside so(7) = 21, for all 16 algebras.
 T2 g2-perp is isomorphic to V as a G2-representation: its Casimir eigenvalue
    equals the Casimir eigenvalue on V.
 T3 delta is INJECTIVE -- rank exactly 98. (If it were not, some connection
    changes would be invisible in the torsion and the count below would fail.)
 T4 cokernel dimension = 147 - 98 = 49, matching V (x) g2-perp = 7 (x) 7.
 T5 the cokernel decomposes under G2 into exactly FOUR irreducibles with
    dimensions 1, 7, 14, 27 -- the Fernandez-Gray classes W1..W4 -- read off
    as Casimir eigenvalue multiplicities.
 T6 all of the above identical across all 16 valid algebras.

Python 3 + NumPy only. Runtime a few seconds.
"""

import itertools
import numpy as np

FANO = [((i) % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]
PAIRS = [(b, c) for b in range(7) for c in range(b + 1, 7)]      # 21
TOL = 1e-9


# ---------------------------------------------------------------- algebra
def structure_constants(signs):
    c = np.zeros((7, 7, 7))
    for (a, b, d), s in zip(FANO, signs):
        for (x, y, z), sg in [((a, b, d), 1), ((b, d, a), 1), ((d, a, b), 1),
                              ((b, a, d), -1), ((a, d, b), -1), ((d, b, a), -1)]:
            c[x, y, z] = s * sg
    return c


def mult(x, y, c):
    x0, xv, y0, yv = x[0], x[1:], y[0], y[1:]
    return np.concatenate([[x0 * y0 - xv @ yv],
                           x0 * yv + y0 * xv + np.einsum('ijk,i,j->k', c, xv, yv)])


def is_composition(c, trials=5, seed=2):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        x, y = rng.normal(size=8), rng.normal(size=8)
        if abs(np.linalg.norm(mult(x, y, c))
               - np.linalg.norm(x) * np.linalg.norm(y)) > 1e-9:
            return False
    return True


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
    A = np.array(rows)
    _, s, Vt = np.linalg.svd(A)
    return [v.reshape(7, 7) for v in Vt[np.sum(s > TOL):]]


def orthonormal_g2(D):
    """Orthonormalise w.r.t. <X,Y> = -tr(XY), positive definite on so(7)."""
    B = np.array([X.ravel() for X in D]).T
    Q, _ = np.linalg.qr(B)
    out = []
    for k in range(Q.shape[1]):
        X = Q[:, k].reshape(7, 7)
        X = 0.5 * (X - X.T)
        X = X / np.sqrt(-np.trace(X @ X))
        out.append(X)
    return out


# ---------------------------------------------------------------- spaces
def so7_basis():
    out = []
    for b, c in PAIRS:
        M = np.zeros((7, 7))
        M[b, c], M[c, b] = 1.0, -1.0
        out.append(M / np.sqrt(2.0))
    return out


def perp_complement(g2, so7):
    """g2-perp inside so(7), w.r.t. the trace form."""
    G = np.array([X.ravel() for X in g2]).T
    Q, _ = np.linalg.qr(G)
    cols = []
    for M in so7:
        v = M.ravel()
        cols.append(v - Q @ (Q.T @ v))
    A = np.array(cols).T
    U, s, _ = np.linalg.svd(A)
    return [U[:, k].reshape(7, 7) for k in range(int(np.sum(s > TOL)))]


def flatten_T(T):
    return np.array([T[a, b, c] for a in range(7) for (b, c) in PAIRS])


def unflatten_T(v):
    T = np.zeros((7, 7, 7))
    i = 0
    for a in range(7):
        for (b, c) in PAIRS:
            T[a, b, c], T[a, c, b] = v[i], -v[i]
            i += 1
    return T


def delta_matrix(g2):
    """V* (x) g2 -> V (x) Lambda^2 V."""
    M = np.zeros((147, 98))
    for bi in range(7):
        for k, G in enumerate(g2):
            T = np.zeros((7, 7, 7))
            for a in range(7):
                for b in range(7):
                    for cc in range(7):
                        T[a, b, cc] = ((b == bi) * G[a, cc]
                                       - (cc == bi) * G[a, b])
            M[:, bi * 14 + k] = flatten_T(T)
    return M


def act_on_T(X, T):
    return (np.einsum('am,mbc->abc', X, T)
            + np.einsum('bm,amc->abc', X, T)
            + np.einsum('cm,abm->abc', X, T))


def casimir_on_space(g2, basis_cols):
    """Restrict -sum rho(X)^2 to the span of basis_cols (147 x d, orthonormal)."""
    d = basis_cols.shape[1]
    C = np.zeros((d, d))
    for X in g2:
        L = np.zeros((d, d))
        for k in range(d):
            T = unflatten_T(basis_cols[:, k])
            L[:, k] = basis_cols.T @ flatten_T(act_on_T(X, T))
        C -= L @ L
    return C


def casimir_on_V(g2):
    C = np.zeros((7, 7))
    for X in g2:
        C -= X @ X
    return C


def cluster(vals, tol=1e-6):
    out = []
    for v in sorted(vals):
        if out and abs(v - out[-1][0]) < tol:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


# ---------------------------------------------------------------- run
def main():
    print("=" * 70)
    print("A-2 LEG (ii) REPLACEMENT :: INTRINSIC TORSION OF THE G2-STRUCTURE")
    print("=" * 70)

    valid = []
    for mask in range(128):
        signs = [1 if (mask >> b) & 1 == 0 else -1 for b in range(7)]
        c = structure_constants(signs)
        if is_composition(c):
            valid.append((mask, c))
    print(f"\n  valid algebras: {len(valid)}")

    so7 = so7_basis()
    all_ok = True
    summary = []

    for mask, c in valid:
        g2 = orthonormal_g2(derivations(c))
        perp = perp_complement(g2, so7)

        # T2: Casimir on g2-perp vs on V
        P = np.array([X.ravel() for X in perp]).T
        Q, _ = np.linalg.qr(P)
        Cp = np.zeros((len(perp), len(perp)))
        for X in g2:
            L = np.zeros((len(perp), len(perp)))
            for k in range(len(perp)):
                M = Q[:, k].reshape(7, 7)
                L[:, k] = Q.T @ (X @ M - M @ X).ravel()
            Cp -= L @ L
        ev_perp = cluster(np.linalg.eigvalsh(Cp))
        ev_V = cluster(np.linalg.eigvalsh(casimir_on_V(g2)))

        # T3/T4: delta and its cokernel
        D = delta_matrix(g2)
        rank = np.linalg.matrix_rank(D, tol=1e-8)
        U, s, _ = np.linalg.svd(D, full_matrices=True)
        coker = U[:, rank:]

        # T5: decompose the cokernel
        C = casimir_on_space(g2, coker)
        groups = cluster(np.linalg.eigvalsh(C), tol=1e-5)
        dims = [g[1] for g in groups]

        ok = (len(g2) == 14 and len(perp) == 7 and rank == 98
              and coker.shape[1] == 49 and sorted(dims) == [1, 7, 14, 27]
              and abs(ev_perp[0][0] - ev_V[0][0]) < 1e-6)
        all_ok &= ok
        summary.append((mask, len(g2), len(perp), rank, coker.shape[1],
                        sorted(dims), ok))

    print("\n  mask  dim g2  perp  rank(delta)  coker   irrep dims       ok")
    print("-" * 70)
    for m, dg, dp, r, dc, dims, ok in summary:
        print(f"   {m:3d}     {dg:3d}   {dp:3d}      {r:4d}      {dc:3d}"
              f"   {str(dims):16s} {'PASS' if ok else 'MISS'}")

    m0 = summary[0]
    print(f"\n  T1 dim g2 = 14, dim g2-perp = 7        : "
          f"{'PASS' if m0[1] == 14 and m0[2] == 7 else 'MISS'}")
    print(f"  T2 g2-perp ~ V (matching Casimir)      : PASS"
          if all_ok else "  T2 : see per-row")
    print(f"  T3 delta injective, rank 98            : "
          f"{'PASS' if m0[3] == 98 else 'MISS'}")
    print(f"  T4 cokernel = 147 - 98 = 49            : "
          f"{'PASS' if m0[4] == 49 else 'MISS'}")
    print(f"  T5 decomposes as 1 + 7 + 14 + 27       : "
          f"{'PASS' if m0[5] == [1, 7, 14, 27] else 'MISS'}")
    print(f"  T6 identical across all 16 algebras    : "
          f"{'PASS' if all_ok else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  Of 147 torsion components, 98 are absorbable by re-choosing the")
    print("  G2-compatible connection and 49 are not. The 49 split into four")
    print("  irreducible classes. So 'is the octonion structure background or")
    print("  dynamical' is not a yes/no: it has exactly four independent")
    print("  switches, and 2^4 = 16 possible partial closures. Any replacement")
    print("  for leg (ii) has to say which of the four it is turning off.")


if __name__ == "__main__":
    main()
