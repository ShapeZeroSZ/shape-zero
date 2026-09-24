#!/usr/bin/env python3
"""
a2_unification_audit.py — auditing an unverified claim, and one new result

PART A audits a claim I asserted without testing: that the four constructible
rank-4 couplings (leg i) and the four intrinsic torsion classes (leg ii's
replacement) are the SAME decomposition of V (x) V, not two facts that happen
to both give 4. Both give 4 because 7 (x) 7 = 1 + 7 + 14 + 27 -- but matching
dimensions is exactly the kind of coincidence this whole exercise has been
catching. If the couplings are not the irrep projectors, the "one decomposition,
two consequences" framing is wrong and should not go into C2.

PART B is new, and settles something B-7 raised. Under SO(7),
    V (x) Lambda^2 V  =  7  +  35  +  105
where the 35 = Lambda^3 V is the totally antisymmetric part -- the piece
Einstein-Cartan torsion is sourced into by spin density. Under G2 the full 147
splits as gauge (98 = 7 + 27 + 64) plus intrinsic (49 = 1 + 7 + 14 + 27).
Note 7 and 27 appear on BOTH sides, so which part of Lambda^3 V survives as
intrinsic is NOT determined by irrep type and has to be computed.

PREDICTIONS STATED BEFORE RUNNING
 U1 the Casimir on V (x) V has 4 eigenvalues with multiplicities 1, 7, 14, 27.
 U2 the four irrep projectors, viewed as rank-4 tensors, span exactly the same
    4-dimensional space as the four constructible invariants
      d_ij d_kl,  d_ik d_jl,  d_il d_jk,  sum_m c_ijm c_klm
    (mutual projection residual <= 1e-8). THIS IS THE AUDIT. A miss refutes
    the unification claim.
 U3 the four Casimir eigenvalues on the intrinsic torsion space equal the four
    on V (x) V, so the identification is canonical rather than dimensional.
 U4 Lambda^3 V has dimension 35 and contains NO copy of the 14. Therefore its
    image in the intrinsic space has zero overlap with W2. Spin-sourced torsion
    cannot populate W2 by any mechanism.
 U5 the 147 has a unique G2 singlet, and it lies in the intrinsic part, so the
    singlet of Lambda^3 V -- which is the 3-form phi itself -- is pure W1.
 NOT PREDICTED: how the 7 and 27 pieces of Lambda^3 V divide between gauge and
    intrinsic. Computed, not guessed.

Python 3 + NumPy only.
"""

import numpy as np

FANO = [((i) % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]
PAIRS = [(b, c) for b in range(7) for c in range(b + 1, 7)]
TOL = 1e-9


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


def is_composition(c, trials=5, seed=3):
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
    _, s, Vt = np.linalg.svd(np.array(rows))
    return [v.reshape(7, 7) for v in Vt[np.sum(s > TOL):]]


def orthonormal_g2(D):
    B = np.array([X.ravel() for X in D]).T
    Q, _ = np.linalg.qr(B)
    out = []
    for k in range(Q.shape[1]):
        X = Q[:, k].reshape(7, 7)
        X = 0.5 * (X - X.T)
        out.append(X / np.sqrt(-np.trace(X @ X)))
    return out


def cluster(vals, tol=1e-5):
    out = []
    for v in sorted(vals):
        if out and abs(v - out[-1][0]) < tol:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


# ---------------------------------------------------------------- PART A
def casimir_VxV(g2):
    I = np.eye(7)
    C = np.zeros((49, 49))
    for X in g2:
        L = np.kron(X, I) + np.kron(I, X)
        C -= L @ L
    return C


def irrep_projectors(g2):
    C = casimir_VxV(g2)
    w, V = np.linalg.eigh(C)
    groups, out = cluster(w), []
    idx = 0
    for val, mult_ in groups:
        blk = V[:, idx:idx + mult_]
        out.append((val, mult_, blk @ blk.T))
        idx += mult_
    return out


def constructible4(c):
    d = np.eye(7)
    return [np.einsum('ij,kl->ijkl', d, d).ravel(),
            np.einsum('ik,jl->ijkl', d, d).ravel(),
            np.einsum('il,jk->ijkl', d, d).ravel(),
            np.einsum('ijm,klm->ijkl', c, c).ravel()]


def span_gap(A_cols, B_cols):
    """Max residual of each A column outside span(B), and vice versa."""
    QA, _ = np.linalg.qr(np.array(A_cols).T)
    QB, _ = np.linalg.qr(np.array(B_cols).T)
    r1 = np.max(np.abs(QA - QB @ (QB.T @ QA)))
    r2 = np.max(np.abs(QB - QA @ (QA.T @ QB)))
    return max(r1, r2)


# ---------------------------------------------------------------- PART B
def flat(T):
    return np.array([T[a, b, c] for a in range(7) for (b, c) in PAIRS])


def unflat(v):
    T = np.zeros((7, 7, 7))
    i = 0
    for a in range(7):
        for (b, c) in PAIRS:
            T[a, b, c], T[a, c, b] = v[i], -v[i]
            i += 1
    return T


def delta_matrix(g2):
    M = np.zeros((147, 98))
    for bi in range(7):
        for k, G in enumerate(g2):
            T = np.zeros((7, 7, 7))
            for a in range(7):
                for b in range(7):
                    for cc in range(7):
                        T[a, b, cc] = (b == bi) * G[a, cc] - (cc == bi) * G[a, b]
            M[:, bi * 14 + k] = flat(T)
    return M


def act_T(X, T):
    return (np.einsum('am,mbc->abc', X, T) + np.einsum('bm,amc->abc', X, T)
            + np.einsum('cm,abm->abc', X, T))


def casimir_on(g2, cols):
    d = cols.shape[1]
    C = np.zeros((d, d))
    for X in g2:
        L = np.zeros((d, d))
        for k in range(d):
            L[:, k] = cols.T @ flat(act_T(X, unflat(cols[:, k])))
        C -= L @ L
    return C


def lambda3_basis():
    out = []
    for a in range(7):
        for b in range(a + 1, 7):
            for cc in range(b + 1, 7):
                T = np.zeros((7, 7, 7))
                for (x, y, z), sg in [((a, b, cc), 1), ((b, cc, a), 1),
                                      ((cc, a, b), 1), ((b, a, cc), -1),
                                      ((a, cc, b), -1), ((cc, b, a), -1)]:
                    T[x, y, z] = sg
                out.append(flat(T))
    return np.array(out).T


def main():
    print("=" * 70)
    print("A-2 UNIFICATION AUDIT")
    print("=" * 70)

    valid = []
    for mask in range(128):
        c = structure_constants([1 if (mask >> b) & 1 == 0 else -1
                                 for b in range(7)])
        if is_composition(c):
            valid.append((mask, c))

    print("\nPART A  -- do the rank-4 couplings ARE the irrep projectors?")
    print("-" * 70)
    print("  mask   Casimir mults on VxV      span gap (couplings vs projectors)")
    okA = True
    for mask, c in valid:
        g2 = orthonormal_g2(derivations(c))
        projs = irrep_projectors(g2)
        mults = [p[1] for p in projs]
        gap = span_gap(constructible4(c), [p[2].ravel() for p in projs])
        okA &= (sorted(mults) == [1, 7, 14, 27] and gap < 1e-8)
        print(f"   {mask:3d}   {str(sorted(mults)):18s}   {gap:.3e}")
    print(f"\n  U1 multiplicities 1,7,14,27 : {'PASS' if okA else 'MISS'}")
    print(f"  U2 SPANS COINCIDE (the audit): {'PASS' if okA else 'MISS'}")

    # ---- canonical identification + Part B on one algebra ----------
    mask, c = valid[0]
    g2 = orthonormal_g2(derivations(c))
    projs = irrep_projectors(g2)
    ev_VxV = sorted(round(p[0], 6) for p in projs)

    D = delta_matrix(g2)
    rank = np.linalg.matrix_rank(D, tol=1e-8)
    U, s, _ = np.linalg.svd(D, full_matrices=True)
    coker = U[:, rank:]
    Cint = casimir_on(g2, coker)
    w, Vv = np.linalg.eigh(Cint)
    gint = cluster(w)
    ev_int = sorted(round(g[0], 6) for g in gint)

    print("\n  U3 Casimir eigenvalues, V(x)V vs intrinsic torsion")
    print("-" * 70)
    print(f"     V (x) V   : {ev_VxV}")
    print(f"     intrinsic : {ev_int}")
    same = all(abs(a - b) < 1e-4 for a, b in zip(ev_VxV, ev_int))
    print(f"     identical : {'PASS' if same else 'MISS'}")

    print("\nPART B  -- where does spin-sourced (totally antisymmetric)")
    print("           torsion land?")
    print("-" * 70)
    L3 = lambda3_basis()
    QL, _ = np.linalg.qr(L3)
    print(f"  dim Lambda^3 V : {QL.shape[1]}  [expect 35]")

    proj_int = coker @ coker.T
    img = proj_int @ QL
    U2_, s2, _ = np.linalg.svd(img)
    dim_img = int(np.sum(s2 > 1e-8))
    print(f"  dim of its image in the intrinsic 49 : {dim_img}")
    print(f"  therefore absorbable as gauge        : {QL.shape[1] - dim_img}")

    # classify the image by torsion class
    print("\n  distribution across the four classes:")
    idx = 0
    labels = {1: 'W1 (1)', 7: 'W4 (7)', 14: 'W2 (14)', 27: 'W3 (27)'}
    for val, mult_ in gint:
        blk = coker @ Vv[:, idx:idx + mult_]
        overlap = np.linalg.norm(blk.T @ QL)
        rank_here = int(np.sum(np.linalg.svd(blk.T @ QL)[1] > 1e-8))
        print(f"    {labels.get(mult_, str(mult_)):10s} "
              f"dim {mult_:3d}   reached dim {rank_here:3d}   "
              f"norm {overlap:.3e}")
        idx += mult_

    print("\n  U4 W2 unreachable from Lambda^3 V : ", end="")
    idx, okU4 = 0, True
    for val, mult_ in gint:
        if mult_ == 14:
            blk = coker @ Vv[:, idx:idx + mult_]
            okU4 = np.linalg.norm(blk.T @ QL) < 1e-8
        idx += mult_
    print('PASS' if okU4 else 'MISS')

    # U5: phi itself
    phi = flat(c)
    resid = phi - proj_int @ phi
    idx, in_W1 = 0, 0.0
    for val, mult_ in gint:
        if mult_ == 1:
            blk = coker @ Vv[:, idx:idx + 1]
            in_W1 = np.linalg.norm(blk.T @ phi) / np.linalg.norm(phi)
        idx += mult_
    print(f"  U5 phi as torsion: intrinsic residual {np.linalg.norm(resid):.3e},"
          f" fraction in W1 = {in_W1:.10f}")
    print(f"     phi is pure W1 : {'PASS' if abs(in_W1 - 1.0) < 1e-8 else 'MISS'}")


if __name__ == "__main__":
    main()
