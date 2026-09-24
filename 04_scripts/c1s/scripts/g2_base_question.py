#!/usr/bin/env python3
"""
g2_base_question.py — before searching for geometry, check which geometry

The Fernandez-Gray four-class structure (1 + 7 + 14 + 27) came out of
V (x) g2-perp with G2 acting on BOTH factors. That is only available when the
base tangent space IS the 7 that G2 acts on. Shape Zero's 7 is Im(O) -- the
imaginary part of an algebra, reached by a ladder over division-algebra
dimensions 1,2,4,8. The arena carried up that ladder is a 2-dimensional cone.
Same number, not obviously the same object.

PART 1 asks what the "distance from G2-valued transport" data looks like when
the base is d-dimensional and G2 does NOT act on it -- the bundle reading,
which is what a connection over the cone would be.

PART 2 is an instrument, not a result: given any so(7) element (a connection
component, or the generator of a Wilson loop), report how much of it lies
outside g2, and confirm that the failure to preserve the multiplication is
controlled by that part alone. This plugs into transport output the program
already produces, and tests leg (ii)'s replacement in the arena that exists
rather than in a 7-manifold that does not.

PREDICTIONS STATED BEFORE RUNNING
 G1 base d-dimensional, G2 not acting on the base: the space of G2-defects is
    7d-dimensional and decomposes as d copies of the 7. NO singlet, NO 14,
    NO 27 -- for any d, including d = 7.
 G2 base = V with G2 acting: 1 + 7 + 14 + 27, as before.
 G3 therefore W1, W2, W3 exist ONLY when the base is the 7 that G2 acts on.
    On a 2-dimensional cone the answer is two copies of the 7 and nothing else.
 G4 instrument check: for X in so(7), the multiplication defect ||X.c|| depends
    ONLY on the g2-perp component -- adding any g2 element leaves it unchanged
    (to 1e-10), and it vanishes exactly when the perp component does.

Python 3 + NumPy only.
"""

import numpy as np

FANO = [((i) % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]
TOL = 1e-9


def structure_constants(signs=(1,) * 7):
    c = np.zeros((7, 7, 7))
    for (a, b, d), s in zip(FANO, signs):
        for (x, y, z), sg in [((a, b, d), 1), ((b, d, a), 1), ((d, a, b), 1),
                              ((b, a, d), -1), ((a, d, b), -1), ((d, b, a), -1)]:
            c[x, y, z] = s * sg
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


def orthonormal_g2(D):
    Q, _ = np.linalg.qr(np.array([X.ravel() for X in D]).T)
    out = []
    for k in range(Q.shape[1]):
        X = Q[:, k].reshape(7, 7)
        X = 0.5 * (X - X.T)
        out.append(X / np.sqrt(-np.trace(X @ X)))
    return out


def so7_basis():
    out = []
    for b in range(7):
        for c_ in range(b + 1, 7):
            M = np.zeros((7, 7))
            M[b, c_], M[c_, b] = 1.0, -1.0
            out.append(M / np.sqrt(2.0))
    return out


def g2_perp(g2, so7):
    Q, _ = np.linalg.qr(np.array([X.ravel() for X in g2]).T)
    cols = [M.ravel() - Q @ (Q.T @ M.ravel()) for M in so7]
    U, s, _ = np.linalg.svd(np.array(cols).T)
    return [U[:, k].reshape(7, 7) for k in range(int(np.sum(s > TOL)))]


def cluster(vals, tol=1e-5):
    out = []
    for v in sorted(vals):
        if out and abs(v - out[-1][0]) < tol:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def casimir_multiplicities(g2, blocks, act):
    """Casimir spectrum on the span of `blocks`, using action `act`."""
    d = len(blocks)
    B = np.array([b.ravel() for b in blocks]).T
    Q, _ = np.linalg.qr(B)
    C = np.zeros((Q.shape[1], Q.shape[1]))
    for X in g2:
        L = np.zeros_like(C)
        for k in range(Q.shape[1]):
            L[:, k] = Q.T @ act(X, Q[:, k]).ravel()
        C -= L @ L
    return cluster(np.linalg.eigvalsh(C))


def main():
    c = structure_constants()
    g2 = orthonormal_g2(derivations(c))
    perp = g2_perp(g2, so7_basis())
    print("=" * 70)
    print("WHICH GEOMETRY? -- BASE DIMENSION DECIDES THE CLASSIFICATION")
    print("=" * 70)
    print(f"\n  dim g2 = {len(g2)}, dim g2-perp = {len(perp)}")

    # ---- PART 1a: bundle reading, G2 does NOT act on the base ------
    print("\nG1  BUNDLE READING: base d-dim, G2 acting only on the fibre")
    print("-" * 70)
    print("     d    dim of defect space    G2 content")
    for d in [1, 2, 3, 4, 7]:
        blocks = []
        for _ in range(d):
            blocks.extend(perp)                      # one copy of the 7 per direction
        # the decomposition is d copies of the 7 by construction; report the count
        print(f"    {d:2d}    {len(blocks):3d}"
              f"                 {d} x (the 7)")

    # verify the single copy really is the 7 (Casimir on g2-perp vs on V)
    def act_adj(X, v):
        M = v.reshape(7, 7)
        return X @ M - M @ X

    Cp = casimir_multiplicities(g2, perp, act_adj)
    CV = cluster(np.linalg.eigvalsh(-sum(X @ X for X in g2)))
    print(f"\n    Casimir on g2-perp : {[(round(a,4), b) for a, b in Cp]}")
    print(f"    Casimir on V       : {[(round(a,4), b) for a, b in CV]}")
    same = abs(Cp[0][0] - CV[0][0]) < 1e-6 and Cp[0][1] == 7
    print(f"    g2-perp is the 7   : {'PASS' if same else 'MISS'}")

    # ---- PART 1b: manifold reading, G2 acts on the base -----------
    print("\nG2  MANIFOLD READING: base = V, G2 acting on both factors")
    print("-" * 70)
    I7 = np.eye(7)
    C = np.zeros((49, 49))
    for X in g2:
        L = np.kron(X, I7) + np.kron(I7, X)
        C -= L @ L
    groups = cluster(np.linalg.eigvalsh(C))
    dims = sorted(g[1] for g in groups)
    print(f"    V (x) g2-perp = V (x) V : {dims}")
    print(f"    four classes present    : "
          f"{'PASS' if dims == [1, 7, 14, 27] else 'MISS'}")

    print("\nG3  READING")
    print("-" * 70)
    print("    On a 2-dimensional base the defect space is 14-dimensional and")
    print("    is two copies of the 7. There is no singlet, no 14, no 27 --")
    print("    W1, W2 and W3 do not exist there. The four-class structure is")
    print("    a fact about a base the group acts on, not about the algebra.")

    # ---- PART 2: the instrument -----------------------------------
    print("\nG4  INSTRUMENT :: G2-defect of an so(7) transport generator")
    print("-" * 70)
    Qg = np.linalg.qr(np.array([X.ravel() for X in g2]).T)[0]
    Qp = np.linalg.qr(np.array([X.ravel() for X in perp]).T)[0]

    def split(X):
        v = X.ravel()
        return (Qg @ (Qg.T @ v)).reshape(7, 7), (Qp @ (Qp.T @ v)).reshape(7, 7)

    def mult_defect(X):
        """||X . c|| -- how much the generator fails to preserve the product."""
        dc = (np.einsum('am,mbc->abc', X, c) + np.einsum('bm,amc->abc', X, c)
              + np.einsum('cm,abm->abc', X, c))
        return np.linalg.norm(dc) / np.linalg.norm(c)

    rng = np.random.default_rng(0)
    print("    trial   |X_g2|   |X_perp|   defect    defect after +g2 noise")
    ok = True
    for t in range(5):
        A = rng.normal(size=(7, 7))
        X = 0.5 * (A - A.T)
        Xg, Xp = split(X)
        d0 = mult_defect(X)
        B = rng.normal(size=len(g2))
        noise = sum(b * G for b, G in zip(B, g2))
        d1 = mult_defect(X + noise)
        ok &= abs(d0 - d1) < 1e-10
        print(f"     {t}     {np.linalg.norm(Xg):.4f}   {np.linalg.norm(Xp):.4f}"
              f"   {d0:.6f}   {d1:.6f}")
    print(f"\n    defect independent of the g2 part : {'PASS' if ok else 'MISS'}")
    pure_g2 = mult_defect(sum(rng.normal() * G for G in g2))
    print(f"    defect of a pure g2 element       : {pure_g2:.3e}"
          f"  {'PASS' if pure_g2 < 1e-10 else 'MISS'}")

    print("\n    Point any transport generator the program already produces at")
    print("    mult_defect(). Zero means G2-valued; nonzero measures leg (ii)")
    print("    failing in the arena that exists.")


if __name__ == "__main__":
    main()
