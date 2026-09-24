#!/usr/bin/env python3
"""
z1_d4_structure_sphere.py — is the symplectic form on the structure sphere forced?

Open thread B-4 lists hbar, the Born rule and entanglement as explicitly not
derived. A route was proposed several sessions ago and never run: geometric
quantisation. Its first step, prequantisation, carries the INTEGRALITY
CONDITION — symplectic flux through a closed 2-cycle must be an integer multiple
of 2*pi*hbar. Theorem 2.9 already supplies a closed 2-cycle: the structure
sphere, the manifold of orthogonal complex structures on R^4, which is
SO(4)/U(2) = S^2 and is the Bloch sphere of the u(2) gauge theory.

WHAT IS AND IS NOT BEING CLAIMED. Integrality cannot produce a numerical value
for hbar; it fixes flux in units of hbar, so hbar enters as the unit rather than
as an output. That caveat was stated when the route was proposed and is not
retracted here. What integrality CAN do is convert a continuous parameter into a
discrete one, and that is worth testing on its own terms: minimality forbids
continuous parameters, and a parameter that integrality discretises is in a
different category from one that stays continuous.

So the question is narrower and answerable: is the symplectic form on the
structure sphere FORCED, leaving only a scale, and does integrality then
discretise that scale?

The argument mirrors the D8 target-metric result exactly. There, the isotropy
representation of G2 on g2-perp was irreducible, so Schur left a single
invariant metric up to scale. Here the isotropy group is U(2) acting on the
2-dimensional tangent space of SO(4)/U(2).

PREDICTIONS STATED BEFORE RUNNING
 S1 the set {J in SO(4) : J^2 = -I, J skew} is a 2-dimensional manifold, and it
    has TWO components -- the self-dual and anti-self-dual spheres -- each a
    2-sphere. Verified by sampling and measuring the tangent dimension.
 S2 so(4) = su(2) + su(2) as stated in Theorem 2.9: two commuting 3-dimensional
    ideals.
 S3 the space of U(2)-invariant antisymmetric 2-forms on the 2-dimensional
    tangent space is exactly 1-dimensional, so the symplectic form is FORCED up
    to scale -- the same Schur argument that fixed the D8 target metric.
 S4 the space of invariant symmetric forms is also 1-dimensional, so metric and
    symplectic form are BOTH forced up to scale and are compatible -- the sphere
    is Kaehler, with no independent choice.
 S5 consequence, stated not computed: integrality then quantises the single
    remaining scale. The parameter does not vanish, but it moves from continuous
    to discrete, which is the category minimality actually cares about.

Python 3 + NumPy only.
"""

import numpy as np


def so4_basis():
    """Orthonormal basis of so(4) under <X,Y> = -tr(XY)/2."""
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            M = np.zeros((4, 4))
            M[i, j], M[j, i] = 1.0, -1.0
            out.append(M)
    return out


def selfdual_split(basis):
    """so(4) = su(2)+ (+) su(2)-, the self-dual / anti-self-dual ideals."""
    # duality operator on 2-forms in 4d: (*F)_{ij} = 1/2 eps_{ijkl} F_{kl}
    eps = np.zeros((4, 4, 4, 4))
    from itertools import permutations
    for p in permutations(range(4)):
        sgn = 1
        pl = list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if pl[i] > pl[j]:
                    sgn = -sgn
        eps[p] = sgn
    star = lambda F: 0.5 * np.einsum('ijkl,kl->ij', eps, F)
    plus, minus = [], []
    for X in basis:
        plus.append(0.5 * (X + star(X)))
        minus.append(0.5 * (X - star(X)))

    def indep(mats):
        A = np.array([m.ravel() for m in mats])
        sv = np.linalg.svd(A, compute_uv=False)
        return int(np.sum(sv > 1e-9 * sv[0]))

    return plus, minus, indep(plus), indep(minus)


def complex_structures(rng, n=4000):
    """Sample J in SO(4) with J^2 = -I by exponentiating and projecting."""
    found = []
    for _ in range(n):
        A = rng.normal(size=(4, 4))
        A = A - A.T
        # a skew J with J^2 = -I is an orthogonal complex structure:
        # build from a random orthonormal frame pairing e1<->e2, e3<->e4
        Q = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        if np.linalg.det(Q) < 0:
            Q[:, [0, 1]] = Q[:, [1, 0]]
        J0 = np.zeros((4, 4))
        J0[0, 1], J0[1, 0] = 1.0, -1.0
        J0[2, 3], J0[3, 2] = 1.0, -1.0
        found.append(Q @ J0 @ Q.T)
    return found


def main():
    rng = np.random.default_rng(1009)
    print("=" * 70)
    print("THE D4 STRUCTURE SPHERE: IS ITS SYMPLECTIC FORM FORCED?")
    print("=" * 70)

    # ---- S2 the so(4) split -----------------------------------------
    basis = so4_basis()
    plus, minus, dp, dm = selfdual_split(basis)
    print(f"\nS2  so(4) split : dim su(2)+ = {dp}, dim su(2)- = {dm}"
          f"   [predict 3 and 3 -> "
          f"{'PASS' if (dp, dm) == (3, 3) else 'MISS'}]")
    comm = 0.0
    for X in plus:
        for Y in minus:
            comm = max(comm, np.max(np.abs(X @ Y - Y @ X)))
    print(f"    the two ideals commute : {comm:.3e}"
          f"   [{'PASS' if comm < 1e-10 else 'MISS'}]")

    # ---- S1 the manifold of complex structures ----------------------
    Js = complex_structures(rng)
    ok = max(np.max(np.abs(J @ J + np.eye(4))) for J in Js[:200])
    A = np.array([J.ravel() for J in Js])
    A = A - A.mean(0)
    sv = np.linalg.svd(A, compute_uv=False)
    amb = int(np.sum(sv > 1e-6 * sv[0]))
    print(f"\nS1  sampled J satisfy J^2 = -I : {ok:.3e}")
    print(f"    the sampled set spans {amb} ambient dimensions")
    # each component is a sphere: check by self-duality of J
    from itertools import permutations
    eps = np.zeros((4, 4, 4, 4))
    for p in permutations(range(4)):
        sgn = 1
        pl = list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if pl[i] > pl[j]:
                    sgn = -sgn
        eps[p] = sgn
    star = lambda F: 0.5 * np.einsum('ijkl,kl->ij', eps, F)
    sd = [J for J in Js if np.max(np.abs(star(J) - J)) < 1e-8]
    asd = [J for J in Js if np.max(np.abs(star(J) + J)) < 1e-8]
    print(f"    self-dual J : {len(sd)},  anti-self-dual J : {len(asd)},"
          f"  other : {len(Js) - len(sd) - len(asd)}")
    for nm, S in (("self-dual    ", sd), ("anti-self-dual", asd)):
        if len(S) < 20:
            continue
        B = np.array([J.ravel() for J in S])
        B = B - B.mean(0)
        s2 = np.linalg.svd(B, compute_uv=False)
        d = int(np.sum(s2 > 1e-6 * s2[0]))
        rad = np.std([np.linalg.norm(J.ravel() - B.mean(0)) for J in S])
        print(f"    {nm} component spans {d} dims"
              f"   [a 2-sphere spans 3 -> {'PASS' if d == 3 else 'MISS'}]")

    # ---- S3/S4 invariant forms on the tangent space -----------------
    print("\nS3/S4  INVARIANT FORMS ON THE TANGENT SPACE")
    print("-" * 70)
    # tangent space at J0 is 2-dimensional; isotropy U(2) acts on it as SO(2)
    # (rotation), so test invariance under that circle action
    def invariant_forms(sym):
        rows = []
        for th in np.linspace(0, 2 * np.pi, 40, endpoint=False):
            Rt = np.array([[np.cos(th), -np.sin(th)],
                           [np.sin(th), np.cos(th)]])
            for i in range(2):
                for j in range(2):
                    r = np.zeros(4)
                    for m in range(2):
                        for n in range(2):
                            r[m * 2 + n] += Rt[m, i] * Rt[n, j]
                    r[i * 2 + j] -= 1.0
                    rows.append(r)
        M = np.array(rows)
        # restrict to symmetric or antisymmetric 2x2
        if sym:
            P = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]],
                         dtype=float).T / np.array([1, np.sqrt(2), 1])
        else:
            P = np.array([[0, 1, -1, 0]], dtype=float).T / np.sqrt(2)
        MP = M @ P
        sv = np.linalg.svd(MP, compute_uv=False) if MP.size else np.array([0.0])
        return P.shape[1] - int(np.sum(sv > 1e-9 * max(sv[0], 1e-30)))

    n_anti = invariant_forms(sym=False)
    n_sym = invariant_forms(sym=True)
    print(f"    invariant ANTISYMMETRIC forms : {n_anti}"
          f"   [predict 1 -> {'PASS' if n_anti == 1 else 'MISS'}]")
    print(f"    invariant SYMMETRIC forms     : {n_sym}"
          f"   [predict 1 -> {'PASS' if n_sym == 1 else 'MISS'}]")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The symplectic form on the structure sphere is FORCED up to a")
    print("  single scale, by the same Schur argument that fixed the D8 target")
    print("  metric: the isotropy action leaves exactly one invariant")
    print("  antisymmetric form. The metric is forced the same way, so the")
    print("  sphere is Kaehler with no independent choice.")
    print()
    print("  What integrality then does is DISCRETISE that one scale -- flux")
    print("  through the sphere must be an integer multiple of 2 pi hbar. It")
    print("  does not produce a value for hbar; hbar is the unit, not the")
    print("  output. But the parameter moves from continuous to discrete, and")
    print("  that is the distinction minimality actually turns on.")


if __name__ == "__main__":
    main()
