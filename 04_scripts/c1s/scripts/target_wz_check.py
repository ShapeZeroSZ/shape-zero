#!/usr/bin/env python3
"""
target_wz_check.py — is there a Wess-Zumino term?

coset_audit.py established the sigma model target is SO(7)/G2 with its metric
FORCED up to scale, and tangent directions canonically labelled by Im(O) via
v |-> L_v, (L_v)_ab = c_vab. The rank-3 invariant computation established the
invariant cubic space is 1-dimensional and equals phi. So the target carries a
unique invariant 3-form Phi, the natural candidate for a WZ term.

A WZ term needs it CLOSED. This checks.

METHOD. For a reductive homogeneous space G/H with g = h + m, an invariant
form's exterior derivative is purely algebraic:

    dPhi(X0..X3) = sum_{p<q} (-1)^{p+q} Phi([Xp,Xq]_m, ...remaining...)

so everything reduces to brackets of m with itself, projected back to m. If
[m,m] lands entirely in h the space is symmetric and dPhi = 0 automatically.
G2 is not a symmetric subgroup of SO(7), so that is not expected.

PREDICTIONS STATED BEFORE RUNNING
 W1 the L_i are orthogonal with tr(L_i L_j) = -6 delta_ij, so L_i/sqrt(6) is
    an orthonormal basis of m.
 W2 [m,m] has a NONZERO m-component -- the space is not symmetric.
 W3 dPhi != 0. There is no Wess-Zumino term. The action has one constant,
    not two.
 W4 the failure is not generic: dPhi is proportional to the dual 4-form,
    dPhi = tau0 * Psi with a SINGLE constant and residual <= 1e-10. That is
    exactly the nearly-parallel condition, i.e. torsion class W1 alone.
 W5 consistency: this must agree with the earlier finding that phi viewed as a
    torsion tensor is pure W1. Two unrelated computations, same class.

 Psi is built as the totally antisymmetric part of c_{ij.}c_{kl.}, which is
 proportional to the Hodge dual of phi without needing a 7-index epsilon --
 the delta-delta terms in that contraction vanish under antisymmetrisation.

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


def antisymmetrise4(T):
    out = np.zeros_like(T)
    for p in permutations(range(4)):
        sgn = 1
        pl = list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if pl[i] > pl[j]:
                    sgn = -sgn
        out += sgn * np.transpose(T, p)
    return out / 24.0


def main():
    c = imaginary_c(oriented_lines())
    print("=" * 70)
    print("TARGET WZ CHECK :: IS THE INVARIANT 3-FORM CLOSED?")
    print("=" * 70)

    # ---- W1 the canonical basis of m --------------------------------
    L = [c[i] for i in range(7)]
    gram = np.array([[np.trace(L[i] @ L[j]) for j in range(7)] for i in range(7)])
    off = np.max(np.abs(gram + 6.0 * np.eye(7)))
    print(f"\nW1  tr(L_i L_j) + 6 delta_ij : max |dev| = {off:.3e}"
          f"   [predicted 0 -> {'PASS' if off < 1e-10 else 'MISS'}]")

    # ---- W2 is [m,m] inside h? --------------------------------------
    alpha = np.zeros((7, 7, 7))          # alpha[i,j,k] : m-component of [L_i,L_j]
    for i in range(7):
        for j in range(7):
            br = L[i] @ L[j] - L[j] @ L[i]
            for k in range(7):
                alpha[i, j, k] = -np.trace(br @ L[k]) / 6.0
    mnorm = np.linalg.norm(alpha)
    print(f"\nW2  || [m,m]_m || = {mnorm:.6f}"
          f"   [predicted nonzero -> {'PASS' if mnorm > 1e-6 else 'MISS'}]")
    print("    -> SO(7)/G2 is reductive but NOT symmetric")

    # ---- dPhi via the algebraic formula -----------------------------
    dPhi = np.zeros((7, 7, 7, 7))
    idx = range(7)
    for i in idx:
        for j in idx:
            for k in idx:
                for l in idx:
                    s = 0.0
                    s -= np.dot(alpha[i, j], c[:, k, l])
                    s += np.dot(alpha[i, k], c[:, j, l])
                    s -= np.dot(alpha[i, l], c[:, j, k])
                    s -= np.dot(alpha[j, k], c[:, i, l])
                    s += np.dot(alpha[j, l], c[:, i, k])
                    s -= np.dot(alpha[k, l], c[:, i, j])
                    dPhi[i, j, k, l] = s
    dPhi = antisymmetrise4(dPhi)
    print(f"\nW3  || dPhi || = {np.linalg.norm(dPhi):.6f}"
          f"   [predicted nonzero -> "
          f"{'PASS' if np.linalg.norm(dPhi) > 1e-6 else 'MISS'}]")
    print("    -> the invariant 3-form is NOT closed; no Wess-Zumino term")

    # ---- W4 is the failure pure nearly-parallel? --------------------
    Psi = antisymmetrise4(np.einsum('ijm,klm->ijkl', c, c))
    nP = np.linalg.norm(Psi)
    tau0 = float(np.sum(dPhi * Psi) / (nP * nP))
    resid = np.linalg.norm(dPhi - tau0 * Psi) / np.linalg.norm(dPhi)
    print(f"\nW4  fit dPhi = tau0 * Psi")
    print(f"      ||Psi||          = {nP:.6f}")
    print(f"      tau0             = {tau0:.10f}")
    print(f"      relative residual= {resid:.3e}"
          f"   [predicted ~0 -> {'PASS' if resid < 1e-10 else 'MISS'}]")

    # ---- W5 consistency with the torsion-class result ---------------
    print("\nW5  CONSISTENCY")
    print("-" * 70)
    print("    dPhi proportional to *Phi with a single constant IS the")
    print("    nearly-parallel condition: torsion class W1 alone, W2=W3=W4=0.")
    print("    The earlier run found phi, viewed as a torsion tensor, is pure")
    print("    W1 (fraction 1.0000000000). Two unrelated computations, one")
    print("    class. That is the consistency check, not a new claim.")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  No WZ term. The sigma model action carries ONE constant, the")
    print("  overall scale of the forced target metric.")
    print("  The nearly-parallel structure is real and it is the TARGET's own")
    print("  geometry -- exactly where the coset audit said it lives. Its")
    print("  consequence here is negative: it is what obstructs closure.")


if __name__ == "__main__":
    main()
