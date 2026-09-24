#!/usr/bin/env python3
"""
a2_leg_ii.py — A-2 leg (ii)
(Shape Zero open threads item A-2, second leg; follows a2_invariance_hinge.py)

Leg (ii) as written: "transport that fails to preserve the multiplication does
not preserve the role assignment."

Contrapositive, which is the testable form:
    g preserves the role assignment  =>  g preserves the multiplication

This is what forces transport to be G2-valued rather than SO(7)-valued, so it
is load-bearing for the coupling class being SELECTED rather than CHOSEN.

Everything here is finite. The role assignment is a proper 3-edge-colouring of
the Heawood graph (the point-line incidence graph of the Fano plane); the
colours induce a cyclic order on each line, hence an orientation, hence the
structure constants. Transport candidates that could preserve a discrete
combinatorial structure are signed permutations: pi in Aut(Fano) acting on the
7 points, together with s in {+-1}^7.

THE CRUX is what "preserves the role assignment" means, and the answer differs:

  READING A (bare colouring). The role assignment is the colouring of the
  abstract incidence structure. Sign flips act on the algebra but not on the
  incidence combinatorics, so they preserve it trivially.

  READING B (colouring + induced orientation). The role assignment is the
  colouring together with the multiplication it induces on the actual basis.
  Preserving it means the colouring still yields the algebra you have.

PREDICTIONS STATED BEFORE RUNNING
 S1 proper 3-edge-colourings of the Heawood graph = 48 (the Koenig count).
 S2 all 48 induce composition algebras -- 48/48, against a 16/128 base rate.
 S3 they induce exactly 16 DISTINCT algebras, 3 colourings each, and those 16
    are precisely the valid sign patterns found by a2_invariance_hinge.py.
    (Reason: of the 6 colour relabellings, the 3 cyclic ones preserve the
    induced cyclic order and the 3 transpositions reverse it.)
 S4 |Aut(Fano)| = 168.
 S5 exactly 8 of the 128 sign patterns preserve a given c -- the kernel of the
    Fano incidence map over F2, whose binary rank is 4.
 S6 |B| = |{signed perms preserving c}| = 8 * 168 = 1344.
 S7 LEG (ii) TEST. Under Reading A, A is NOT contained in B: A contains all
    128 sign flips while B contains only 8. Leg (ii) FAILS on Reading A.
    Under Reading B it holds, but by construction.

Python 3 + NumPy only. Runtime a few seconds.
"""

import itertools
import numpy as np

FANO = [((i) % 7, (i + 1) % 7, (i + 3) % 7) for i in range(7)]


# ---------------------------------------------------------------- colourings
def incidences():
    return [(li, p) for li, L in enumerate(FANO) for p in L]


def enumerate_colourings():
    """Proper 3-edge-colourings of the Heawood graph, by backtracking."""
    inc = incidences()
    lines_of_point = {p: [li for li, L in enumerate(FANO) if p in L]
                      for p in range(7)}
    out = []
    col = {}

    def rec(k):
        if k == len(inc):
            out.append(dict(col))
            return
        li, p = inc[k]
        used_line = {col[(li, q)] for q in FANO[li] if (li, q) in col}
        used_pt = {col[(lj, p)] for lj in lines_of_point[p] if (lj, p) in col}
        for c in (0, 1, 2):
            if c not in used_line and c not in used_pt:
                col[(li, p)] = c
                rec(k + 1)
                del col[(li, p)]

    rec(0)
    return out


def colouring_to_c(chi):
    """Colours give a cyclic order on each line -> orientation -> constants."""
    c = np.zeros((7, 7, 7))
    for li, L in enumerate(FANO):
        order = sorted(L, key=lambda p: chi[(li, p)])
        a, b, d = order
        for (x, y, z), sg in [((a, b, d), 1), ((b, d, a), 1), ((d, a, b), 1),
                              ((b, a, d), -1), ((a, d, b), -1), ((d, b, a), -1)]:
            c[x, y, z] = sg
    return c


# ---------------------------------------------------------------- algebra
def mult(x, y, c):
    x0, xv = x[0], x[1:]
    y0, yv = y[0], y[1:]
    r0 = x0 * y0 - xv @ yv
    rv = x0 * yv + y0 * xv + np.einsum('ijk,i,j->k', c, xv, yv)
    return np.concatenate([[r0], rv])


def is_composition(c, trials=6, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        x, y = rng.normal(size=8), rng.normal(size=8)
        if abs(np.linalg.norm(mult(x, y, c))
               - np.linalg.norm(x) * np.linalg.norm(y)) > 1e-9:
            return False
    return True


# ---------------------------------------------------------------- groups
def fano_automorphisms():
    lineset = {frozenset(L) for L in FANO}
    out = []
    for pi in itertools.permutations(range(7)):
        if {frozenset(pi[p] for p in L) for L in FANO} == lineset:
            out.append(pi)
    return out


def act(c, pi, s):
    """Push c forward by the signed permutation e_p -> s_p e_{pi(p)}."""
    out = np.zeros((7, 7, 7))
    for i in range(7):
        for j in range(7):
            for k in range(7):
                out[pi[i], pi[j], pi[k]] = s[i] * s[j] * s[k] * c[i, j, k]
    return out


def preserves_c(c, pi, s):
    return np.allclose(act(c, pi, s), c, atol=1e-12)


def permute_colouring(chi, pi):
    """pi acts on points, hence on lines, hence on incidences."""
    lmap = {}
    for li, L in enumerate(FANO):
        img = frozenset(pi[p] for p in L)
        lmap[li] = next(lj for lj, M in enumerate(FANO) if frozenset(M) == img)
    return {(lmap[li], pi[p]): col for (li, p), col in chi.items()}


def main():
    print("=" * 70)
    print("A-2 LEG (ii) :: DOES ROLE-PRESERVATION IMPLY MULT-PRESERVATION?")
    print("=" * 70)

    cols = enumerate_colourings()
    print(f"\nS1  proper 3-edge-colourings : {len(cols)}"
          f"   [predicted 48 -> {'PASS' if len(cols) == 48 else 'MISS'}]")

    cs = [colouring_to_c(chi) for chi in cols]
    nvalid = sum(is_composition(c) for c in cs)
    print(f"S2  of those, composition algebras : {nvalid}/{len(cols)}"
          f"   [predicted all -> {'PASS' if nvalid == len(cols) else 'MISS'}]")

    keys = {c.tobytes() for c in cs}
    print(f"S3  distinct algebras induced : {len(keys)}"
          f"   [predicted 16 -> {'PASS' if len(keys) == 16 else 'MISS'}]"
          f"   ({len(cols)//max(len(keys),1)} colourings each)")

    auts = fano_automorphisms()
    print(f"S4  |Aut(Fano)| : {len(auts)}"
          f"   [predicted 168 -> {'PASS' if len(auts) == 168 else 'MISS'}]")

    c0 = cs[0]
    ident = tuple(range(7))
    sign_ok = [s for s in itertools.product([1, -1], repeat=7)
               if preserves_c(c0, ident, s)]
    print(f"S5  sign patterns preserving c : {len(sign_ok)}/128"
          f"   [predicted 8 -> {'PASS' if len(sign_ok) == 8 else 'MISS'}]")

    B = [(pi, s) for pi in auts for s in itertools.product([1, -1], repeat=7)
         if preserves_c(c0, pi, s)]
    print(f"S6  |B| = signed perms preserving c : {len(B)}"
          f"   [predicted 1344 -> {'PASS' if len(B) == 1344 else 'MISS'}]")

    print("\nS7  LEG (ii) TEST")
    print("-" * 70)
    chi0 = cols[0]
    stab_pi = [pi for pi in auts if permute_colouring(chi0, pi) == chi0]
    A_readingA = len(stab_pi) * 128
    print(f"  Reading A  (bare colouring)")
    print(f"    stabiliser of the colouring in Aut(Fano) : {len(stab_pi)}")
    print(f"    sign flips acting trivially on it        : 128")
    print(f"    |A| = {A_readingA}")
    inA_notB = [s for s in itertools.product([1, -1], repeat=7)
                if not preserves_c(c0, ident, s)]
    print(f"    elements of A that are NOT in B          : "
          f">= {len(inA_notB)} (sign flips alone)")
    print(f"    A contained in B ? {'YES' if not inA_notB else 'NO'}"
          f"  ->  leg (ii) {'holds' if not inA_notB else 'FAILS'}")

    print(f"\n  Reading B  (colouring + induced orientation)")
    A_readingB = [(pi, s) for (pi, s) in B]
    print(f"    role-preserving <=> colouring still yields the same c")
    print(f"    |A| = {len(A_readingB)}, |B| = {len(B)}, equal: "
          f"{len(A_readingB) == len(B)}")
    print(f"    A contained in B ? YES  ->  leg (ii) holds, by construction")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  Leg (ii) is not true or false as written -- it is reading-")
    print("  dependent. On the weak reading it fails outright. On the strong")
    print("  reading it holds but is definitional: the role assignment was")
    print("  built to determine the multiplication, so preserving one")
    print("  preserves the other by construction, not by argument.")


if __name__ == "__main__":
    main()
