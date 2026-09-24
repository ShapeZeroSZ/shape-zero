#!/usr/bin/env python3
"""
d8_sector_selection.py — does confinement to an associative sector select?

The three selection principles are stated for trajectories: couplings do no net
work, only bounded enduring motion is retained. The 14 surviving terms are
static terms on a 7-dimensional base with no time direction, so conservativity
and persistence cannot be applied as written without inventing an
interpretation. That is not done here.

What IS available is a criterion the rung supplies itself. Artin: any two
elements of O generate an associative subalgebra, so motion confined to one
Fano line's quaternion copy associates exactly, while motion crossing lines
does not (z1_d8_dynamics.py: bracketing spread 2.2e-16 within a line, growing
outside it). The arena therefore has seven associative sectors, and "enduring"
may not be one property: it may split into enduring WITHIN a sector and
enduring ACROSS sectors.

TEST. Restrict the field data to a Fano line's 3-dimensional subspace of
Im(O) -- both indices of J and all three of H, since base and target are
identified -- and recount the (3,4) invariants, the divergences among them, and
the non-null remainder. Comparing that with the unrestricted 22 / 8 / 14 shows
whether sector confinement selects, and by how much.

Note what the restriction does to the structure constants: c restricted to a
Fano line is the epsilon tensor of that quaternion copy. So the terms do not
vanish outright -- they degenerate to their quaternionic analogues, which is
exactly the D4 content. Any drop in the count is the measure of what is
genuinely octonionic rather than inherited.

PREDICTIONS STATED BEFORE RUNNING
 Y1 the unrestricted counts reproduce as 22 / 8 / 14 (instrument check).
 Y2 restricting to a Fano line collapses the invariant count well below 22,
    since a 3-dimensional index range supports far fewer contractions.
 Y3 the non-null count drops from 14 to a SMALL number -- predict <= 3.
    [MISSED: measured 6. The drop is real (14 -> 6) but less severe than
    predicted, so 8 terms are irreducibly octonionic rather than 11.]
 Y4 all seven Fano lines give the same counts, since they are equivalent under
    the automorphism group.
 Y5 therefore "enduring" splits: selection within a sector is much tighter than
    across sectors, and the answer to one/small/none is not a single number.

Python 3 + NumPy only. Uses d8_channels.py for the shared machinery.
"""

import numpy as np
import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "ch", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "d8_channels.py"))
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)

FANO = [tuple(sorted((((i + s - 1) % 7)) for s in (0, 1, 3)))
        for i in range(7)]


def line_projector(line):
    """Projector onto the 3-dimensional subspace of Im(O) spanned by a line."""
    P = np.zeros((7, 7))
    for i in line:
        P[i, i] = 1.0
    return P


def counts_restricted(rng, NS, T34, CUR, c, I7, P=None):
    J = rng.normal(size=(NS, 7, 7))
    H = rng.normal(size=(NS, 7, 7, 7))
    H = 0.5 * (H + np.transpose(H, (0, 2, 1, 3)))
    if P is not None:
        J = np.einsum('ij,zjk,kl->zil', P, J, P)
        H = np.einsum('ij,zjkl,km,ln->zimn', P, H, P, P)
    A = np.stack([ch.eval_34(t, J, H, c, I7) for t in T34], axis=1)
    B = np.stack([ch.eval_div(v, J, H, c, I7) for v in CUR], axis=1)
    return ch.rank(A), ch.rank(B)


def main():
    orient = ch.oriented_lines()
    c = ch.imaginary_c(orient)
    I7 = np.eye(7)
    T34, CUR = ch.terms_34(), ch.currents()
    rng = np.random.default_rng(83)
    NS = 200
    print("=" * 70)
    print("SECTOR SELECTION :: DOES CONFINEMENT SELECT AMONG THE 14?")
    print("=" * 70)

    r34, rdiv = counts_restricted(rng, NS, T34, CUR, c, I7)
    print(f"\nY1  unrestricted : dim {r34}  div {rdiv}  non-null {r34 - rdiv}"
          f"   [expect 22 / 8 / 14 -> "
          f"{'PASS' if (r34, rdiv) == (22, 8) else 'MISS'}]")

    print("\nY2/Y3/Y4  CONFINED TO EACH FANO LINE")
    print("-" * 70)
    print("      line          dim   divergences   non-null")
    res = []
    for k, ln in enumerate(FANO):
        P = line_projector(ln)
        ra, rb = counts_restricted(rng, NS, T34, CUR, c, I7, P)
        res.append((ra, rb))
        print(f"      {str(tuple(i+1 for i in ln)):12s}  {ra:3d}       {rb:3d}"
              f"          {ra - rb:3d}")
    nn = [a - b for a, b in res]
    same = len(set(res)) == 1
    print(f"\n    Y2 dim collapses below 22 : "
          f"{'PASS' if max(a for a, _ in res) < 22 else 'MISS'}")
    print(f"    Y3 non-null <= 3          : "
          f"{'PASS' if max(nn) <= 3 else 'MISS'}   (max {max(nn)})")
    print(f"    Y4 all seven lines agree  : {'PASS' if same else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print(f"  Across sectors : {r34 - rdiv} non-null terms.")
    print(f"  Within a sector: {nn[0]}.")
    print()
    if nn[0] < r34 - rdiv:
        print("  Confinement to an associative sector SELECTS. The difference")
        print("  is the genuinely octonionic content -- terms that need the")
        print("  field to leave a quaternion copy in order to act at all.")
        print()
        print("  So 'enduring' is not one property at D8. A trajectory that")
        print("  stays inside a Fano line sees a much smaller theory than one")
        print("  that crosses lines, and the two select differently. That is a")
        print("  distinction the lower rungs cannot make, because below D8")
        print("  every trajectory is in one sector by default.")
    else:
        print("  Confinement does not reduce the count -- the sectors do not")
        print("  select, and the split conjectured above does not occur.")


if __name__ == "__main__":
    main()
