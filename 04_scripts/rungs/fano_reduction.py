#!/usr/bin/env python3
"""
fano_reduction.py — A4: what triadic closure does and does not entail
(Shape Zero action log item A4)

Claims verified here:

1. Triadic closure (any two elements determine a unique third; triads closed
   under the pair->third map) is exactly the Steiner-triple-system axiom set.
2. COUNTEREXAMPLE: AG(2,3) — 9 points, 12 lines — satisfies triadic closure
   perfectly yet contains disjoint (parallel) triads. So triadic closure
   alone does NOT entail the Fano plane.
3. REDUCTION THEOREM (checked on the two concrete systems; proof in spec):
   within triadic closure + nondegeneracy, these are equivalent and each
   forces the unique STS(7) = Fano plane:
     (A) any two triads share an element      (global irreducibility)
     (B) every element lies in exactly 3 triads (local uniformity, r = 3)
"""

from itertools import combinations

def check_sts(points, lines, name):
    """Triadic closure = every pair of points on exactly one 3-element line."""
    ok_size = all(len(L) == 3 for L in lines)
    ok_pairs = all(sum(1 for L in lines if p in L and q in L) == 1
                   for p, q in combinations(points, 2))
    disjoint = [(L, M) for L, M in combinations(lines, 2) if not (L & M)]
    r_values = {p: sum(1 for L in lines if p in L) for p in points}
    print(f'{name}: n={len(points)}, lines={len(lines)}')
    print(f'  triadic closure (STS axioms): {ok_size and ok_pairs}')
    print(f'  disjoint triad pairs: {len(disjoint)}'
          + (f'  e.g. {sorted(disjoint[0][0])} | {sorted(disjoint[0][1])}'
             if disjoint else ''))
    print(f'  triads per element r: {sorted(set(r_values.values()))}')
    print(f'  axiom (A) any two triads meet: {not disjoint};'
          f'  axiom (B) r = 3 everywhere: {set(r_values.values()) == {3}}')
    return ok_size and ok_pairs

# --- AG(2,3): points Z3 x Z3; x,y,z collinear iff x + y + z = 0 -------------
pts9 = [(a, b) for a in range(3) for b in range(3)]
lines9 = set()
for p, q in combinations(pts9, 2):
    r = ((-p[0] - q[0]) % 3, (-p[1] - q[1]) % 3)   # unique third: x∘y = −x−y
    lines9.add(frozenset({p, q, r}))
assert check_sts(set(pts9), lines9, '\nAG(2,3)  [triadic closure holds, Fano fails]')

# --- Fano plane: points 1..7, lines = quadratic residue construction --------
lines7 = {frozenset({(i + s) % 7 for s in (0, 1, 3)}) for i in range(7)}
assert check_sts(set(range(7)), lines7, '\nFano STS(7)  [both supplements hold]')

print('\nConclusion: triadic closure alone admits AG(2,3) — the Fano plane is')
print('not entailed. Adding EITHER (A) or (B) forces n = 7 and the Fano plane')
print('(proof: (A) + a point p off a line L gives exactly the three lines pa,')
print('pb, pc through p, so r = 3; in any STS r = (n-1)/2, hence n = 7; the')
print('STS(7) is unique. (B) gives n = 2r + 1 = 7 directly.)')
