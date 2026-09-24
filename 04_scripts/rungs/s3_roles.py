#!/usr/bin/env python3
"""
s3_roles.py — S3: role-theoretic derivation of supplement (B)
(Shape Zero action log, Phase S)

Formalization of the genesis derivation: a triad IS the role set
{observer, observed, observation}. Two postulates, both direct
formalizations of the inside-up derivation:
  ROLE COMPLETENESS  every element occupies every role (superposition
                     clause: nothing is intrinsically role-restricted)
  ROLE MINIMALITY    each role exactly once per element (parameter-free
                     principle: no unforced multiplicity)
Together: exactly 3 triads per element = axiom (B) of the A4 reduction
=> n = 7 => the Fano plane, uniquely.

Machine checks:
 1. The required role structure EXISTS on the Fano plane: a proper
    3-edge-coloring of the 3-regular bipartite incidence graph
    (guaranteed by Konig; constructed explicitly here).
 2. AG(2,3) is EXCLUDED by pigeonhole: r = 4 incidences per point,
    3 roles => some role repeats => minimality violated. The A4
    counterexample is ruled out for a principled reason.
"""

from itertools import permutations, combinations

ROLES = ('observer', 'observed', 'observation')
lines = [tuple(sorted(((i+s) % 7) for s in (0, 1, 3))) for i in range(7)]

def color_fano():
    """Assign a role-permutation to each line s.t. no point repeats a role."""
    used = {p: set() for p in range(7)}
    assign = {}
    def bt(li):
        if li == len(lines):
            return True
        L = lines[li]
        for perm in permutations(range(3)):
            if all(perm[j] not in used[L[j]] for j in range(3)):
                for j in range(3): used[L[j]].add(perm[j])
                assign[L] = perm
                if bt(li + 1): return True
                for j in range(3): used[L[j]].discard(perm[j])
                del assign[L]
        return False
    return assign if bt(0) else None

if __name__ == '__main__':
    A = color_fano()
    assert A is not None
    print('Fano role structure exists (Konig, constructed):')
    for L, perm in sorted(A.items()):
        print('  triad', L, ' -> ', {L[j]: ROLES[perm[j]] for j in range(3)})
    # verify: each point occupies each role exactly once
    tally = {p: [0, 0, 0] for p in range(7)}
    for L, perm in A.items():
        for j in range(3): tally[L[j]][perm[j]] += 1
    assert all(v == [1, 1, 1] for v in tally.values())
    print('verified: every element occupies every role EXACTLY once (axiom B).')

    # AG(2,3): pigeonhole exclusion
    pts9 = [(a, b) for a in range(3) for b in range(3)]
    l9 = set()
    for p, q in combinations(pts9, 2):
        r = ((-p[0]-q[0]) % 3, (-p[1]-q[1]) % 3)
        l9.add(frozenset({p, q, r}))
    r_count = sum(1 for L in l9 if pts9[0] in L)
    print(f'\nAG(2,3): incidences per point r = {r_count} > 3 roles')
    print('=> pigeonhole: some role must repeat => role minimality violated.')
    print('The A4 counterexample is excluded by the postulates, not by fiat.')
    print('\nTheorem (modulo the two named postulates): the Fano plane is the')
    print('unique triadic-closure structure in which every distinction')
    print('realizes every role exactly once.')
