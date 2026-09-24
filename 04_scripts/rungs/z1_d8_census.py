#!/usr/bin/env python3
"""z1_d8_census.py — decisive D8 checks (log: D8 ATTEMPT record).
P3 sign-corrected triadic grading; P4 holonomy sector dim; P5 THE CENSUS:
all Konig role colorings vs all 128 orientations. Established: 16/128
orientations valid; ALL role colorings valid, 3:1 with the 16."""
import numpy as np
from itertools import permutations, product
LINES = [tuple(sorted(((i+s-1)%7)+1 for s in (0,1,3))) for i in range(1,8)]
rng = np.random.default_rng(11)
def eps_from(oriented):
    E = np.zeros((8,8,8)); E[0,:,:] = np.eye(8); E[:,0,:] = np.eye(8)
    for i in range(1,8): E[i,i,0] = -1
    for (a,b,c) in oriented:
        for x,y,z in ((a,b,c),(b,c,a),(c,a,b)): E[x,y,z]=1; E[y,x,z]=-1
    return E
mulE = lambda E,u,v: np.einsum('ijk,i,j->k',E,u,v)
def valid(E,n=8):
    for _ in range(n):
        x,y = rng.normal(size=8), rng.normal(size=8)
        if abs(np.linalg.norm(mulE(E,x,y))-np.linalg.norm(x)*np.linalg.norm(y))>1e-9:
            return False
    return True
STD = [(i,(i%7)+1,((i+2)%7)+1) for i in range(1,8)]
E0 = eps_from(STD); I8 = np.eye(8)
Lm = lambda a: np.column_stack([mulE(E0,a,I8[j]) for j in range(8)])
la,lb = 1,2; prod_ = mulE(E0,I8[la],I8[lb]); k = int(np.argmax(np.abs(prod_)))
G = np.sign(prod_[k])*np.linalg.inv(Lm(I8[k]))@Lm(I8[la])@Lm(I8[lb])
ev,V = np.linalg.eigh((G+G.T)/2)
plus = V[:,ev>0.5]; plane = np.column_stack([I8[0],I8[la],I8[lb],I8[k]])
sv = np.linalg.svd(plus.T@plane, compute_uv=False)
print(f'P3 grading: split {np.sum(ev>0.5)}+{np.sum(ev<-0.5)}; cosines {np.round(sv,6)}')
Ls = [Lm(I8[i]) for i in range(1,8)]; basis=[]
def add(M):
    v=M.flatten()
    for b0 in basis: v=v-(v@b0)*b0
    if np.linalg.norm(v)>1e-8: basis.append(v/np.linalg.norm(v)); return True
    return False
for i in range(7):
    for j in range(i+1,7): add(Ls[i]@Ls[j]-Ls[j]@Ls[i])
print(f'P4 holonomy dim = {len(basis)} (pred 21)')
nval = sum(valid(eps_from([(a,b,c) if s==0 else (a,c,b)
            for (a,b,c),s in zip(LINES,bits)])) for bits in product((0,1),repeat=7)
           for _ in [0])
print(f'P5 base rate: {nval}/128 valid')
sols=[]; used={p:set() for p in range(1,8)}; assign={}
def bt(li):
    if li==len(LINES): sols.append(dict(assign)); return
    Ln=LINES[li]
    for perm in permutations(range(3)):
        if all(perm[j] not in used[Ln[j]] for j in range(3)):
            for j in range(3): used[Ln[j]].add(perm[j])
            assign[Ln]=perm; bt(li+1)
            for j in range(3): used[Ln[j]].discard(perm[j])
            del assign[Ln]
bt(0)
nv = sum(valid(eps_from([tuple(Ln[A[Ln].index(r)] for r in (0,1,2))
         for Ln in LINES]), n=6) for A in sols)
print(f'P5 census: {nv}/{len(sols)} role colorings induce valid octonion algebras')
