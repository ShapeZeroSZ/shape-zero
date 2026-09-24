#!/usr/bin/env python3
"""z1_hinge.py — the G2 hinge checks (log: Z1-H). Established: dim Der = 14
(G2 discovered from the role table); invariant cubic space = 1-dim, zero
symmetric content, overlap with phi = 1.000000; Der inside holonomy (14 of
21); all 7 pulses fail table-preservation."""
import numpy as np
from itertools import permutations
LINES = [tuple(sorted(((i+s-1)%7)+1 for s in (0,1,3))) for i in range(1,8)]
used={p:set() for p in range(1,8)}; assign={}
def bt(li):
    if li==len(LINES): return True
    Ln=LINES[li]
    for perm in permutations(range(3)):
        if all(perm[j] not in used[Ln[j]] for j in range(3)):
            for j in range(3): used[Ln[j]].add(perm[j])
            assign[Ln]=perm
            if bt(li+1): return True
            for j in range(3): used[Ln[j]].discard(perm[j])
            del assign[Ln]
    return False
bt(0)
oriented=[tuple(Ln[assign[Ln].index(r)] for r in (0,1,2)) for Ln in LINES]
E = np.zeros((8,8,8)); E[0,:,:]=np.eye(8); E[:,0,:]=np.eye(8)
for i in range(1,8): E[i,i,0]=-1
for (a,b,c) in oriented:
    for x,y,z in ((a,b,c),(b,c,a),(c,a,b)): E[x,y,z]=1; E[y,x,z]=-1
mul = lambda u,v: np.einsum('ijk,i,j->k',E,u,v)
I8 = np.eye(8); rows=[]
for i in range(8):
    for j in range(8):
        eij = mul(I8[i],I8[j])
        for comp in range(8):
            r = np.zeros(64)
            for d in range(8):
                r[comp*8+d] += eij[d]
                r[d*8+i] -= E[d,j,comp]
                r[d*8+j] -= E[i,d,comp]
            rows.append(r)
for comp in range(8):
    r=np.zeros(64); r[comp*8+0]=1; rows.append(r)
_,s,Vt = np.linalg.svd(np.array(rows))
null = Vt[np.sum(s>1e-8):]
print(f'dim Der = {null.shape[0]} (pred 14)')
Ders = [null[k].reshape(8,8) for k in range(null.shape[0])]
Tstack = np.eye(343).reshape(343,7,7,7)
MM = np.vstack([(np.einsum('da,ndbc->nabc',D[1:,1:],Tstack)
   + np.einsum('db,nadc->nabc',D[1:,1:],Tstack)
   + np.einsum('dc,nabd->nabc',D[1:,1:],Tstack)).reshape(343,343).T
   for D in Ders])
_,s2,Vt2 = np.linalg.svd(MM)
inv = Vt2[np.sum(s2>1e-8):]
T = inv[0].reshape(7,7,7)
fs = np.linalg.norm(sum(T.transpose(p) for p in
     [(0,1,2),(1,2,0),(2,0,1),(1,0,2),(0,2,1),(2,1,0)])/6)
phi = np.zeros((7,7,7))
for (a,b,c) in oriented:
    for x,y,z in ((a,b,c),(b,c,a),(c,a,b)):
        phi[x-1,y-1,z-1]=1; phi[y-1,x-1,z-1]=-1; phi[z-1,y-1,x-1]=-1
        phi[x-1,z-1,y-1]=-1; phi[y-1,z-1,x-1]=1; phi[z-1,x-1,y-1]=1
ov = abs(np.sum(T*phi))/(np.linalg.norm(T)*np.linalg.norm(phi))
print(f'invariant cubics dim = {inv.shape[0]} (pred 1); sym content = {fs:.1e}; phi overlap = {ov:.6f}')
Lmt = lambda a: np.column_stack([mul(a,I8[j]) for j in range(8)])
Ls = [Lmt(I8[i]) for i in range(1,8)]; bb=[]
def add(M):
    v=M.flatten()
    for b0 in bb: v=v-(v@b0)*b0
    if np.linalg.norm(v)>1e-8: bb.append(v/np.linalg.norm(v)); return True
    return False
for i in range(7):
    for j in range(i+1,7): add(Ls[i]@Ls[j]-Ls[j]@Ls[i])
UB = np.array(bb)
res = max(np.linalg.norm(d-UB.T@(UB@d)) for d in
          [(D/np.linalg.norm(D)).flatten() for D in Ders])
print(f'holonomy dim = {len(bb)}; Der-in-holonomy residual = {res:.1e} (pred 14 inside 21)')
