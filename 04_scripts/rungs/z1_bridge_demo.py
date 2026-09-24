#!/usr/bin/env python3
"""z1_bridge_demo.py — selection rule (log: Z1-B). Resonant transfer on a
Fano line (5 orders) vs zero on an equally-resonant off-line triple."""
import numpy as np
LINES = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
g = np.zeros((8,8,8))
for (a,b,c) in LINES:
    for x,y,z in ((a,b,c),(b,c,a),(c,a,b)): g[x,y,z]=1; g[y,x,z]=-1
w = np.zeros(8); w[1],w[2],w[4],w[5],w[3],w[6],w[7] = 1.0,1.3,2.3,2.3,0.7,1.9,3.1
lam,DT,T = 0.1,1e-3,20.0
A = np.zeros(8,complex); A[1]=A[2]=0.5; A[4]=A[5]=1e-3
triples = [(a,b,c,g[a,b,c]) for a in range(1,8) for b in range(a+1,8)
           for c in range(1,8) if g[a,b,c]]
def rhs(A):
    dA = -1j*w*A
    for (a,b,c,s) in triples:
        dA[c] += -1j*lam*s*A[a]*A[b]
        dA[a] += -1j*lam*s*np.conj(A[b])*A[c]
        dA[b] += -1j*lam*s*np.conj(A[a])*A[c]
    return dA
for _ in range(int(T/DT)):
    k1=rhs(A); k2=rhs(A+DT/2*k1); k3=rhs(A+DT/2*k2); k4=rhs(A+DT*k3)
    A = A + DT/6*(k1+2*k2+2*k3+k4)
print(f'line (1,2,4): |A4|^2 = {abs(A[4])**2:.4f} (from 1e-6)')
print(f'off-line ctrl: |A5|^2 = {abs(A[5])**2:.2e} (equally resonant, zero coupling)')
