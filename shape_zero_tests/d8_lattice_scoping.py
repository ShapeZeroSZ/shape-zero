#!/usr/bin/env python3
"""d8_lattice_scoping.py -- scoping checks for realising the D8 flow in the lattice (2026-09-26, not a
prediction test): the orientation class (Pfaffian sign) of the J-sector complex structure JJ (n = 4)
against left and right multiplication by unit imaginaries, in the D8 rung table and cd(3).
Same Pfaffian sign => conjugate by an orientation-preserving relabelling of the node components.
usage: python3 d8_lattice_scoping.py (from the repository root or shape_zero_tests/)"""
import os, sys, importlib.util, numpy as np; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0,"04_scripts/session"); import model as M
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
fl=load("fl","04_scripts/rungs/z1_d8_flow.py"); v2=load("v2","04_scripts/session/d16_spectrum_v2.py")
def pf(A):
    n=A.shape[0]
    if n==0: return 1.0
    s=0.0
    for j in range(1,n):
        if abs(A[0,j])<1e-14: continue
        idx=[k for k in range(n) if k not in (0,j)]
        s+=(-1)**(j+1)*A[0,j]*pf(A[np.ix_(idx,idx)])
    return s
J=M.rho(1j*np.eye(4))
for name,E,mul in (("D8 rung table",fl.oct_table(fl.ch.oriented_lines()),None),("cd(3)",v2.cd(3),None)):
    Ls=[np.einsum("ijk,i->kj",E,np.eye(8)[k]) for k in range(1,8)]
    Rs=[np.einsum("ijk,j->ki",E,np.eye(8)[k]) for k in range(1,8)]
    print(name,"Pf(JJ)=",round(pf(J),3),"Pf(L_e)=",sorted({round(pf(L),3) for L in Ls}),"Pf(R_e)=",sorted({round(pf(R),3) for R in Rs}),
      "L^2=-1:",max(np.abs(L@L+np.eye(8)).max() for L in Ls))
