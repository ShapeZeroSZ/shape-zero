#!/usr/bin/env python3
"""residual_tower_check.py -- (2026-09-26, report check) (a) across the Cayley-Dickson levels
C, H, O, S, D32: how far left multiplication L_g fails to commute with a complex structure JJ
(L_e1, and the lattice rho(iI)), its charge-2 fraction, and whether L_g^2 = -|g|^2 and
L_g L_g x = g(gx) hold (they fail only for generic, non-octonionic g past O); (b) the code s
residual at D16: the block structure of L_g in the octonion / upper halves, and in the gate-11
run which half s U(1) charge drifts.
usage: python3 residual_tower_check.py"""
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, importlib.util, numpy as np
sys.path.insert(0,"04_scripts/session"); import model as M
s=importlib.util.spec_from_file_location("v2","04_scripts/session/d16_spectrum_v2.py"); v2=importlib.util.module_from_spec(s); s.loader.exec_module(v2)
Lm=lambda E,x: np.einsum("ijk,i->kj",E,x)
n=np.linalg.norm; rng=np.random.default_rng(4)
print("level  D   J        ||[J,L_g]||/||L_g||  charge-2 fraction  L_g^2=-|g|^2  alternative? L_g L_g x = g(gx)")
for k in (1,2,3,4,5):
    E=v2.cd(k); D=2**k
    for Jlab in ("L_e1","lattice rho(iI)"):
        J = Lm(E,np.eye(D)[1]) if Jlab=="L_e1" else M.rho(1j*np.eye(D//2))
        if np.abs(J@J+np.eye(D)).max()>1e-9: print(f"{k} {D:3d} {Jlab}: not a complex structure"); continue
        for glab in ("g in O (octonionic)","g in Im(level) generic"):
            if glab.startswith("g in O") and D<8: continue
            g=np.zeros(D); m = min(D,8) if glab.startswith("g in O") else D
            g[1:m]=rng.normal(size=m-1); g/=n(g)
            L=Lm(E,g); A=0.5*(L+J@L@J)
            sq=n(L@L+np.eye(D))
            x=rng.normal(size=D); alt=n(L@(L@x)-np.einsum("ijk,i,j->k",E,np.einsum("ijk,i,j->k",E,g,g),x))
            print(f"{k} {D:3d} {Jlab:16s} {glab:24s} {n(J@L-L@J)/n(L):.3f}   {n(A)/n(L):.3f}   {sq:.1e}   {alt:.1e}")
s=importlib.util.spec_from_file_location("v2","04_scripts/session/d16_spectrum_v2.py"); v2=importlib.util.module_from_spec(s); s.loader.exec_module(v2)
E16=v2.cd(4); rg=np.random.default_rng(5); gv=np.zeros(16); gv[1:8]=rg.normal(size=7); gv/=np.linalg.norm(gv)
L=np.einsum("ijk,i->kj",E16,gv); n=np.linalg.norm
print("L_g blocks (lower=octonion half, upper): ||LL||,||LU||,||UL||,||UU|| =",[round(n(L[a:a+8,b:b+8]),3) for a in (0,8) for b in (0,8)])
J=M.rho(1j*np.eye(8)); Jl=J[:8,:8]; Ll=L[:8,:8]
print("J block-diagonal in halves:", n(J[:8,8:])+n(J[8:,:8])==0, "; lower block: ||[J,L_g]||/||L_g|| =", round(n(Jl@Ll-Ll@Jl)/n(Ll),3),
      "; upper block:", round(n(J[8:,8:]@L[8:,8:]-L[8:,8:]@J[8:,8:])/n(L[8:,8:]),3))
def Qh(u,v,Jb): return float(np.einsum("na,ab,nb->",v,Jb,u))
for kap in (0.0, M.KAPPA):
    lr=M.Lattice(n=8,N=512,q=1,kappa=kap,C_r=0.05,tower=(E16,gv))
    u,v=lr.packet(amp=1e-3); u[:,8:]+=0.3*u[:,:8]
    def parts(u,v):
        lo=Qh(u[:,:8],v[:,:8],Jl)-0.5*kap*float((u[:,:8]**2).sum())
        up=Qh(u[:,8:],v[:,8:],J[8:,8:])-0.5*kap*float((u[:,8:]**2).sum())
        return lo,up
    a=parts(u,v); u2,v2,_=lr.run(u,v,T=20.0); b=parts(u2,v2)
    print(f"kappa {kap:.3f} C_r 0.05: lower-half (D<=8) charge {a[0]:.3e} -> {b[0]:.3e} (rel {b[0]/a[0]-1:+.2e}); upper-half {a[1]:.3e} -> {b[1]:.3e}")
    # with only the upper-half (beyond-D8) packet: does the octonion half's charge change?
