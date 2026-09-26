#!/usr/bin/env python3
"""residual_j_projection.py -- the JJ-commuting part of the residual generator L_g (2026-09-26; a
report, not a prediction test). C = (L_g - JJ L_g JJ)/2 is the projection of L_g onto the commutant of
JJ (u(n)). Reports: its size; whether it is an octonionic multiplication (least-squares fit by L_x, R_y,
L_x + R_y), a complex structure (C^2 prop. to -I) or a derivation (in g2); the same in relabelled
identifications JJ = L_e1, R_e1; and gate-11 dynamics with C in place of L_g at kappa = 0 and kappa*,
including the FULL Noether charge Q = v.JJ u + (1/2) u.(G JJ) u, G = kappa JJ + C_r X (the charge
recorded in MODEL_SPEC 4b.1, FINDING 2026-09-26, omitted the C_r part).
usage: python3 residual_j_projection.py (from anywhere; ~3 min)"""
import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import sys, importlib.util, numpy as np
sys.path.insert(0,"04_scripts/session"); import model as M
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
v2=load("v2","04_scripts/session/d16_spectrum_v2.py")
E16=v2.cd(4); E8=v2.cd(3)
rg=np.random.default_rng(5); gv=np.zeros(16); gv[1:8]=rg.normal(size=7); gv/=np.linalg.norm(gv)
def Lm(E,x): return np.einsum("ijk,i->kj",E,x)
def Rm(E,x): return np.einsum("ijk,j->ki",E,x)
def fit(E,X):
    D=E.shape[0]; I=np.eye(D)
    Lb=np.stack([Lm(E,I[i]).ravel() for i in range(D)],1); Rb=np.stack([Rm(E,I[i]).ravel() for i in range(D)],1)
    out={}
    for nm,A in (("L",Lb),("R",Rb),("L+R",np.hstack([Lb,Rb]))):
        c,*_=np.linalg.lstsq(A,X.ravel(),rcond=None); out[nm]=round(np.linalg.norm(A@c-X.ravel())/np.linalg.norm(X),3)
    return out
def report(tag,J,Lg,E):
    C=0.5*(Lg-J@Lg@J); A=0.5*(Lg+J@Lg@J)
    D=J.shape[0]; n=np.linalg.norm
    ev=np.unique(np.round(np.abs(np.linalg.eigvals(C).imag),6))
    CC=C@C
    print(f"{tag}: |C|/|L|={n(C)/n(Lg):.3f} |A|/|L|={n(A)/n(Lg):.3f}; [J,C]={n(J@C-C@J):.1e}; C antisym {np.abs(C+C.T).max():.1e};"
          f" C^2 ∝ -I? spread {np.ptp(np.diag(CC)):.3f}, offdiag {n(CC-np.diag(np.diag(CC))):.3f}; |freq| {ev}; fit {fit(E,C)}")
    if D==16:
        O=C[:8,:8]; X=C[8:,:8]
        print(f"     octonion block preserved? mixing O->upper {n(X)/n(C):.3f}")
# present identification
J16=M.rho(1j*np.eye(8)); report("n=8 present id (sedenion)",J16,Lm(E16,gv),E16)
g8=gv[:8]; J8=M.rho(1j*np.eye(4)); report("n=4 present id (octonion)",J8,Lm(E8,g8),E8)
# relabelled: J = L_e1 and J = R_e1 in the octonion table
e=np.zeros(8); e[1]=1
for nm,J in (("J=L_e1",Lm(E8,e)),("J=R_e1",Rm(E8,e))):
    for gg,lab in ((g8,"generic g"),(np.r_[0,0,*g8[2:]]/np.linalg.norm(g8[2:]),"g ⟂ e1")):
        report(f"n=4 {nm}, {lab}",J,Lm(E8,gg),E8)

print("\n-- derivation test (is C in g2 = Der(O)?) and dynamics with C in place of L_g")
C8 = 0.5*(Lm(E8,g8) - J8@Lm(E8,g8)@J8)
mul = lambda x,y: np.einsum("ijk,i,j->k",E8,x,y)
X = rg.normal(size=(20,8)); Y = rg.normal(size=(20,8))
der = max(np.linalg.norm(C8@mul(x,y) - mul(C8@x,y) - mul(x,C8@y)) for x,y in zip(X,Y))
print(f"  n=4 present id: |C(xy) - C(x)y - xC(y)| max {der:.2f} (0 would mean C in g2)")
C16 = 0.5*(Lm(E16,gv) - J16@Lm(E16,gv)@J16)
Et = np.zeros((16,16,16)); Et[0] = C16.T          # einsum('ijk,i,nj->nk', Et, e0, v) = C16 v
e0 = np.zeros(16); e0[0] = 1
def charge(u, v, kap):
    psi = u[:,0::2] + 1j*u[:,1::2]; dps = v[:,0::2] + 1j*v[:,1::2]
    return float(np.sum(np.imag(np.conj(psi)*dps) - 0.5*kap*np.abs(psi)**2))
for kap in (0.0, M.KAPPA):
    for Cr in (0.05, 0.20):
        for lab, tw in (("L_g (as model)", (E16, gv)), ("J-commuting part C", (Et, e0))):
            lr = M.Lattice(n=8, N=512, q=1, kappa=kap, C_r=Cr, tower=tw)
            uu, vv = lr.packet(amp=1e-3); uu[:,8:] += 0.3*uu[:,:8]
            N0 = charge(uu,vv,kap); uu, vv, dq = lr.run(uu, vv, T=20.0)
            print(f"  kappa {kap:.3f} C_r {Cr:.2f} {lab:20s}: B {lr.residual_B(uu).mean():.3e}  dN/N {charge(uu,vv,kap)/N0-1:+.2e}  dE {dq:.1e}")

print("\n-- full Noether charge Q = v.JJ u + 1/2 u.(G JJ) u, G = kappa JJ + C_r X (X = L_g or C)")
def Q(u, v, G):
    return float(np.einsum("na,ab,nb->", v, J16, u) + 0.5*np.einsum("na,ab,nb->", u, G@J16, u))
for kap in (0.0, M.KAPPA):
    for Cr in (0.05, 0.20):
        for lab, tw, X in (("L_g (as model)", (E16, gv), Lm(E16, gv)), ("J-commuting part C", (Et, e0), C16)):
            G = kap*J16 + Cr*X
            lr = M.Lattice(n=8, N=512, q=1, kappa=kap, C_r=Cr, tower=tw)
            uu, vv = lr.packet(amp=1e-3); uu[:,8:] += 0.3*uu[:,:8]
            Q0 = Q(uu, vv, G); uu, vv, dq = lr.run(uu, vv, T=20.0)
            print(f"  kappa {kap:.3f} C_r {Cr:.2f} {lab:20s}: dQ/Q {Q(uu,vv,G)/Q0-1:+.2e}")
