#!/usr/bin/env python3
"""
phi_gauge_u3_working.py — u(3) at Rank 3, WORKING VERSION

This is the version that produced the measured result quoted in
INPUT_LEDGER.md 2b2. The companion phi_gauge_u3.py is the annotated handoff
copy whose dynamics functions raise NotImplementedError by design; it documents
the four defects and their fixes. THIS file has them applied:

  FIX-1  RK4 stepper (velocity-Verlet is not symplectic for velocity-dependent
         forces; drift went from 7.4e-01 to ~3e-07)
  FIX-2  density-matrix readout over the whole lattice, not a single site
  FIX-3  spectral projectors from eigh(LAM[axis]) -- three eigenvalues, not two
  FIX-4  free-case control
  plus   composition order flipped (second segment first in the product)
  plus   SEG_START = (60, 80): the inter-segment gap must be short, because the
         prediction has no free-evolution operator between segments. At 60 sites
         the Abelian control reads 62 deg where theory says 0; at 20 sites it
         reads 0.0169 deg.

MEASURED (segments at 20-site separation):
    Abelian control            0.0169 deg   (theory: 0)
    sim vs pred, both orders   0.31, 0.32 deg
    non-commuting sim vs pred  0.61, 0.57 deg
    ordering splitting         65.12 deg measured, 64.97 predicted
    energy drift               ~3e-07
    u(2) benchmark             59.86 measured, 59.84 predicted

SCOPE: synthetic u(3) on a lattice fibre, not colour SU(3) on quark
representations. "Consistency", not QCD.
"""
import numpy as np
SQ5=np.sqrt(5); N=200; C=1.0; KAPPA=0.5; DT=0.02; K0=np.pi/2
AMP=1e-3; W_ENV=8.0; N0=20; SEG_START=(60,80)
RAMP=[0.25,0.5,0.75]+[1.0]*6+[0.75,0.5,0.25]
NDIM=3; D=2*NDIM
OMEGA=0.5*(-KAPPA+np.sqrt(KAPPA**2+4*(SQ5+2*C*(1-np.cos(K0)))))
def rho(A):
    n=A.shape[0]; M=np.zeros((2*n,2*n))
    for j in range(n):
        for l in range(n):
            a=A[j,l]; M[2*j:2*j+2,2*l:2*l+2]=[[a.real,-a.imag],[a.imag,a.real]]
    return M
def gell_mann():
    E=lambda i,j:(np.eye(3,dtype=complex)[:,[i]]@np.eye(3,dtype=complex)[[j],:])
    return [E(0,1)+E(1,0),-1j*(E(0,1)-E(1,0)),E(0,0)-E(1,1),E(0,2)+E(2,0),
            -1j*(E(0,2)-E(2,0)),E(1,2)+E(2,1),-1j*(E(1,2)-E(2,1)),
            (E(0,0)+E(1,1)-2*E(2,2))/np.sqrt(3)]
LAM=gell_mann(); JJ=rho(1j*np.eye(NDIM))
def k_branch(g,w=OMEGA):
    Q=w*w+KAPPA*w-SQ5
    f=lambda k:2*C*(1-np.cos(k))-2*C*g*w*np.sin(k)-Q
    lo,hi=0.2,np.pi-0.2
    for _ in range(80):
        mid=0.5*(lo+hi)
        if f(lo)*f(mid)<=0: hi=mid
        else: lo=mid
    return 0.5*(lo+hi)
def make_links(spec):
    W=np.zeros((N,D,D))
    for start,axis,g in spec:
        R=rho(LAM[axis])
        for j,wgt in enumerate(RAMP): W[start+j]=g*wgt*R
    return W
def force(u,v,W,Wm):
    up,um=np.roll(u,-1,axis=0),np.roll(u,1,axis=0)
    vp,vm=np.roll(v,-1,axis=0),np.roll(v,1,axis=0)
    f=-(SQ5*u+u*u)+KAPPA*(v@JJ.T)+C*(up+um-2*u)
    f+=C*(np.einsum('nab,nb->na',W,vp)-np.einsum('nab,nb->na',Wm,vm))
    return f
def energy(u,v):
    up=np.roll(u,-1,axis=0)
    return (0.5*(v*v).sum()+0.5*SQ5*(u*u).sum()+(u**3).sum()/3+0.5*C*((up-u)**2).sum())
def U_segment(axis,g_max):
    R=rho(LAM[axis])
    ev,V=np.linalg.eigh(R)
    # INTERLEAVED extraction: rho puts (Re z0, Im z0, Re z1, Im z1, ...)
    cvals=[]; cvecs=[]; used=set()
    for i,lam in enumerate(ev):
        if i in used: continue
        part=[j for j in range(len(ev)) if j not in used and j!=i
              and abs(ev[j]-lam)<1e-8]
        if not part: continue
        j=part[0]; used.add(i); used.add(j)
        vr=V[0::2,i]; vi=V[1::2,i]
        vc=vr+1j*vi
        n=np.linalg.norm(vc)
        if n<1e-12:
            vr=V[0::2,j]; vi=V[1::2,j]; vc=vr+1j*vi; n=np.linalg.norm(vc)
            if n<1e-12: continue
        cvals.append(lam); cvecs.append(vc/n)
    if len(cvals)!=3:
        eigs,vecs=np.linalg.eigh(LAM[axis])
        cvals=list(eigs); cvecs=[vecs[:,k] for k in range(3)]
    U=np.eye(3,dtype=complex)
    for wgt in RAMP:
        P=[np.outer(v,v.conj()) for v in cvecs]
        U=sum(np.exp(1j*k_branch(g_max*wgt*cvals[k]))*P[k] for k in range(3))@U
    return U

def readout(u,v):
    psi=u[:,0::2]+1j*u[:,1::2]; dps=v[:,0::2]+1j*v[:,1::2]
    chi=psi+(1j/OMEGA)*dps; bar=psi-(1j/OMEGA)*dps
    rho_s=chi.T@chi.conj(); tr=np.real(np.trace(rho_s))+1e-30
    return (np.array([np.real(np.trace(LAM[a]@rho_s))/tr for a in range(8)]),
            float(np.clip(1-np.linalg.norm(bar)/(np.linalg.norm(chi)+1e-30),0,1)))
def run(spec,ts):
    W=make_links(spec); Wm=np.roll(W,1,axis=0)
    x=np.arange(N); env=np.exp(-0.5*((x-N0)/W_ENV)**2); ph=K0*(x-N0)
    u=np.zeros((N,D)); v=np.zeros((N,D))
    ur=AMP*env*np.cos(ph); ui=AMP*env*np.sin(ph)
    u[:,0]=ur; u[:,1]=ui; v[:,0]=OMEGA*ui; v[:,1]=-OMEGA*ur
    E0=energy(u,v); snaps={}; t=0.0; i=0; mx=max(ts)+1.0
    while t<mx:
        if i<len(ts) and t>=ts[i]: snaps[ts[i]]=(u.copy(),v.copy()); i+=1
        k1v=force(u,v,W,Wm); k1u=v
        k2v=force(u+.5*DT*k1u,v+.5*DT*k1v,W,Wm); k2u=v+.5*DT*k1v
        k3v=force(u+.5*DT*k2u,v+.5*DT*k2v,W,Wm); k3u=v+.5*DT*k2v
        k4v=force(u+DT*k3u,v+DT*k3v,W,Wm); k4u=v+DT*k3v
        u=u+(DT/6)*(k1u+2*k2u+2*k3u+k4u); v=v+(DT/6)*(k1v+2*k2v+2*k3v+k4v); t+=DT
    if i<len(ts): snaps[ts[i]]=(u.copy(),v.copy())
    return snaps,abs(energy(u,v)-E0)/(abs(E0)+1e-30)
def ang(c1,c2):
    n1,n2=np.linalg.norm(c1),np.linalg.norm(c2)
    if n1<1e-12 or n2<1e-12: return 0.0
    return np.degrees(np.arccos(np.clip(c1@c2/(n1*n2),-1,1)))
gA,gB=0.15,0.10
print("  CORRECTED CONTROL: same axis, DIFFERENT strengths, both orders")
sAB=[(SEG_START[0],0,gA),(SEG_START[1],0,gB)]
sBA=[(SEG_START[0],0,gB),(SEG_START[1],0,gA)]
a,_=run(sAB,[180.0]); b,d=run(sBA,[180.0])
cA,pA=readout(*a[180.0]); cB,_=readout(*b[180.0])
print(f"      control angle   {ang(cA,cB):8.4f} deg   drift {d:.2e}  purity {pA:.4f}")
print("\n  NON-COMMUTING: different axes, same strengths, both orders")
s12=[(SEG_START[0],0,gA),(SEG_START[1],1,gA)]
s21=[(SEG_START[0],1,gA),(SEG_START[1],0,gA)]
a,_=run(s12,[180.0]); b,d=run(s21,[180.0])
c12,p12=readout(*a[180.0]); c21,_=readout(*b[180.0])
print(f"      ordering split  {ang(c12,c21):8.4f} deg   drift {d:.2e}  purity {p12:.4f}")
psi0=np.zeros(3,dtype=complex); psi0[0]=1.0
def gm(p):
    r=np.outer(p,p.conj()); tr=np.real(np.trace(r))+1e-30
    return np.array([np.real(np.trace(LAM[a]@r))/tr for a in range(8)])
U12=U_segment(0,gA)@U_segment(1,gA); U21=U_segment(1,gA)@U_segment(0,gA)
print(f"      predicted       {ang(gm(U12@psi0),gm(U21@psi0)):8.4f} deg")

print("\n"+"="*66)
print("SIM vs INDEPENDENT PREDICTION")
print("="*66)
psi0=np.zeros(3,dtype=complex); psi0[0]=1.0
print("\nABELIAN CONTROL (lam1 at two strengths, both orders)")
g1,g2=0.12,0.08
sp12=[(SEG_START[0],0,g1),(SEG_START[1],0,g2)]
sp21=[(SEG_START[0],0,g2),(SEG_START[1],0,g1)]
s12,d12=run(sp12,[180.0]); s21,d21=run(sp21,[180.0])
c12s,p12=readout(*s12[180.0]); c21s,_=readout(*s21[180.0])
U12=U_segment(0,g2)@U_segment(0,g1); U21=U_segment(0,g1)@U_segment(0,g2)
c12p=gm(U12@psi0); c21p=gm(U21@psi0)
print(f"      energy drift          {max(d12,d21):.2e}")
print(f"      sim vs pred (order 1) {ang(c12s,c12p):8.4f} deg")
print(f"      sim vs pred (order 2) {ang(c21s,c21p):8.4f} deg")
print(f"      run-vs-run splitting  {ang(c12s,c21s):8.4f} deg")
print(f"      PREDICTED splitting   {ang(c12p,c21p):8.4f} deg   <- theory says ~0 (commuting)")
print(f"      chirality purity      {p12:.4f}")
print("\nNON-COMMUTING (lam1 <-> lam2, equal strength)")
g=0.15
sp12=[(SEG_START[0],0,g),(SEG_START[1],1,g)]
sp21=[(SEG_START[0],1,g),(SEG_START[1],0,g)]
s12,d12=run(sp12,[180.0]); s21,d21=run(sp21,[180.0])
c12s,p12=readout(*s12[180.0]); c21s,_=readout(*s21[180.0])
U12=U_segment(1,g)@U_segment(0,g); U21=U_segment(0,g)@U_segment(1,g)
c12p=gm(U12@psi0); c21p=gm(U21@psi0)
print(f"      energy drift          {max(d12,d21):.2e}")
print(f"      sim vs pred (order 1) {ang(c12s,c12p):8.4f} deg")
print(f"      sim vs pred (order 2) {ang(c21s,c21p):8.4f} deg")
print(f"      measured splitting    {ang(c12s,c21s):8.4f} deg")
print(f"      predicted splitting   {ang(c12p,c21p):8.4f} deg")
print(f"      chirality purity      {p12:.4f}")

print("\n"+"="*66)
print("SINGLE-SEGMENT FIDELITY  (the isolation test)")
print("="*66)
g_test=0.15; start=40; T1=90.0
psi0=np.zeros(3,dtype=complex); psi0[0]=1.0
snaps,drift=run([(start,0,g_test)],[T1])
c_sim,pur=readout(*snaps[T1])
c_pred=gm(U_segment(0,g_test)@psi0)
a=ang(c_sim,c_pred)
print(f"      energy drift          {drift:.2e}")
print(f"      sim vs pred (1 seg)   {a:.4f} deg")
print(f"      chirality purity      {pur:.4f}")
print()
if a>5.0:
    print("  SINGLE-SEGMENT FAILED. Ordering experiment is premature.")
    print("  The defect is already in the mapping of ONE segment.")
else:
    print("  SINGLE-SEGMENT PASSED. Defect is in composition or timing.")
# extra: free case with the pure component, as a floor
snaps0,d0=run([],[T1]); c0,p0=readout(*snaps0[T1])
c0p=gm(psi0)
print(f"\n  FLOOR: free case (no segment) sim vs initial state: {ang(c0,c0p):.4f} deg")
print(f"         drift {d0:.2e}  purity {p0:.4f}")
