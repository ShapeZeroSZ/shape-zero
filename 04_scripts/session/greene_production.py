import numpy as np
from scipy.optimize import root, brentq
GOLDEN=(np.sqrt(5)-1)/2
def sm_lift(x,p,K):
    p_new=p+(K/(2*np.pi))*np.sin(2*np.pi*(x%1.0)); return x+p_new,p_new
def sm(x,p,K):
    p_new=p+(K/(2*np.pi))*np.sin(2*np.pi*x); return (x+p_new)%1.0,p_new
def find_periodic_points(K,q,n_grid=56):
    xs=np.linspace(0.0,1.0,n_grid,endpoint=False); ps=np.linspace(-0.75,0.75,n_grid)
    cands=[]; seen=set()
    for x0 in xs:
        for p0 in ps:
            def func(z):
                x,p=float(z[0]),float(z[1])
                if not(np.isfinite(x) and np.isfinite(p)): return [1e6,1e6]
                for _ in range(q):
                    x,p=sm_lift(x,p,K)
                    if not(np.isfinite(x) and np.isfinite(p)): return [1e6,1e6]
                dx=x-z[0]-round(x-z[0]); return [dx,p-z[1]]
            try: sol=root(func,[x0,p0],tol=1e-10,method="hybr",options={"maxfev":180})
            except Exception: continue
            if not sol.success: continue
            pt=sol.x.copy(); pt[0]%=1.0
            if not(np.isfinite(pt[0]) and np.isfinite(pt[1])): continue
            key=(round(pt[0],8),round(pt[1],8))
            if key in seen: continue
            seen.add(key); cands.append(pt)
    return cands
def rotation_number(pt,K,q):
    x,p=float(pt[0]),float(pt[1]); x0=x
    for _ in range(q): x,p=sm_lift(x,p,K)
    return ((x-x0)/q)%1.0
def residue(pt,K,q):
    x,p=float(pt[0]),float(pt[1]); M=np.eye(2)
    for _ in range(q):
        c=np.cos(2*np.pi*x)
        J=np.array([[1.0+K*c,1.0],[K*c,1.0]]); M=J@M; x,p=sm(x,p,K)
    return (2.0-np.trace(M))/4.0
def select_golden(K,q,n_grid=56):
    pts=find_periodic_points(K,q,n_grid=n_grid)
    if not pts: return None
    best=None; bs=1e9
    for pt in pts:
        rot=rotation_number(pt,K,q); R=residue(pt,K,q)
        score=abs(rot-GOLDEN)-0.015*(1.0 if R>0 else 0.0)
        if score<bs: bs=score; best=(pt,R,rot)
    return best
def Kc(q,n_grid=48):
    def f(K):
        o=select_golden(K,q,n_grid=n_grid)
        return -1.0 if o is None else abs(o[1])-1.0
    try:
        if f(0.04)>=0: return 0.04
        if f(1.7)<0: return np.nan
        return brentq(f,0.04,1.7,xtol=1e-4)
    except Exception: return np.nan
print("  Greene residues, golden-mean convergent family")
print(f"  {'q':>4}  {'R@K=0.5':>13}  {'rot':>9}  {'|rot-phi|':>10}  {'K_c':>9}")
print("  "+"-"*52)
for q in (2,3,5,8,13):
    o=select_golden(0.5,q,n_grid=56 if q<=13 else 64)
    if o is None: print(f"  {q:4d}  location failed"); continue
    _,R,rot=o
    k=Kc(q,n_grid=44)
    print(f"  {q:4d}  {R:13.5e}  {rot:9.6f}  {abs(rot-GOLDEN):10.6f}  {k:9.5f}")
print("\n  CALIBRATION: Greene's K_c for the golden torus = 0.971635")
