import numpy as np
from scipy.optimize import root
GOLDEN=(np.sqrt(5)-1)/2
def standard_map_lifted(x,p,K):
    p_new=p+(K/(2*np.pi))*np.sin(2*np.pi*(x%1.0)); x_new=x+p_new
    return x_new,p_new
def map_q_lifted(z,K,q):
    x,p=float(z[0]),float(z[1])
    for _ in range(q): x,p=standard_map_lifted(x,p,K)
    return np.array([x,p])
def find_periodic_points(K,q,n_grid=40):
    xs=np.linspace(0.0,1.0,n_grid,endpoint=False); ps=np.linspace(-0.7,0.7,n_grid)
    candidates=[]; seen=set()
    for x0 in xs:
        for p0 in ps:
            def func(z):
                zq=map_q_lifted(z,K,q)
                dx=zq[0]-z[0]-round(zq[0]-z[0])
                return [dx,zq[1]-z[1]]
            try: sol=root(func,[x0,p0],tol=1e-11,method="hybr")
            except Exception: continue
            if not sol.success: continue
            pt=sol.x.copy(); pt[0]%=1.0
            key=(round(pt[0],8),round(pt[1],8))
            if key in seen: continue
            seen.add(key); candidates.append(pt)
    return candidates
def rotation_number(pt,K,q):
    x,p=float(pt[0]),float(pt[1]); x_start=x
    for _ in range(q): x,p=standard_map_lifted(x,p,K)
    return (x-x_start)/q
def residue(pt,K,q):
    def sm(x,p,K):
        p_new=p+(K/(2*np.pi))*np.sin(2*np.pi*x); return (x+p_new)%1.0,p_new
    M=np.eye(2); x,p=float(pt[0]),float(pt[1]); eps=1e-8
    for _ in range(q):
        def f(z): return np.array(sm(z[0],z[1],K))
        J=np.zeros((2,2)); f0=f([x,p])
        for i in range(2):
            dz=np.zeros(2); dz[i]=eps
            fp=f([x+dz[0],p+dz[1]])
            dx=(fp[0]-f0[0]+0.5)%1-0.5
            J[:,i]=[dx/eps,(fp[1]-f0[1])/eps]
        M=J@M; x,p=sm(x,p,K)
    return (2.0-np.trace(M))/4.0
def select_golden_orbit(K,q,n_grid=40):
    pts=find_periodic_points(K,q,n_grid=n_grid)
    if not pts: return None
    best=None; best_dist=1e9
    for pt in pts:
        rot=rotation_number(pt,K,q); rot_mod=rot%1.0
        dist=min(abs(rot_mod-GOLDEN),abs(rot_mod-GOLDEN-1),abs(rot_mod-GOLDEN+1))
        R=residue(pt,K,q)
        score=dist-0.01*(1 if R>0 else 0)
        if score<best_dist: best_dist=score; best=(pt,R,rot_mod)
    return best
print("Corrected lifting - golden-mean family")
print(f"{'q':>4}  {'R@0.5':>12}  {'rot':>10}  {'|rot-phi-1|':>12}")
print("-"*45)
K=0.5
for q in [2,3,5,8,13]:
    orb=select_golden_orbit(K,q,n_grid=48)
    if orb is None: print(f"{q:4d}  failed"); continue
    _,R,rot=orb
    print(f"{q:4d}  {R:12.5e}  {rot:10.6f}  {abs(rot-GOLDEN):12.6f}")
