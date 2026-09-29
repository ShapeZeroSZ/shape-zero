# POST HOC (2026-09-29, after the smooth run): successive-difference ratio |co(A)-co(A/2)| / |co(A/2)-co(A/4)|; 4 for A^2, 2 for A.
# usage: python3 ampA_smooth_ratio.py smooth|node
import os,sys,json,numpy as np
os.environ['SZ_J_WELL']=sys.argv[1]
sys.path.insert(0,'.'); import amp_scaling as S, model as M
for q,rows in ((1,json.load(open(S.OUT1))),(3,S.q3_rows())):
  for n in (2,3):
    for axes in ([0,1],[0,0]):
      rr=sorted([r for r in rows if r['n']==n and r['axes']==axes],key=lambda r:-r['amp'])
      for o in ('AB','BA'):
        C=[np.array(r['co'][o]) for r in rr]
        d1=np.linalg.norm(C[0]-C[1]); d2=np.linalg.norm(C[1]-C[2])
        print(f"q={q} u({n}) axes{axes} {o}: |co(A)-co(A/2)| = {np.degrees(d1):.2e} deg, |co(A/2)-co(A/4)| = {np.degrees(d2):.2e} deg, ratio {d1/d2:.2f}")
