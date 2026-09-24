import numpy as np; from sim import run
B=np.round(np.arange(0.055,0.0951,0.0025),4)
for A in (0.20,0.30,0.40):
  t,r=run(A,B,T=2500.0,mode='exact',every=5.0)
  mn=r[:,:,0].min(0)
  w=np.clip(1-mn,0,None); cen=(B*w).sum()/w.sum() if w.sum()>0 else np.nan
  print(f"A={A} exact IC, T=2500, min_t R: "+" ".join(f"{b:.4f}:{x:.3f}" for b,x in zip(B,mn))+f"\n   argmin beta={B[np.argmin(mn)]:.4f}  depletion-weighted centre={cen:.4f}",flush=True)
