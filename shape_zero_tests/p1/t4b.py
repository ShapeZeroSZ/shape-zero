import numpy as np; from sim import run
B=np.round(np.arange(0.055,0.0951,0.0025),4)
for A in (0.30,0.40):
  t,r=run(A,B,T=800.0)
  R=r[:,:,0]
  print(f"A={A} script IC, fine grid:")
  print("   beta   "+" ".join(f"{b:.4f}" for b in B))
  for T in (200,300,400,500,600,800):
    i=int(np.argmin(np.abs(t-T))); row=R[i]
    print(f"  R(T={T:3d}) "+" ".join(f"{x:6.3f}" for x in row)+f"  argmin beta={B[np.argmin(row)]:.4f}")
  mn=R.min(0); print("  min_t R  "+" ".join(f"{x:6.3f}" for x in mn)+f"  argmin beta={B[np.argmin(mn)]:.4f}")
