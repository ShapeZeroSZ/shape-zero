import numpy as np; from hill import growth
B=np.arange(0.055,0.090,0.00025)
res=[]
for H in (6,8):
  print("H=",H)
  for A in (0.10,0.15,0.20,0.25,0.30,0.35,0.40):
    g=np.array([growth(A,b,H=H,js=[0])[0] for b in B]); on=g>1e-6
    bs=B[on]; gs=g[on]; i=int(np.argmax(g))
    cen=np.sum(bs*gs)/np.sum(gs)
    print(f"  A={A:.2f}: window [{bs.min():.4f},{bs.max():.4f}] centre {0.5*(bs.min()+bs.max()):.4f} peak g={g[i]:.5f} at {B[i]:.4f}; e-folds by T=400: {400*g[i]:.2f}")
    if H==8: res.append((A,0.5*(bs.min()+bs.max())))
A=np.array([r[0] for r in res]); c=np.array([r[1] for r in res])
p=np.polyfit(A**2,c,1); print("fit centre = %.4f + %.4f A^2"%(p[1],p[0]))
