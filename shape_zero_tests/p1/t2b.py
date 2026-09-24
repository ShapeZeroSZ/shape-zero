import numpy as np; from hill import growth
BETAS=np.round(np.arange(0.00,0.1201,0.0025),4)
for H in (4,6,8):
  print("H=",H)
  for A in (0.10,0.20,0.30,0.40):
    g=np.array([growth(A,b,H=H,js=[0])[0] for b in BETAS]); i=int(np.argmax(g))
    y0,y1,y2=g[i-1:i+2]; d=0.5*(y0-y2)/(y0-2*y1+y2); bc=BETAS[i]+d*0.0025
    on=BETAS[g>1e-7]
    print(f"  A={A:.2f}: (0,pi) max growth {g[i]:.5f} at beta={bc:.4f}; window [{on.min():.4f},{on.max():.4f}]  gain*400={400*g[i]:.2f}")
