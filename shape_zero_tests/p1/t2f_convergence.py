# Hill truncation check for the (0, pi) band at A = 0.2 and 0.4.
import numpy as np; from hill import growth
B=np.arange(0.060,0.095,0.00025)
for H,Hw in ((8,8),(10,10),(12,12)):
  for A in (0.2,0.4):
    g=np.array([growth(A,b,H=H,Hw=Hw,js=[0])[0] for b in B]); bs=B[g>1e-6]
    print(H,A,f'[{bs.min():.4f},{bs.max():.4f}] centre {0.5*(bs.min()+bs.max()):.4f} peak {B[np.argmax(g)]:.4f} g={g.max():.5f}')
