# Growth of the (0, pi) channel vs the fastest other pair channel (q, 2k0 - q).
from hill import growth
import numpy as np
B=np.round(np.arange(0.05,0.1001,0.005),3)
for A in (0.3,0.4):
  print('A=',A)
  for b in B:
    g0=growth(A,b,H=8,js=[0])[0]; go=growth(A,b,H=8,js=range(1,16))
    print(f'  beta={b:.3f}  (0,pi) g={g0:.5f}   best other pair g={go[0]:.5f} (q=2pi*{go[1]}/64)')
