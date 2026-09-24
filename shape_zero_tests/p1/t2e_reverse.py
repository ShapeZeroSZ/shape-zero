# Linear stability of the REVERSE (-k) pump on all channels.
from hill import growth
for A in (0.3,0.4):
  for b in (0.05,0.07,0.08):
    print(f'A={A} beta={b}: -k pump (0,pi) g={growth(A,b,H=8,sign=-1,js=[0])[0]:.5f}  other pairs g={growth(A,b,H=8,sign=-1,js=range(1,16))[0]:.5f}')
