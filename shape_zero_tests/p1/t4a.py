import numpy as np; from sim import run
A=0.4; B=[0.07,0.08]
for lab,kw in [('script seed0',{}),('script seed7',{'seed':7}),('script noise=0',{'noise':0.0}),('script noise=1e-3',{'noise':1e-3}),('exact-wave IC seed0',{'mode':'exact'})]:
    t,r=run(A,B,**kw)
    print(f"{lab:22s} R(T=400) b=.07:{r[-1,0,0]:.4f} b=.08:{r[-1,1,0]:.4f} | early (t<20) max |u_pi| b=.07: {r[t<20,0,2].max():.2e}  osc amp of u0 (std, t<20): {r[t<20,0,1].std():.2e}")
