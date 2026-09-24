import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '04_scripts', 'platform'))
import numpy as np, phi_gauge_delta as D
from phi_gauge_decaymap import beta_res, W_SUM
print("W_SUM (constant, linear w(0)+w(pi)) =", W_SUM, np.sqrt(D.SQ5)+np.sqrt(D.SQ5+4))
b0=beta_res(1e-4); print("beta_res(0)=%.5f"%b0)
for A in (0.1,0.2,0.3,0.4):
    b=beta_res(A); print(f"A={A}: beta_res={b:.4f}  slope (b-b0)/A^2={(b-b0)/A**2:+.4f}")
s=(D.w_pt(D.K,b0,1e-3)-D.w_lin(D.K,b0))/1e-6
dwdb=(D.w_lin(D.K,b0+1e-6)-D.w_lin(D.K,b0))/1e-6
print(f"pump self-shift at beta_res0: {s:+.4f} A^2 ; dw/dbeta={dwdb:+.4f} ; implied slope = -s/dwdb*... = {-s/dwdb:+.4f}")
