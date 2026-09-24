# Size of the PT asymmetry term vs the symmetric softening at beta_res, A = 0.4,
# and beta_res with MODEL_SPEC 5's (retracted) kappa = +0.0799 swapped in.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '04_scripts', 'platform'))
import numpy as np, phi_gauge_delta as D
from phi_gauge_decaymap import W_SUM
from scipy.optimize import brentq
b=0.0629; A=0.4
sym=(D.w_pt(D.K,b,A)+D.w_pt(-D.K,b,A))/2-(D.w_lin(D.K,b)+D.w_lin(-D.K,b))/2
asy=((D.w_pt(D.K,b,A)-D.w_pt(-D.K,b,A))-(D.w_lin(D.K,b)-D.w_lin(-D.K,b)))/2
print(f'PT +k shift at A=0.4: symmetric (centre) {sym:+.5f}, antisymmetric half {asy:+.5f}')
kap=0.0799; asy5=-2*b*kap*A*A/2
print(f'Sec.5 kappa=+0.0799 antisymmetric half for +k: {asy5:+.5f}')
def f(bb,sgn):
    s=(D.w_pt(D.K,bb,A)+D.w_pt(-D.K,bb,A))/2-(D.w_lin(D.K,bb)+D.w_lin(-D.K,bb))/2
    a=-2*bb*kap*A*A/2 if sgn else ((D.w_pt(D.K,bb,A)-D.w_pt(-D.K,bb,A))-(D.w_lin(D.K,bb)-D.w_lin(-D.K,bb)))/2
    return 2*(D.w_lin(D.K,bb)+s+a)-W_SUM
print('beta_res(0.4) PT asym: %.4f ; with Sec.5 kappa: %.4f ; linear: %.4f'%(brentq(lambda x:f(x,0),0,.2),brentq(lambda x:f(x,1),0,.2),brentq(lambda x:2*D.w_lin(D.K,x)-W_SUM,0,.2)))
