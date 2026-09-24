import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '04_scripts', 'session'))
import numpy as np, pinned_asymmetry_reference as R
from scipy.integrate import solve_ivp
def run_dir_fixed(beta,A,direction,T=900.0,dt=0.008,rtol=1e-9):
    N=R.N; n=np.arange(N); eom=R.eom_factory(beta)
    B=2*R.c*beta*np.sin(R.k); d=np.sqrt(B*B+4*R.W2)
    # reference lattice gyro = x[n-1]-x[n+1]: +k is the UPPER root there
    west=(B+d)/2 if direction=="+" else (-B+d)/2
    s=1.0 if direction=="+" else -1.0
    x0=R.phi+A*np.cos(R.k*n); v0=s*A*west*np.sin(R.k*n)
    te=np.arange(0.0,T,dt)
    sol=solve_ivp(eom,[0,T],np.concatenate([x0,v0]),t_eval=te,method="DOP853",rtol=rtol,atol=rtol)
    X=np.fft.fft(sol.y[:N],axis=0)/N
    return R.freq_phase(X[R.m_pos] if direction=="+" else X[R.m_neg],te)
for A in (0.2,0.3):
    wp,rp=run_dir_fixed(0.05,A,"+"); wm,rm=run_dir_fixed(0.05,A,"-")
    d=wp-wm; print(f"A={A}: delta={d:+.7f} |d/d0|={abs(d)/0.1:.6f} kappa={(abs(d)-0.1)/(0.1*A*A):+.4f} resid={max(rp,rm):.4f}",flush=True)
