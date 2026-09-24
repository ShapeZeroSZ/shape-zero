#!/usr/bin/env python3
"""
pinned_asymmetry_reference.py — REFERENCE IMPLEMENTATION (external)

Source: independent reimplementation supplied by the author, unmodified except
for trimming the main block to the kappa extraction. This is the script that
produced kappa = 0.078-0.082 and the beta-collapse to four digits.

It is the reference because it (a) seeds UNIDIRECTIONAL travelling waves with
direction-specific velocities, (b) projects onto a single Fourier mode m = +/-16
so the two directions never mix, and (c) uses weighted complex-phase regression,
which harness.py calibrates to 0.3% against the exact Duffing shift.

The package's own phi_gauge_nonlinear.py uses an FFT-peak estimator that fails
calibration (0.4985 relative error) and gives 0.0305. This file supersedes it for
any quoted coefficient.

q-AWARENESS. set_base(q, side) switches the lattice to q spatial dimensions:
the elastic Laplacian then sums over ALL q axes while the gyroscopic term acts
along the PROPAGATION axis (axis 0) only, since k -> -k reverses that axis.

TRANSVERSE_WIDTH controls the seed. Leave it None and the seed is a
transverse-UNIFORM plane wave -- whose transverse Laplacian is identically zero,
so the q=3 problem reduces EXACTLY to q=1 and the outputs are bit-identical.
That is not a q=3 measurement. Set it to a float (e.g. side/4) for a genuinely
localised 3D packet.
"""
import numpy as np
from scipy.integrate import solve_ivp
import warnings; warnings.filterwarnings("ignore")

N=64; c=1.0; k=np.pi/2.0
phi=(1.0+np.sqrt(5.0))/2.0
m_pos=16; m_neg=48
W2=(2.0*phi-1.0)+2.0*c*(1.0-np.cos(k))
omega_lin0=np.sqrt(W2)

SHAPE = (N,)          # set by set_base(); the propagation axis is axis 0


def set_base(q, side=None):
    """Switch the reference lattice to q spatial dimensions.

    The asymmetry is a PROPAGATION-DIRECTION effect, so the gyroscopic term
    acts along ONE axis (axis 0) while the elastic Laplacian sums over all q.
    That is the q-dimensional generalisation of the 1D chain: reversing k
    means reversing axis 0.
    """
    global N, SHAPE, m_pos, m_neg
    if q == 1:
        SHAPE = (N,)
        return
    side = side or int(round(N ** (1.0 / q)))
    SHAPE = (side,) * q
    N = side ** q
    m_pos = SHAPE[0] // 4
    m_neg = SHAPE[0] - m_pos


def _lap(x):
    if len(SHAPE) == 1:
        return np.roll(x, -1) + np.roll(x, 1) - 2 * x
    y = x.reshape(SHAPE)
    out = np.zeros_like(y)
    for ax in range(len(SHAPE)):
        out += np.roll(y, -1, axis=ax) + np.roll(y, 1, axis=ax) - 2 * y
    return out.reshape(-1)


def _dir0(x):
    """Antisymmetric difference along the PROPAGATION axis only."""
    if len(SHAPE) == 1:
        return np.roll(x, 1) - np.roll(x, -1)
    y = x.reshape(SHAPE)
    return (np.roll(y, 1, axis=0) - np.roll(y, -1, axis=0)).reshape(-1)


def eom_factory(beta):
    def eom(t,y):
        x=y[:N]; v=y[N:]
        force=-(x*x-x-1.0)
        elastic=c*_lap(x)
        gyro=(beta*c)*_dir0(v)
        return np.concatenate([v,force+elastic+gyro])
    return eom

def freq_phase(mode,t,discard=0.08):
    n=len(t); i0=int(discard*n)
    tt=t[i0:]; mm=mode[i0:]; amp=np.abs(mm)
    ma=np.mean(amp)
    if ma<1e-14: return np.nan,99.0
    msk=amp>0.08*ma; tt=tt[msk]; mm=mm[msk]; amp=amp[msk]
    phase=np.unwrap(np.angle(mm)); w=amp**2
    sw=np.sum(w); swt=np.sum(w*tt); swt2=np.sum(w*tt*tt)
    swp=np.sum(w*phase); swtp=np.sum(w*tt*phase)
    den=sw*swt2-swt*swt
    if abs(den)<1e-30: return np.nan,99.0
    slope=(sw*swtp-swt*swp)/den
    resid=phase-(slope*tt+(swt2*swp-swt*swtp)/den)
    return -slope,float(np.sqrt(np.mean(resid**2)))

TRANSVERSE_WIDTH = None      # None = plane wave (1D-equivalent); float = localised


def run_dir(beta,A,direction="+",T=900.0,dt=0.008,rtol=1e-9):
    eom=eom_factory(beta); nidx=np.arange(N)
    if len(SHAPE)==1:
        coord=nidx.astype(float)
    else:
        c0=np.indices(SHAPE)[0].astype(float)
        coord=c0.reshape(-1)
    env=1.0
    if len(SHAPE)>1 and TRANSVERSE_WIDTH:
        cc=np.indices(SHAPE).astype(float); e=np.ones(SHAPE)
        for a in range(1,len(SHAPE)):
            d=cc[a]-SHAPE[a]/2.0
            d=(d+SHAPE[a]/2)%SHAPE[a]-SHAPE[a]/2
            e*=np.exp(-0.5*(d/TRANSVERSE_WIDTH)**2)
        env=e.reshape(-1)
    x0=phi+A*env*np.cos(k*coord)
    sign=1.0 if direction=="+" else -1.0
    if abs(beta)<1e-12: west=omega_lin0
    else:
        B=2*c*beta*np.sin(k); d=np.sqrt(B*B+4*W2)
        west=(-B+d)/2 if direction=="+" else (B+d)/2
    v0=sign*A*env*west*np.sin(k*coord)
    t_eval=np.arange(0.0,T,dt)
    sol=solve_ivp(eom,[0,T],np.concatenate([x0,v0]),t_eval=t_eval,
                  method="DOP853",rtol=rtol,atol=rtol)
    if not sol.success: return None
    Y=sol.y[:N]
    if len(SHAPE)>1:
        Y=Y.reshape(SHAPE+(Y.shape[1],)).mean(axis=tuple(range(1,len(SHAPE))))
    X=np.fft.fft(Y,axis=0)/Y.shape[0]
    mp = m_pos if len(SHAPE)==1 else SHAPE[0]//4
    mn = m_neg if len(SHAPE)==1 else SHAPE[0]-SHAPE[0]//4
    mode=X[mp] if direction=="+" else X[mn]
    return freq_phase(mode,t_eval)

def delta(beta,A,**kw):
    rp=run_dir(beta,A,"+",**kw); rn=run_dir(beta,A,"-",**kw)
    if rp is None or rn is None: return None,None
    return rp[0]-rn[0], max(rp[1],rn[1])

if __name__=="__main__":
    print(f"  exact sqrt(W^2) = {omega_lin0:.10f}")
    print("\n  AMPLITUDE SWEEP beta=0.05")
    print(f"  {'A':>6} {'d_omega':>12} {'resid':>11} {'kappa':>9} {'q':>6}")
    d0=-0.1; ks=[]
    for A in (0.10,0.20,0.30,0.40):
        d,q=delta(0.05,A)
        if d is None: continue
        r=d-d0; kap=r/(d0*A*A); ks.append(kap)
        print(f"  {A:6.2f} {d:12.7f} {r:11.7f} {kap:9.5f} {q:6.3f}")
    print(f"\n  kappa mean = {np.mean(ks):.5f} +/- {np.std(ks):.5f}")
    print("\n  BETA SWEEP A=0.30 (collapse)")
    print(f"  {'beta':>6} {'d_omega':>12} {'kappa':>9} {'d/d0':>10}")
    for b in (0.02,0.05,0.10,0.20):
        d,q=delta(b,0.30)
        if d is None: continue
        th=-2*c*b*np.sin(k)
        print(f"  {b:6.2f} {d:12.7f} {(d-th)/(th*0.09):9.5f} {d/th:10.6f}")
