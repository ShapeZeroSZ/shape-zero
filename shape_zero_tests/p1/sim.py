import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '04_scripts', 'platform'))
import numpy as np, phi_gauge_delta as D
from hill import wave
N,DT=D.N,D.DT; n=np.arange(N)
def force(x,v,b):  # batched over rows; b column vector
    xp,xm=np.roll(x,-1,1),np.roll(x,1,1); vp,vm=np.roll(v,-1,1),np.roll(v,1,1)
    return -(x*x-x-1.0)+D.C*(xp+xm-2*x)+D.C*b*(vp-vm)
def ic(A,betas,mode='script',seed=0,noise=1e-6):
    nb=len(betas); rng=np.random.default_rng(seed)
    X=np.zeros((nb,N)); V=np.zeros((nb,N))
    for i,b in enumerate(betas):
        if mode=='script':
            w0=D.w_lin(D.K,b); X[i]=D.PHI+A*np.cos(D.K*n); V[i]=A*w0*np.sin(D.K*n)
        else:  # exact harmonic-balance travelling wave
            w,c=wave(A,b); th=D.K*n
            X[i]=D.PHI+sum((c[m]*np.exp(1j*m*th)).real for m in c)
            V[i]=sum((c[m]*(-1j*m*w)*np.exp(1j*m*th)).real for m in c)
        rs=np.random.default_rng(seed)
        X[i]+=noise*rs.standard_normal(N); V[i]+=noise*rs.standard_normal(N)
    return X,V
def run(A,betas,T=400.0,mode='script',seed=0,noise=1e-6,every=1.0):
    b=np.asarray(betas)[:,None]; x,v=ic(A,betas,mode,seed,noise)
    a0=A*N/2; rec=[]; ts=[]; st=int(round(every/DT))
    for s in range(int(T/DT)+1):
        if s%st==0:
            u=np.fft.fft(x-D.PHI,axis=1); ts.append(s*DT)
            rec.append(np.stack([abs(u[:,D.M])/a0, abs(u[:,0]-u[:,0].mean()*0)/N, abs(u[:,N//2])/N],1))
        k1v=force(x,v,b);k1x=v
        k2v=force(x+.5*DT*k1x,v+.5*DT*k1v,b);k2x=v+.5*DT*k1v
        k3v=force(x+.5*DT*k2x,v+.5*DT*k2v,b);k3x=v+.5*DT*k2v
        k4v=force(x+DT*k3x,v+DT*k3v,b);k4x=v+DT*k3v
        x=x+DT/6*(k1x+2*k2x+2*k3x+k4x); v=v+DT/6*(k1v+2*k2v+2*k3v+k4v)
    return np.array(ts),np.array(rec)  # rec[t, beta, (ret, |u0|, |upi|)]
