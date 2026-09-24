import numpy as np
SQ5=np.sqrt(5); C=1.0; N=64; K=np.pi/2
def Q(q): return SQ5+2*C*(1-np.cos(q))
def L(q,w,b): return -w*w+Q(q)-2*C*b*w*np.sin(q)
def conv(c,H):  # c: dict m->complex for m=-H..H ; returns (u^2)_m
    out={}
    for m in range(-H,H+1):
        out[m]=sum(c[j]*c[m-j] for j in range(-H,H+1) if -H<=m-j<=H)
    return out
def wave(A,b,H=8,sign=+1):
    """exact travelling wave u=sum c_m e^{im(sign*K n - w t)} by harmonic balance"""
    k=sign*K
    c={m:0j for m in range(-H,H+1)}; c[1]=c[-1]=A/2
    bb=C*b*np.sin(k); w=-bb+np.sqrt(bb*bb+Q(k))
    for it in range(400):
        s=conv(c,H)
        rhs=-(s[1]/(A/2)).real
        q=Q(k)-rhs; w=-bb+np.sqrt(bb*bb+q)
        new={m:0j for m in range(-H,H+1)}; new[1]=new[-1]=A/2
        for m in range(-H,H+1):
            if abs(m)==1: continue
            new[m]=-s[m]/L(m*k,m*w,b)
        c=new
    return w,c
def growth(A,b,H=6,Hw=8,sign=+1,js=range(0,16)):
    w,c=wave(A,b,Hw,sign); k=sign*K
    best=(0,None)
    ms=np.arange(-H,H+1); n=len(ms)
    for j in js:
        q=2*np.pi*j/N
        Bm=np.zeros((n,n)); Cm=np.zeros((n,n),complex)
        for a,m in enumerate(ms):
            qq=q+m*k; s=np.sin(qq)
            Bm[a,a]=2*m*w+2*C*b*s
            Cm[a,a]=m*m*w*w+2*C*b*m*w*s-Q(qq)
            for bi,mm in enumerate(ms):
                jj=m-mm
                if abs(jj)<=Hw: Cm[a,bi]+=-2*c[jj]
        M=np.block([[np.zeros((n,n)),np.eye(n)],[-Cm,-np.diag(np.diag(Bm))]])
        g=np.max(np.linalg.eigvals(M).imag)
        if g>best[0]: best=(g,j)
    return best
if __name__=='__main__':
    # checks: self-shift, linear limit
    w0=wave(1e-4,0.0)[0]; wA=wave(0.1,0.0)[0]
    print("HB self-shift at beta=0: %+.4f A^2"%((wA-w0)/0.01))
    for bet in (0.025,0.05):
        dp=wave(0.1,bet,sign=+1)[0]; dm=wave(0.1,bet,sign=-1)[0]
        d0=wave(1e-4,bet,sign=+1)[0]-wave(1e-4,bet,sign=-1)[0]
        print(f"HB asym beta={bet}: dw(A=0)={d0:+.6f}  correction/(beta A^2)={((dp-dm)-d0)/(bet*0.01):+.4f}  -> kappa_equiv={((dp-dm)-d0)/(d0*0.01):+.4f}")
    BETAS=np.round(np.arange(0.020,0.1101,0.0025),4)
    for A in (0.10,0.15,0.20,0.25,0.30,0.35,0.40):
        G=[growth(A,b) for b in BETAS]
        g=np.array([x[0] for x in G]); i=int(np.argmax(g))
        # parabolic refine
        if 0<i<len(g)-1:
            y0,y1,y2=g[i-1:i+2]; d=0.5*(y0-y2)/(y0-2*y1+y2); bc=BETAS[i]+d*(BETAS[1]-BETAS[0])
        else: bc=BETAS[i]
        on=BETAS[g>1e-6]
        print(f"A={A:.2f}: max growth {g[i]:.5f} at beta~{bc:.4f} (j={G[i][1]}); unstable beta in [{on.min() if len(on) else np.nan:.4f},{on.max() if len(on) else np.nan:.4f}]; gain*T400={g[i]*400:.1f}")
