import json, sys, numpy as np, j_compat_test as T
M=T.M
M.K0=np.pi/4
T.N_SITES, T.N0, T.SEG0 = 1200, 160, 300
def ramp(L):
    q=L//4
    return [j/(q+1) for j in range(1,q+1)] + [1.0]*(L-2*q) + [j/(q+1) for j in range(q,0,-1)]
assert ramp(12)==[0.25,0.5,0.75]+[1.0]*6+[0.75,0.5,0.25]
THR=1e-6
OLD_EXIT=T.exit_time
MODE={'readout':'clear'}
_cache={}
def clear_time_(lat):
    """Run the no-segment packet until < THR of its weight lies within 10 sites
    of the segment (either side, so a wrapped front also counts); +10% margin."""
    u,v=lat.packet(amp=1e-3,n0=T.N0,width=T.WIDTH,colour=0)
    lo,hi=T.SEG0-10,T.SEG0+len(M.RAMP)+10
    idx=np.arange(lat.N); win=(idx>=lo)&(idx<hi)
    t, seen = 0.0, False
    while t < 4000:
        u,v,_=lat.run(u,v,5.0); t+=5.0
        w=(u*u+(v*v)/lat.omega**2).sum(axis=1)
        frac=w[win].sum()/w.sum()
        if frac>1e-3: seen=True
        if seen and frac<THR: return 1.1*t
    raise RuntimeError("packet never cleared the segment window")
def clear_time(lat):
    if MODE['readout']=='old': return OLD_EXIT(lat)
    key=(T.WIDTH,len(M.RAMP),lat.N,M.K0)
    if key not in _cache: _cache[key]=clear_time_(lat)
    return _cache[key]
T.exit_time = clear_time
def config(W,L):
    T.WIDTH=W; M.RAMP=ramp(L)
def one(W,L,g,seed=1):
    config(W,L)
    _,Wc,Wx,_=T.split_W(seed); H=T.to_complex(Wc)
    r=T.measure(M.SQ5,g,Wc,Wx,H)
    return dict(W=W,L=L,g=g,eff_c=r["eff_c"],eff_x=r["eff_x"],ratio=r["eff_x"]/r["eff_c"],leak_x=r["leak_x"],
        rot_pred=M.angle(np.array([0,0,1.]),r["co_pred"]),rot_meas=M.angle(r["co_ref"],r["co_c"]),
        inst_err=M.angle(r["co_c"],r["co_pred"]),drift=max(r["drift_c"],r["drift_x"]),centroid=r["centroid_ref"],T=r["T"])
if __name__!="__main__":
    pass
elif sys.argv[1]=="checknew":
    for mode in ('old','clear'):
        MODE['readout']=mode
        for g in (0.0025,0.005,0.015):
            r=one(8,12,g); r['mode']=mode; print(json.dumps(r),flush=True)
elif sys.argv[1]=="check200":
    T.N_SITES, T.N0, T.SEG0 = 200, 20, 60
    for W,L,g in ((8,12,0.0025),(8,12,0.005),(8,12,0.015)):
        print(json.dumps(one(W,L,g)),flush=True)
elif sys.argv[1]=="check":
    # same-geometry baseline must reproduce the N=200 result's ratio closely
    for W,L,g in ((8,12,0.0025),(8,12,0.015),(32,12,0.015),(8,48,0.015)):
        print(json.dumps(one(W,L,g)),flush=True)
else:
    for W,L in ((8,12),(16,12),(32,12),(8,24),(8,48)):
        for g in (0.0025,0.005): print(json.dumps(one(W,L,g)),flush=True)
