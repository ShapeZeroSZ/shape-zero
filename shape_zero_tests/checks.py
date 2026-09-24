import json, numpy as np, j_compat_test as T
M=T.M
def row(tag,K,g,seed):
    W,Wc,Wx,J=T.split_W(seed); H=T.to_complex(Wc)
    r=T.measure(K,g,Wc,Wx,H)
    lat=T.KLattice(n=2,K=K,N=200)
    ck=lat.kappa*lat.omega/M.C + np.cos(M.K0)      # cos k' of the opposite-chirality mode at the packet's frequency
    print(json.dumps(dict(tag=tag,K=K,seed=seed,g=g,omega=r["omega"],cos_kprime=ck,channel_open=bool(abs(ck)<=1),
          eff_c=r["eff_c"],eff_x=r["eff_x"],ratio=r["eff_x"]/r["eff_c"],leak_x=r["leak_x"],
          inst_err=M.angle(r["co_c"],r["co_pred"]),drift=max(r["drift_c"],r["drift_x"]))),flush=True)
for seed in (2,3):
    for K in (M.SQ5, 4*M.SQ5):
        for g in (0.0025,0.01): row("seed",K,g,seed)
for K in (2.8,3.2):
    for g in (0.0025,0.01): row("threshold",K,g,1)
