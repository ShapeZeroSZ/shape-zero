import json, sys, numpy as np, j_compat_test as T
M=T.M
def setk(k0): M.K0 = k0            # packet phase, omega, exit time and k_branch all read M.K0 at call time
def one(k0,g,seed=1):
    setk(k0); W,Wc,Wx,J=T.split_W(seed); H=T.to_complex(Wc)
    r=T.measure(M.SQ5,g,Wc,Wx,H)
    lat=T.KLattice(n=2,K=M.SQ5,N=200)
    return dict(k0=k0,g=g,seed=seed,omega=r["omega"],cos_kp=np.cos(k0)+lat.kappa*lat.omega/M.C,
        eff_c=r["eff_c"],eff_x=r["eff_x"],ratio=r["eff_x"]/r["eff_c"],leak_x=r["leak_x"],
        rot_pred=M.angle(np.array([0,0,1.]),r["co_pred"]),rot_meas=M.angle(r["co_ref"],r["co_c"]),
        inst_err=M.angle(r["co_c"],r["co_pred"]),drift=max(r["drift_c"],r["drift_x"]),centroid=r["centroid_ref"],T=r["T"])
mode=sys.argv[1]
if mode=="check":
    r=one(np.pi/2,0.01); print("REPRO pi/2 g=0.01 ratio %.4f (earlier 0.0883)"%r["ratio"],flush=True)
    for k0 in (np.pi/4,3*np.pi/4):
        for g in (0.03,0.06):
            print(json.dumps(one(k0,g)),flush=True)
else:
    for k0 in (np.pi/4,3*np.pi/4):
        for g in (0.0025,0.005,0.01,0.02,0.04): print(json.dumps(one(k0,g)),flush=True)
        for seed in (2,3):
            for g in (0.0025,0.01): print(json.dumps(one(k0,g,seed)),flush=True)
