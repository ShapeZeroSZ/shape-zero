import json, numpy as np, j_compat_test as T
M=T.M
W,Wc,Wx,J=T.split_W(1); H=T.to_complex(Wc)
rows=[]
for K in (M.SQ5/4, M.SQ5, 4*M.SQ5, 16*M.SQ5):
    for g in (0.0025, 0.005, 0.01, 0.02, 0.04):
        r=T.measure(K,g,Wc,Wx,H)
        rot_pred=M.angle(np.array([0,0,1.]),r["co_pred"]); rot_meas=M.angle(r["co_ref"],r["co_c"])
        row=dict(K=K,g=g,omega=r["omega"],g_over_omega=g/r["omega"],eff_c=r["eff_c"],eff_x=r["eff_x"],
                 ratio=r["eff_x"]/r["eff_c"],leak_x=r["leak_x"],leak_c=r["leak_c"],rot_pred=rot_pred,rot_meas=rot_meas,
                 inst_err=M.angle(r["co_c"],r["co_pred"]),drift=max(r["drift_c"],r["drift_x"]),centroid=r["centroid_ref"],T=r["T"])
        rows.append(row); print(json.dumps(row),flush=True)
json.dump(rows,open("grid.json","w"),indent=1)
