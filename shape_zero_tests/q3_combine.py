import os,sys,json,numpy as np
# model.py pinned to archive 948b09e8 (the q = 3 runs); MODEL_DIR overrides it.
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.environ.get("MODEL_DIR",os.path.join(HERE,"model_versions","948b09e8"))); import model as M
G={2:(0.12,0.08),3:(0.15,0.15)}
def res(n):
    R={j:json.load(open(os.path.join(HERE,"q3",f"{n}_{j}.json"))) for j in ("AB","BA","fAB","fBA")}
    lat=M.Lattice(n=n,N=200); psi0=np.zeros(n,complex); psi0[0]=1
    gA,gB=G[n]; out=[]
    for k in ("fixed","centroid","clear"):
        a,b,fa,fb=(R[j][k] for j in ("AB","BA","fAB","fBA"))
        cA,cB=np.array(a["co"]),np.array(b["co"])
        pAB=M.coords_of_state(lat,M.U_segment(lat,1,gB,a["Qt"])@M.U_segment(lat,0,gA,a["Qt"])@psi0)
        pBA=M.coords_of_state(lat,M.U_segment(lat,0,gA,b["Qt"])@M.U_segment(lat,1,gB,b["Qt"])@psi0)
        split=M.angle(cA,cB); psplit=M.angle(pAB,pBA)
        out.append(dict(readout=k,t=(a["t"],b["t"],fa["t"],fb["t"]),split=split,pred_split=psplit,split_err=abs(split-psplit),
            floor=M.angle(np.array(fa["co"]),np.array(fb["co"])),simprod=(M.angle(cA,pAB),M.angle(cB,pBA)),
            maxwin=max(max(x["windows"]) for x in (a,b,fa,fb)),drift=max(x["drift"] for x in (a,b,fa,fb))))
    return out
allrows=[]
for n in [int(a) for a in sys.argv[1:]]:
    for r in res(n):
        r["n"]=n; allrows.append(r)
        print(f"u({n}) {r['readout']:8} t={'/'.join(str(int(x)) for x in r['t']):16} max window {r['maxwin']:.1e} | split {r['split']:6.2f} pred {r['pred_split']:6.2f} err {r['split_err']:5.2f} | Abelian floor {r['floor']:5.3f} | sim-vs-product {r['simprod'][0]:.2f}/{r['simprod'][1]:.2f} | drift {r['drift']:.1e}")
json.dump(allrows,open(os.path.join(HERE,"q3_old_vs_new.json"),"w"),indent=1)
