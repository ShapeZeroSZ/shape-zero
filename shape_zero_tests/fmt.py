import sys, json, numpy as np
for l in sys.stdin:
    if not l.startswith("{"):
        print(l.strip()); continue
    r = json.loads(l)
    print("k0={:.2f}pi seed={} w={:.3f} cos kp={:.3f} g={:.4f} | eff_c={:7.3f} eff_x={:8.5f} ratio={:.4f} leak_x={:+.1e} | Wc rot pred {:6.2f} meas {:6.2f} err {:.3f} | drift {:.1e} cen {:.0f} T {:.0f}".format(
        r["k0"]/np.pi, r["seed"], r["omega"], r["cos_kp"], r["g"], r["eff_c"], r["eff_x"], r["ratio"], r["leak_x"],
        r["rot_pred"], r["rot_meas"], r["inst_err"], r["drift"], r["centroid"], r["T"]))
