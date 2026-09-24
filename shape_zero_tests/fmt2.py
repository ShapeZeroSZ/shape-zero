import sys, json
for l in sys.stdin:
    if not l.startswith("{"): print(l.strip()); continue
    r=json.loads(l)
    print(r.get("mode","")+" W={:2d} L={:2d} g={:.4f} | eff_c={:7.3f} eff_x={:8.5f} ratio={:.5f} leak_x={:+.1e} | Wc rot pred {:6.2f} meas {:6.2f} err {:.3f} | drift {:.1e} cen {:.0f} T {:.0f}".format(
        r["W"],r["L"],r["g"],r["eff_c"],r["eff_x"],r["ratio"],r["leak_x"],r["rot_pred"],r["rot_meas"],r["inst_err"],r["drift"],r["centroid"],r["T"]))
