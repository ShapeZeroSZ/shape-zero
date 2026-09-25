#!/usr/bin/env python3
"""
q3_floor_fix_summary.py -- before / after the two readout fixes (per-mode launch;
one common readout time per pair), from saved outputs only.

  q3_gate.py   before: q3_gate_runs/result_260x8[_kstar].json  (carrier launch,
                        each run read at its own clearing time)
               after : q3_gate_runs/result_260x8_{kstar,k0.5}_v2.json
  model.py     before: model_gates_kstar.txt, jcompat_gates_k0.5.txt (carrier launch)
               after : model_gates_kstar_v2.txt, model_gates_k0.5_v2.txt
                        (per-mode launch in gate 3 and ordering_test)

usage:  python3 q3_floor_fix_summary.py
"""
import json
import re


def q3(tag):
    runs = json.load(open(f"q3_gate_runs_260x8{tag}.json"))["runs"]
    res = json.load(open(f"q3_gate_result_260x8_averaged{tag}.json"))["rows"]
    rows = {r["check"]: r for r in res}
    out = {}
    for n in (2, 3):
        rr = [r for r in runs if r["n"] == n]
        out[n] = dict(
            order=(rows[f"u({n}) AB order"]["value"], rows[f"u({n}) BA order"]["value"]),
            split_err=rows[f"u({n}) split"]["value"], floor=rows[f"u({n}) Abelian floor"]["value"],
            split=(rows[f"u({n}) split measured / predicted"]["value"],
                   rows[f"u({n}) split measured / predicted"]["predicted"]),
            t={r["job"]: r["t"] for r in rr},
            pur=(min(r["pur"] for r in rr), max(r["pur"] for r in rr)),
            pur_mode=((min(r["pur_mode"] for r in rr), max(r["pur_mode"] for r in rr))
                      if "pur_mode" in rr[0] else None))
    return out


def gates(path):
    t = open(path).read()
    g = {}
    g["3"] = re.search(r"chirality purity ([0-9.]+)", t).group(1)
    for n, lab in ((2, "u\\(2\\)"), (3, "u\\(3\\)")):
        m = re.search(lab + r" ordering.*\n\s+split ([0-9.]+) deg vs predicted ([0-9.]+) deg \| per-order "
                      r"([0-9.]+) / ([0-9.]+) deg \| Abelian floor ([0-9.]+)", t)
        g[f"7 u({n})"] = m.groups()
    g["8"] = re.search(r"8\. Abelian control.*\n\s+([0-9.]+) deg", t).group(1)
    return g


def main():
    print("=" * 92)
    print("q3_gate.py (260 x 8 x 8, spectrum-averaged predictor)")
    print("=" * 92)
    for kap, before, after in (("0.5", "", "_k0.5_v2"), ("kappa*", "_kstar", "_kstar_v2")):
        b, a = q3(before), q3(after)
        print(f"\n  kappa = {kap}")
        for n in (2, 3):
            print(f"   u({n}) readout times   before {b[n]['t']}\n                       after  {a[n]['t']}")
            print(f"   u({n}) Abelian floor   {b[n]['floor']:.4f} -> {a[n]['floor']:.4f} deg")
            print(f"   u({n}) split meas/pred {b[n]['split'][0]:.2f} / {b[n]['split'][1]:.2f} -> "
                  f"{a[n]['split'][0]:.2f} / {a[n]['split'][1]:.2f}   (error {b[n]['split_err']:.3f} -> "
                  f"{a[n]['split_err']:.3f} deg)")
            print(f"   u({n}) per-order AB/BA {b[n]['order'][0]:.3f} / {b[n]['order'][1]:.3f} -> "
                  f"{a[n]['order'][0]:.3f} / {a[n]['order'][1]:.3f} deg")
            pm = a[n]["pur_mode"]
            print(f"   u({n}) purity (carrier readout) {b[n]['pur'][0]:.5f}-{b[n]['pur'][1]:.5f} -> "
                  f"{a[n]['pur'][0]:.5f}-{a[n]['pur'][1]:.5f};  per-mode readout after: "
                  f"{pm[0]:.6f}-{pm[1]:.6f}")
    print("\n" + "=" * 92)
    print("model.py gates 3, 7, 8 (carrier launch -> per-mode launch)")
    print("=" * 92)
    for kap, before, after in (("0.5", "jcompat_gates_k0.5.txt", "model_gates_k0.5_v2.txt"),
                               ("kappa*", "model_gates_kstar.txt", "model_gates_kstar_v2.txt")):
        b, a = gates(before), gates(after)
        print(f"\n  kappa = {kap}")
        print(f"   gate 3 chirality purity   {b['3']} -> {a['3']}")
        for n in (2, 3):
            k = f"7 u({n})"
            print(f"   gate 7 u({n}) split / pred / per-order / floor   "
                  f"{b[k][0]} / {b[k][1]} / {b[k][2]}, {b[k][3]} / {b[k][4]}  ->  "
                  f"{a[k][0]} / {a[k][1]} / {a[k][2]}, {a[k][3]} / {a[k][4]}")
        print(f"   gate 8 Abelian control    {b['8']} -> {a['8']} deg")


if __name__ == "__main__":
    main()
