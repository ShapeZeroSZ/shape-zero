#!/usr/bin/env python3
"""
kstar_rerun_summary.py -- every kappa-dependent result re-run at the reference
operating point kappa = kappa* = 2c/sqrt(K + 2c) = 0.971737 (adopted 2026-09-25,
MODEL_SPEC §3), side by side with the kappa = 0.5 value it supersedes.

Reads only saved outputs (no simulation):
  model.py gates       jcompat_gates_k0.5.txt   vs  model_gates_kstar.txt
  gate7_readout.py     gate7.jsonl              vs  gate7_kstar.jsonl
  q3_gate.py           q3_gate_result_260x8_{averaged,carrier}.json  vs  ..._kstar.json
  J-compat table       kscan.jsonl, open_pi2.jsonl, open_3pi4.jsonl  vs  *_kstar.jsonl
  instrument check     0.0883 (README)          vs  kscan_check_kstar.txt
  P-3                  -0.0896 / -0.0899 (predictions_v1 P-3)  vs  p3_kstar.txt
The kappa = 0.5 values are superseded by the change of operating point, not
retracted: they were correct at kappa = 0.5.

usage:  python3 kstar_rerun_summary.py
"""
import json
import re


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip().startswith("{")]


def gate_lines(p):
    return {m.group(1): m.group(2).strip() for m in
            re.finditer(r"\[(?:PASS|FAIL)\] (\d+\.[^\n]*)\n\s+([^\n]*)", open(p).read())}


def main():
    print("=" * 88)
    print("kappa-DEPENDENT RESULTS: kappa = 0.5 (superseded)  ->  kappa* = 0.971737")
    print("=" * 88)

    print("\n1. model.py gates (status line and its value line)")
    a, b = open("jcompat_gates_k0.5.txt").read(), open("model_gates_kstar.txt").read()
    sa = re.findall(r"\[(PASS|FAIL)\] ([^\n]*)\n\s+([^\n]*)", a)
    sb = re.findall(r"\[(PASS|FAIL)\] ([^\n]*)\n\s+([^\n]*)", b)
    for (pa, na, va), (pb, nb, vb) in zip(sa, sb):
        tag = "same" if va == vb else "CHANGED"
        print(f"  {na[:44]:44s} {pa}->{pb}  {tag}")
        if va != vb:
            print(f"      0.5 : {va}\n      k*  : {vb}")

    print("\n2. gate7_readout.py (q = 1; split / predicted / floor, degrees)")
    for old, new in zip(jl("gate7.jsonl"), jl("gate7_kstar.jsonl")):
        tag = old.get("tag")
        pred = f"{old.get('pred_split', float('nan')):.2f} -> {new.get('pred_split', float('nan')):.2f}"
        print(f"  u({old['n']}) {tag:8s} split {old['split']:.2f} -> {new['split']:.2f}   "
              f"pred {pred}   floor {old['floor']:.5f} -> {new['floor']:.5f}   "
              f"per-order {old['simprod_AB']:.2f}/{old['simprod_BA']:.2f} -> "
              f"{new['simprod_AB']:.2f}/{new['simprod_BA']:.2f}")

    print("\n3. q3_gate.py (260 x 8 x 8)")
    for pr in ("averaged", "carrier"):
        try:
            o = json.load(open(f"q3_gate_result_260x8_{pr}.json"))
            n = json.load(open(f"q3_gate_result_260x8_{pr}_kstar.json"))
        except FileNotFoundError as e:
            print(f"  {pr}: missing {e.filename}")
            continue
        print(f"  predictor {pr}: gate {'PASS' if o['passed'] else 'FAIL'} -> {'PASS' if n['passed'] else 'FAIL'}")
        for ro, rn in zip(o["rows"], n["rows"]):
            if "tol" in ro:
                print(f"     {ro['check']:24s} {ro['value']:8.3f} -> {rn['value']:8.3f} deg (tol {ro['tol']})")
            else:
                print(f"     {ro['check']:34s} {ro['value']:.2f}/{ro['predicted']:.2f} -> "
                      f"{rn['value']:.2f}/{rn['predicted']:.2f}")

    print("\n4. J-compatibility table (seed 1; ratio = eff_x/eff_c, leak_x)")
    print("   old readout (kscan.py; the pi/2 row is the check mode at g = 0.01):")
    for k0, name in ((0.7853981633974483, "pi/4"), (2.356194490192345, "3pi/4")):
        o = [d for d in jl("kscan.jsonl") if abs(d["k0"] - k0) < 1e-9 and d["seed"] == 1]
        n = [d for d in jl("kscan_kstar.jsonl") if abs(d["k0"] - k0) < 1e-9 and d["seed"] == 1]
        print(f"   {name:5s} cos k' {o[0]['cos_kp']:+.3f} -> {n[0]['cos_kp']:+.3f}")
        for do, dn in zip(o, n):
            print(f"      g {do['g']:6.4f}  ratio {do['ratio']:.4f} -> {dn['ratio']:.4f}   "
                  f"leak {do['leak_x']:.1e} -> {dn['leak_x']:.1e}")
    print("   clearing readout (openrows.py, 1200 sites):")
    for name, fo, fn in (("pi/2", "open_pi2.jsonl", "open_pi2_kstar.jsonl"),
                         ("3pi/4", "open_3pi4.jsonl", "open_3pi4_kstar.jsonl")):
        o = [d for d in jl(fo) if d["seed"] == 1]
        n = [d for d in jl(fn) if d["seed"] == 1]
        for do, dn in zip(o, n):
            print(f"   {name:5s} g {do['g']:6.4f}  ratio {do['ratio']:.4f} -> {dn['ratio']:.4f}   "
                  f"leak {do['leak_x']:.1e} -> {dn['leak_x']:.1e}")
        so = [d for d in jl(fo) if d["seed"] != 1]
        sn = [d for d in jl(fn) if d["seed"] != 1]
        for do, dn in zip(so, sn):
            print(f"   {name:5s} seed {do['seed']} g {do['g']:6.4f}  ratio {do['ratio']:.4f} -> {dn['ratio']:.4f}")

    print("\n5. README instrument check (kscan.py check, pi/2, g = 0.01)")
    m = re.search(r"ratio ([0-9.]+)", open("kscan_check_kstar.txt").read())
    print(f"   0.0883 -> {m.group(1)}")

    print("\n6. P-3 spinor self-precession coefficient C")
    t = open("p3_kstar.txt").read()
    cp = re.search(r"C = ([+-][0-9.]+)", t).group(1)
    cm = re.search(r"C_measured = ([+-][0-9.]+)", t).group(1)
    res = re.search(r"max residual ([0-9.]+)%", t).group(1)
    print(f"   predicted -0.0896 -> {cp};  measured -0.0899 -> {cm}  (max residual {res}%)")


if __name__ == "__main__":
    main()
