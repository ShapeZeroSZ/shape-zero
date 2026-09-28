#!/usr/bin/env python3
"""spice_diag.py -- POST HOC diagnostics, written after spice_verify.py's output was seen. Not part of the
verification plan. Locates the SPICE/own-model disagreements (P2 R 1.083 vs 1.031; c-hat 0.2475; product rule
4e-3; stability threshold) by swapping, in SPICE, the GIC for an ideal inductor and/or the op-amp VCCS for ideal
G-sources. Variant C (both ideal) solves the same equations as the own linear model with ideal parts: it checks the
netlist-to-model mapping and the estimator.
usage: python3 spice_diag.py"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gen_ring as GR
import spice_verify as SV

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"A_idealGIC": dict(gic=False, vccs="opamp"), "B_idealVCCS": dict(gic=True, vccs="ideal"),
            "C_bothideal": dict(gic=False, vccs="ideal")}
WIN = dict(f=(54.0e3, 69.8e3, 25.0))


def run(name, **kw):
    out = os.path.join(HERE, "out", name + ".txt")
    if not os.path.exists(out):
        cir = out.replace(".txt", ".cir")
        open(cir, "w").write(GR.netlist(out=out, **kw))
        subprocess.run(["ngspice", "-b", cir], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return SV.ac_data(np.loadtxt(out, skiprows=1))


def main():
    e = SV.eta("gauss", 4)
    jobs = []
    for v, kw in VARIANTS.items():
        jobs += [(f"d_{v}_band", dict(gyro=False, **kw)), (f"d_{v}_p1", kw), (f"d_{v}_S0", dict(**WIN, **kw)),
                 (f"d_{v}_g4p", dict(R5=SV.bump_R5(e), **WIN, **kw)), (f"d_{v}_g4m", dict(R5=SV.bump_R5(-e), **WIN, **kw)),
                 (f"d_{v}_Q450", dict(Q=450, **kw))]
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda j: run(j[0], **j[1]), jobs))
    print("POST HOC SPICE DIAGNOSTICS (variant: c-hat, beta-hat, product rule, P1 max, P2 R, min decay rate at Q = 450)")
    print("  original (both real, from spice_verify): c-hat 0.24747, beta-hat 0.10533, product 4.0e-03, P1 1.59e-03, "
          "R 1.0826, Q450 -21.7 s^-1")
    for v in VARIANTS:
        fb, Vb = run(f"d_{v}_band", **VARIANTS[v])
        ms = list(range(17))
        band = SV.lines(fb, Vb, ms, gyro=False)
        wb = np.array([np.pi * (band[(m, 1)] + band[(m, -1)]) for m in ms])
        X = np.vstack([np.ones(17), 2 * (1 - np.cos(2 * np.pi * np.array(ms) / 32))]).T
        (K, c), *_ = np.linalg.lstsq(X, wb ** 2, rcond=None)
        fp, Vp = run(f"d_{v}_p1", **VARIANTS[v])
        L = SV.lines(fp, Vp, range(1, 16))
        dw = {m: 2 * np.pi * (L[(m, 1)] - L[(m, -1)]) for m in range(1, 16)}
        b = dw[8] / 2
        p1 = max(abs(dw[m] / dw[8] - np.sin(2 * np.pi * m / 32)) for m in range(1, 16))
        prod = max(abs(4 * np.pi ** 2 * L[(m, 1)] * L[(m, -1)] / (K + 2 * c * (1 - np.cos(2 * np.pi * m / 32))) - 1) for m in range(1, 16))
        d = {}
        for tag in ("S0", "g4p", "g4m"):
            ff, VV = run(f"d_{v}_{tag}", **VARIANTS[v])
            Lx = SV.lines(ff, VV, [8])
            d[tag] = 2 * np.pi * (Lx[(8, 1)] - Lx[(8, -1)])
        shift = 0.5 * (d["g4p"] + d["g4m"]) - d["S0"]
        R = shift / (-0.25 * b * np.mean((K * SV.S * e) ** 2) / c ** 2)
        fq, Vq = run(f"d_{v}_Q450", **VARIANTS[v])
        rates = []
        for m in range(1, 16):
            for sign in (1, -1):
                po, Rr = SV.vf(fq, Vq, SV.design_f(m, sign))
                rates.append(2 * np.pi * SV.pick(po, Rr, m, sign, SV.design_f(m, sign))[0].imag)
        print(f"  {v:<12s}: c-hat {c / K:.5f}, beta-hat {b / np.sqrt(K):.5f}, product {prod:.1e}, P1 {p1:.2e}, R {R:.4f}, "
              f"Q450 {min(rates):+.1f} s^-1")


if __name__ == "__main__":
    main()
