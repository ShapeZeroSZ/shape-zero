#!/usr/bin/env python3
"""stab_variantA.py -- POST HOC: stability window of variant A (ideal inductor in place of the GIC, i.e. the proposed
passive trim element; real OPA4197 VCCS), 32 nodes, from the AC pole widths (as spice_verify.py), Q = 100, 200, 300;
Q = 450 is in out_gic_v1/d_A_idealGIC_Q450 (the GIC does not enter variant A)."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gen_ring as GR
import spice_verify as SV

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "out_variantA"), exist_ok=True)


def run(Q):
    out = os.path.join(HERE, "out_variantA", f"A_Q{Q}.txt")
    if not os.path.exists(out):
        cir = out.replace(".txt", ".cir")
        open(cir, "w").write(GR.netlist(out=out, Q=Q, gic=False, f=(48e3, 75e3, 25.0)))
        subprocess.run(["ngspice", "-b", cir], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out


with ThreadPoolExecutor(3) as ex:
    files = dict(zip((100, 200, 300), ex.map(run, (100, 200, 300))))
files[450] = os.path.join(HERE, "out_gic_v1", "d_A_idealGIC_Q450.txt")
print("POST HOC: variant A (ideal on-site inductor, real OPA4197 VCCS), 32 nodes: min signed decay rate over m = 1..15")
for Q in (100, 200, 300, 450):
    f, V = SV.ac_data(np.loadtxt(files[Q], skiprows=1))
    rates = []
    for m in range(1, 16):
        for sign in (1, -1):
            po, R = SV.vf(f, V, SV.design_f(m, sign))
            rates.append(2 * np.pi * SV.pick(po, R, m, sign, SV.design_f(m, sign))[0].imag)
    print(f"  Q = {Q}: min decay rate {min(rates):+.3e} s^-1 ({'stable' if min(rates) > 0 else 'unstable'})")
