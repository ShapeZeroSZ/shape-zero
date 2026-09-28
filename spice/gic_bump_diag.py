#!/usr/bin/env python3
"""gic_bump_diag.py -- POST HOC, after spice_diag.py: the GIC's realised on-site stiffness change against the design
value, at the two probe-line frequencies (56.5 and 67.2 kHz), for the extreme R5 values of the Gaussian bump (+-S),
and the band-fit c-hat of the own linear model with the same parasitics (does the design's own C_p explain c-hat
0.248?)."""
import os
import subprocess
import sys

import numpy as np

import gen_ring as GR
import spice_verify as SV

HERE = os.path.dirname(os.path.abspath(__file__))
e = SV.eta("gauss", 4)
R5s = {"base": GR.r5_for(GR.LGIC0), "max+": SV.bump_R5(e).min(), "min-": SV.bump_R5(-e).max()}
lines = ["* GIC impedance at the probe frequencies", ".include OPAx197.LIB", "VCC vcc 0 12", "VEE vee 0 -12"]
for i, (k, r5) in enumerate(R5s.items()):
    lines += [f"I{i} 0 a{i} dc 0 ac 1", f"R1{i} a{i} b{i} 2k", f"R2{i} b{i} c{i} 2k", f"R3{i} c{i} d{i} 2k", f"C4{i} d{i} e{i} 1n",
              f"R5{i} e{i} 0 {r5:.6g}", f"XA{i} a{i} c{i} vcc vee b{i} OPAx197", f"XB{i} c{i} e{i} vcc vee d{i} OPAx197"]
out = os.path.join(HERE, "out", "gic_bump.txt")
lines += [".control", "set wr_singlescale", "set wr_vecnames", "ac lin 3 56536 67206",
          "wrdata " + out + " " + " ".join(f"v(a{i})" for i in range(len(R5s))), ".endc", ".end"]
cir = out.replace(".txt", ".cir")
open(cir, "w").write("\n".join(lines) + "\n")
subprocess.run(["ngspice", "-b", cir], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
d = np.atleast_2d(np.loadtxt(out, skiprows=1))
print("POST HOC: GIC admittance 1/Z (as an inductance term 1/L = -w Im(1/Z)... ) at the probe frequencies")
for row in d:
    f = row[0]; w = 2 * np.pi * f
    Z = row[1::2] + 1j * row[2::2]
    invL = {k: -w * np.imag(1 / z) for k, z in zip(R5s, Z)}
    g = {k: np.real(1 / z) for k, z in zip(R5s, Z)}
    for k in ("max+", "min-"):
        des = 1 / (GR.GIC_K * R5s[k]) - 1 / (GR.GIC_K * R5s["base"])
        real = invL[k] - invL["base"]
        print(f"  f = {f / 1e3:.3f} kHz, {k}: d(1/L) realised {real:.4e}, design {des:.4e}, ratio {real / des:.4f}; "
              f"conductance change {g[k] - g['base']:+.2e} S")
sys.path.insert(0, os.path.join(HERE, "..", "shape_zero_tests"))
import circuit_error_budget as CB  # noqa: E402
p = SV.own_model()
Kc, cc, bc = CB.calibrate(p)
print(f"own model with the same parasitics (C_p 10 pF, C_in 6.5 pF, Q 100, 9.25 MHz pole): band-fit c-hat {cc / Kc:.5f}")
