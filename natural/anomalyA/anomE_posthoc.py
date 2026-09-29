#!/usr/bin/env python3
"""anomE_posthoc.py -- POST-HOC (written after anomE_purity_output.txt): is the residual per-mode purity loss of the
isolated reference (0.0032, reached by t ~ 250 and then flat) a nonlinear DRESSING that scales with amplitude, rather
than a continuing loss? Same set-up, amplitudes 0.025, 0.05, 0.1, T = 1000; and the elementwise-free check: the same
packet under a LINEAR force (well term removed) must keep per-mode purity 1."""
import os, sys
import numpy as np
WT = os.environ["MAINWT"]
sys.path.insert(0, os.path.join(WT, "04_scripts", "session")); sys.path.insert(0, os.path.join(WT, "shape_zero_tests"))
import model as M
import tower_populated_test as TP

def series(amp, linear=False):
    lat = M.Lattice(n=4, N=TP.N, well="node")
    if linear:
        lat._onsite_nl = lambda u: 0 * u
        lat._onsite_cubic = lambda u: 0.0
    u, v = lat.packet(amp=amp, n0=TP.N // 2, width=8.0, per_mode=True)
    out = [lat.readout_modes(u, v)[1]]
    for _ in range(4):
        u, v, _ = lat.run(u, v, 250.0); out.append(lat.readout_modes(u, v)[1])
    return out

print("POST-HOC -- per-mode purity loss (1 - purity) at t = 0, 250, 500, 750, 1000")
for amp in (0.025, 0.05, 0.1):
    s = series(amp)
    print(f"   A = {amp:<6}: " + "  ".join(f"{1-x:.2e}" for x in s) + f"   (loss / A at T: {(1-s[-1])/amp:.4f})")
s = series(0.05, linear=True)
print("   A = 0.05, force linearised (no well nonlinearity): " + "  ".join(f"{1-x:.1e}" for x in s))
