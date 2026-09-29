#!/usr/bin/env python3
"""anomE_purity.py -- anomaly E: tower_populated's isolated reference (main's own code: D8, A', ring N = 128, kappa*,
amp 0.05, width 8, k0 = pi/2, per-mode launch, T = 2000), chirality purity by the single-carrier readout (as recorded)
and by the per-mode readout readout_modes. Predictions: ANOMALY_A_PREDICTIONS.md (534805c)."""
import os, sys
import numpy as np
WT = os.environ.get("MAINWT", "/tmp/claude-0/-home-user-shape-zero/7f4f19ab-d202-57e0-98c4-ef10d02be803/scratchpad/mainwt")
sys.path.insert(0, os.path.join(WT, "04_scripts", "session")); sys.path.insert(0, os.path.join(WT, "shape_zero_tests"))
import model as M
import tower_populated_test as TP

lat = M.Lattice(n=4, N=TP.N, well="node")
u, v = lat.packet(amp=TP.AMP, n0=TP.N // 2, width=8.0, per_mode=True)
def pur_sc(u, v):
    chi, bar = TP.chi_low(lat, u, v); return 1 - np.linalg.norm(bar) / np.linalg.norm(chi)
rows = [(0.0, pur_sc(u, v), lat.readout_modes(u, v)[1])]
E0 = lat.energy(u, v); t = 0.0
while t < TP.T - 1e-9:
    u, v, _ = lat.run(u, v, 250.0); t += 250.0
    rows.append((t, pur_sc(u, v), lat.readout_modes(u, v)[1]))
print("ANOMALY E -- isolated reference purity: single-carrier (recorded 0.9872 -> 0.9494) and per-mode readout")
for t, a, b in rows:
    print(f"   t = {t:6.0f}: single-carrier {a:.4f}   per-mode {b:.6f}")
print(f"   change 0 -> T: single-carrier {rows[-1][1]-rows[0][1]:+.4f}; per-mode {rows[-1][2]-rows[0][2]:+.2e}; "
      f"energy drift {abs(lat.energy(u, v)-E0)/abs(E0):.1e}")
