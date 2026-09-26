#!/usr/bin/env python3
"""
gate7A_predict.py -- predictions for model.py's gate 7/8 under the radial well
(default since 2026-09-26) with the clearing readout, from
model.precession_prediction (first-order self-precession; derivation in its
docstring and MODEL_SPEC sec 4d). No free parameter. Computes each pair's common
clearing time (the readout protocol) and the predicted final Bloch vectors; it does
not run the measured evolutions.

Disclosure: the measured clearing-readout values for this setup already exist
(shape_zero_tests/radialA_runs.json: u(2) floor 0.1603 deg, u(3) 0.0965 deg; per-order
errors against the LINEAR prediction 3.4-3.8 / 2.3-3.3 deg). The correction was
derived from the model's equations, not fitted to them.

Criterion (gate 7, clearing): per-order error < 1 deg, |split - predicted| < 1 deg,
|floor - predicted floor| < 0.05 deg. Also predicted: the elementwise well with the
fixed readout (SZ_J_WELL=elementwise SZ_GATE7_READOUT=fixed) reproduces
model_gates_kstar_v2.txt's gate lines exactly.

usage:  python3 gate7A_predict.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M

assert M.J_WELL == "radial"
for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10)):
    lat = M.Lattice(n=n, N=1200)
    psi0 = np.zeros(n, complex); psi0[0] = 1
    for axes, kind in (((0, 1), "split"), ((0, 0), "floor")):
        a0, a1 = axes
        runs = (("AB", [(60, a0, gA), (80, a1, gB)]), ("BA", [(60, a1, gB), (80, a0, gA)]))
        T = 1.1 * max(M.clear_time(lat, sp, (60, 80), 8.0, 20, 1e-3) for _, sp in runs)
        pr = {nm: M.precession_prediction(lat, sp, T, 8.0, 20, 1e-3, psi0) for nm, sp in runs}
        lat_e = M.Lattice(n=n, N=1200, well="elementwise")
        lin = {nm: M.precession_prediction(lat_e, sp, T, 8.0, 20, 1e-3, psi0) for nm, sp in runs}
        print(f"u({n}) {kind:5s} axes {axes}: clearing T = {T:.0f}; predicted {kind} "
              f"{M.angle(pr['AB'], pr['BA']):.4f} deg (linear product: {M.angle(lin['AB'], lin['BA']):.4f}); "
              f"precession moves AB by {M.angle(pr['AB'], lin['AB']):.3f} deg, BA by {M.angle(pr['BA'], lin['BA']):.3f} deg")
