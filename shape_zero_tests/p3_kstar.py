#!/usr/bin/env python3
"""
p3_kstar.py -- P-3 (spinor self-precession) at the model's reference operating
point kappa = kappa* = 2c/sqrt(K + 2c) = 0.971737 (MODEL_SPEC §3).

Runs 04_scripts/platform/phi_gauge_precession.py unchanged (prediction C and the
nine-run global fit), after setting phi_gauge_chiral's KAPPA, OMEGA and VG --
module globals read at call time -- to kappa*. The platform scripts keep their own
KAPPA = 0.5; their output at 0.5 (C = -0.0896 predicted, -0.0899 measured) is
superseded by the change of operating point, not retracted.

usage:  python3 p3_kstar.py
"""
import os
import runpy
import sys

import numpy as np

PLAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "platform")
sys.path.insert(0, PLAT)
import phi_gauge_chiral as P

P.KAPPA = 2 * P.C / np.sqrt(P.SQ5 + 2 * P.C)
P.OMEGA = 0.5 * (-P.KAPPA + np.sqrt(P.KAPPA ** 2 + 4 * (P.SQ5 + 2 * P.C * (1 - np.cos(P.K0)))))
P.VG = 2 * P.C * np.sin(P.K0) / (2 * P.OMEGA + P.KAPPA)
print(f"kappa = kappa* = {P.KAPPA:.6f}, omega = {P.OMEGA:.6f}, v_g = {P.VG:.6f}", flush=True)
runpy.run_path(os.path.join(PLAT, "phi_gauge_precession.py"), run_name="__main__")
