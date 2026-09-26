#!/usr/bin/env python3
"""
scale_invariance.py -- numerical check of the model's rescaling invariance (MODEL_SPEC sec 1b).

Every force term in model.py is homogeneous of degree 1 in the displacement except the
on-site nonlinearity (u*u elementwise, |psi| psi radial), which is of degree 2. So two
scalings are free: amplitude u = lam w and time t = s tau, with lam = K'/K and s = 1/sqrt(lam)
taking the linear stiffness K = sqrt5 to any K' while keeping the nonlinear coefficient 1.
The lattice spacing is fixed (no length scaling). Held fixed, the invariant combinations
    c/K, kappa/sqrtK, beta c/sqrtK, g c/sqrtK, C_r/sqrtK, A/K, T sqrtK
(and DT sqrtK, so RK4 maps step for step). The same run is made at K = sqrt5 (model.py),
K' = 2 (the z^2 - 1 form of the well), K' = 1 and K' = 7.3, and mapped back; the fields
must agree to rounding. Sectors: radial u(2) lattice with gauge links at kappa*;
elementwise u(2) lattice; scalar beta sector (n = 1, beta = 0.05); residual sector
(n = 8, C_r = 0.05, octonionic tower, kappa = 0). Control: c/K changed by 20% alone.
Monkeypatches model.SQ5, model.C and model.DT (module globals).

usage:  python3 scale_invariance.py   (from any directory; ~1 min)
"""
import os
import sys, importlib, importlib.util, numpy as np
SESSION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session")
sys.path.insert(0, SESSION)
import model as M
K0v, C0 = np.sqrt(5.0), 1.0

def run(Kp, case):
    """Run `case` in units where the linear stiffness is Kp (nonlinear coeff 1).
    Invariants held fixed: c/K, kappa/sqrtK, beta c/sqrtK, g c/sqrtK, C_r/sqrtK, A/K, T sqrtK."""
    lam = Kp / K0v                      # amplitude scale; time scale s = 1/sqrt(lam)
    M.SQ5 = Kp; M.C = C0 * lam; M.DT = 0.02 / np.sqrt(lam)
    sq = np.sqrt(lam)
    n, well, beta, g, Cr = case["n"], case["well"], case["beta"], case["g"], case["Cr"]
    kw = dict(n=n, N=200, kappa=case["kappa"] * sq, well=well)
    if beta is not None: kw["gyro_scalar"] = beta * sq / lam      # beta*c scales as sqrt(lam)
    if Cr:
        sp = importlib.util.spec_from_file_location("v2", os.path.join(SESSION, "d16_spectrum_v2.py"))
        v2m = importlib.util.module_from_spec(sp); sp.loader.exec_module(v2m)
        gv = np.zeros(16); gv[1:8] = np.random.default_rng(5).normal(size=7); gv /= np.linalg.norm(gv)
        kw.update(C_r=Cr * sq, tower=(v2m.cd(4), gv))
    lat = M.Lattice(**kw)
    u, v = lat.packet(amp=3e-2 * lam, width=6.0, n0=40, per_mode=(beta is None))
    if Cr: u[:, 8:] += 0.3 * u[:, :8]
    W = Wm = None
    if g:
        W, Wm = M.make_links(lat, [(80, 0, g * sq / lam), (100, 1, 0.8 * g * sq / lam)])
    u, v, _ = lat.run(u, v, 60.0 / sq + 0.5 * M.DT, W, Wm)
    return u / lam, v / (lam * sq)        # back to the reference units

cases = {"radial dimer lattice, u(2), kappa*, gauge": dict(n=2, well="radial", beta=None, g=0.12, Cr=0, kappa=M.KAPPA),
         "elementwise, u(2), gauge": dict(n=2, well="elementwise", beta=None, g=0.12, Cr=0, kappa=M.KAPPA),
         "scalar sector n=1, beta=0.05": dict(n=1, well=None, beta=0.05, g=0, Cr=0, kappa=M.KAPPA),
         "residual n=8, C_r=0.05, kappa=0": dict(n=8, well="radial", beta=None, g=0, Cr=0.05, kappa=0.0)}
for name, cs in cases.items():
    ua, va = run(K0v, cs)
    for Kp in (2.0, 1.0, 7.3):
        ub, vb = run(Kp, cs)
        print(f"{name:45s} K'={Kp}: max rel diff u {np.abs(ub-ua).max()/np.abs(ua).max():.1e}, v {np.abs(vb-va).max()/np.abs(va).max():.1e}")
# negative control: change c/K only
ua, _ = run(K0v, cases["radial dimer lattice, u(2), kappa*, gauge"])
M.SQ5 = K0v; M.C = 1.2; M.DT = 0.02
lat = M.Lattice(n=2, N=200, kappa=M.KAPPA, well="radial"); u, v = lat.packet(amp=3e-2, width=6.0, n0=40, per_mode=True)
W, Wm = M.make_links(lat, [(80, 0, 0.12), (100, 1, 0.096)]); ub, _, _ = lat.run(u, v, 60.0, W, Wm)
print("control, c/K changed by 20%: max rel diff", np.abs(ub-ua).max()/np.abs(ua).max())
