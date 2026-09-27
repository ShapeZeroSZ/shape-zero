#!/usr/bin/env python3
"""
continuum_checks.py -- the continuum limit and the frame readings of kappa and beta
(MODEL_SPEC sec 1c, 2026-09-27).

STATUS: DERIVED, THEN CHECKED -- NOT PREDICTED. No predictions were committed before the
derivation; the checks below were written after it.

T1  KAPPA IS A ROTATING FRAME (Larmor's theorem), exact on the lattice at all orders.
    In the J sector without links, psi(t) = exp(mu t JJ) phi(t), mu = kappa/2, maps
        u'' = F(u) + kappa JJ u'        onto        w'' = F(w) - mu^2 w     (kappa = 0, K -> K + mu^2)
    whenever F commutes with the rotation: the elastic term and K always; the well for node
    forms A' ('node') and A ('radial'); NOT for the elementwise form (the control).
    Both systems are integrated (RK4, amplitude ~1, q = 1 and q = 2, n = 2) with model.py's
    own force and compared at T = 20; agreement at RK4 accuracy (error x16 per halving of dt).
T2  BETA (scalar sector).  (a) At fixed frequency, EXACTLY on the lattice, a Peierls phase
    growing with frequency:
        w^2 = K + 2c - 2c sqrt(1 + beta^2 w^2) cos(k + arctan(beta w)).
    (b) At long wavelength, a uniform drift V = beta c -- a Galilean boost of Klein-Gordon with
    s^2 = c + V^2:  w = V k + sqrt(K + (c + V^2) k^2); lattice departures O(k^3).
T3  THE KAPPA FLOOR is the no-resonance condition: an a-wave at k0 cannot reach the b-branch at
    the same lab frequency; compared with the recorded closed-iff rule kappa w > c(1 - cos k0)
    on a 60 x 60 (kappa, k0) grid.

usage:  python3 continuum_checks.py      (output: continuum_checks_output.txt)
"""
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M

def rk4(f, u, v, dt, nst):
    for _ in range(nst):
        a1 = f(u, v); k1u, k1v = v, a1
        a2 = f(u + .5*dt*k1u, v + .5*dt*k1v); k2u, k2v = v + .5*dt*k1v, a2
        a3 = f(u + .5*dt*k2u, v + .5*dt*k2v); k3u, k3v = v + .5*dt*k2v, a3
        a4 = f(u + dt*k3u, v + dt*k3v); k4u, k4v = v + dt*k3v, a4
        u = u + dt/6*(k1u + 2*k2u + 2*k3u + k4u); v = v + dt/6*(k1v + 2*k2v + 2*k3v + k4v)
    return u, v

# ---- T1: kappa is a rotating frame (exact on the lattice, nonlinear) ----
print("T1  kappa = rotating frame: psi(t) = exp(mu t JJ) phi(t), phi under kappa = 0 with K -> K + mu^2")
rng = np.random.default_rng(3)
for well in ("node", "radial", "elementwise"):
    for q, shape in ((1, None), (2, 12)):
        n = 2
        latk = M.Lattice(n, N=48 if q == 1 else 144, kappa=M.KAPPA, q=q, shape=shape, well=well)
        lat0 = M.Lattice(n, N=48 if q == 1 else 144, kappa=0.0, q=q, shape=shape, well=well)
        mu = latk.kappa / 2; JJ = latk.JJ
        u0 = 0.6 * rng.standard_normal((latk.N, 2 * n)); v0 = 0.6 * rng.standard_normal((latk.N, 2 * n))
        fk = lambda u, v: latk.force(u, v)
        f0 = lambda w, wd: lat0.force(w, wd) - mu ** 2 * w
        T = 20.0
        errs = []
        for dt in (0.01, 0.005):
            uT, _ = rk4(fk, u0, v0, dt, int(round(T / dt)))
            w0, wd0 = u0, v0 - mu * u0 @ JJ.T          # v = R(wdot + mu JJ w) at t = 0
            wT, _ = rk4(f0, w0, wd0, dt, int(round(T / dt)))
            R = expm(mu * T * JJ)
            errs.append(np.abs(uT - wT @ R.T).max())
        print(f"    well={well:11s} q={q}: max|psi - R phi| at T=20: dt .01 {errs[0]:.2e}, dt .005 {errs[1]:.2e}"
              f"  (ratio {errs[0]/max(errs[1],1e-300):.1f}; 16 = pure RK4 error)   |u| ~ {np.abs(uT).max():.2f}")

# ---- T2: scalar beta = frequency-dependent Peierls phase (exact), and a Galilean frame (continuum) ----
print("\nT2  scalar beta sector")
K, c, beta = np.sqrt(5), 1.0, 0.05
ks = np.linspace(-np.pi, np.pi, 2001)
b = 2 * beta * c * np.sin(ks)
Q = K + 2 * c * (1 - np.cos(ks))
wl = 0.5 * (b + np.sqrt(b ** 2 + 4 * Q))                  # lattice, upper root (as joint5_cq)
A = np.arctan(beta * wl); r = np.sqrt(1 + (beta * wl) ** 2)
res = wl ** 2 - (K + 2 * c - 2 * c * r * np.cos(ks + A))
print(f"    Peierls identity  w^2 = K + 2c - 2c sqrt(1+beta^2 w^2) cos(k + arctan(beta w)): max resid {np.abs(res).max():.1e}")
V = beta * c
for kk in (0.4, 0.2, 0.1, 0.05):
    bb = 2 * beta * c * np.sin(kk); QQ = K + 2 * c * (1 - np.cos(kk))
    w_lat = 0.5 * (bb + np.sqrt(bb ** 2 + 4 * QQ))
    w_gal = V * kk + np.sqrt(K + (c + V ** 2) * kk ** 2)
    print(f"    k={kk:5.2f}: lattice {w_lat:.10f}  boosted KG (V = beta c, s^2 = c + V^2) {w_gal:.10f}  diff {w_lat - w_gal:+.2e}")

# ---- T3: the floor is the no-resonance condition of a J-breaking term in the rotating frame ----
print("\nT3  kappa floor = 'a-wave at k0 cannot reach the b-branch at equal lab frequency'")
K = np.sqrt(5); worst = 0
for kap in np.linspace(0.05, 3, 60):
    for k0 in np.linspace(0.05, np.pi - 0.05, 60):
        Q0 = K + 2 * c * (1 - np.cos(k0)); wa = 0.5 * (-kap + np.sqrt(kap ** 2 + 4 * Q0))
        # b-branch positive frequency 0.5(kap + sqrt(kap^2 + 4Q)) >= 0.5(kap + sqrt(kap^2+4K)); reachable iff some k' has equality
        wb_min = 0.5 * (kap + np.sqrt(kap ** 2 + 4 * K))
        open_res = wa >= wb_min
        open_rec = kap * wa <= c * (1 - np.cos(k0))
        worst += open_res != open_rec
print(f"    disagreements between resonance condition and recorded closed-iff (kappa w > c(1 - cos k0)): {worst} of 3600")
