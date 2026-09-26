#!/usr/bin/env python3
"""
slice_precession.py -- SLICE-RESOLVED first-order self-precession under the radial well
(MODEL_SPEC sec 4d; follows the impulsive correction model.precession_prediction, which
over-predicts gate 7's floor ~1.8x).

THE MODEL (fixed before any comparison). The packet is cut into slices: one per lattice
site of the launched packet (at q = 3 one per (x, y, z) site), each carrying a unit
node-state vector s (in C^n, s = e_0 at launch) and the weight F(x, 0)^2. Every slice
moves along axis 0 at the carrier group velocity v_g = 2c sin k0 / (2w + kappa)
(at q = 3, w includes the packet's mean transverse term Qt). Along the way, per time
step dt:
  1. PRECESSION at its LOCAL amplitude: each dimer component precesses at
     delta w_j = |psi_j| / (2w + kappa), |psi_j| = F_loc |s_j|, where F_loc is the
     EXACT LINEAR free envelope |psi(x, t)| (per mode on the a-branch) at the slice's
     current position (linear interpolation along axis 0):
         s_j <- exp(-i F_loc |s_j| dt / (2w + kappa)) s_j
  2. ROTATION as it goes: when the slice crosses link site m + 1/2 of a segment (ramp
     weight wgt), it takes that link's unitary sum_j exp(i k(g wgt lambda_j)) P_j -- one
     factor of model.U_segment (at q = 3 with the transverse term Qt, as
     q3_gate._kx_in_segment) -- so a slice inside a segment is partly rotated and
     precesses in that partly rotated state.
Readout at the run's own readout time T: rho = sum_slices F(x,0)^2 s s^+, Bloch
coordinates as model.coords_of_state. The same model with the precession switched off
gives the linear baseline; the PREDICTED SLOPES (degrees per 1e-3 of amplitude) are
    floor      angle(AB, BA), commuting axes (0, 0)       [baseline: 0]
    split      split(with) - split(without), signed
    per-order  angle(with, without), AB and BA
computed at A = 1e-3 (the model is first order, so this is its slope).
Neglected: dispersion of the slice trajectories (all slices move at the carrier v_g;
the envelope's own dispersion is kept through F_loc), the segments' change of the
group velocity and amplitude inside a segment, reflections, O(A^2).

Geometry (as the runs): q = 1 gate 7 -- N = 1200, n0 = 20, width 8, segments 60/80,
u(2) (0.12, 0.08), u(3) (0.15, 0.10), T = 1314.5 / 1309.0 (amp_scaling_1d.json).
q = 3 q3_gate -- 260 x 8 x 8, x0 = 30, width 3, segments 50/70, G / GFLOOR strengths,
T per pair from q3_gate_runs_260x8_ampA1.json.

usage:  python3 slice_precession.py predict   -> slice_precession_predictions.{txt,json}
        python3 slice_precession.py compare   -> against certify_gates_slopes.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M
import q3_gate as Q

AMP, DT = 1e-3, 0.25
PRED = os.path.join(HERE, "slice_precession_predictions.json")


# ------------------------------------------------------------------ geometry
def geom_q1(n):
    gA, gB = {2: (0.12, 0.08), 3: (0.15, 0.10)}[n]
    T = {r["n"]: r["T"] for r in json.load(open(os.path.join(HERE, "amp_scaling_1d.json")))}[n]
    lat = M.Lattice(n=n, N=1200)
    x = np.arange(lat.N)
    psi = AMP * np.exp(-0.5 * ((x - 20) / 8.0) ** 2) * np.exp(1j * M.K0 * (x - 20))
    split = {"AB": [(60, 0, gA), (80, 1, gB)], "BA": [(60, 1, gB), (80, 0, gA)]}
    floor = {"AB": [(60, 0, gA), (80, 0, gB)], "BA": [(60, 0, gB), (80, 0, gA)]}
    return dict(lat=lat, shape=(lat.N,), psi=psi, Qt=0.0, T={"split": T, "floor": T},
                specs={"split": split, "floor": floor})


def geom_q3(n):
    d = json.load(open(os.path.join(HERE, "q3_gate_runs_260x8_ampA1.json")))
    lat = Q.Slab(n, d["L0"], d["S"], d["kappa"])
    u, v = lat.packet3(AMP)
    psi = (u[:, 0] + 1j * u[:, 1])
    T = {r["job"]: r["t"] for r in d["runs"] if r["n"] == n}
    specs = {"split": {"AB": Q.spec_for(n, "AB"), "BA": Q.spec_for(n, "BA")},
             "floor": {"AB": Q.spec_for(n, "fAB"), "BA": Q.spec_for(n, "fBA")}}
    # mean transverse term of the launched packet (power-weighted), as model.transverse_Q
    Qt = M.transverse_Q(lat, u, v)
    return dict(lat=lat, shape=lat.shape, psi=psi, Qt=Qt, T={"split": T["AB"], "floor": T["fAB"]},
                specs=specs)


# ------------------------------------------------------------------ the slice model
def free_envelope(g):
    """F(x, t) = |psi| under exact linear free evolution (a-branch per mode)."""
    lat, shape = g["lat"], g["shape"]
    P = np.fft.fftn(g["psi"].reshape(shape))
    w = lat.branch_omega()
    return lambda t: np.abs(np.fft.ifftn(P * np.exp(-1j * w * t)))


def link_unitaries(g, spec):
    """[(site + 1/2, U_link)] for every ramp link of every segment, in crossing order."""
    lat, Qt = g["lat"], g["Qt"]
    w = omega_c(g)
    out = []
    for start, axis, gs in spec:
        eigs, vecs = np.linalg.eigh(lat.G[axis])
        for j, wgt in enumerate(M.RAMP):
            ks = [Q._kx_in_segment(np.array([M.K0]), w, Qt, gs * wgt * e, lat.kappa)[0] for e in eigs]
            U = (vecs * np.exp(1j * np.array(ks))) @ vecs.conj().T
            out.append((start + j + 0.5, U))
    return sorted(out, key=lambda p: p[0])


def omega_c(g):
    lat = g["lat"]
    return 0.5 * (-lat.kappa + np.sqrt(lat.kappa ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(M.K0)) + g["Qt"])))


def slice_run(g, spec, T, env, precess=True):
    lat, shape, n = g["lat"], g["shape"], g["lat"].n
    w = omega_c(g)
    den = 2 * w + lat.kappa
    vg = 2 * M.C * np.sin(M.K0) / den
    F0 = env(0.0)
    keep = F0 ** 2 > 1e-14 * (F0 ** 2).max()
    idx = np.argwhere(keep)                       # slice labels (launch sites)
    wts = F0[keep] ** 2
    x0 = idx[:, 0].astype(float)
    trans = tuple(idx[:, a] for a in range(1, len(shape)))
    L0 = shape[0]
    s = np.zeros((len(wts), n), complex); s[:, 0] = 1.0
    links = link_unitaries(g, spec)
    nst = int(round(T / DT))
    for i in range(nst):
        t0, t1 = i * DT, (i + 1) * DT
        if precess:
            tm = 0.5 * (t0 + t1)
            F = env(tm)
            xm = x0 + vg * tm
            xl = np.floor(xm).astype(int)
            fr = xm - xl
            Fl = F[(xl % L0,) + trans]
            Fr = F[((xl + 1) % L0,) + trans]
            Floc = (1 - fr) * Fl + fr * Fr
            s = np.exp(-1j * Floc[:, None] * np.abs(s) * DT / den) * s
        # links crossed during (t0, t1]
        xa, xb = x0 + vg * t0, x0 + vg * t1
        for pos, U in links:
            hit = (xa < pos) & (xb >= pos)
            if hit.any():
                s[hit] = s[hit] @ U.T
    rho = np.einsum("m,mi,mj->ij", wts, s, s.conj())
    tr = np.real(np.trace(rho))
    return np.array([np.real(np.trace(S @ rho)) / tr for S in lat.G])


def predict_case(q, n):
    g = geom_q1(n) if q == 1 else geom_q3(n)
    env = free_envelope(g)
    res = {}
    co = {}
    for kind in ("split", "floor"):
        for o in ("AB", "BA"):
            for p in (True, False):
                co[(kind, o, p)] = slice_run(g, g["specs"][kind][o], g["T"][kind], env, precess=p)
    ang = M.angle
    res[f"u({n}) floor"] = ang(co[("floor", "AB", True)], co[("floor", "BA", True)])
    res[f"u({n}) floor (baseline, precession off)"] = ang(co[("floor", "AB", False)], co[("floor", "BA", False)])
    res[f"u({n}) split"] = (ang(co[("split", "AB", True)], co[("split", "BA", True)])
                            - ang(co[("split", "AB", False)], co[("split", "BA", False)]))
    for o in ("AB", "BA"):
        res[f"u({n}) per-order {o}"] = ang(co[("split", o, True)], co[("split", o, False)])
    return res


def predict():
    out = {}
    lines = ["SLICE-RESOLVED FIRST-ORDER SELF-PRECESSION -- PREDICTED SLOPES (deg per 1e-3 of amplitude)",
             "(computed before any comparison with the measured slopes; model: this file's docstring)", ""]
    for q in (1, 3):
        out[q] = {}
        for n in (2, 3):
            out[q].update(predict_case(q, n))
        lines.append(f"  q = {q} ({'gate 7' if q == 1 else 'q3_gate'}):")
        for k, v in out[q].items():
            lines.append(f"     {k:42s} {v:+.5f}")
        print("\n".join(lines[-len(out[q]) - 1:]), flush=True)
    json.dump(out, open(PRED, "w"), indent=1)
    open(os.path.join(HERE, "slice_precession_predictions.txt"), "w").write("\n".join(lines) + "\n")


def compare():
    pred = json.load(open(PRED))
    meas = json.load(open(os.path.join(HERE, "certify_gates_slopes.json")))
    imp = {"1": {"u(2) floor": 0.294, "u(3) floor": 0.175}}
    print("SLICE MODEL vs MEASURED SLOPES (deg per 1e-3; ratio = predicted / measured)")
    for q in ("1", "3"):
        print(f"  q = {q}:")
        for k, m in meas[q].items():
            p = pred[q][k]
            extra = f"   (impulsive: {imp[q][k]:.3f})" if q in imp and k in imp[q] else ""
            print(f"     {k:22s} predicted {p:+.5f}   measured {m:+.5f}   ratio {p / m:6.3f}{extra}")


if __name__ == "__main__":
    {"predict": predict, "compare": compare}[sys.argv[1]]()
