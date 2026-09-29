#!/usr/bin/env python3
"""
ampA_link_predict.py -- anomaly A (census 2026-09-29): PREDICTIONS, committed before any run,
of the certification test's amplitude slopes under the node form A' (force -(sqrt5 + |psi|) psi,
|psi| the whole node's radius), and of the smooth-force diagnostic (force -(sqrt5 + |psi|^2) psi).

Hypothesis (the user's, 2026-09-29). Under A' the local frequency shift is common to every
component, so it is a global phase and cannot move the Bloch vector by itself. It enters
the gauge readout only through the link sector: the per-direction Peierls angle
tan(theta_j) = g w h_j depends on w, so a frequency shift dw moves each eigen-direction's
phase per site by
        d theta_j = g h_j dw / (1 + (g w h_j)^2),      dw = <|psi|> / (2w + kappa),
and the directions' phases no longer move together. No new parameter.

Construction (design choices ours, fixed now):
  * <|psi|> at a ramp site = A_eff(t) = sum F^3 / sum F^2 of the LINEAR free envelope (the
    |psi|^2-weighted mean of |psi|, the first-order density-matrix average), taken at the
    time the packet centre crosses that site, t = (site - x0) / v_g, v_g = 2c sin k0/(2w + kappa).
    q = 1: model.free_Aeff (N = 1200, width 8, n0 = 20). q = 3: the same on q3_gate's
    260 x 8 x 8 slab, width-3 per-mode packet at x0 = 30.
  * q = 1: single-carrier product, as the certification's linear reference
    (model.U_segment): per ramp site, phase k_j = k_branch(g wgt h_j) + d theta_j.
  * q = 3: q3_gate's spectrum-averaged predictor with, per Fourier mode (its own w),
    the same d theta_j added per ramp site.
  * The certification's own fits (amp_scaling.fit, certify_gates.evaluate_rows) are applied
    to the predicted Bloch vectors at A = 1e-3, 5e-4, 2.5e-4, with the linear predictions as
    reference: the predicted split-error slope, per-order slopes and floor slopes.
  * PRIMARY: the angle law above. SECONDARY (labelled; its difference from the primary is
    the hopping renormalisation the angle law leaves out): the full branch equation at
    w + dw with Q unchanged (Q = w^2 + kappa w - sqrt5 - |psi| is invariant to first order),
    i.e. k_j = arctan(g w' h_j) + arccos((1 - Q/2c)/sqrt(1 + (g w' h_j)^2)), w' = w + dw.
  * SMOOTH diagnostic: dw = <|psi|^2> / (2w + kappa), <|psi|^2> = sum F^4 / sum F^2. Every
    deviation it produces scales as A^2.

usage: python3 ampA_link_predict.py   -> ampA_link_predictions.txt / .json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M
import amp_scaling as S
import q3_gate as Q

AMPS = S.AMPS
SEGS1 = (60, 80)
N0_1, W_1 = 20, 8.0


# ------------------------------------------------------------------ envelopes
def vg(om, kappa):
    return 2 * M.C * np.sin(M.K0) / (2 * om + kappa)


def aeff_1d(amp, times, p=1):
    """p = 1: sum F^3/sum F^2 (A'); p = 2: sum F^4/sum F^2 (smooth)."""
    lat = M.Lattice(n=2, N=1200, well="elementwise")
    x = np.arange(lat.N)
    psi = amp * np.exp(-0.5 * ((x - N0_1) / W_1) ** 2) * np.exp(1j * M.K0 * (x - N0_1))
    P = np.fft.fft(psi)
    q = 2 * np.pi * np.fft.fftfreq(lat.N)
    w = 0.5 * (-lat.kappa + np.sqrt(lat.kappa ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(q)))))
    out = []
    for t in times:
        F = np.abs(np.fft.ifft(P * np.exp(-1j * w * t)))
        out.append((F ** (2 + p)).sum() / (F ** 2).sum())
    return np.array(out), lat


def aeff_3d(amp, times, kappa, p=1):
    lat = Q.Slab(2, 260, 8, kappa)
    u, v = lat.packet3(amp)
    psi = (u[:, 0] + 1j * u[:, 1]).reshape(lat.shape)
    dps = (v[:, 0] + 1j * v[:, 1]).reshape(lat.shape)
    Pk = np.fft.fftn(psi)
    w = lat.branch_omega()
    out = []
    for t in times:
        F = np.abs(np.fft.ifftn(Pk * np.exp(-1j * w * t)))
        out.append((F ** (2 + p)).sum() / (F ** 2).sum())
    return np.array(out), lat


# ------------------------------------------------------------------ link phases
def dtheta(g, h, om, dw):
    return g * h * dw / (1 + (g * om * h) ** 2)


def k_full(g, h, om, dw, Qv):
    """secondary: exact branch at w' = w + dw, Q fixed."""
    wp = om + dw
    th = np.arctan(g * wp * h)
    return th + np.arccos((1 - Qv / (2 * M.C)) / np.sqrt(1 + (g * wp * h) ** 2))


def k_lin_exact(g, h, om, Qv):
    return k_full(g, h, om, 0.0, Qv)


# ------------------------------------------------------------------ q = 1
def pred_q1(n, segs_spec, amp, mode):
    """segs_spec: [(start, axis, g)]; mode in lin / primary / secondary / smooth."""
    lat = M.Lattice(n=n, N=1200, well="elementwise")
    om, kap = lat.omega, lat.kappa
    Qv = om ** 2 + kap * om - M.SQ5
    v = vg(om, kap)
    psi = np.zeros(n, complex); psi[0] = 1
    for start, ax, g in segs_spec:
        eigs, vecs = np.linalg.eigh(lat.G[ax])
        times = [(start + i - N0_1) / v for i in range(len(M.RAMP))]
        if mode == "lin":
            dws = np.zeros(len(times))
        else:
            A, _ = aeff_1d(amp, times, p=2 if mode == "smooth" else 1)
            dws = A / (2 * om + kap)
        ph = np.zeros(n)
        for i, wgt in enumerate(M.RAMP):
            for j in range(n):
                ge = g * wgt * eigs[j]
                k0 = M.k_branch(lat, ge)
                if mode == "secondary":
                    ph[j] += k0 + (k_full(g * wgt, eigs[j], om, dws[i], Qv) - k_lin_exact(g * wgt, eigs[j], om, Qv))
                else:
                    ph[j] += k0 + dtheta(g * wgt, eigs[j], om, dws[i])
        psi = vecs @ (np.exp(1j * ph) * (vecs.conj().T @ psi))
    return M.coords_of_state(lat, psi)


# ------------------------------------------------------------------ q = 3
_CACHE = {}


def pred_q3(n, segs, amp, mode, kappa):
    """segs: [(axis, g)] at q3_gate's SEGS; spectrum-averaged with per-mode d theta."""
    key = n
    if key not in _CACHE:
        _CACHE[key] = Q.spectrum(n, 260, 8, kappa)
    lat, P, KX, KY, KZ = _CACHE[key]
    mask = (P > 1e-12 * P.max()) & (KX > 0) & (KX < np.pi)
    p, kx, ky, kz = P[mask], KX[mask], KY[mask], KZ[mask]
    Qt = 2 * M.C * ((1 - np.cos(ky)) + (1 - np.cos(kz)))
    om = 0.5 * (-lat.kappa + np.sqrt(lat.kappa ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(kx)) + Qt)))
    Qx = om ** 2 + lat.kappa * om - M.SQ5 - Qt
    v = vg(lat.omega, lat.kappa)
    state = np.zeros((len(p), n), complex); state[:, 0] = 1
    for (axis, g), start in zip(segs, Q.SEGS):
        eigs, vecs = np.linalg.eigh(lat.G[axis])
        times = [(start + i - Q.X0) / v for i in range(len(M.RAMP))]
        if mode == "lin":
            Aw = np.zeros(len(times))
        else:
            Aw, _ = aeff_3d(amp, times, lat.kappa, p=2 if mode == "smooth" else 1)
        ph = np.zeros((len(p), n))
        for i, wgt in enumerate(M.RAMP):
            dw = Aw[i] / (2 * om + lat.kappa)
            for j in range(n):
                ph[:, j] += Q._kx_in_segment(kx, om, Qt, g * wgt * eigs[j], lat.kappa)
                if mode == "secondary":
                    ph[:, j] += k_full(g * wgt, eigs[j], om, dw, Qx) - k_lin_exact(g * wgt, eigs[j], om, Qx)
                elif mode != "lin":
                    ph[:, j] += dtheta(g * wgt, eigs[j], om, dw)
        state = ((state @ vecs.conj()) * np.exp(1j * ph)) @ vecs.T
    rho = np.einsum("m,mi,mj->ij", p, state, state.conj())
    return np.array([np.real(np.trace(Sg @ rho)) / np.real(np.trace(rho)) for Sg in lat.G])


# ------------------------------------------------------------------ rows -> certification fits
def rows_for(q, mode, kappa3):
    rows = []
    for amp in AMPS:
        for n in (2, 3):
            if q == 1:
                gA, gB = dict((c[0], c[1:]) for c in S.CASES)[n]
                for axes in ((0, 1), (0, 0)):
                    a0, a1 = axes
                    sp = {"AB": [(SEGS1[0], a0, gA), (SEGS1[1], a1, gB)],
                          "BA": [(SEGS1[0], a1, gB), (SEGS1[1], a0, gA)]}
                    rows.append(dict(n=n, axes=list(axes), amp=amp,
                                     co={k: pred_q1(n, s, amp, mode).tolist() for k, s in sp.items()},
                                     lin={k: pred_q1(n, s, amp, "lin").tolist() for k, s in sp.items()}))
            else:
                for axes, (gA, gB) in (((0, 1), Q.G[n]), ((0, 0), Q.GFLOOR[n])):
                    a0, a1 = axes
                    sp = {"AB": [(a0, gA), (a1, gB)], "BA": [(a1, gB), (a0, gA)]}
                    rows.append(dict(n=n, axes=list(axes), amp=amp,
                                     co={k: pred_q3(n, s, amp, mode, kappa3).tolist() for k, s in sp.items()},
                                     lin={k: pred_q3(n, s, amp, "lin", kappa3).tolist() for k, s in sp.items()}))
    return rows


def main():
    import certify_gates as CG
    kappa3 = float(M.KAPPA)
    lines, res = [], {}
    lines.append("PREDICTIONS (committed before any run) -- anomaly A: certification slopes under A' from the")
    lines.append("link-sector angle law, and the smooth-force diagnostic. Slopes in deg per 1e-3 of amplitude.")
    for mode in ("primary", "secondary", "smooth"):
        res[mode] = {}
        for q in (1, 3):
            rows = rows_for(q, mode, kappa3)
            ok, ls, slopes = CG.evaluate_rows(rows, q)
            res[mode][q] = slopes
            # the predicted deviations themselves at A = 1e-3 (split error and per-order angles)
            dev = {}
            for r in rows:
                if r["amp"] != 1e-3 or r["axes"] != [0, 1]:
                    continue
                ang = lambda a, b: M.angle(np.array(a), np.array(b))
                dev[f"u({r['n']}) split error at 1e-3"] = ang(r["co"]["AB"], r["co"]["BA"]) - ang(r["lin"]["AB"], r["lin"]["BA"])
                for o in ("AB", "BA"):
                    dev[f"u({r['n']}) per-order {o} at 1e-3"] = ang(r["co"][o], r["lin"][o])
            res[mode][f"dev{q}"] = dev
            lines.append(f"\n  [{mode}] q = {q}")
            lines += ["   " + l for l in ls]
            for k, vv in dev.items():
                lines.append(f"      {k}: {vv:+.6f} deg")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "ampA_link_predictions.txt"), "w").write(out + "\n")
    json.dump(res, open(os.path.join(HERE, "ampA_link_predictions.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
