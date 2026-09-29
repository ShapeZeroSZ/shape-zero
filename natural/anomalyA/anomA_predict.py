#!/usr/bin/env python3
"""anomA_predict.py -- DERIVATION ONLY (no gate run): predicted certification slopes under A' from the common
frequency shift. Every quantity is linear: free envelopes, segment unitaries. No parameter is fitted.

Mechanism. Under A' the local stiffness is sqrt5 + |psi|, common to all components, so a packet of envelope F carries a
frequency shift dw = A_eff / (2w + kappa), A_eff = sum F^3 / sum F^2 (density-weighted |psi|). The frequency is
conserved into the links; at fixed frequency each link site's Peierls angle per eigen-direction j is
theta_j = arctan(g w_s w h_j) (w_s = the RAMP weight; main's link-sector derivation 93debc6), so
    d theta_j = g w_s h_j dw / (1 + (g w_s w h_j)^2)            (predictor V1, the stated law)
V2 (refinement, same physics, no parameter): the model's own fixed-frequency wavenumber k_branch at w + dw with Q fixed
(the stiffness shift and the frequency shift cancel in Q), which adds the hopping renormalisation's w-dependence.
dw is taken separately at each segment's centre-crossing time (the envelope disperses).
Quantities (deg per 1e-3 of amplitude): split error slope = split(dw) - split(0); per-order slope = angle(co(dw),
co(0)); floors (commuting axes) = 0 exactly under this mechanism.
"""
import math, os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "04_scripts", "session"))
import model as M

K, KA, C = M.SQ5, M.KAPPA, M.C
K0 = M.K0
RAMP = np.array(M.RAMP)
AMP = 1e-3


def k_branch(glam, w, Q):
    f = lambda k: 2 * C * (1 - np.cos(k)) - 2 * C * glam * w * np.sin(k) - Q
    lo, hi = 0.05, np.pi - 0.05
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)


def U_seg(n, axis, g, w, Q, mode):
    G = M.generators(n)[axis]
    eigs, vecs = np.linalg.eigh(G)
    ph = []
    for lam in eigs:
        if mode == "V1":
            ph.append(sum(math.atan(g * ws * w * lam) for ws in RAMP))
        else:
            ph.append(sum(k_branch(g * ws * lam, w, Q) for ws in RAMP))
    return (vecs * np.exp(1j * np.array(ph))) @ vecs.conj().T


def coords(n, s):
    r = np.outer(s, s.conj()); tr = np.real(np.trace(r))
    return np.array([np.real(np.trace(S @ r)) / tr for S in M.generators(n)])


def ang(a, b):
    return M.angle(np.asarray(a), np.asarray(b))


def predict(n, gA, gB, axes, w, Q, dws, mode):
    psi0 = np.zeros(n, complex); psi0[0] = 1
    a0, a1 = axes
    orders = {"AB": [(a0, gA), (a1, gB)], "BA": [(a1, gB), (a0, gA)]}
    co = {}
    for dsh in (0, 1):
        for o, seq in orders.items():
            s = psi0.copy()
            for i, (ax, g) in enumerate(seq):
                s = U_seg(n, ax, g, w + dsh * dws[i], Q, mode) @ s
            co[(dsh, o)] = coords(n, s)
    split0, split1 = ang(co[(0, "AB")], co[(0, "BA")]), ang(co[(1, "AB")], co[(1, "BA")])
    return dict(split_slope=split1 - split0, per_AB=ang(co[(1, "AB")], co[(0, "AB")]),
                per_BA=ang(co[(1, "BA")], co[(0, "BA")]), split0=split0)


def q1_setup():
    lat = M.Lattice(n=2, N=1200, well="node")
    w = lat.omega; Q = w * w + KA * w - K
    vg = 2 * C * np.sin(K0) / (2 * w + KA)
    ts, Ae = M.free_Aeff(lat, 8.0, 20, AMP, 400.0, dt=1.0)
    tcs = [(s + len(RAMP) / 2 - 20) / vg for s in (60, 80)]
    Aeff = [float(np.interp(t, ts, Ae)) for t in tcs]
    return w, Q, vg, tcs, Aeff, Ae[0]


def q3_setup():
    L0, S, W, X0 = 260, 8, 3.0, 30
    sh = (L0, S, S)
    c = np.indices(sh).astype(float)
    r2 = np.zeros(sh)
    for a, ctr in zip(range(3), (X0, S / 2.0, S / 2.0)):
        d = c[a] - ctr; d = (d + sh[a] / 2) % sh[a] - sh[a] / 2; r2 += d ** 2
    psi = AMP * np.exp(-0.5 * r2 / W ** 2) * np.exp(1j * K0 * (c[0] - X0))
    k = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m) for m in sh], indexing="ij")
    Qk = K + 2 * C * sum(1 - np.cos(x) for x in k)
    wk = 0.5 * (-KA + np.sqrt(KA ** 2 + 4 * Qk))
    P = np.fft.fftn(psi)
    pw = np.abs(P) ** 2 / (np.abs(P) ** 2).sum()
    Qt = float((pw * 2 * C * ((1 - np.cos(k[1])) + (1 - np.cos(k[2])))).sum())
    w = 0.5 * (-KA + math.sqrt(KA ** 2 + 4 * (K + 2 * C * (1 - math.cos(K0)) + Qt)))
    Q = w * w + KA * w - K - Qt
    vg = 2 * C * math.sin(K0) / (2 * w + KA)
    tcs = [(s + len(RAMP) / 2 - X0) / vg for s in (50, 70)]
    Aeff = []
    for t in [0.0] + tcs:
        F = np.abs(np.fft.ifftn(P * np.exp(-1j * wk * t)))
        Aeff.append(float((F ** 3).sum() / (F ** 2).sum()))
    return w, Q, vg, tcs, Aeff[1:], Aeff[0], Qt


def main():
    out = {}
    print("=" * 96)
    print("ANOMALY A -- PREDICTED CERTIFICATION SLOPES UNDER A' (derivation only; deg per 1e-3 of amplitude)")
    print("=" * 96)
    w, Q, vg, tcs, Ae, A0 = q1_setup()
    print(f"q = 1: w = {w:.6f}, 2w + kappa = {2*w+KA:.4f}, v_g = {vg:.4f}; A_eff/A at t = 0 {A0/AMP:.4f}, "
          f"at crossings t = {tcs[0]:.1f}, {tcs[1]:.1f}: {Ae[0]/AMP:.4f}, {Ae[1]/AMP:.4f}")
    dws = [a / (2 * w + KA) for a in Ae]
    print(f"       dw at the two segment positions: {dws[0]:.4e}, {dws[1]:.4e}")
    cases1 = [("u(2)", 2, 0.12, 0.08), ("u(3)", 3, 0.15, 0.10)]
    for mode in ("V1", "V2"):
        for name, n, gA, gB in cases1:
            p = predict(n, gA, gB, (0, 1), w, Q, dws, mode)
            f = predict(n, gA, gB, (0, 0), w, Q, dws, mode)
            out[f"q1 {name} {mode}"] = dict(p, floor=ang(0, 0) if False else 0.0, floor_check=f["split_slope"])
            print(f"   {mode} q=1 {name}: split slope {p['split_slope']:+.5f}; per-order AB {p['per_AB']:.4f}, "
                  f"BA {p['per_BA']:.4f}; floor {f['split_slope']:+.1e} (commuting); split(0) = {p['split0']:.3f}")
    w3, Q3, vg3, tcs3, Ae3, A03, Qt = q3_setup()
    print(f"q = 3: Qt = {Qt:.4f}, w = {w3:.6f}, v_g = {vg3:.4f}; A_eff/A at t = 0 {A03/AMP:.4f}, at crossings "
          f"t = {tcs3[0]:.1f}, {tcs3[1]:.1f}: {Ae3[0]/AMP:.4f}, {Ae3[1]/AMP:.4f}")
    dws3 = [a / (2 * w3 + KA) for a in Ae3]
    cases3 = [("u(2)", 2, 0.12, 0.08), ("u(3)", 3, 0.15, 0.15)]
    for mode in ("V1", "V2"):
        for name, n, gA, gB in cases3:
            p = predict(n, gA, gB, (0, 1), w3, Q3, dws3, mode)
            fA, fB = {2: (0.12, 0.08), 3: (0.15, 0.10)}[n]
            f = predict(n, fA, fB, (0, 0), w3, Q3, dws3, mode)
            p["floor_check"] = f["split_slope"]
            out[f"q3 {name} {mode}"] = p
            print(f"   {mode} q=3 {name}: split slope {p['split_slope']:+.5f}; per-order AB {p['per_AB']:.4f}, "
                  f"BA {p['per_BA']:.4f}; floor {f['split_slope']:+.1e}; split(0) = {p['split0']:.3f}")
    # smooth-force diagnostic: |psi|^2 psi -> dw = sum F^4 / sum F^2 / (2w + kappa), ~ A^2
    print("SMOOTH-FORCE DIAGNOSTIC (|psi|^2 psi; diagnostic only): predicted split-error change from the A -> 0 value")
    lat = M.Lattice(n=2, N=1200, well="node")
    x = np.arange(1200)
    for name, n, gA, gB in cases1:
        for A in (0.04, 0.02, 0.01, 0.005):
            psi = A * np.exp(-0.5 * ((x - 20) / 8.0) ** 2) * np.exp(1j * K0 * (x - 20))
            P = np.fft.fft(psi); q = 2 * np.pi * np.fft.fftfreq(1200)
            wq = 0.5 * (-KA + np.sqrt(KA ** 2 + 4 * (K + 2 * C * (1 - np.cos(q)))))
            d2 = []
            for t in tcs:
                F = np.abs(np.fft.ifft(P * np.exp(-1j * wq * t))); d2.append((F ** 4).sum() / (F ** 2).sum() / (2 * w + KA))
            p = predict(n, gA, gB, (0, 1), w, Q, d2, "V1")
            out[f"smooth q1 {name} A={A}"] = p
            print(f"   q=1 {name} A = {A}: dw = {d2[0]:.3e}; split change {p['split_slope']:+.5f}, per-order AB {p['per_AB']:.5f}")
    w3_, Q3_ = w3, Q3
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "anomA_predicted.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
