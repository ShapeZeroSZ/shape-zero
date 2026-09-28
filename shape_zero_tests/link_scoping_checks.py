#!/usr/bin/env python3
"""link_scoping_checks.py -- checks for the link-sector scoping (MODEL_SPEC sec 9); derivation, hypotheses and
predictions in link_scoping_predictions.txt, committed with this script before any run.
C1: plane-wave residual of model.py's force against Derivation 1 (uniform links, q = 1).
C2: dynamic frequencies (model.run, RK4) and the splitting (S) and product (P) rules.
usage: python3 link_scoping_checks.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M  # noqa: E402

N, G_LINK = 128, 0.3
MS = (6, 32, 51)


def gens(n):
    G = M.generators(n)
    if n == 2:
        return {"sigma_1": G[0], "sigma_3": G[2]}
    return {"lambda_1": G[0], "lambda_8": G[7]}


def omega(branch, k, h, g):
    B = M.KAPPA + 2 * M.C * g * h * np.sin(k)
    Q = M.SQ5 + 2 * M.C * (1 - np.cos(k))
    R = np.sqrt(B * B + 4 * Q)
    return (-B + R) / 2 if branch == "a" else (B + R) / 2


def links(lat, H, g):
    W = np.repeat((g * M.rho(H))[None], lat.N, axis=0)
    return W, np.roll(W, 1, axis=0)


def state(lat, chi, k, w, branch, amp):
    x = np.arange(lat.N)
    psi = amp * np.outer(np.exp(1j * k * x), chi)                     # (N, n)
    dps = (-1j if branch == "a" else 1j) * w * psi
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    u[:, 0::2], u[:, 1::2] = psi.real, psi.imag
    v[:, 0::2], v[:, 1::2] = dps.real, dps.imag
    return u, v, psi


def measure_w(lat, chi, k, w0, branch, H, g, T=200.0, dts=0.5):
    # INSTRUMENT FIX (after the first run, link_scoping_checks_output_aliased.txt): dts was 10, so the phase
    # advanced ~16 rad between samples and np.unwrap aliased; 0.5 keeps the advance below pi. Predictions
    # unchanged.
    W, Wm = links(lat, H, g)
    u, v, psi0 = state(lat, chi, k, w0, branch, 1e-6)
    ref = psi0 / np.linalg.norm(psi0)
    ts, ph = [0.0], [0.0]
    t = 0.0
    while t < T - 1e-9:
        u, v, _ = lat.run(u, v, dts, W, Wm); t += dts
        psi = u[:, 0::2] + 1j * u[:, 1::2]
        ts.append(t); ph.append(np.angle(np.vdot(ref, psi)))
    ph = np.unwrap(np.array(ph))
    slope = np.polyfit(np.array(ts), ph, 1)[0]
    return -slope if branch == "a" else slope


def main():
    print(f"LINK SCOPING CHECKS -- model.py force, node form {M.J_WELL}, kappa* = {M.KAPPA:.6f}, N = {N}, g = {G_LINK}")
    print("C1 plane-wave residual |F + w^2 psi| / |w^2 psi| at t = 0 (amplitude 1e-10)")
    worst = 0.0
    for n in (2, 3):
        lat = M.Lattice(n=n, N=N, well="node")
        for name, H in gens(n).items():
            hs, V = np.linalg.eigh(H)
            W, Wm = links(lat, H, G_LINK)
            for m in MS:
                k = 2 * np.pi * m / N
                for j, h in enumerate(hs):
                    for br in "ab":
                        w = omega(br, k, h, G_LINK)
                        u, v, psi = state(lat, V[:, j], k, w, br, 1e-10)
                        F = lat.force(u, v, W, Wm)
                        Fc = F[:, 0::2] + 1j * F[:, 1::2]
                        r = np.linalg.norm(Fc + w * w * psi) / np.linalg.norm(w * w * psi)
                        worst = max(worst, r)
    print(f"  max residual over n = 2 (sigma_1, sigma_3), n = 3 (lambda_1, lambda_8), m = {MS}, both branches, "
          f"every eigen-direction: {worst:.2e}")
    print("C2 dynamic frequencies (RK4, T = 200, amplitude 1e-6); (S) w_b - w_a vs kappa + 2cgh sin k; (P) w_a w_b / Q(k)")
    devw, devS, devP = 0.0, 0.0, 0.0
    for n in (2, 3):
        lat = M.Lattice(n=n, N=N, well="node")
        for name, H in list(gens(n).items()) + [("none (g = 0)", np.eye(n))]:
            g = 0.0 if name.startswith("none") else G_LINK
            hs, V = np.linalg.eigh(H)
            for m in MS:
                k = 2 * np.pi * m / N
                Q = M.SQ5 + 2 * M.C * (1 - np.cos(k))
                for j, h in enumerate(hs):
                    wa = measure_w(lat, V[:, j], k, omega("a", k, h, g), "a", H, g)
                    wb = measure_w(lat, V[:, j], k, omega("b", k, h, g), "b", H, g)
                    da = wa / omega("a", k, h, g) - 1; db = wb / omega("b", k, h, g) - 1
                    S = (wb - wa) / (M.KAPPA + 2 * M.C * g * h * np.sin(k)) - 1
                    P = wa * wb / Q - 1
                    devw = max(devw, abs(da), abs(db)); devS = max(devS, abs(S)); devP = max(devP, abs(P))
                    print(f"  n={n} {name:<12s} m={m:<2d} h={h:+.4f}: w_a {wa:.10f}, w_b {wb:.10f}; w_b - w_a "
                          f"{wb - wa:.10f} (S dev {S:+.1e}); w_a w_b / Q - 1 {P:+.1e}")
    print(f"  max |w/w_derived - 1| {devw:.1e}; max (S) deviation {devS:.1e}; max (P) deviation {devP:.1e}")


if __name__ == "__main__":
    main()
