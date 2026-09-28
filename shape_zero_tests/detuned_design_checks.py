#!/usr/bin/env python3
"""detuned_design_checks.py -- POST HOC, written after circuit_error_budget.py and its checks were seen. Not part of
any pre-registered budget.
Finding: at the design point c-hat = 0.25, beta-hat = 0.10 (both platforms), the k = pi/2 probe mode w(+pi/2) lies
8.4e-4 sqrt(K) from the w(-13 pi/16) mode (m = -13 at N = 32, -26 at N = 64) -- an accidental near-degeneracy
about the size of the P2 signal (1e-3 sqrt K), which disorder couples. At beta-hat = 0.106 the nearest other mode is
8.8e-3 sqrt(K) away at both N = 32 and 64. This reruns the key budget items there.
usage: python3 detuned_design_checks.py"""
import numpy as np

import circuit_error_budget as CB
import platform_error_budget as PB

BH = 0.106


def circuit():
    CB.G0 = BH * np.sqrt(CB.C0 / CB.LG0); CB.B0 = CB.G0 / CB.C0
    print(f"CIRCUIT at beta-hat = {BH} (G = {CB.G0:.4e} S, R_set = {1 / CB.G0:.0f} Ohm)")
    for NN, SS in ((32, 0.05), (64, 0.02)):
        gg = CB.eta_shape(NN, "gauss", NN / 8)
        shp = [("gauss", NN * 3 / 32), ("gauss", NN * 5 / 32), ("sech2", NN / 8), ("two", NN * 2.5 / 32)]
        print(f"  N = {NN}, S = {SS}: ideal C ratio {CB.C_ratio(CB.ideal(NN), gg, S=SS):.4f}; ideal shape spread "
              f"{np.ptp([CB.C_ratio(CB.ideal(NN), CB.eta_shape(NN, s, a), S=SS) for s, a in shp]):.4f}")
        for lab, mk in (("1% tolerance", lambda r: CB.with_tol(NN, 0.01, r)), ("0.1% tolerance", lambda r: CB.with_tol(NN, 0.001, r)),
                        ("realistic (0.1%, Q 100, 5 MHz, parasitics)", lambda r: CB.realistic(NN, r))):
            rng = np.random.default_rng(99)
            R, P1, SP = [], [], []
            for _ in range(30):
                q = mk(rng)
                R.append(CB.C_ratio(q, gg, S=SS)); P1.append(CB.p1_residual(q))
                v = np.array([CB.C_ratio(q, CB.eta_shape(NN, s, a), S=SS) for s, a in shp]); SP.append((v.max() - v.min()) / abs(v.mean()))
            R, P1, SP = map(np.array, (R, P1, SP))
            print(f"    {lab:<42s}: P1 max {P1.max():.1e}; P2 C ratio median {np.median(R):.4f}, sd {R.std():.4f}, "
                  f"max |dev| {np.abs(R - np.median(R)).max():.4f}; P3 spread mean {SP.mean():.3f}, max {SP.max():.3f}")
    print("  1% component decomposition (N = 32, S = 0.05, 30 realisations): P2 sd")
    g4 = CB.eta_shape(32, "gauss", 4)
    for lab, keys in (("C", ["C"]), ("L_g", ["Lg"]), ("L_c", ["Lc"]), ("gyrator", ["Ga", "Gb", "gs"])):
        rng = np.random.default_rng(31)
        r = []
        for _ in range(30):
            p = dict(CB.ideal(32))
            for k in keys:
                if k == "gs":
                    p["gs"] = CB.G0 * 0.01 * (rng.standard_normal(32) + rng.standard_normal(32))
                else:
                    p[k] = dict(C=CB.C0, Lg=CB.LG0, Lc=CB.LC0, Ga=CB.G0, Gb=CB.G0)[k] * (1 + 0.01 * rng.standard_normal(32))
            r.append(CB.C_ratio(p, g4))
        print(f"    {lab:<8s}: sd {np.std(r):.4f}")


def pendulum():
    PB.B0 = BH * np.sqrt(PB.K0)
    print(f"PENDULUM RING at beta-hat = {BH} (b = {PB.B0:.4f} s^-1; rotor speed x1.06)")
    g = np.sqrt(PB.K0 + 2 * PB.C0) / 500
    for lab, dis in (("nominal (K 0.3%, springs 1%, rotors 0.5%)", dict(K=0.003, c=0.01, b=0.005)),
                     ("springs 0.3%", dict(K=0.003, c=0.003, b=0.005)), ("springs 0.1%, K 0.1%", dict(K=0.001, c=0.001, b=0.005))):
        rng = np.random.default_rng(2026)
        R, P1 = [], []
        for _ in range(30):
            N = 32
            Kn = PB.K0 * (1 + dis["K"] * rng.standard_normal(N)); cn = PB.C0 * (1 + dis["c"] * rng.standard_normal(N))
            bn = PB.B0 * (1 + dis["b"] * rng.standard_normal(N))
            R.append(PB.C_ratio(Kn, cn, bn, g, PB.eta_shape(N, "gauss", 4))[0]); P1.append(PB.p1_residual(Kn, cn, bn, g))
        R = np.array(R)
        print(f"  {lab:<42s}: P1 max {max(P1):.1e}; P2 C ratio median {np.median(R):.4f}, sd {R.std():.4f}, max |dev| "
              f"{np.abs(R - np.median(R)).max():.4f}")


if __name__ == "__main__":
    circuit()
    pendulum()
