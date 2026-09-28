#!/usr/bin/env python3
"""platform_error_budget_checks.py -- POST HOC diagnostics on platform_error_budget.py, written after its output
was seen (expectations 1c63b6d). Not part of the pre-registered budget.
 1. Which disorder component drives the P2 spread (sd 19% with the +-S protocol): K, c or b alone, 20 realisations.
 2. The P2 spread against the K-disorder level (0.3%, 0.1%, 0.03%, 0.01%), N = 32 and 64, S = 0.05 and 0.02.
 3. P3 at reduced disorder: spread of the four localised shapes, N = 32 and 64.
 4. Anharmonicity alone: ideal ring, no damping, time domain, A = 0.001, 0.02, 0.1 rad.
 5. (appended after 1-4 were seen) The P2 spread against spring disorder: python3 platform_error_budget_checks.py 5
usage: python3 platform_error_budget_checks.py"""
import numpy as np

import platform_error_budget as PB


def spread(N, dis, S, n=20, seed=7):
    rng = np.random.default_rng(seed)
    g = PB.np.sqrt(PB.K0 + 2 * PB.C0) / 500
    eta = PB.eta_shape(N, "gauss", N / 8)
    r = []
    for _ in range(n):
        Kn = PB.K0 * (1 + dis.get("K", 0) * rng.standard_normal(N))
        cn = PB.C0 * (1 + dis.get("c", 0) * rng.standard_normal(N))
        bn = PB.B0 * (1 + dis.get("b", 0) * rng.standard_normal(N))
        r.append(PB.C_ratio(Kn, cn, bn, g, eta, S=S)[0])
    r = np.array(r)
    return r.mean(), r.std()


def shapes(N, dis, n=10, seed=11):
    rng = np.random.default_rng(seed)
    g = np.sqrt(PB.K0 + 2 * PB.C0) / 500
    sh = [("gauss", N * 3 / 32), ("gauss", N * 5 / 32), ("sech2", N / 8), ("two", N * 2.5 / 32)]
    sp = []
    for _ in range(n):
        Kn = PB.K0 * (1 + dis.get("K", 0) * rng.standard_normal(N))
        cn = PB.C0 * (1 + dis.get("c", 0) * rng.standard_normal(N))
        bn = PB.B0 * (1 + dis.get("b", 0) * rng.standard_normal(N))
        v = np.array([PB.C_ratio(Kn, cn, bn, g, PB.eta_shape(N, s, p), S=0.02)[0] for s, p in sh])
        sp.append((v.max() - v.min()) / abs(v.mean()))
    return np.mean(sp), np.max(sp)


def check5():
    """5. (appended after checks 1-4 were seen) P2 spread against SPRING disorder, K 0.03%, b 0.1%."""
    print("5. P2 spread against spring disorder (K 0.03%, b 0.1%, 20 realisations)")
    for N, S in ((32, 0.05), (32, 0.02), (64, 0.02)):
        for dc in (0.003, 0.001, 0.0003):
            m, s = spread(N, {"K": 0.0003, "c": dc, "b": 0.001}, S)
            print(f"  N = {N:<3d} S = {S}: spring disorder {dc:.2%}: C ratio mean {m:.4f}, sd {s:.4f}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["5"]:
        check5(); raise SystemExit
    print("POST HOC. 1. P2 spread by disorder component (N = 32, S = 0.05, +-S protocol, 20 realisations)")
    for lab, dis in (("K 0.3% only", {"K": 0.003}), ("c 1% only", {"c": 0.01}), ("b 0.5% only", {"b": 0.005})):
        m, s = spread(32, dis, 0.05)
        print(f"  {lab:<12s}: C ratio mean {m:.4f}, sd {s:.4f}")
    print("2. P2 spread against K disorder (c 1%, b 0.5% kept)")
    for N in (32, 64):
        for S in (0.05, 0.02):
            for dK in (0.003, 0.001, 0.0003, 0.0001):
                m, s = spread(N, {"K": dK, "c": 0.01, "b": 0.005}, S)
                print(f"  N = {N:<3d} S = {S}: K disorder {dK:.2%}: C ratio mean {m:.4f}, sd {s:.4f}")
    print("3. P3 spread of the four localised shapes (S = 0.02, 10 realisations)")
    for N in (32, 64):
        for dis in ({}, {"K": 0.0003, "c": 0.001, "b": 0.001}, {"K": 0.0001, "c": 0.0003, "b": 0.0003}):
            m, mx = shapes(N, dis)
            print(f"  N = {N:<3d} disorder {dis or 'none'}: spread mean {m:.4f}, max {mx:.4f}")
    print("4. anharmonicity alone (ideal ring, gamma = 0, T = 300 s)")
    o = np.ones(32)
    base = None
    for A in (0.001, 0.02, 0.1):
        d = PB.dw_time(PB.K0 * o, PB.C0 * o, PB.B0 * o, 0.0, 8, A)
        base = d if base is None else base
        print(f"  A = {A}: Delta w(pi/2) {d:.8f}, relative to A = 0.001: {d / base - 1:+.2e}")

