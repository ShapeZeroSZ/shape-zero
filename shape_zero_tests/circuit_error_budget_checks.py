#!/usr/bin/env python3
"""circuit_error_budget_checks.py -- POST HOC diagnostics on circuit_error_budget.py, written after its output was
seen (expectations 65ee4eb). Not part of the pre-registered budget.
 1. Which component drives the P2 spread at 1% (sd 53%) and 0.1% (sd 1.3%): C, L_g, L_c, or the gyrator
    (G_a, G_b and Howland shunt) alone; N = 32, S = 0.05, 20 realisations.
 2. The gyrator mismatch's Delta w change against its effect on the mean G: is it absorbed by the calibration?
 3. (appended after 1-2 were seen) python3 circuit_error_budget_checks.py 3: the 0.1% P2 spread with 60 realisations.
 4. (appended after 3 was seen) python3 circuit_error_budget_checks.py 4: which non-ideality lowers that spread.
usage: python3 circuit_error_budget_checks.py"""
import numpy as np

import circuit_error_budget as CB

N = 32
g4 = CB.eta_shape(N, "gauss", 4)


def one(tol, keys, rng):
    p = dict(CB.ideal(N))
    base = dict(C=CB.C0, Lg=CB.LG0, Lc=CB.LC0, Ga=CB.G0, Gb=CB.G0)
    for k in keys:
        if k == "gs":
            p["gs"] = CB.G0 * tol * (rng.standard_normal(N) + rng.standard_normal(N))
        else:
            p[k] = base[k] * (1 + tol * rng.standard_normal(N))
    return p


def check3():
    """3. (appended after 1-2 were seen) The 0.1% P2 spread with more realisations: X2 (seed 2028) gave sd 1.3%,
    X8 (seed 88, plus parasitics) 0.22%. 60 realisations each of X2 and X8 conditions; sd, median |dev|, max |dev|."""
    print("3. 0.1% P2 spread, 60 realisations (N = 32, S = 0.05)")
    for lab, mk in (("X2 tolerance only", lambda r: CB.with_tol(N, 0.001, r)), ("X8 realistic", lambda r: CB.realistic(N, r))):
        rng = np.random.default_rng(99)
        v = np.array([CB.C_ratio(mk(rng), g4) for _ in range(60)])
        d = np.abs(v - np.median(v))
        print(f"  {lab:<18s}: mean {v.mean():.4f}, median {np.median(v):.4f}, sd {v.std():.4f}, median |dev| {np.median(d):.4f}, "
              f"max |dev| {d.max():.4f}, fraction |dev| > 0.02: {np.mean(d > 0.02):.2f}")


def check4():
    """4. (appended after 3 was seen) Which added non-ideality lowers the 0.1% spread: tolerance plus Q = 100 only,
    plus C_p/C_in only, plus the 5 MHz pole only; 60 realisations; also the mode overlaps of the chosen +-k modes."""
    print("4. 0.1% tolerance plus one non-ideality at a time, 60 realisations (N = 32, S = 0.05)")
    W2 = CB.W2

    def q100(p):
        p["Rg"] = W2 * p["Lg"] / 100; p["Rc"] = W2 * p["Lc"] / 100; return p

    def par(p):
        p["Cp"] = 10e-12 * np.ones(N); p["Cin"] = 3e-12; return p

    def pole(p):
        p["tau"] = 1 / (2 * np.pi * 5e6); return p
    for lab, f in (("+ Q = 100", q100), ("+ C_p, C_in", par), ("+ 5 MHz pole", pole)):
        rng = np.random.default_rng(99)
        v = np.array([CB.C_ratio(f(CB.with_tol(N, 0.001, rng)), g4) for _ in range(60)])
        d = np.abs(v - np.median(v))
        print(f"  {lab:<13s}: median {np.median(v):.4f}, sd {v.std():.4f}, max |dev| {d.max():.4f}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["3"]:
        check3(); raise SystemExit
    if sys.argv[1:] == ["4"]:
        check4(); raise SystemExit
    print("POST HOC. 1. P2 spread by component (N = 32, S = 0.05, 20 realisations)")
    for tol in (0.01, 0.001):
        for lab, keys in (("C only", ["C"]), ("L_g only", ["Lg"]), ("L_c only", ["Lc"]), ("gyrator only", ["Ga", "Gb", "gs"])):
            rng = np.random.default_rng(31)
            r = np.array([CB.C_ratio(one(tol, keys, rng), g4) for _ in range(20)])
            print(f"  {tol:.1%} {lab:<13s}: C ratio mean {r.mean():.4f}, sd {r.std():.4f}")
    print("2. gyrator mismatch 1%: Delta w change vs the mean-G change")
    rng = np.random.default_rng(5)
    d0 = CB.dw(CB.ideal(N), 8)
    for _ in range(5):
        p = one(0.01, ["Ga", "Gb", "gs"], rng)
        mg = 0.5 * (p["Ga"].mean() + p["Gb"].mean()) / CB.G0 - 1
        print(f"  Delta w change {CB.dw(p, 8) / d0 - 1:+.2e}; mean-G change {mg:+.2e}")

