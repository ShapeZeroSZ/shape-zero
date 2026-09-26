#!/usr/bin/env python3
"""
persist_radial.py -- does a RADIAL phi-well in the J sector (shape_zero_tests/
model_radial.py: F_j = -(sqrt5 + |psi_j|) psi_j per dimer) make J-compatibility hold
at every order, and what happens to the persistence windows?

SYMMETRY. The radial force is U(1)-equivariant, as are the gyroscopic term and the
Laplacian, so the J sector conserves the Noether charge of psi -> e^{i theta} psi,
    N = sum_n [ Im(conj(psi) dpsi/dt) - (kappa/2) |psi|^2 ]
(derived from L = |dpsi|^2/2 - V - (kappa/2) Im(conj(psi) dpsi/dt), which gives
model.py's kappa JJ v). An a-wave (e^{-i w_a t}) contributes -(w_a + kappa/2)|A|^2 and a
b-wave (e^{+i w_b t}) +(w_b - kappa/2)|B|^2: the two branches carry OPPOSITE charge, so
every resonant process must conserve n_a - n_b.
  FORBIDDEN (Delta(n_a - n_b) != 0): every three-wave channel; a+a -> a+b, a+a -> b+b;
    1 -> 3 a->aaa, b->aaa, b->aab, a->abb, b->bbb, a->bbb.
  ALLOWED: same-branch quartets (aa->aa, bb->bb); 1 -> 3 a->aab, b->abb; and PAIR
    CREATION -- n a-waves -> (n+1) a-waves + 1 b-wave -- at any order, whenever energy
    and momentum allow: n w_a,max >= (n+2) w_a,min + kappa for even n (all incoming at
    the zone corner, all outgoing at k = 0 conserves crystal momentum).
So the hypothesis is half right: phase charge is conserved at every order, and
chirality CONVERSION (a -> b) is forbidden; the b-branch can still be POPULATED by
charge-neutral pair creation, and for any kappa some order n allows it.
CAVEAT: r psi = |psi| psi is homogeneous of degree 2 and non-analytic at psi = 0; a
multi-wave expansion has all harmonic orders at the same power of the amplitude, so
"high order" means high harmonic order, not a higher power of A.

usage:  python3 persist_radial.py predict | run | report
"""
import json
import math
import os
# model.py's J sector defaults to the radial well since 2026-09-26; this script's
# "model.py" results are the ELEMENTWISE form, so pin it (before model is imported).
os.environ["SZ_J_WELL"] = "elementwise"
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_radial as MR          # built by radial_model.py build
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import persist_resonance as PR
import persist_sim as PS

K, C = PR.K, 1.0
KS = PR.kappa_star(1.0)
N = PS.N
OUT = os.path.join(HERE, "persist_radial_runs.json")
A1, T1 = 0.01, 300.0
M1, M2 = 131, 164                  # k1 = 1.25472 (R1's resonant k), k2 = pi/2


# ------------------------------------------------------------------ analysis
def pair_threshold(kap, c, q, nmax=40):
    """Smallest n (incoming a-waves) for which n a -> (n+1) a + b is energetically
    open with crystal momentum conserved by the corner/zero configurations."""
    wmin = PR.wa(K, kap)
    wmax = PR.wa(K + 4 * q * c, kap)
    for n in range(1, nmax + 1):
        if n % 2 == 0:
            ok = n * wmax - (n + 2) * wmin - kap >= 0          # all in at corner, all out at 0
        else:
            ok = (n - 1) * wmax - (n + 1) * wmin - kap >= 0    # one outgoing b at the corner
        if ok:
            return n
    return None


def radial_windows():
    """The persistence table's channels with the U(1) selection rule applied."""
    rows = []
    for q in (1, 3):
        for c in ((0.1, 0.25, 0.5, 1.0, 2.0, 3.0) if q == 1 else (0.1, 0.25, 0.5, 1.0, 2.0)):
            kst = PR.kappa_star(c)
            ok = []
            for kap in np.round(np.linspace(max(kst, 0.02), 8.0, 41), 3):
                r = PR.ranges_1d(kap, c) if q == 1 else PR.ranges_3d(kap, c)
                open_13 = r["G3"][0] - 1e-9 <= kap <= r["G3"][1] + 1e-9     # a->aab, b->abb (m = 1)
                ok.append((kap, not open_13))
            rows.append((q, c, kst, ok))
    return rows


def spans(ok):
    out, s, prev = [], None, None
    for k, m in ok:
        if m and s is None:
            s = k
        if not m and s is not None:
            out.append(f"[{s:.2f},{prev:.2f}]"); s = None
        prev = k
    if s is not None:
        out.append(f"[{s:.2f},{ok[-1][0]:.1f}+]")
    return " ".join(out) or "none"


# ------------------------------------------------------------------ simulations
def charge(u, v, kap):
    psi = u[:, 0] + 1j * u[:, 1]
    dps = v[:, 0] + 1j * v[:, 1]
    return float(np.sum(np.imag(np.conj(psi) * dps) - 0.5 * kap * np.abs(psi) ** 2))


def launch(kap, modes):
    q = 2 * np.pi * np.fft.fftfreq(N)
    coef = np.zeros(N, complex)
    for m, a in modes:
        coef[m % N] = a
    psi = np.fft.ifft(coef) * N
    dpsi = np.fft.ifft(-1j * PS.wa(q, kap) * coef) * N
    return np.stack([psi.real, psi.imag], 1), np.stack([dpsi.real, dpsi.imag], 1)


RUNS = {
    "P1 radial, R1 rerun (k1 only)": ("radial", [(M1, A1)]),
    "P2 radial, two waves k1 + k2": ("radial", [(M1, A1), (M2, A1)]),
    "P2c elementwise, two waves k1 + k2": ("elementwise", [(M1, A1), (M2, A1)]),
}


def predict():
    print("=" * 92)
    print("PREDICTIONS (committed before any run) -- radial phi-well in the J sector")
    print("=" * 92)
    k1 = 2 * np.pi * M1 / N
    r = PS.r_coef(2 * k1, KS)
    print(f"  kappa* = {KS:.6f}, N = {N}, k1 = {k1:.5f} (R1's resonant wavenumber), k2 = pi/2, A = {A1}")
    print("  P1  radial, single a-wave at k1: |psi| = A is constant, so F = -(sqrt5 + A) psi and the")
    print("      plane wave is an EXACT solution (frequency shifted, Q -> Q + A). The b-mode at 2k1 does")
    print("      not grow at all: max |B|/A over T < 1e-10 (elementwise R1 gave 0.207 at T; slope")
    print(f"      r A^2, r = {r:.5f}). a-amplitude at k1 constant to < 1e-10 relative.")
    print("  P2  radial, two a-waves (k1, k2): phase charge N conserved, |dN/N| < 1e-6 over T (RK4);")
    print("      the b-branch weight at 2k1 shows no secular growth: |B(2k1)|(T) < 1e-3 A (it would be")
    print(f"      {r * A1 ** 2 * T1 / A1:.3f} A under the elementwise well). All b content is non-resonant.")
    print("  P2c elementwise, same two waves (control): |B(2k1)| grows linearly, slope within 10% of")
    print("      r A^2 as in R1; N NOT conserved (|dN/N| > 1e-4).")
    print("\n  ANALYSIS (no simulation): pair creation n a -> (n+1) a + b, lowest open n, c = 1:")
    for q in (1, 3):
        line = ", ".join(f"kappa {kap:.3f}: n = {pair_threshold(kap, 1.0, q)}"
                         for kap in (KS, 2.0, 5.0, 7.5, 10.0))
        print(f"     q = {q}: {line}")
    print("     -> no kappa closes every order: n grows with kappa but some n always opens.")


def run():
    predict()
    res = {}
    for name, (well, modes) in RUNS.items():
        Lat = MR.Lattice if well == "radial" else PS.MS.Lattice
        lat = Lat(n=1, N=N, kappa=KS)
        u, v = launch(KS, modes)
        N0 = charge(u, v, KS)
        ts, B, Aa, dN = [], [], [], []
        t = 0.0
        while t < T1 - 1e-9:
            u, v, _ = lat.run(u, v, 2.0)
            t += 2.0
            a, b = PS.per_mode(u, v, KS)
            ts.append(t)
            B.append(float(abs(b[(-2 * M1) % N])))
            Aa.append(float(abs(a[M1])))
            dN.append(charge(u, v, KS) / N0 - 1)
        res[name] = dict(t=ts, B=B, A=Aa, dN=dN)
        print(f"  {name} done", flush=True)
    json.dump(res, open(OUT, "w"))


def report():
    res = json.load(open(OUT))
    k1 = 2 * np.pi * M1 / N
    r = PS.r_coef(2 * k1, KS)
    print("MEASURED")
    for name, d in res.items():
        t, B, Aa, dN = map(np.array, (d["t"], d["B"], d["A"], d["dN"]))
        slope = np.polyfit(t[: len(t) // 2], B[: len(t) // 2], 1)[0] / (r * A1 ** 2)
        print(f"  {name}: max|B(2k1)|/A = {B.max() / A1:.3e}, at T {B[-1] / A1:.3e}; slope/(r A^2) = {slope:+.4f}; "
              f"max|dN/N| = {np.abs(dN).max():.2e}; a(k1) relative change {abs(Aa[-1] / Aa[0] - 1):.2e}")
    print("\n  PERSISTENCE TABLE UNDER THE RADIAL WELL (orders of the original table: three-wave,")
    print("  four-wave pump pairs, 1->3). Every branch-changing channel is forbidden by the U(1)")
    print("  selection rule; same-branch quartets remain open (strict form still unsatisfiable).")
    print("  The only remaining requirement besides kappa >= kappa* is 1->3 a->aab / b->abb closed:")
    for q, c, kst, ok in radial_windows():
        print(f"     q = {q}, c = {c:4.2f} (c/sqrt5 = {c / K:.3f}): kappa* = {kst:.4f}; allowed: {spans(ok)}")


if __name__ == "__main__":
    {"predict": predict, "run": run, "report": report}[sys.argv[1]]()
