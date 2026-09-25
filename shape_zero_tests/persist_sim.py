#!/usr/bin/env python3
"""
persist_sim.py -- direct simulations of the resonant channels found by
persist_resonance.py, with every prediction printed (and committed) before any run.

Model: 04_scripts/session/model.py's Lattice (q = 1, one dimer n = 1, c = 1, the
elementwise quadratic nonlinearity, gyroscopic ratio kappa). One dimer suffices:
dimers do not couple on a free lattice (persist_resonance.py docstring).
Launch: every Fourier mode on its own a-branch frequency (per-mode launch).
Readout: per-mode split psi_q = a_q e^{-i w_a t} + b e^{+i w_b t},
    a_q = (w_b psi_q + i dpsi_q) / (w_a + w_b)     (a-branch at wavevector q)
    b   = (w_a psi_q - i dpsi_q) / (w_a + w_b)     (b-branch at wavevector -q)

PART 1 -- three-wave, second harmonic into the b-branch (a + a -> b).
An a-wave psi = A e^{i(kx - w_a t)} drives, through the -(1-i)/4 conj(psi)^2 term,
the b-mode at physical wavevector 2k (psi-wavevector -2k, psi-frequency -w_b). In
the slowly varying amplitude B of that mode, i(2 w_b - kappa) dB/dt = -(1-i)/4 A^2 e^{i delta t},
delta = 2 w_a(k) - w_b(2k). So
    resonant (delta ~ 0):  |B| = r A^2 t,   r = sqrt2 / (4 (2 w_b(2k) - kappa))
    detuned:               |B| = 2 r A^2 |sin(delta t / 2)| / |delta|  (max 2 r A^2/|delta|)
Runs (N = 656 sites, A = 0.01, T = 300):
  R1 kappa*, k = 2 pi 131/656 = 1.25472 (delta = 8e-7: resonant)
  R2 kappa*, k = pi/2                     (detuned)
  R3 kappa = 0.5, k = 1.25472             (detuned; b->aa closed at kappa = 0.5)
  R4 kappa = 3.0, k = 1.25472             (detuned; b->aa closed above ~2.4)

PART 2 -- four-wave, a pump's own pair (k, k -> k+p, k-p), from a 1e-7 noise floor.
The same-branch channel aa -> aa is open for some pumps at every kappa and c
(persist_resonance.py); aa -> ab is open at kappa* for pumps whose D(k, p) range
reaches kappa. Growth rates need the effective quartic coefficient, which is not
derived here, so Part 2's predictions are qualitative: which modes grow and which do
not. A = 0.2, T = 2500, N = 656:
  R5 kappa*, pump in the open band (k, p* printed below): sidebands at k +- p* grow
  R6 kappa*, pump with every channel closed (printed below): nothing grows beyond
     the harmonics the pump drives directly

usage:  python3 persist_sim.py predict | run | report
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as MS

K = MS.SQ5
C = MS.C
KS = 2 * C / math.sqrt(K + 2 * C)
N = 656
OUT = os.path.join(HERE, "persist_sim_runs.json")


def wa(k, kap):
    return 0.5 * (-kap + np.sqrt(kap * kap + 4 * (K + 2 * C * (1 - np.cos(k)))))


def wb(k, kap):
    return wa(k, kap) + kap


def r_coef(k2, kap):
    return math.sqrt(2) / (4 * (2 * wb(k2, kap) - kap))


RUNS1 = {"R1": (KS, 131), "R2": (KS, 164), "R3": (0.5, 131), "R4": (3.0, 131)}
A1, T1 = 0.01, 300.0


def part2_pumps():
    """Pick the Part-2 pumps on the N = 656 grid at kappa*."""
    q = 2 * np.pi * np.arange(N) / N
    q = np.where(q > np.pi, q - 2 * np.pi, q)

    def chans(m):
        k = 2 * np.pi * m / N
        p = q[np.abs(q) > 1e-9]
        D = 2 * wa(k, KS) - wa(k + p, KS) - wa(k - p, KS)
        shg = 2 * wa(k, KS) - wb(2 * k, KS)
        return k, p, D, shg
    # open pump: D changes sign (aa->aa), SHG detuned by > 0.05, aa->ab closed (kappa not in D range)
    best_open, best_closed = None, None
    for m in range(5, N // 2 - 5):
        k, p, D, shg = chans(m)
        sign_change = D.min() < 0 < D.max()
        ab = D.min() <= KS <= D.max() or D.min() <= -KS <= D.max()
        if abs(shg) < 0.1:
            continue
        if sign_change and not ab and best_open is None and k > 1.2:
            best_open = m
        if not sign_change and not ab and best_closed is None and k > 1.75:
            best_closed = m
    return best_open, best_closed


def predict():
    print("=" * 90)
    print("PREDICTIONS (committed before any run)")
    print("=" * 90)
    print(f"  N = {N}, c = 1, kappa* = {KS:.6f}")
    print("\n  PART 1 -- a + a -> b (second harmonic into the b-branch), A = 0.01, T = 300")
    print("     run  kappa     k        2k       delta = 2w_a(k) - w_b(2k)   r        |B|/A at T (pred.)    law")
    pred = {}
    for name, (kap, m) in RUNS1.items():
        k = 2 * np.pi * m / N
        d = 2 * wa(k, kap) - wb(2 * k, kap)
        r = r_coef(2 * k, kap)
        if abs(d) * T1 < 0.1:
            B = r * A1 ** 2 * T1
            law = "linear growth, slope r A^2"
        else:
            B = 2 * r * A1 ** 2 * abs(math.sin(d * T1 / 2)) / abs(d)
            law = f"bounded, max {2 * r * A1 ** 2 / abs(d) / A1:.2e} A"
        pred[name] = dict(kap=kap, m=m, k=k, delta=d, r=r, B_T=B)
        print(f"     {name}   {kap:.4f}   {k:.5f}  {2 * k:.5f}   {d:+.3e}                    {r:.5f}  "
              f"{B / A1:.3e}             {law}")
    print("     Criterion: R1's |B|(t) linear with slope within 10% of r A^2 (pump depletion is second order,")
    print("     |B|/A reaches 0.24 by T); R2-R4 stay below their predicted maximum x 1.2.")
    mo, mc = part2_pumps()
    print("\n  PART 2 -- the pump's own four-wave pair, A = 0.2, noise 1e-7 per mode, T = 2500")
    for tag, m in (("R5 open", mo), ("R6 closed", mc)):
        k = 2 * np.pi * m / N
        q = 2 * np.pi * np.arange(1, N // 2) / N
        D = 2 * wa(k, KS) - wa(k + q, KS) - wa(k - q, KS)
        zs = q[np.where(np.diff(np.sign(D)))[0]]
        shg = 2 * wa(k, KS) - wb(2 * k, KS)
        print(f"     {tag}: m = {m}, k = {k:.4f}; D(k, p) range over p != 0 [{D.min():+.4f}, {D.max():+.4f}]; "
              f"resonant p* = {', '.join(f'{z:.4f}' for z in zs) or 'none'}; SHG detuning {shg:+.4f}; "
              f"aa->ab {'OPEN' if D.min() <= KS <= D.max() else 'closed'}")
        pred[tag.split()[0]] = dict(m=int(m), k=k, pstar=[float(z) for z in zs])
    print("     R5: the a-branch sidebands at k +- p* (within the nonlinear-shift width) grow above the")
    print("     noise floor -- the largest-growing a-mode outside the pump's harmonics lies within")
    print("     |p - p*| < 0.15. R6: no a-mode with |D(k, p)| > 0.05 and no b-mode grows more than")
    print("     10x above its initial noise. CAVEAT: 'closed' is a linear-resonance statement; at")
    print("     finite amplitude, sidebands with |D| of order the nonlinear shift (small p) may still")
    print("     grow modulationally -- that is not excluded by R6's prediction.")
    return pred


def per_mode(u, v, kap):
    psi = u[:, 0] + 1j * u[:, 1]
    dps = v[:, 0] + 1j * v[:, 1]
    P, D = np.fft.fft(psi) / N, np.fft.fft(dps) / N
    q = 2 * np.pi * np.fft.fftfreq(N)
    w_a, w_b = wa(q, kap), wb(q, kap)
    a = (w_b * P + 1j * D) / (w_a + w_b)
    b = (w_a * P - 1j * D) / (w_a + w_b)
    return a, b


def launch(kap, m, A, noise=0.0, seed=1):
    x = np.arange(N)
    q = 2 * np.pi * np.fft.fftfreq(N)
    coef = np.zeros(N, complex)
    coef[m % N] = A
    if noise:
        rng = np.random.default_rng(seed)
        coef = coef + noise * np.exp(2j * np.pi * rng.random(N)) * (np.arange(N) != m % N)
    psi = np.fft.ifft(coef) * N
    dpsi = np.fft.ifft(-1j * wa(q, kap) * coef) * N
    u = np.stack([psi.real, psi.imag], axis=1)
    v = np.stack([dpsi.real, dpsi.imag], axis=1)
    return u, v


def run():
    pred = predict()
    res = {}
    for name, (kap, m) in RUNS1.items():
        lat = MS.Lattice(n=1, N=N, kappa=kap)
        u, v = launch(kap, m, A1)
        ts, Bs = [], []
        t = 0.0
        while t < T1 - 1e-9:
            u, v, _ = lat.run(u, v, 2.0)
            t += 2.0
            _, b = per_mode(u, v, kap)
            Bs.append(float(abs(b[(-2 * m) % N])))
            ts.append(t)
        res[name] = dict(t=ts, B=Bs)
        print(f"  {name} done", flush=True)
    for name in ("R5", "R6"):
        m = pred[name]["m"]
        lat = MS.Lattice(n=1, N=N, kappa=KS)
        u, v = launch(KS, m, 0.2, noise=1e-7)
        a0, b0 = per_mode(u, v, KS)
        snaps = []
        for T in (500.0, 1000.0, 1500.0, 2000.0, 2500.0):
            u, v, drift = lat.run(u, v, 500.0)
            a, b = per_mode(u, v, KS)
            snaps.append(dict(T=T, a=np.abs(a).tolist(), b=np.abs(b).tolist(), drift=drift))
            print(f"  {name} T = {T:.0f}", flush=True)
        res[name] = dict(a0=np.abs(a0).tolist(), b0=np.abs(b0).tolist(), snaps=snaps)
    json.dump(dict(pred=pred, res=res), open(OUT, "w"))


def report():
    d = json.load(open(OUT))
    pred, res = d["pred"], d["res"]
    print("=" * 90)
    print("MEASURED")
    print("=" * 90)
    print("\n  PART 1   run   |B|/A at T (pred / meas)      fitted slope / (r A^2)    max |B|/A (pred bound x1.2)")
    for name in RUNS1:
        t, B = np.array(res[name]["t"]), np.array(res[name]["B"])
        p = pred[name]
        slope = np.polyfit(t[: len(t) // 2], B[: len(t) // 2], 1)[0]
        bound = 2 * p["r"] * A1 ** 2 / abs(p["delta"]) if abs(p["delta"]) * T1 >= 0.1 else None
        verdict = ("PASS" if abs(slope / (p["r"] * A1 ** 2) - 1) < 0.10 else "FAIL") if bound is None else \
                  ("PASS" if B.max() <= 1.2 * bound else "FAIL")
        print(f"            {name}    {p['B_T'] / A1:.3e} / {B[-1] / A1:.3e}          "
              f"{slope / (p['r'] * A1 ** 2):.4f}                    {B.max() / A1:.3e}"
              + (f" ({1.2 * bound / A1:.3e})" if bound else "") + f"   {verdict}")
    print("\n  PART 2   growth factor = |mode| at T / initial noise, a-branch (b-branch)")
    for name in ("R5", "R6"):
        m = pred[name]["m"]
        k = pred[name]["k"]
        a0, b0 = np.array(res[name]["a0"]), np.array(res[name]["b0"])
        last = res[name]["snaps"][-1]
        a, b = np.array(last["a"]), np.array(last["b"])
        harm = {(j * m) % N for j in range(-6, 7)}
        g = np.array([a[i] / a0[i] if i not in harm and a0[i] > 0 else 0 for i in range(N)])
        gb = np.array([b[i] / b0[i] if b0[i] > 0 and i not in harm and (-i) % N not in harm else 0
                       for i in range(N)])
        top = np.argsort(g)[::-1][:6]
        q = 2 * np.pi * np.fft.fftfreq(N)
        print(f"   {name} (pump k = {k:.4f}, predicted p* = {pred[name]['pstar']}), drift {last['drift']:.1e}:")
        print("      largest a-branch growth, excluding the pump's harmonics: "
              + ", ".join(f"q = {q[i]:+.4f} (p = {q[i] - k:+.4f}) x{g[i]:.1e}" for i in top))
        print(f"      largest b-branch growth: x{gb.max():.1e} at q = {q[int(np.argmax(gb))]:+.4f}")
        for s in res[name]["snaps"]:
            aa = np.array(s["a"])
            print(f"      T = {s['T']:.0f}: max a-growth outside harmonics x"
                  f"{max(aa[i] / a0[i] for i in range(N) if i not in harm and a0[i] > 0):.2e}")


if __name__ == "__main__":
    {"predict": predict, "run": run, "report": report}[sys.argv[1]]()
