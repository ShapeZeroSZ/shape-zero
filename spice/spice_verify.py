#!/usr/bin/env python3
"""spice_verify.py -- independent check of the 32-node LC-gyrator ring design in ngspice (TI OPAx197 macro-models)
against our own linear simulation (shape_zero_tests/circuit_error_budget.py). This verifies the DESIGN, not the
model: ngspice solves the circuit, and the circuit was designed to realise the model's own equations.

Runs (AC: 1 A drive into node 0; every node voltage written):
  band   gyrators unpowered, 48-75 kHz, 25 Hz steps     -> K, c (band fit), zero check
  p1     gyrators powered, same sweep                   -> w+-(m), m = 1..15; P1; b; product rule
  bump   +-S (Gaussian sigma 4, S = 0.05) via the GIC R5 -> P2 ; four localised shapes and a ramp -> P3
  tran   pulse into node 0, 3 ms, for Q = 100, 300, 600, 1000 -> growth/decay rate (stability window)
Estimator (as CIRCUIT_PREREGISTRATION.md sec 2): P(f) = sum_n V_n e^{-+2 pi i m n/32}; rational 3/3 fit
(Sanathanan-Koerner) over +-2 kHz around the design line; the pole nearest the design frequency.
usage: python3 spice_verify.py            (needs ngspice and OPAx197.LIB in this directory)
STATUS: results/spice_verify_output_gic_v1.txt is this script run with GIC version 1 (op-amp B + at g5), which
oscillates in the time domain; a rerun with version 2 (amendment 2) was stopped because version 2 latches at DC
(results/transcribed_runs.txt). The stability section later moved to Gear integration and the 4-node transients.
"""
import os
import subprocess
import sys

import numpy as np

import gen_ring as GR

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "shape_zero_tests"))
import circuit_error_budget as CB  # noqa: E402

N = 32
BH = 0.106
S = 0.05
K, CC = 1e11, 0.25e11
B = BH * np.sqrt(K)


def design_f(m, sign):
    k = 2 * np.pi * m / N
    Qk = K + 2 * CC * (1 - np.cos(k))
    bb = B if sign else 0.0
    return (sign * bb * np.sin(k) + np.sqrt((bb * np.sin(k)) ** 2 + Qk)) / (2 * np.pi)


def run(name, **kw):
    out = os.path.join(HERE, "out", name + ".txt")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    if not os.path.exists(out):
        cir = os.path.join(HERE, "out", name + ".cir")
        open(cir, "w").write(GR.netlist(out=out, **kw))
        subprocess.run(["ngspice", "-b", cir], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    d = np.loadtxt(out, skiprows=1)
    return d


def ac_data(d):
    f = d[:, 0]
    V = d[:, 1::2] + 1j * d[:, 2::2]        # wr_singlescale: freq, then (re, im) per vector
    return f, V


def sk_fit(f, P, f0, hw=2000.0, deg=3, it=6):
    m = np.abs(f - f0) <= hw
    x = (f[m] - f0) / hw
    y = P[m]
    w = np.ones_like(x)
    for _ in range(it):
        A = np.hstack([np.vander(x, deg + 1, increasing=True), -(y[:, None] * np.vander(x, deg, increasing=True))])
        rhs = y * x ** deg
        coef, *_ = np.linalg.lstsq(A * w[:, None], rhs * w, rcond=None)
        b = np.concatenate([coef[deg + 1:], [1.0]])
        w = 1 / np.abs(np.polyval(b[::-1], x))
    r = np.roots(b[::-1])
    r = r[np.abs(r.imag) < 1.0]
    j = np.argmin(np.abs(r.real))
    return f0 + hw * r[j].real, hw * abs(r[j].imag)


def vf(f, V, f0, hw=2000.0, npol=16, it=8):
    """Common-pole rational fit (vector fitting, Sanathanan-Koerner weights) of all 32 node responses over
    f0 +- hw. Returns the poles (Hz, complex) and the residue vectors (one row per pole)."""
    msk = np.abs(f - f0) <= hw
    x = (f[msk] - f0) / hw
    Y = V[msk]
    npts = len(x)
    w = np.ones(npts)
    for _ in range(it):
        Vd = np.vander(x, npol + 1, increasing=True)
        Dd = np.vander(x, npol, increasing=True)
        Qv, _ = np.linalg.qr(Vd * w[:, None])
        Mw = -(Y * w[:, None]).T[:, :, None] * Dd[None]                  # (channels, points, npol)
        Mw = Mw - np.einsum("pk,ckj->cpj", Qv, np.einsum("pk,cpj->ckj", Qv.conj(), Mw))
        rw = (Y * (x ** npol * w)[:, None]).T                             # (channels, points)
        rw = rw - np.einsum("pk,ck->cp", Qv, rw @ Qv.conj())
        bc, *_ = np.linalg.lstsq(Mw.reshape(-1, npol), -rw.reshape(-1), rcond=None)
        b = np.concatenate([bc, [1.0]])
        w = 1 / np.abs(np.polyval(b[::-1], x))
    poles = np.roots(b[::-1])
    Bm = np.hstack([1 / (x[:, None] - poles[None, :]), np.ones((npts, 1))])
    R, *_ = np.linalg.lstsq(Bm, Y, rcond=None)
    return f0 + hw * poles, R[:-1]


def pick(poles, R, m, sign, f0, hw=2000.0):
    """The pole whose residue vector (mode shape) overlaps most with the +k (sign = +1) or -k plane wave. Phasor
    convention e^{+jwt}: the +k mode's shape is ~ e^{-ikn}."""
    pw = np.exp(1j * sign * 2 * np.pi * m * np.arange(len(R[0])) / len(R[0]))
    ok = (np.abs(poles.real - f0) < hw) & (np.abs(poles.imag) < hw)
    ov = np.abs(R @ pw) / (np.linalg.norm(R, axis=1) * np.sqrt(len(pw)) + 1e-300)
    ov[~ok] = -1
    j = int(np.argmax(ov))
    return poles[j], ov[j]


def proj(V, m, s):
    return V @ np.exp(-s * 2j * np.pi * m * np.arange(N) / N)


def lines(f, V, ms, which="both", gyro=True):
    """w+(m), w-(m) in Hz from the vector fit (pole real parts). With the gyrators off the design lines are the
    unsplit band (b = 0)."""
    out = {}
    for m in ms:
        for sign in (1, -1):
            if which != "both" and which != sign:
                continue
            f0 = design_f(m, sign) if gyro else design_f(m, 0)
            po, R = vf(f, V, f0)
            out[(m, sign)] = pick(po, R, m, sign, f0)[0].real
    return out

def signed_rate(f, P, f0, hw=2000.0):
    """Decay rate 2 pi |Im p| of the fitted pole, signed so that decay is positive. With the e^{+jwt} phasor
    convention a decaying mode's pole lies at f0 + i gamma/(2 pi), gamma > 0 (checked on the Q = 100 run)."""
    m = np.abs(f - f0) <= hw
    x = (f[m] - f0) / hw
    y = P[m]
    w = np.ones_like(x)
    deg = 3
    for _ in range(6):
        A = np.hstack([np.vander(x, deg + 1, increasing=True), -(y[:, None] * np.vander(x, deg, increasing=True))])
        coef, *_ = np.linalg.lstsq(A * w[:, None], y * x ** deg * w, rcond=None)
        b = np.concatenate([coef[deg + 1:], [1.0]])
        w = 1 / np.abs(np.polyval(b[::-1], x))
    r = np.roots(b[::-1])
    r = r[np.abs(r.imag) < 1.0]
    j = np.argmin(np.abs(r.real))
    return 2 * np.pi * hw * r[j].imag


def run4(Q, gic=True):
    """4-node transient (the full ring was too slow). Gear integration: with trapezoidal integration the Q = 100
    run stalled at 134 us and the Q = 1000 run aborted at 20 us ("timestep too small" in a GIC op-amp macro-model)."""
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    out = os.path.join(HERE, "out", f"tran4_Q{Q}_{'gear' if gic else 'idealGIC'}.txt")
    if not os.path.exists(out):
        net = GR.netlist(analysis="tran", Q=Q, tran=(0.35e-3, 4e-7), out=out, nodes=4, gic=gic)
        if gic:
            net = net.replace(".options reltol=1e-3 rshunt=1e9", ".options method=gear reltol=1e-3 rshunt=1e9")
        cir = out.replace(".txt", ".cir")
        open(cir, "w").write(net)
        subprocess.run(["ngspice", "-b", cir], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.loadtxt(out, skiprows=1)

def bump_R5(eta):
    L = 1 / (1 / GR.LGIC0 + K * GR.C * S * eta)
    return GR.r5_for(L)


def eta(kind, p=None):
    return CB.eta_shape(N, kind, p)


def own_model():
    CB.G0 = BH * np.sqrt(CB.C0 / CB.LG0); CB.B0 = CB.G0 / CB.C0
    p = dict(CB.ideal(N))
    p["Rg"] = CB.W2 * p["Lg"] / 100; p["Rc"] = CB.W2 * p["Lc"] / 100
    p["tau"] = 1 / (2 * np.pi * 9.25e6)          # the SPICE unit test's equivalent VCCS pole
    p["Cp"] = 10e-12 * np.ones(N); p["Cin"] = 6.5e-12
    return p


def prefetch():
    """Run every SPICE job, four at a time, before the analysis (results are cached in out/)."""
    from concurrent.futures import ThreadPoolExecutor
    win = dict(f=(54.0e3, 69.8e3, 25.0))
    jobs = [("band", dict(gyro=False)), ("p1", {}), ("m8_S0", win)]
    for kind, prm in [("gauss", 4), ("gauss", 3), ("gauss", 5), ("sech2", 4), ("two", 2.5), ("ramp", None)]:
        e = eta(kind, prm)
        jobs += [(f"m8_{kind}{prm}_p", dict(R5=bump_R5(e), **win)), (f"m8_{kind}{prm}_m", dict(R5=bump_R5(-e), **win))]
    jobs += [(f"stab_Q{Q}", dict(Q=Q, f=(48e3, 75e3, 25.0))) for Q in (100, 200, 300, 450, 600, 1000)]
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(run4, Q, g) for Q in (100, 1000) for g in (True, False)] + [ex.submit(run, n, **kw) for n, kw in jobs]
        for f in futs:
            f.result()


def main():
    print("SPICE VERIFICATION OF THE 32-NODE DESIGN (ngspice 42, TI OPAx197 macro-model). This checks the design;")
    print("the simulator solves the circuit equations, which the design was built to make the model's equations.")
    # ---- band and P1 --------------------------------------------------------------------------------------------
    fb, Vb = ac_data(run("band", gyro=False))
    fp, Vp = ac_data(run("p1"))
    ms = list(range(0, 17))
    band = lines(fb, Vb, ms, gyro=False)
    wb = np.array([0.5 * (band[(m, 1)] + band[(m, -1)]) for m in ms]) * 2 * np.pi
    zero = max(abs(band[(m, 1)] - band[(m, -1)]) for m in range(1, 16)) / (2 * B / (2 * np.pi))
    X = np.vstack([np.ones(17), 2 * (1 - np.cos(2 * np.pi * np.array(ms) / N))]).T
    (Kc, cc), *_ = np.linalg.lstsq(X, wb ** 2, rcond=None)
    fitres = np.abs(np.sqrt(X @ [Kc, cc]) / wb - 1).max()
    L = lines(fp, Vp, range(1, 16))
    dw = {m: (L[(m, 1)] - L[(m, -1)]) * 2 * np.pi for m in range(1, 16)}
    bc = dw[8] / 2
    p1 = {m: dw[m] / dw[8] - np.sin(2 * np.pi * m / N) for m in range(1, 16)}
    prod = max(abs(L[(m, 1)] * L[(m, -1)] * 4 * np.pi ** 2 / (Kc + 2 * cc * (1 - np.cos(2 * np.pi * m / N))) - 1) for m in range(1, 16))
    print("\nCALIBRATION (SPICE)")
    print(f"  zero check, gyrators off: max |f+ - f-| = {zero:.1e} of Delta f(pi/2)")
    print(f"  band fit: K = {Kc:.5e} s^-2 (design 1e11), c-hat = {cc / Kc:.5f} (design 0.25), max band-fit residual {fitres:.1e}")
    print(f"  b = {bc:.5e} s^-1, beta-hat = {bc / np.sqrt(Kc):.5f} (design 0.106)")
    print(f"  product rule w+ w- / Q(k) - 1: max {prod:.1e}")
    # ---- own model, same quantities ---------------------------------------------------------------------------
    p = own_model()
    lam, Vm = CB.modes(p)
    own = {(m, s): CB.mode_w(lam, Vm, m, s) / (2 * np.pi) for m in range(1, 16) for s in (1, -1)}
    print("\nRESONANCES: SPICE vs own linear model (own: ideal GIC; VCCS pole 9.25 MHz; C_p 10 pF; C_in 6.5 pF; Q 100)")
    print("   m   f+ SPICE   f+ own    diff(Hz)   f- SPICE   f- own    diff(Hz)   P1 SPICE   P1 own")
    own_dw8 = own[(8, 1)] - own[(8, -1)]
    for m in range(1, 16):
        o1 = (own[(m, 1)] - own[(m, -1)]) / own_dw8 - np.sin(2 * np.pi * m / N)
        print(f"  {m:2d}  {L[(m, 1)]:9.2f} {own[(m, 1)]:9.2f} {L[(m, 1)] - own[(m, 1)]:+8.2f}   {L[(m, -1)]:9.2f} "
              f"{own[(m, -1)]:9.2f} {L[(m, -1)] - own[(m, -1)]:+8.2f}   {p1[m]:+.2e}  {o1:+.2e}")
    print(f"  P1 max |residual|: SPICE {max(abs(v) for v in p1.values()):.2e}")
    # ---- P2, P3 -------------------------------------------------------------------------------------------------
    print("\nP2 AND P3 (SPICE, +-S even part at m = 8; ratio R to -(1/4) b <dK^2>/c^2 with the design dK)")
    shapes = [("gauss", 4), ("gauss", 3), ("gauss", 5), ("sech2", 4), ("two", 2.5), ("ramp", None)]
    win = dict(f=(54.0e3, 69.8e3, 25.0))
    fd, Vd = ac_data(run("m8_S0", **win))
    L0 = lines(fd, Vd, [8])
    d0 = (L0[(8, 1)] - L0[(8, -1)]) * 2 * np.pi
    CB.G0 = BH * np.sqrt(CB.C0 / CB.LG0); CB.B0 = CB.G0 / CB.C0
    for kind, prm in shapes:
        e = eta(kind, prm)
        res = []
        for sgn, tag in ((1, "p"), (-1, "m")):
            ff, VV = ac_data(run(f"m8_{kind}{prm}_{tag}", R5=bump_R5(sgn * e), **win))
            Lx = lines(ff, VV, [8])
            res.append((Lx[(8, 1)] - Lx[(8, -1)]) * 2 * np.pi)
        shift = 0.5 * (res[0] + res[1]) - d0
        pred = -0.25 * bc * np.mean((Kc * S * e) ** 2) / cc ** 2
        own_r = CB.C_ratio(p, e)
        ideal_r = CB.C_ratio(CB.ideal(N), e)
        print(f"  {kind} {prm}: d(Delta f) = {shift / (2 * np.pi):+.3f} Hz; R SPICE {shift / pred:.4f}; own (realistic nominal) "
              f"{own_r:.4f}; own (ideal) {ideal_r:.4f}")
    # ---- stability ----------------------------------------------------------------------------------------------
    # Full-ring transients with 192 macro-models were too slow (0.1 ms of simulated time took > 10 min), so the
    # stability window is read from the AC runs -- the sign of the fitted poles' imaginary parts (growth vs decay) --
    # with a transient confirmation on a 4-node ring of the same cells (an 8-node ring was also too slow).
    print("\nSTABILITY from AC pole widths (full ring, gyrators on): minimum over m = 1..15 and both branches of the")
    print("signed decay rate 2 pi Im(pole) from the vector fit (negative = growing; convention checked on Q = 100)")
    for Q in (100, 200, 300, 450, 600, 1000):
        ff, VV = ac_data(run(f"stab_Q{Q}", Q=Q, f=(48e3, 75e3, 25.0)))
        rates = []
        for m in range(1, 16):
            for sign in (1, -1):
                po, R = vf(ff, VV, design_f(m, sign))
                pl, _ = pick(po, R, m, sign, design_f(m, sign))
                rates.append(2 * np.pi * pl.imag)
        pown = own_model(); pown["Rg"] = CB.W2 * pown["Lg"] / Q; pown["Rc"] = CB.W2 * pown["Lc"] / Q
        print(f"  Q = {Q:5d}: SPICE min decay rate {min(rates):+.3e} s^-1 ({'unstable' if min(rates) < 0 else 'stable'}); "
              f"own max Re lambda {CB.max_growth(pown):+.3e} s^-1")
    print("\nSTABILITY, time domain: 4-node ring of the same cells, 10 uA triangular kick at 5-15 us, 0.35 ms, Gear")
    for Q in (100, 1000):
        for gic in (True, False):
            d = run4(Q, gic)
            t, V = d[:, 0], d[:, 1:]
            pre = np.abs(V[t < 5e-6]).max()
            E = (V ** 2).sum(axis=1)
            bins = np.arange(0.05e-3, t[-1], 30e-6)
            idx = np.digitize(t, bins)
            et = np.array([t[idx == i].mean() for i in range(1, len(bins))])
            ev = np.array([E[idx == i].max() for i in range(1, len(bins))])
            print(f"  Q = {Q:4d}, {'real GIC ' if gic else 'ideal GIC'}: max |V| before the kick {pre:.2e} V, overall {np.abs(V).max():.2e} V; "
                  f"growth rate after 50 us {0.5 * np.polyfit(et, np.log(ev), 1)[0]:+.3e} s^-1")

if __name__ == "__main__":
    prefetch()
    main()
