# circuit_analysis.py -- self-contained analysis for the 32-node LC-gyrator ring (CIRCUIT_BUILD.md).
# Paste into one Colab cell and run. Needs only numpy. (The demonstration takes a few minutes.)
#
# INPUT: frequency-response files, one per configuration, each with a header line and then rows
#   f_Hz  Re V0  Im V0  Re V1  Im V1 ... Re V31  Im V31
# (the node voltages for a 1 A-equivalent current drive into node 0; ngspice's `wrdata` with wr_singlescale
# writes exactly this). Several files for the same configuration (repeat sweeps) give error bars from their
# scatter; with one file the error bar comes from bootstrapping the fit residuals.
#   band*.txt            gyrators unpowered, 48-75 kHz
#   p1*.txt              gyrators powered, 48-75 kHz
#   m8_S0*.txt           gyrators powered, no bump, 54-70 kHz
#   m8_<shape>_p*.txt    bump +S ; m8_<shape>_m*.txt  bump -S     shapes: gauss4 gauss3 gauss5 sech24 two2.5 rampNone
#   dK_<shape>_p.txt, dK_<shape>_m.txt   the 32 measured dK_n (s^-2) of that bump at +S and at -S (single-node
#                        spectroscopy); <dK^2> is averaged over both. If absent, the design dK_n = K S eta_n is used and
#                        flagged (SPICE: the GIC realises the bump 1.5-3.1% larger than designed, so measure it).
# Leave DATA_DIR = None to run on synthetic data from a built-in model of the circuit (a demonstration only).
#
# OUTPUT: calibrations and preconditions, then P1, P2, P3 with error bars against the pre-registered
# predictions of CIRCUIT_PREREGISTRATION.md (committed before any measurement).
import glob
import os

import numpy as np

DATA_DIR = None          # e.g. "/content/data"
N, S = 32, 0.05
RNG = np.random.default_rng(0)

# ---- pre-registered predictions (CIRCUIT_PREREGISTRATION.md) -------------------------------------------------
PRED = dict(beta_hat=0.106, c_hat=0.250, K=1e11,
            P1_tol=5e-3, P2_R=1.031, P2_tol=0.015,
            P3={"gauss4": 1.035, "gauss3": 1.058, "gauss5": 1.026, "sech24": 1.058, "two2.5": 1.077, "rampNone": 0.894},
            P3_tol=0.02, P3_ramp_tol=0.03)


def eta_shape(kind, p=None):
    x = np.arange(N, dtype=float)
    if kind == "gauss":
        e = np.exp(-0.5 * ((x - N / 2) / p) ** 2)
    elif kind == "sech2":
        e = 1.0 / np.cosh((x - N / 2) / p) ** 2
    elif kind == "two":
        e = np.exp(-0.5 * ((x - N / 4) / p) ** 2) + np.exp(-0.5 * ((x - 3 * N / 4) / p) ** 2)
    else:
        e = x.copy()
    e = e - e.mean()
    return e / np.sqrt((e ** 2).mean())


SHAPES = {"gauss4": ("gauss", 4), "gauss3": ("gauss", 3), "gauss5": ("gauss", 5), "sech24": ("sech2", 4),
          "two2.5": ("two", 2.5), "rampNone": ("ramp", None)}


def design_f(m, sign, K=1e11, c=0.25e11, b=0.106 * np.sqrt(1e11)):
    """Design line frequency (Hz); sign = +1/-1 for the +k/-k branch, 0 for the unsplit band (gyrators off)."""
    k = 2 * np.pi * m / N
    Qk = K + 2 * c * (1 - np.cos(k))
    bb = b if sign else 0.0
    return (sign * bb * np.sin(k) + np.sqrt((bb * np.sin(k)) ** 2 + Qk)) / (2 * np.pi)


# ---- estimator (fixed in the pre-registration: a rational fit around each line, pole real part; implemented as a
# common-pole fit of all 32 node responses, the line identified by its residue vector) -------------------------
def vf(f, V, f0, hw=2000.0, npol=16, it=8):
    """Common-pole rational fit (vector fitting with Sanathanan-Koerner weights) of all node responses over
    f0 +- hw. Returns poles (Hz, complex), residue vectors (one row per pole), the fitted responses and the mask."""
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
    return f0 + hw * poles, R[:-1], Bm @ R, msk


def pick(poles, R, m, sign, f0, hw=2000.0):
    """The pole whose residue vector overlaps most with the plane wave of the +k (sign = +1) or -k mode. Phasor
    convention e^{+jwt}: the +k mode e^{i(kn - wt)} has shape ~ e^{-ikn}. For the band (sign 0) either works."""
    s = sign if sign else 1
    pw = np.exp(1j * s * 2 * np.pi * m * np.arange(N) / N)
    ok = (np.abs(poles.real - f0) < hw) & (np.abs(poles.imag) < hw)
    ov = np.abs(R @ pw) / (np.linalg.norm(R, axis=1) * np.sqrt(N) + 1e-300)
    ov[~ok] = -1
    return poles[int(np.argmax(ov))]


def line(f, V, m, sign, gyro=True, nboot=10):
    """Line frequency (Hz) with a noise-injection standard error. With >= 2 repeat files, lines_all also uses
    their scatter (preferred: take >= 3 repeat sweeps)."""
    f0 = design_f(m, sign if gyro else 0)
    po, R, fit, msk = vf(f, V, f0)
    fr = pick(po, R, m, sign, f0).real
    # noise-injection bootstrap: the relative noise level from fourth differences along frequency (model-free;
    # removes the smooth resonance shape), injected into the data and the line refitted. A residual or wild
    # bootstrap overestimated the error ~50x on synthetic data with known scatter: the fit residuals are not noise.
    d4 = V[4:] - 4 * V[3:-1] + 6 * V[2:-2] - 4 * V[1:-3] + V[:-4]
    sig = np.median(np.abs(d4) / np.abs(V[2:-2])) / (np.sqrt(70) * np.sqrt(2 * np.log(2)))
    boots = []
    for _ in range(nboot):
        Vb = V * (1 + sig * (RNG.standard_normal(V.shape) + 1j * RNG.standard_normal(V.shape)))
        pb, Rb, _, _ = vf(f, Vb, f0)
        boots.append(pick(pb, Rb, m, sign, f0).real)
    return fr, np.std(boots)


# ---- data ---------------------------------------------------------------------------------------------------
def load(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    out = []
    for fn in files:
        d = np.loadtxt(fn, skiprows=1)
        out.append((d[:, 0], d[:, 1::2] + 1j * d[:, 2::2]))
    return out


def synth(config, noise=1e-3):
    """Synthetic response of the design circuit (ideal gyrators with a 9.25 MHz pole, Q = 100, C_p = 10 pF,
    0.1% component scatter, a random 0.1% relative noise on every voltage). A demonstration only."""
    C, Lg, Lc, G = 10e-9, 1e-3, 4e-3, 0.106 * np.sqrt(10e-9 / 1e-3)
    w2 = np.sqrt(1.5e11)
    rng = np.random.default_rng(42)
    Cn, Lgn, Lcn = (v * (1 + 1e-3 * rng.standard_normal(N)) for v in (C, Lg, Lc))
    gyro = not config.startswith("band")
    f = np.arange(48e3, 75e3 + 1, 25.0) if config in ("band", "p1") else np.arange(54e3, 69.8e3 + 1, 25.0)
    invL = 1 / Lgn
    if config.startswith("m8_") and config != "m8_S0":
        name, sg = config[3:-2], (1 if config.endswith("_p") else -1)
        invL = invL * (1 + sg * S * eta_shape(*SHAPES[name]))
    V = np.zeros((len(f), N), complex)
    for i, fi in enumerate(f):
        w = 2 * np.pi * fi
        Gw = G / (1 + 1j * w / (2 * np.pi * 9.25e6))
        Y = np.diag(1j * w * Cn + 1 / (w2 * Lgn / 100 + 1j * w / invL))
        for n in range(N):
            m = (n + 1) % N
            yc = 1 / (w2 * Lcn[n] / 100 + 1j * w * Lcn[n]) + 1j * w * 10e-12
            Y[n, n] += yc; Y[m, m] += yc; Y[n, m] -= yc; Y[m, n] -= yc
            if gyro:
                Y[n, m] += Gw; Y[m, n] -= Gw
        I = np.zeros(N, complex); I[0] = 1
        V[i] = np.linalg.solve(Y, I)
    V *= 1 + noise * (rng.standard_normal(V.shape) + 1j * rng.standard_normal(V.shape))
    return [(f, V)]


def get(config):
    if DATA_DIR is None:
        return synth(config)
    d = load(config + "*.txt")
    if not d:
        raise FileNotFoundError(f"no files for {config} in {DATA_DIR}")
    return d


def lines_all(data, ms, sign, gyro=True):
    """Mean over repeat files; error = scatter of repeats (if >= 2 files) combined with the bootstrap error."""
    out = {}
    for m in ms:
        vals = [line(f, V, m, sign, gyro) for f, V in data]
        fr = np.mean([v[0] for v in vals])
        err = np.sqrt(np.mean([v[1] ** 2 for v in vals]) / len(vals)
                      + (np.var([v[0] for v in vals], ddof=1) / len(vals) if len(vals) > 1 else 0.0))
        out[m] = (fr, err)
    return out


# ---- analysis -----------------------------------------------------------------------------------------------
def main():
    print("CIRCUIT ANALYSIS --", "SYNTHETIC DEMONSTRATION DATA" if DATA_DIR is None else f"data from {DATA_DIR}")
    ms = range(0, 17)
    band = get("band")
    bp, bm = lines_all(band, ms, 1, gyro=False), lines_all(band, ms, -1, gyro=False)
    wb = np.array([np.pi * (bp[m][0] + bm[m][0]) for m in ms])
    zero = max(abs(bp[m][0] - bm[m][0]) for m in range(1, 16))
    X = np.vstack([np.ones(17), 2 * (1 - np.cos(2 * np.pi * np.array(ms) / N))]).T
    (K, c), *_ = np.linalg.lstsq(X, wb ** 2, rcond=None)
    fitres = np.abs(np.sqrt(X @ [K, c]) / wb - 1).max()
    p1 = get("p1")
    Lp, Lm = lines_all(p1, range(1, 16), 1), lines_all(p1, range(1, 16), -1)
    dw = {m: 2 * np.pi * (Lp[m][0] - Lm[m][0]) for m in range(1, 16)}
    dwe = {m: 2 * np.pi * np.hypot(Lp[m][1], Lm[m][1]) for m in range(1, 16)}
    b = dw[8] / 2
    prod = max(abs(4 * np.pi ** 2 * Lp[m][0] * Lm[m][0] / (K + 2 * c * (1 - np.cos(2 * np.pi * m / N))) - 1) for m in range(1, 16))
    print("\nCALIBRATIONS AND PRECONDITIONS")
    print(f"  1 zero check: max |f+ - f-| gyrators off = {zero / (dw[8] / 2 / np.pi):.1e} of Delta f(pi/2)   "
          f"[{'ok' if zero / (dw[8] / 2 / np.pi) < 1e-3 else 'FAIL - run void'}]")
    print(f"  2 band: K = {K:.5e} s^-2, c-hat = {c / K:.4f}, max fit residual {fitres:.1e}   "
          f"[{'ok' if abs(c / K - 0.25) <= 0.002 and fitres <= 2e-3 else 'FAIL - run void'}]")
    print(f"  3 beta-hat = {b / np.sqrt(K):.4f}   [{'ok' if abs(b / np.sqrt(K) - 0.106) <= 0.001 else 'FAIL - run void'}]")
    print(f"  4 product rule: max |w+ w- / Q(k) - 1| = {prod:.1e}   [{'ok' if prod <= 1e-3 else 'FAIL - run void'}]")
    print("\nP1 -- Delta w(m)/Delta w(8) - sin k   (failure: |residual| > 5e-3 + 3 sigma)")
    fails = 0
    for m in range(1, 16):
        r = dw[m] / dw[8] - np.sin(2 * np.pi * m / N)
        sig = abs(dw[m] / dw[8]) * np.hypot(dwe[m] / abs(dw[m]), dwe[8] / dw[8])
        bad = abs(r) > PRED["P1_tol"] + 3 * sig
        fails += bad
        print(f"  m = {m:2d}: Delta f = {dw[m] / (2 * np.pi):9.2f} Hz; residual {r:+.2e} +- {sig:.1e}  {'FAIL' if bad else 'pass'}")
    print(f"  P1: {'FAILS' if fails else 'passes'} ({fails} wavenumber(s) outside)")
    print("\nP2, P3 -- R = d(Delta w) / (-(1/4) b <dK^2> / c^2) at m = 8, +-S even part")
    base = get("m8_S0")
    b0p, b0m = lines_all(base, [8], 1)[8], lines_all(base, [8], -1)[8]
    d0, e0 = 2 * np.pi * (b0p[0] - b0m[0]), 2 * np.pi * np.hypot(b0p[1], b0m[1])
    for name in ["gauss4", "gauss3", "gauss5", "sech24", "two2.5", "rampNone"]:
        vals = []
        for sg in ("p", "m"):
            dd = get(f"m8_{name}_{sg}")
            lp, lm = lines_all(dd, [8], 1)[8], lines_all(dd, [8], -1)[8]
            vals.append((2 * np.pi * (lp[0] - lm[0]), 2 * np.pi * np.hypot(lp[1], lm[1])))
        shift = 0.5 * (vals[0][0] + vals[1][0]) - d0
        se = np.sqrt(0.25 * vals[0][1] ** 2 + 0.25 * vals[1][1] ** 2 + e0 ** 2)
        fp = None if DATA_DIR is None else os.path.join(DATA_DIR, f"dK_{name}_p.txt")
        fm = None if DATA_DIR is None else os.path.join(DATA_DIR, f"dK_{name}_m.txt")
        if fp and os.path.exists(fp) and os.path.exists(fm):
            # the realised bump differs between +S and -S (the GIC is slightly nonlinear): average <dK^2> over both
            dK2, flag = 0.5 * (np.mean(np.loadtxt(fp) ** 2) + np.mean(np.loadtxt(fm) ** 2)), ""
        else:
            dK2, flag = np.mean((K * S * eta_shape(*SHAPES[name])) ** 2), "  (design dK used -- measure it)"
        pred = -0.25 * b * dK2 / c ** 2
        R, sR = shift / pred, se / abs(pred)
        if name == "gauss4":
            bad = abs(R - PRED["P2_R"]) > PRED["P2_tol"] + 3 * sR
            print(f"  P2 (gauss4): d(Delta f) = {shift / (2 * np.pi):+.2f} Hz; R = {R:.4f} +- {sR:.4f}; predicted "
                  f"{PRED['P2_R']} (window +-{PRED['P2_tol']} + 3 sigma)  {'FAILS' if bad else 'passes'}{flag}")
        else:
            tol = PRED["P3_ramp_tol"] if name == "rampNone" else PRED["P3_tol"]
            bad = abs(R - PRED["P3"][name]) > tol + 3 * sR
            print(f"  P3 ({name}): R = {R:.4f} +- {sR:.4f}; predicted {PRED['P3'][name]} (window +-{tol} + 3 sigma)  "
                  f"{'FAILS' if bad else 'passes'}{flag}")


main()
