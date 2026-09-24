#!/usr/bin/env python3
"""
kappa_side_gpu.py -- why does kappa depend on box size for a localised beam?

SELF-CONTAINED. Paste into Colab and run; it runs itself. Uses PyTorch on the
GPU if present (float64), otherwise falls back to NumPy on the CPU (slow).

================================================================================
THE QUESTION
================================================================================
kappa is the amplitude coefficient of the propagation asymmetry,
|dw|/|dw_0| = 1 + kappa*A^2. For a PLANE WAVE it is -0.0187, confirmed four
independent ways, and box-independent. For a transversely LOCALISED beam
(Gaussian, width w = 2, on a periodic cubic lattice of side L), the measured
kappa falls about tenfold from L = 8 to L = 32 and does not settle:

    L       8         12        16        24        32
    kappa  -0.00447  -0.00289  -0.00204  -0.00115  -0.00047

(kappa_readout_test.py with Fourier-space seeding, T = 300, uniform readout.)
Already ruled out: seeding (the seed changes kappa's sign and scale but not
this shape), the readout (the uniform readout's phase fit is clean), and
proportionality to the fill fraction (predicted, and it failed).

================================================================================
WHAT THIS SCRIPT DECIDES
================================================================================
Is the box-size dependence there FROM THE START, or does it DEVELOP during the
run?

  STATIC  -- a fixed property of how a narrow packet sits on the lattice.
             kappa is the same early and late in the run, does not depend on
             run length, scales as A^2 (mild rise only, like the plane wave),
             and the measured wave neither gains nor loses amplitude.
  DYNAMIC -- something happens during the run, e.g. the beam's energy leaking
             into the sideways waves a bigger box has room for. kappa drifts
             between early and late, depends on run length, grows faster than
             A^2, and the measured wave gains or loses amplitude as energy
             moves between modes.

Note what it does NOT test: "the beam spreads, so kappa drops". The readout is
the average over the whole box, and spreading cannot change that average --
total energy is conserved, so the box-mean intensity is fixed however the beam
spreads. Spreading can only matter through nonlinear energy transfer, which is
the DYNAMIC case.

PREDICTION, stated before running: DYNAMIC. Reason: the P-1 investigation
found broad instability at this amplitude (A = 0.3) from a clean start, and a
localised packet seeds many sideways modes directly. This prediction can fail.

================================================================================
HOW IT WORKS
================================================================================
Four long runs, each read over several time windows:
  PW  : plane wave,        L = 8,  A = 0.02 / 0.30,               T = 600
  L8  : localised packet,  L = 8,  A = 0.02 / 0.30,               T = 600
  L16 : localised packet,  L = 16, A = 0.02 / 0.10 / 0.20 / 0.30, T = 1200
  L32 : localised packet,  L = 32, A = 0.02 / 0.30,               T = 600
Each amplitude is run in both directions. The [0, 300] window of each run is
exactly the earlier measurement, so it doubles as VALIDATION.

Physics, matching kappa_readout_test.py exactly:
  force = -(x^2 - x - 1) + c*lap(x) + beta*c*(v[n-1] - v[n+1])   along axis 0
  -> reference sign convention: +k has the UPPER root, -k the LOWER root.
  seed: Fourier space, every wavevector at its own branch frequency
        w(k) = beta*c*sin(k0) + sqrt(beta^2 c^2 sin^2 k0 + sqrt5 + 2c*sum_a(1-cos k_a))
  readout: transverse mean, FFT along axis 0, mode L/4 (k = pi/2),
           amplitude-weighted phase regression (identical to the original).
  kappa = (|dw(A)| / |2 c beta sin k| - 1) / A^2   (the original formula).
Integrator: fixed-step RK4, dt = 0.01, float64. Recorded every 0.1 time units,
matching the original's sampling.

================================================================================
VALIDATION (must pass, or nothing after it is trusted)
================================================================================
  V1  plane wave, [0,300]:  linear ratio 1.00000,  kappa -0.01869
  V2  localised L = 8:      kappa -0.00447
  V3  localised L = 16:     kappa -0.00204
  V4  localised L = 32:     kappa -0.00047
  V5  energy drift small in every run
If any fails, the script prints FAIL and stops before the tests.

TESTED LOCALLY (NumPy backend) before release: V1 plane wave -0.01869,
V2 L=8 -0.00447, V3 L=16 -0.00204 -- each reproduced to five decimals, energy
drift ~1e-9. The GPU backend uses the same algorithm; V1-V5 confirm it on
your machine before any test result is shown.

Runtime: minutes on a GPU; long on CPU.
"""

import math
import sys
import time

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

# ----------------------------------------------------------------- parameters
PHI = (1.0 + math.sqrt(5.0)) / 2.0
SQ5 = math.sqrt(5.0)
C = 1.0
K = math.pi / 2
BETA = 0.05
W_TRANS = 2.0
DT = 0.01
REC_EVERY = 10                  # record every 0.1 time units
A_LIN = 0.02
A_NL = 0.30
TH = abs(2 * C * BETA * math.sin(K))

REF = {"pw": -0.01869, 8: -0.00447, 16: -0.00204, 32: -0.00047}

if HAVE_TORCH:
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
else:
    DEV = None


# ------------------------------------------------------------------ backend
def to_backend(a):
    if HAVE_TORCH:
        return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    return a


def to_numpy(a):
    if HAVE_TORCH:
        return a.detach().cpu().numpy()
    return a


def roll(a, s, ax):
    if HAVE_TORCH:
        return torch.roll(a, shifts=s, dims=ax)
    return np.roll(a, s, axis=ax)


def mode_series(x, side):
    """Uniform readout: transverse mean, FFT along lattice axis 0, mode L/4.
    Array layout: (batch, axis0, axis1, axis2)."""
    m = side // 4
    if HAVE_TORCH:
        prof = x.mean(dim=(2, 3))
        return torch.fft.fft(prof, dim=1)[:, m] / side
    prof = x.mean(axis=(2, 3))
    return np.fft.fft(prof, axis=1)[:, m] / side


def force(x, v):
    lap = (roll(x, 1, 1) + roll(x, -1, 1) + roll(x, 1, 2) + roll(x, -1, 2)
           + roll(x, 1, 3) + roll(x, -1, 3) - 6.0 * x)
    gy = roll(v, 1, 1) - roll(v, -1, 1)           # v[n-1] - v[n+1], axis 0
    return -(x * x - x - 1.0) + C * lap + BETA * C * gy


def energy(x, v):
    """Conserved energy (the gyroscopic term does no work)."""
    u = x
    pot = u ** 3 / 3.0 - u ** 2 / 2.0 - u
    grad = 0.0
    for ax in (1, 2, 3):
        d = roll(x, -1, ax) - x
        grad = grad + 0.5 * C * d * d
    e = 0.5 * v * v + pot + grad
    if HAVE_TORCH:
        return to_numpy(e.sum(dim=(1, 2, 3)))
    return e.sum(axis=(1, 2, 3))


# ------------------------------------------------------------------ seeding
def seed(side, plane, amp, sign):
    """Fourier-space seed: every wavevector at its own branch frequency.
    Reference convention: +k upper root, -k lower root."""
    idx = np.indices((side,) * 3).astype(float)
    if plane:
        env = np.ones((side,) * 3)
    else:
        env = np.ones((side,) * 3)
        for a in (1, 2):
            d = idx[a] - side / 2.0
            d = (d + side / 2) % side - side / 2
            env = env * np.exp(-0.5 * (d / W_TRANS) ** 2)
    psi = amp * env * np.exp(1j * sign * K * idx[0])
    k = 2 * np.pi * np.fft.fftfreq(side)
    k0, k1, k2 = np.meshgrid(k, k, k, indexing="ij")
    b = BETA * C * np.sin(k0)
    om = b + np.sqrt(b * b + SQ5 + 2 * C * ((1 - np.cos(k0)) + (1 - np.cos(k1))
                                          + (1 - np.cos(k2))))
    vhat = -1j * om * np.fft.fftn(psi)
    u0 = psi.real
    v0 = np.fft.ifftn(vhat).real
    return PHI + u0, v0


# --------------------------------------------------------------- integration
def run(side, plane, amps, T, label):
    """Evolve all (amp, sign) cases as one batch. Returns (t, series, drift)
    where series[:, j] is the mode time series of case j, cases ordered
    (amp0,+), (amp0,-), (amp1,+), ..."""
    xs, vs = [], []
    for a in amps:
        for s in (+1, -1):
            x0, v0 = seed(side, plane, a, s)
            xs.append(x0)
            vs.append(v0)
    x = to_backend(np.stack(xs))
    v = to_backend(np.stack(vs))
    e0 = energy(x, v)
    nsteps = int(round(T / DT))
    rec_t = [0.0]
    rec = [to_numpy(mode_series(x, side))]
    t0 = time.time()
    h = DT
    for step in range(nsteps):
        k1v = force(x, v); k1x = v
        x2 = x + 0.5 * h * k1x; v2 = v + 0.5 * h * k1v
        k2v = force(x2, v2); k2x = v2
        x3 = x + 0.5 * h * k2x; v3 = v + 0.5 * h * k2v
        k3v = force(x3, v3); k3x = v3
        x4 = x + h * k3x; v4 = v + h * k3v
        k4v = force(x4, v4); k4x = v4
        x = x + (h / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + (h / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        if (step + 1) % REC_EVERY == 0:
            rec_t.append((step + 1) * h)
            rec.append(to_numpy(mode_series(x, side)))
    e1 = energy(x, v)
    drift = float(np.max(np.abs(e1 - e0) / np.abs(e0)))
    print(f"    {label}: side {side}, T = {T:g}, {len(amps)*2} cases, "
          f"{time.time()-t0:.0f} s, energy drift {drift:.1e}", flush=True)
    return np.array(rec_t), np.array(rec), drift


# ------------------------------------------------------------------ readout
def omega_of(series, t):
    """Amplitude-weighted phase regression -- identical to the original."""
    ph = np.unwrap(np.angle(series))
    w = np.abs(series)
    good = w > 0.05 * w.max()
    if good.sum() < 10:
        return float("nan"), float("nan")
    tt, pp, ww = t[good], ph[good], w[good]
    Am = np.vstack([tt, np.ones_like(tt)]).T
    Wm = np.diag(ww)
    sol, *_ = np.linalg.lstsq(Wm @ Am, Wm @ pp, rcond=None)
    resid = float(np.sqrt(np.mean((Am @ sol - pp) ** 2)))
    return abs(sol[0]), resid


def kappa_window(t, S, amps, amp, ta, tb):
    """kappa for amplitude `amp` over time window [ta, tb], plus the linear
    ratio and the worst phase residual. Uses the original formula."""
    sel = (t >= ta - 1e-9) & (t <= tb + 1e-9)
    tt = t[sel]

    def dw(a):
        j = amps.index(a)
        wp, rp = omega_of(S[sel, 2 * j], tt)
        wm, rm = omega_of(S[sel, 2 * j + 1], tt)
        return wp - wm, max(rp, rm)

    dl, rl = dw(amps[0])
    dn, rn = dw(amp)
    ratio = abs(dl) / TH
    kap = (abs(dn) / TH - 1.0) / amp ** 2
    return kap, ratio, max(rl, rn)


def retention(t, S, amps, amp, at):
    """|F(at)| / |F(0)| for amplitude `amp`: the direction that moved furthest
    from 1. Checks change in EITHER direction -- a local test found the
    measured wave GAINING amplitude, which a loss-only check would pass."""
    j = amps.index(amp)
    i = int(np.argmin(np.abs(t - at)))
    r = [abs(S[i, col]) / abs(S[0, col]) for col in (2 * j, 2 * j + 1)]
    return max(r, key=lambda x: abs(x - 1.0))


def close(val, ref, rel=0.04, absol=8e-5):
    return abs(val - ref) <= max(absol, rel * abs(ref))


# ---------------------------------------------------------------------- main
def main():
    print("=" * 78)
    print("KAPPA SIDE-DEPENDENCE -- static or dynamic?")
    print("=" * 78)
    if HAVE_TORCH:
        name = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"
        print(f"  backend: PyTorch float64 on {name}")
    else:
        print("  backend: NumPy on CPU (no PyTorch found -- this will be slow)")
    print("  prediction, stated before running: DYNAMIC\n")

    print("Runs:")
    amps2 = [A_LIN, A_NL]
    amps4 = [A_LIN, 0.10, 0.20, A_NL]
    tPW, SPW, dPW = run(8, True, amps2, 600.0, "PW ")
    t8, S8, d8 = run(8, False, amps2, 600.0, "L8 ")
    t16, S16, d16 = run(16, False, amps4, 1200.0, "L16")
    t32, S32, d32 = run(32, False, amps2, 600.0, "L32")
    print()

    # ------------------------------------------------------ validation
    print("=" * 78)
    print("VALIDATION -- the [0, 300] window must reproduce the known numbers")
    print("=" * 78)
    ok = True
    kpw, rpw, respw = kappa_window(tPW, SPW, amps2, A_NL, 0, 300)
    v1 = close(kpw, REF["pw"], rel=0.02) and abs(rpw - 1.0) < 2e-5
    print(f"  V1 plane wave   kappa {kpw:+.5f} (ref {REF['pw']:+.5f})  "
          f"linear ratio {rpw:.6f}  resid {respw:.3f}   {'PASS' if v1 else 'FAIL'}")
    ok &= v1
    runs = {8: (t8, S8, amps2), 16: (t16, S16, amps4), 32: (t32, S32, amps2)}
    for i, side in enumerate((8, 16, 32), start=2):
        tt, SS, aa = runs[side]
        kv, rv, res = kappa_window(tt, SS, aa, A_NL, 0, 300)
        vi = close(kv, REF[side]) and abs(rv - 1.0) < 2e-5
        print(f"  V{i} localised L={side:<2d} kappa {kv:+.5f} (ref {REF[side]:+.5f})  "
              f"linear ratio {rv:.6f}  resid {res:.3f}   {'PASS' if vi else 'FAIL'}")
        ok &= vi
    worst = max(dPW, d8, d16, d32)
    v5 = worst < 1e-6
    print(f"  V5 energy drift  worst {worst:.1e}   {'PASS' if v5 else 'FAIL'}")
    ok &= v5
    if not ok:
        print("\n  VALIDATION FAILED -- the tests below would not be trustworthy.")
        print("  Stopping. Paste this output back so the discrepancy can be found.")
        return
    print("  all validation checks pass -- the tests below can be trusted\n")

    # ------------------------------------------------------ test 1
    print("=" * 78)
    print("TEST 1 -- run length (L = 16): kappa read over [0, T]")
    print("=" * 78)
    t1 = []
    for T in (150, 300, 600, 1200):
        kv, rv, res = kappa_window(t16, S16, amps4, A_NL, 0, T)
        t1.append(kv)
        print(f"   T = {T:5d}   kappa {kv:+.5f}   resid {res:.3f}")
    spread1 = (max(t1) - min(t1)) / abs(np.mean(t1))
    print(f"   spread across run length: {spread1*100:.0f}%\n")

    # ------------------------------------------------------ test 2
    print("=" * 78)
    print("TEST 2 -- amplitude (L = 16, [0, 300]): a true A^2 coefficient is flat")
    print("=" * 78)
    t2 = []
    for a in (0.10, 0.20, A_NL):
        kv, rv, res = kappa_window(t16, S16, amps4, a, 0, 300)
        t2.append(kv)
        print(f"   A = {a:.2f}   kappa {kv:+.5f}   resid {res:.3f}")
    spread2 = (max(t2) - min(t2)) / abs(np.mean(t2))
    print(f"   spread across amplitude: {spread2*100:.0f}%   "
          f"(the plane wave rises ~14% from A = 0.1 to 0.4)\n")

    # ------------------------------------------------------ test 3
    print("=" * 78)
    print("TEST 3 -- early vs late: kappa over [0, 300] vs [300, 600]")
    print("=" * 78)
    rows = [("plane wave (control)", tPW, SPW, amps2)] + \
           [(f"localised L = {s}", *runs[s]) for s in (8, 16, 32)]
    halves = {}
    for lbl, tt, SS, aa in rows:
        ke, _, re_ = kappa_window(tt, SS, aa, A_NL, 0, 300)
        kl, _, rl_ = kappa_window(tt, SS, aa, A_NL, 300, 600)
        ch = (kl - ke) / abs(ke)
        halves[lbl] = (ke, kl, ch)
        print(f"   {lbl:22s} early {ke:+.5f}   late {kl:+.5f}   "
              f"change {ch*100:+5.0f}%   resid {max(re_, rl_):.3f}")
    print()

    # ------------------------------------------------------ retention
    print("=" * 78)
    print("AMPLITUDE -- does the measured wave keep its size? (A = 0.30)")
    print("  (a change in EITHER direction means energy is moving between modes)")
    print("=" * 78)
    rets = {}
    for lbl, tt, SS, aa in rows:
        tend = tt[-1]
        r300 = retention(tt, SS, aa, A_NL, 300)
        rend = retention(tt, SS, aa, A_NL, tend)
        rets[lbl] = max(abs(r300 - 1.0), abs(rend - 1.0))
        print(f"   {lbl:22s} at T = 300: {r300:.4f}   at T = {tend:.0f}: {rend:.4f}")
    print()

    # ------------------------------------------------------ verdict
    res = {
        "pw_change": halves["plane wave (control)"][2],
        "k150": t1[0], "k300": t1[1],
        "spread_T": spread1, "spread_A": spread2,
        "late_changes": {k: halves[k][:2] for k in halves if k.startswith("localised")},
        "amp_dev": max(rets[k] for k in rets if k.startswith("localised")),
    }
    verdict(res)
    print("\n  Paste the whole output back for the record.")


def verdict(res):
    """Report the ORIGIN of the box-size dependence and any SECONDARY drift
    separately. An earlier version merged them: it called the result DYNAMIC
    whenever anything dynamic happened, without asking whether the dynamics
    CAUSED the box-size dependence. On the first real run that gave the wrong
    answer -- the dependence was present from t = 0 while a slow, separate
    energy transfer was also under way."""
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    if abs(res["pw_change"]) > 0.05:
        print("  UNRELIABLE: the plane-wave control changes by "
              f"{abs(res['pw_change'])*100:.0f}% between halves; the early/late")
        print("  method fails its own control.")
        return
    same_start = abs(res["k150"] - res["k300"]) < max(8e-5, 0.05 * abs(res["k300"]))
    flat_A = res["spread_A"] < 0.10
    static_origin = same_start and flat_A
    if static_origin:
        print("  ORIGIN: STATIC. The box-size dependence is present from the start")
        print(f"  (kappa over [0,150] = {res['k150']:+.5f}, over [0,300] = "
              f"{res['k300']:+.5f}) and is a clean A^2 coefficient")
        print(f"  (flat in amplitude to {res['spread_A']*100:.0f}%). It is a fixed property")
        print("  of how the packet sits on the lattice, not something that develops.")
        print("  The prediction (DYNAMIC origin) FAILED.")
    else:
        print("  ORIGIN: DYNAMIC. kappa is not steady from the start and/or is not")
        print("  a clean A^2 coefficient: the dependence develops during the run.")
        print("  The prediction (DYNAMIC origin) HELD.")
    def small(ke, kl):
        return abs(kl - ke) < max(8e-5, 0.10 * abs(ke))
    drifting = [k for k, (ke, kl) in res["late_changes"].items() if not small(ke, kl)]
    print()
    if drifting or res["amp_dev"] > 0.01 or res["spread_T"] > 0.05:
        print("  SECONDARY DYNAMICS: PRESENT. Separately from the origin, the measured")
        print(f"  wave's amplitude changes by up to {res['amp_dev']*100:.1f}% during the run")
        print(f"  (energy moving between modes), and kappa drifts by {res['spread_T']*100:.0f}%")
        if static_origin:
            print("  over long runs. A slow effect riding on top -- not the cause.")
        else:
            print("  over long runs.")
    else:
        print("  SECONDARY DYNAMICS: none detected.")


if __name__ == "__main__":
    main()
