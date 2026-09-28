#!/usr/bin/env python3
"""
irrev_q3_colab.py -- SELF-CONTAINED (no repository imports) q = 3 distance law for target 5, for a Colab GPU.
Predictions: natural/IRREV_HYPOTHESES.md (1c9e5bf), I4 at q = 3: the tower's depletion around an irreversibly absorbing lump
is a ballistic shadow ~ 1/r^2 (exponent in [-2.6, -1.4]); inverse-square force would need a deficit ~ 1/r (exponent -1).
The side-32 container pilot (irrev_q3_output.txt) was too small for a clean law; this run is side 96.

Model (identical to natural/irrev_q3.py, re-implemented here):
  main's force, node form A':  f = -(sqrt5 + |u|) u + c lap(u) + kappa JJ v,   c = 1, kappa = kappa* = 2/sqrt(sqrt5 + 2),
  u in R^(2n) per site, n = 6 complex components: 0-3 = D <= 8 part (lump in comp 0), 4-5 = tower (main's incoherent
  population, rms 0.005 per component, both chirality branches, random phases). RK4, dt = 0.02.
  GENERIC dissipation, deterministic limit: after each step, each tower component's radial velocity w is damped,
  w -> w exp(-gamma dt), gamma = Gamma e_low (D <= 8 energy density), the kinetic energy removed booked into eps;
  heat conduction eps += dt K lap(eps / C). Gamma = 10, C = 1000, K = 1. Reference run: Gamma = 0 (same seed = CRN).
Measurement: the tower energy density averaged over t in [60, 99] (the cone v_max t reaches 48 = half-box at t = 99);
  deficit d(r) = <e_G> - <e_ref>, radially averaged; local slopes and a fit over r = 4..24.
Usage (Colab): !pip install torch (preinstalled on Colab); python irrev_q3_colab.py [--side 96] [--T 99] [--seeds 2]
Runtime estimate: ~1-2 h on a T4 for two seeds x (G, ref) at side 96; set --side 64 --T 66 for a ~20 min check.
"""
import argparse
import math
import time

import numpy as np

try:
    import torch
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    XP = "torch"
except ImportError:  # NumPy fallback
    torch = None
    DEV = "cpu"
    XP = "numpy"

SQ5 = math.sqrt(5.0)
C_EL = 1.0
KAPPA = 2.0 / math.sqrt(SQ5 + 2.0)
DT = 0.02
NC = 6
A_U = 0.005


def branch_omega(side):
    k = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(side)] * 3, indexing="ij")
    Q = SQ5 + 2 * C_EL * sum(1 - np.cos(x) for x in k)
    return 0.5 * (-KAPPA + np.sqrt(KAPPA ** 2 + 4 * Q))


def initial_state(side, seed):
    """main's incoherent_upper (tower comps 4..NC-1) plus an a-branch k0 = 0 lump (width 3, amp 0.05) in comp 0."""
    rng = np.random.default_rng(1000 + seed)
    wa = branch_omega(side); wb = wa + KAPPA
    sh = (side,) * 3
    u = np.zeros(sh + (2 * NC,)); v = np.zeros(sh + (2 * NC,))
    for c in range(4, NC):
        a = rng.normal(size=sh) + 1j * rng.normal(size=sh)
        b = rng.normal(size=sh) + 1j * rng.normal(size=sh)
        psi = np.fft.ifftn(a + b); dps = np.fft.ifftn(-1j * wa * a + 1j * wb * b)
        s = A_U / np.sqrt(np.mean(np.abs(psi) ** 2))
        u[..., 2 * c], u[..., 2 * c + 1] = (s * psi).real, (s * psi).imag
        v[..., 2 * c], v[..., 2 * c + 1] = (s * dps).real, (s * dps).imag
    c3 = np.indices(sh).astype(float)
    r2 = sum(((c3[a] - side // 2 + side / 2) % side - side / 2) ** 2 for a in range(3))
    psi = 0.05 * np.exp(-0.5 * r2 / 9.0) + 0j
    dps = np.fft.ifftn(-1j * wa * np.fft.fftn(psi))
    u[..., 0] += psi.real; u[..., 1] += psi.imag; v[..., 0] += dps.real; v[..., 1] += dps.imag
    return u, v


class Ops:
    def __init__(self):
        if XP == "torch":
            self.t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=DEV)
            self.roll = lambda a, s, ax: torch.roll(a, s, ax)
            self.norm = lambda a: torch.linalg.norm(a, dim=-1)
            self.exp = torch.exp
            self.np = lambda a: a.detach().cpu().numpy()
        else:
            self.t = np.asarray
            self.roll = np.roll
            self.norm = lambda a: np.linalg.norm(a, axis=-1)
            self.exp = np.exp
            self.np = lambda a: a


O = Ops()


def lap(y):
    return sum(O.roll(y, 1, ax) + O.roll(y, -1, ax) - 2 * y for ax in range(3))


def jj(v):
    out = v * 0
    out[..., 0::2] = -v[..., 1::2]
    out[..., 1::2] = v[..., 0::2]
    return out


def force(u, v):
    r = O.norm(u)[..., None]
    return -(SQ5 + r) * u + C_EL * lap(u) + KAPPA * jj(v)


def rk4(u, v):
    k1v, k1u = force(u, v), v
    k2v, k2u = force(u + .5 * DT * k1u, v + .5 * DT * k1v), v + .5 * DT * k1v
    k3v, k3u = force(u + .5 * DT * k2u, v + .5 * DT * k2v), v + .5 * DT * k2v
    k4v, k4u = force(u + DT * k3u, v + DT * k3v), v + DT * k3v
    return u + DT / 6 * (k1u + 2 * k2u + 2 * k3u + k4u), v + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)


def edens(u, v, lo, hi):
    uu, vv = u[..., lo:hi], v[..., lo:hi]
    g = 0
    for ax in range(3):
        b = ((O.roll(uu, -1, ax) - uu) ** 2).sum(-1)
        g = g + 0.5 * (b + O.roll(b, 1, ax))
    r = O.norm(uu)
    return 0.5 * (vv * vv).sum(-1) + 0.5 * SQ5 * (uu * uu).sum(-1) + r ** 3 / 3 + 0.5 * C_EL * g


def energy(u, v):
    return float(O.np(edens(u, v, 0, 2 * NC).sum()))


def run(side, seed, Gamma, T, w0, Cth=1000.0, K=1.0):
    u, v = (O.t(a) for a in initial_state(side, seed))
    eps = O.t(np.full((side,) * 3, Cth * 1e-5))
    E0 = energy(u, v) + float(O.np(eps.sum()))
    acc = O.t(np.zeros((side,) * 3)); n = 0
    steps = int(round(1.0 / DT))
    t0 = time.time()
    for k in range(int(T)):
        for _ in range(steps):
            u, v = rk4(u, v)
            if Gamma:
                gam = Gamma * edens(u, v, 0, 8)
                a = O.exp(-gam * DT)
                for c in range(4, NC):
                    uc, vc = u[..., 2 * c:2 * c + 2], v[..., 2 * c:2 * c + 2]
                    rr = O.norm(uc)[..., None]
                    rh = uc / (rr + 1e-300)
                    w = (vc * rh).sum(-1)
                    w2 = a * w
                    v[..., 2 * c:2 * c + 2] = vc + ((w2 - w)[..., None]) * rh
                    eps = eps + 0.5 * (w * w - w2 * w2)
                eps = eps + DT * K * lap(eps / Cth)
        if k + 1 >= w0:
            acc = acc + edens(u, v, 8, 2 * NC); n += 1
        if (k + 1) % 10 == 0:
            print(f"      seed {seed} Gamma {Gamma}: t = {k+1}, {time.time()-t0:.0f} s", flush=True)
    E1 = energy(u, v) + float(O.np(eps.sum()))
    return O.np(acc / max(n, 1)), abs(E1 - E0) / abs(E0 - float(side ** 3 * Cth * 1e-5))


def gate():
    """Validation gates first: energy conservation of the Hamiltonian part and of the full model on a small box."""
    side = 12
    u, v = (O.t(a) for a in initial_state(side, 0))
    E0 = energy(u, v)
    for _ in range(250):
        u, v = rk4(u, v)
    d = abs(energy(u, v) - E0) / abs(E0)
    print(f"GATE 1 Hamiltonian energy drift over t = 5 (side 12): {d:.1e}  (must be < 1e-6)")
    _, d2 = run(side, 0, 10.0, 5, 99)
    print(f"GATE 2 full-model (mechanics + heat) drift over t = 5: {d2:.1e} of the mechanical energy (must be < 1e-5)")
    return d < 1e-6 and d2 < 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=96)
    ap.add_argument("--T", type=float, default=99.0)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    side, T = a.side, a.T
    w0 = int(round(T * 60 / 99))
    print(f"backend {XP} on {DEV}; side {side}, T {T}, averaging window [{w0}, {T:.0f}], seeds {a.seeds}")
    if not gate():
        print("GATES FAILED -- stopping"); return
    defs = []
    for s in range(a.seeds):
        g, dg = run(side, s, 10.0, T, w0)
        r0, dr = run(side, s, 0.0, T, w0)
        print(f"   seed {s}: energy drifts {dg:.1e} (G), {dr:.1e} (ref)")
        defs.append(g - r0)
    dm = np.mean(defs, axis=0)
    c = np.indices((side,) * 3)
    d = np.minimum(np.abs(c - side // 2), side - np.abs(c - side // 2))
    rad = np.sqrt((d ** 2).sum(0))
    rs = np.arange(1, side // 2 + 1)
    prof = np.array([dm[(rad >= r - 0.5) & (rad < r + 0.5)].mean() for r in rs])
    print("PREDICTION (1c9e5bf): deficit ~ r^p with p in [-2.6, -1.4] (ballistic shadow); inverse-square force needs p = -1")
    print("   r    deficit        local slope")
    for i in range(len(rs)):
        sl = (math.log(abs(prof[i + 1]) / abs(prof[i])) / math.log(rs[i + 1] / rs[i])
              if i + 1 < len(rs) and prof[i] < 0 and prof[i + 1] < 0 else float("nan"))
        print(f"   {rs[i]:3d}  {prof[i]:+.4e}   {sl:+.3f}")
    sel = (rs >= 4) & (rs <= min(24, side // 4)) & (prof < 0)
    p = np.polyfit(np.log(rs[sel]), np.log(-prof[sel]), 1)[0]
    print(f"FIT r = 4..{min(24, side//4)}: exponent {p:.3f}")


if __name__ == "__main__":
    main()
