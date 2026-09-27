#!/usr/bin/env python3
"""
joint5_rate.py -- a FIXED localised stiffness bump in a large lattice: golden-rule scattering rate vs
frequency shift (predictions: joint5_rate_predictions.txt, committed first, 61dbc5d).

K(x) = s (1 + S eta), eta = exp(-r^2/2 sigma^2), sigma = 2 fixed, peak 1; s = sqrt5, c = 1, beta = 0.05;
probe (k0, 0, ..), k0 = pi/2, +k upper root, -k lower root. <p|V|k> = s S Phi(p-k)/V with the exact lattice
transform Phi(q) = prod_a phi1(q_a), phi1(q) = sum_n exp(-n^2/2 sigma^2) e^{-iqn}. From the kernel of
joint5_kernel.py with d_p -> d_p(w + i0):
  rate   V Gamma / S^2 = (pi s^2 / (V |d'_k|)) sum_p |Phi(p-k)|^2 delta(d_p(w_k))
         -> continuum (pi s^2 / ((2 pi)^q |d'_k|)) int d^q p |Phi|^2 delta(d_p)   [roots in p_x per p_perp]
  shift  V dw2 / S^2 = (s^2 / V) sum_{p != k} |Phi(p-k)|^2 / (d_p d'_k)            [second order]
         first order  V dw1 / S = -s Phi(0) / d'_k
Finite-L sums use a Lorentzian delta of width eps. usage: python3 joint5_rate.py
"""
import numpy as np

S0, C, B, K0, SIG = np.sqrt(5.0), 1.0, 0.05, np.pi / 2, 2.0
NMAX = int(10 * SIG)


def phi1(q):
    n = np.arange(-NMAX, NMAX + 1)
    return (np.exp(-0.5 * n[None, :] ** 2 / SIG ** 2) * np.exp(-1j * np.outer(np.atleast_1d(q), n))).sum(1)


def probe(sign):
    w = sign * B * C * np.sin(K0) + np.sqrt(B ** 2 * C ** 2 * np.sin(K0) ** 2 + S0 + 2 * C * (1 - np.cos(K0)))
    return w, 2 * B * C * np.sin(sign * K0) - 2 * w


def continuum_rate(q, sign, npt=4001):
    w, dpr = probe(sign)
    kx = sign * K0
    A = 2 * C * np.hypot(1.0, B * w)
    psi = np.arctan2(B * w, 1.0)          # -2c cos p + 2 b c w sin p = -A cos(p + psi)
    t = np.linspace(-np.pi, np.pi, npt, endpoint=False) + np.pi / npt
    if q == 1:
        grids = [np.zeros(1)]
        wt = 1.0
    elif q == 2:
        grids = [t]; wt = 2 * np.pi / npt
    else:
        ty, tz = np.meshgrid(t[::4], t[::4], indexing="ij"); grids = [ty.ravel(), tz.ravel()]; wt = (2 * np.pi / (npt / 4)) ** 2
    trans = sum(2 * C * (1 - np.cos(g)) for g in grids)
    R = w * w - S0 - trans
    x = (2 * C - R) / A
    ok = np.abs(x) < 1
    tot = 0.0
    phit = np.ones_like(trans, dtype=float)
    if q > 1:                              # [fixed: at q = 1 there is no transverse factor]
        for g in grids:
            phit = phit * np.abs(phi1(g)) ** 2
    for root_sign in (+1, -1):
        p = root_sign * np.arccos(np.clip(x, -1, 1)) - psi
        jac = np.abs(A * np.sin(p + psi))
        if q == 1:
            # [fixed: the forward root p = k is the probe itself, not a scattered state]
            ok1 = ok & (np.abs(((p - kx + np.pi) % (2 * np.pi)) - np.pi) > 1e-6)
        else:
            ok1 = ok
        val = np.where(ok1, np.abs(phi1(p - kx)) ** 2 * phit / np.where(ok1, jac, 1), 0.0)
        tot += val.sum() * wt
    return np.pi * S0 ** 2 / ((2 * np.pi) ** q * abs(dpr)) * tot


def box_sums(q, L, sign, eps_list):
    w, dpr = probe(sign)
    ks = 2 * np.pi * np.fft.fftfreq(L)
    P = np.meshgrid(*([ks] * q), indexing="ij")
    d = S0 + 2 * C * sum(1 - np.cos(p) for p in P) + 2 * B * C * w * np.sin(P[0]) - w * w
    ph = np.abs(phi1(ks - sign * K0)) ** 2
    amp = ph.reshape((L,) + (1,) * (q - 1))
    for a in range(1, q):
        amp = amp * (np.abs(phi1(ks)) ** 2).reshape(tuple(L if b == a else 1 for b in range(q)))
    V = L ** q
    probe_idx = tuple([int(round(sign * K0 * L / (2 * np.pi))) % L] + [0] * (q - 1))
    amp = amp.copy(); amp[probe_idx] = 0.0          # [fixed: the probe does not scatter into itself]
    rates = [np.pi * S0 ** 2 / (V * abs(dpr)) * (amp * (e / np.pi) / (d * d + e * e)).sum() for e in eps_list]
    shift2 = S0 ** 2 / V * (amp / (np.where(amp > 0, d, 1.0) * dpr)).sum()
    # broadened (principal-value) shift: Re of the sum with d -> d + i eps  [POST HOC, added after the
    # raw fixed-bump shift did not settle]
    shiftPV = [S0 ** 2 / V * (amp * d / ((d * d + e * e) * dpr)).sum() for e in eps_list]
    return rates, shift2, shiftPV


def main():
    eps = [0.2, 0.1, 0.05, 0.025]
    for q, Ls in ((1, (64, 256, 1024)), (2, (32, 64, 128, 256, 512)), (3, (16, 32, 48, 64, 96))):
        cont = {s: continuum_rate(q, s) for s in (+1, -1)}
        print(f"\nq = {q}: continuum V Gamma/S^2: +k {cont[1]:.5e}, -k {cont[-1]:.5e}, difference {cont[1] - cont[-1]:+.3e}")
        for L in Ls:
            rp, sp, pp = box_sums(q, L, +1, eps); rm, sm, pm = box_sums(q, L, -1, eps)
            ext = 2 * rp[-1] - rp[-2]                    # linear eps -> 0 from the two smallest eps
            extm = 2 * rm[-1] - rm[-2]
            print(f"  L={L:4d}: V Gamma/S^2 (+k) at eps " + ", ".join(f"{e}: {r:.4e}" for e, r in zip(eps, rp))
                  + f" -> eps->0 {ext:.4e} (-k {extm:.4e}, diff {ext - extm:+.3e})")
            print(f"          raw shift V dw2/S^2: +k {sp:+.4e}, -k {sm:+.4e}, Delta {sp - sm:+.4e} | broadened Delta at eps "
                  + ", ".join(f"{e}: {a - b:+.4e}" for e, a, b in zip(eps, pp, pm)))
    for s in (+1, -1):
        w, dpr = probe(s)
        print(f"first order: V dw1/S ({'+' if s > 0 else '-'}k) = {-S0 * abs(phi1(0.0)[0]) ** 1 / dpr:+.5f}  (x Phi(0)^(q-1) at q > 1)")


if __name__ == "__main__":
    main()
