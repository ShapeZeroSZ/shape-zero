#!/usr/bin/env python3
"""
jcompat_kappa.py -- what does "J-compatibility derived at every wavelength"
require of the gyroscopic ratio kappa?

THE CONDITION (MODEL_SPEC §3). The J-breaking part of a coupling is suppressed
exactly when the opposite chirality has no travelling wave at the packet's
frequency. The packet's branch (j_compat_test.KLattice) is
    w^2 + kappa w - Q(k0) = 0,     Q(k) = K + 2c(1 - cos k),
and the opposite chirality at the same w needs w^2 - kappa w = Q(k'), so
    cos k' = cos k0 + kappa w / c.
The channel is closed iff cos k' > 1, i.e. kappa w(k0; kappa) > c (1 - cos k0).

CLOSED FORM. Put x = c(1 - cos k0). On the boundary kappa w = x, and
w^2 + kappa w = K + 2x gives x^2/kappa^2 = K + x, so
    kappa_req(k0) = x / sqrt(K + x),   x = c(1 - cos k0).
d/dx [x / sqrt(K + x)] = (K + x/2) / (K + x)^(3/2) > 0, so kappa_req rises
monotonically with 1 - cos k0 and is largest at the band edge k0 = pi (x = 2c):
    kappa* = 2c / sqrt(K + 2c).
At kappa = kappa* exactly the only marginal point is k0 = pi, where k' = 0
(zero group velocity); every other k0 is closed.

q DIMENSIONS. On a q-dimensional lattice Q(k) = K + 2c sum_a (1 - cos k_a). If the
coupling conserves transverse momentum (a transversely uniform segment), k' keeps
k0's transverse part and the condition is the 1-D one along the propagation axis
at a higher w -- no harder than kappa*. If it does not, k' may be anywhere in the
zone, the channel is closed iff 2 kappa w > Q(k0) - K, and the worst case is the
zone corner (pi, ..., pi):
    kappa*_q = 2 q c / sqrt(K + 2 q c).

NUMERICAL CHECK (j_compat_test.py's lattice, force law, packet, segment split and
Bures readout; n = 2, q = 1, K = sqrt5, c = 1). At k0 = pi/2, 0.75 pi, 0.9 pi and
kappa 15% below and 15% above kappa_req(k0), the J-breaking segment (Wx) is
compared with the J-respecting one (Wc) at g = 0.04 and 0.01. Closed channel:
the ratio eff_x/eff_c falls in proportion to g (x0.25) and the weight leaked into
the opposite chirality is tiny. Open channel: the ratio levels off. A wider packet
(width 16, 400 sites) keeps its wavenumber spread (~0.06 rad) inside the margin.

SCAN (python3 jcompat_kappa.py scan). Near the band edge the two-point ratio test
reads "unclear" for the open cases (the open channel there is weakly coupled: its
k' is far from k0, so the smooth segment's Fourier weight at the mismatch is small
and the leaked piece does not yet dominate eff_x at g = 0.01-0.04). Two sharper
checks: (a) the open cases extended to g = 0.0025, where the leaked piece must
dominate and the ratio must stop falling; (b) a kappa sweep through kappa_req(k0)
at fixed g = 0.04, where the leaked weight must drop by orders of magnitude at
kappa/kappa_req = 1 (smeared by the packet's wavenumber spread, ~1/width).

usage:  python3 jcompat_kappa.py        (closed form + the 15%-either-side check)
        python3 jcompat_kappa.py scan   (the g ladder and the kappa sweep)
"""

import math
import numpy as np

import j_compat_test as J

M = J.M
K_STIFF = M.SQ5
C = M.C


def kappa_req(k0, K=K_STIFF, c=C):
    x = c * (1 - math.cos(k0))
    return x / math.sqrt(K + x)


def kappa_star(q=1, K=K_STIFF, c=C):
    return 2 * q * c / math.sqrt(K + 2 * q * c)


def omega(k0, kappa, K=K_STIFF, c=C):
    Q = K + 2 * c * (1 - math.cos(k0))
    return 0.5 * (-kappa + math.sqrt(kappa * kappa + 4 * Q))


def cos_kprime(k0, kappa):
    return math.cos(k0) + kappa * omega(k0, kappa) / C


def measure(k0, kappa, g, Wc, Wx, width=16.0, N=400):
    M.K0 = k0
    lat = J.KLattice(n=2, K=K_STIFF, N=N, kappa=kappa)
    vg = 2 * C * math.sin(k0) / (2 * lat.omega + lat.kappa)
    T = (J.SEG0 + len(M.RAMP) + 3 * width - J.N0) / vg
    u0, v0 = lat.packet(amp=1e-3, n0=J.N0, width=width, colour=0)
    ur, vr, _ = lat.run(u0.copy(), v0.copy(), T)
    ref = J.internal_state(lat, ur, vr)
    out = {}
    for name, Wm_ in (("c", Wc), ("x", Wx)):
        W, Wm = J.links(lat, Wm_, g)
        u, v, _ = lat.run(u0.copy(), v0.copy(), T, W, Wm)
        st = J.internal_state(lat, u, v)
        out["eff_" + name] = J.bures_deg(ref, st)
        out["leak_" + name] = float(np.real(np.trace(st[2:, 2:])) - np.real(np.trace(ref[2:, 2:])))
    out["T"] = T
    return out


def main():
    print("=" * 84)
    print("J-COMPATIBILITY AT EVERY WAVELENGTH -- the kappa it requires")
    print("=" * 84)
    print(f"  K = sqrt5 = {K_STIFF:.6f}, c = {C}")
    print(f"  kappa* = 2c/sqrt(K + 2c) = {kappa_star():.6f}   (q = 1; also any coupling that")
    print(f"           conserves transverse momentum)")
    print(f"  kappa*_q = 2qc/sqrt(K + 2qc): q = 2 {kappa_star(2):.6f}, q = 3 {kappa_star(3):.6f}"
          f"   (couplings that do not)")
    print(f"  current model.py KAPPA = 0.5; recorded 'k_c ~ 1.45 rad at kappa = 0.5'")
    print("\n  kappa_req(k0) = x/sqrt(K + x), x = c(1 - cos k0):")
    print("     k0/pi    kappa_req     cos k' at kappa = 0.5    at kappa = kappa*")
    for f in (0.08, 0.25, 0.4617, 0.5, 0.75, 0.9, 0.99, 1.0):
        k0 = f * math.pi
        print(f"     {f:5.3f}    {kappa_req(k0):.6f}      {cos_kprime(k0, 0.5):+.4f}"
              f"                  {cos_kprime(k0, kappa_star()):+.6f}")
    # where the channel closes at kappa = 0.5
    lo, hi = 0.01, math.pi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if kappa_req(mid) < 0.5:
            lo = mid
        else:
            hi = mid
    print(f"\n  at kappa = 0.5 the channel is closed for k0 < {lo:.4f} rad = {lo/math.pi:.4f} pi")
    # 3-D corner check against exact branch frequencies
    ks = kappa_star(3)
    Qc = K_STIFF + 12 * C
    w = 0.5 * (-ks + math.sqrt(ks * ks + 4 * Qc))
    print(f"  q = 3 corner check at kappa*_3: w^2 - kappa w = {w*w - ks*w:.6f}  vs band bottom K = {K_STIFF:.6f}")

    print("\n  NUMERICAL CHECK (j_compat_test machinery; n = 2, q = 1, width 16, 400 sites)")
    _, Wc, Wx, _ = J.split_W(seed=1)
    print("     k0/pi   kappa     vs kappa_req   cos k'    g      eff_x/eff_c   leak_x       verdict")
    for f in (0.5, 0.75, 0.9):
        k0 = f * math.pi
        kr = kappa_req(k0)
        for kap, tag in ((0.85 * kr, "below"), (1.15 * kr, "above")):
            ratios = {}
            for g in (0.04, 0.01):
                o = measure(k0, kap, g, Wc, Wx)
                ratios[g] = (o["eff_x"] / o["eff_c"], o["leak_x"])
                print(f"     {f:4.2f}   {kap:.4f}   {tag:5s} ({kap/kr:.2f})   {cos_kprime(k0, kap):+.4f}  "
                      f"{g:5.3f}   {ratios[g][0]:.5f}       {ratios[g][1]:+.2e}", flush=True)
            fall = ratios[0.01][0] / ratios[0.04][0]
            verdict = "CLOSED (ratio ∝ g)" if fall < 0.4 else ("OPEN (levels off)" if fall > 0.7 else "unclear")
            print(f"            ratio(0.01)/ratio(0.04) = {fall:.3f}  ->  {verdict}"
                  f"   [predicted {'closed' if cos_kprime(k0, kap) > 1 else 'open'}]", flush=True)


def scan():
    print("=" * 84)
    print("SCAN -- g ladder for the open cases, and a kappa sweep through kappa_req")
    print("=" * 84)
    _, Wc, Wx, _ = J.split_W(seed=1)
    print("\n  (a) open cases (kappa = 0.85 kappa_req), g down to 0.0025")
    print("     k0/pi   kappa     g        eff_x/eff_c   ratio/g      leak_x")
    for f in (0.75, 0.9):
        k0 = f * math.pi
        kap = 0.85 * kappa_req(k0)
        for g in (0.04, 0.01, 0.0025):
            o = measure(k0, kap, g, Wc, Wx)
            r = o["eff_x"] / o["eff_c"]
            print(f"     {f:4.2f}   {kap:.4f}   {g:6.4f}   {r:.5f}       {r / g:7.3f}      {o['leak_x']:+.2e}",
                  flush=True)
    print("\n  (b) kappa sweep at g = 0.04 (width 16: packet spread ~0.06 rad)")
    print("     k0/pi   kappa/kappa_req   kappa     cos k'     eff_x/eff_c   leak_x")
    for f in (0.75, 0.9):
        k0 = f * math.pi
        kr = kappa_req(k0)
        for m in (0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10):
            kap = m * kr
            o = measure(k0, kap, 0.04, Wc, Wx)
            print(f"     {f:4.2f}   {m:5.2f}             {kap:.4f}    {cos_kprime(k0, kap):+.4f}    "
                  f"{o['eff_x'] / o['eff_c']:.5f}       {o['leak_x']:+.2e}", flush=True)


if __name__ == "__main__":
    import sys
    scan() if sys.argv[1:] == ["scan"] else main()
