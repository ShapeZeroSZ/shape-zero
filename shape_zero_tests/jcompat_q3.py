#!/usr/bin/env python3
"""
jcompat_q3.py -- which J-compatibility bound on kappa applies at q = 3?

THE QUESTION (MODEL_SPEC §3, candidate): the floor is kappa* = 2c/sqrt(K + 2c)
= 0.972 if the coupling segments conserve transverse momentum, and
2qc/sqrt(K + 2qc) = 2.091 at q = 3 if they do not.

FROM THE CODE. model.make_links builds each segment as a SLAB: every site whose
axis-0 coordinate is start + j carries the same link matrix, and the links act
along axis 0 only (Lattice._shift, axis = 0). Every other term (the phi-well,
kappa JJ v, the Laplacian, the scalar beta sector, the C_r residual) is on-site or
uniform. q3_gate.py's Slab uses model.make_links, and its spectrum-averaged
predictor already assumes each component keeps its transverse term. So the linear
dynamics commutes with transverse translations and transverse crystal momentum is
conserved: the opposite-chirality wave must keep the packet's k_perp.

WHY A k_perp = 0 PACKET CANNOT TELL. For k_perp = 0 the two conditions coincide
(kappa w > c(1 - cos kx) either way). They differ for a packet carrying transverse
momentum: a non-conserving coupling can drop it into the opposite chirality at
lower k_perp. The zone-corner bound 2.091 comes from exactly that.

THE TEST. n = 2, q = 3, lattice L0 x S x S (S = 4), packet with carrier
kx = pi/2 and k_perp = (pi, pi) (sign alternating in y and z), Gaussian in x
(width 16), uniform transversely. Branch: w^2 + kappa w = K + 2c(1 - cos kx) + T,
T = 2c sum(1 - cos k_perp) = 8c. Two segment geometries, same RAMP, same Wc / Wx
split as j_compat_test (seed 1):
  slab   -- model.make_links geometry (uniform over y, z): conserves k_perp
  stagger -- the same links times (-1)^(y+z): passive (each link still symmetric),
             but carries transverse momentum (pi, pi)
Channel predictions for this packet:
  slab    closed iff kappa w > c(1 - cos kx)            -> kappa_req = 0.2988
  stagger closed iff 2 kappa w > Q(k0) - K = 2 + 8 = 10  -> kappa_req = 1.8590
                  (the opposite chirality then lands at k_perp = (0, 0))
Runs at kappa = 0.25 (both open), 1.2 (slab closed, stagger open), 2.3 (both
closed). Readout as j_compat_test: whole-lattice internal state, Bures angle to
the no-segment reference, eff_x/eff_c at g = 0.04 and 0.01 (closed: falls x0.25),
and the weight leaked into the bar chirality. Also, for the slab, the fraction of
weight that left k_perp = (pi, pi) (conservation: machine zero).

usage:  python3 jcompat_q3.py predict | measure
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import math
import sys
from multiprocessing import Pool

import numpy as np

import j_compat_test as J

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as MS   # the working model.py (make_links, Lattice with q = 3)

L0, S = 200, 4
X0, WIDTH, SEG0 = 40, 16.0, 80
KX = math.pi / 2
KPERP = (math.pi, math.pi)
KAPPAS = (0.25, 1.2, 2.3)
GS = (0.04, 0.01)
K = MS.SQ5
C = MS.C
T_PERP = 2 * C * sum(1 - math.cos(k) for k in KPERP)


def omega(kap):
    Q = K + 2 * C * (1 - math.cos(KX)) + T_PERP
    return 0.5 * (-kap + math.sqrt(kap * kap + 4 * Q))


def kappa_req(geom):
    """Solve kappa w(kappa) = rhs on the packet branch."""
    rhs = C * (1 - math.cos(KX)) if geom == "slab" else 0.5 * (2 * C * (1 - math.cos(KX)) + T_PERP)
    Q = K + 2 * C * (1 - math.cos(KX)) + T_PERP
    # kappa w = rhs and w^2 + kappa w = Q  ->  w = sqrt(Q - rhs), kappa = rhs / w
    return rhs / math.sqrt(Q - rhs)


class Box(MS.Lattice):
    def __init__(self, kap):
        super().__init__(n=2, N=S ** 3, q=3, shape=S, kappa=kap)
        self.shape = (L0, S, S)
        self.N = L0 * S * S
        self.omega = omega(kap)

    def packet(self, amp=1e-3):
        c = np.indices(self.shape).astype(float)
        env = np.exp(-0.5 * ((c[0] - X0) / WIDTH) ** 2).reshape(-1)
        ph = (KX * (c[0] - X0) + KPERP[0] * c[1] + KPERP[1] * c[2]).reshape(-1)
        u = np.zeros((self.N, self.D)); v = np.zeros((self.N, self.D))
        u[:, 0], u[:, 1] = amp * env * np.cos(ph), amp * env * np.sin(ph)
        v[:, 0], v[:, 1] = self.omega * u[:, 1], -self.omega * u[:, 0]
        return u, v


def links(lat, Wmat, g, geom):
    Wg = np.zeros(lat.shape + (lat.D, lat.D))
    for j, wgt in enumerate(MS.RAMP):
        Wg[(SEG0 + j) % L0, ...] = g * wgt * Wmat
    if geom == "stagger":
        y, z = np.indices((S, S))
        Wg *= ((-1.0) ** (y + z))[None, :, :, None, None]
    Wm = np.roll(Wg, 1, axis=0).reshape(lat.N, lat.D, lat.D)
    return Wg.reshape(lat.N, lat.D, lat.D), Wm


def run_time(lat):
    vg = 2 * C * math.sin(KX) / (2 * lat.omega + lat.kappa)
    return (SEG0 + len(MS.RAMP) + 3 * WIDTH - X0) / vg


def perp_leak(lat, u, v):
    """Fraction of the chi-weight away from k_perp = (pi, pi)."""
    psi = (u[:, 0::2] + 1j * u[:, 1::2]) + (1j / lat.omega) * (v[:, 0::2] + 1j * v[:, 1::2])
    F = np.abs(np.fft.fft2(psi.reshape(L0, S, S, 2), axes=(1, 2))) ** 2
    P = F.sum(axis=(0, 3))
    return float(1 - P[S // 2, S // 2] / P.sum())


def one(job):
    kap, geom, name, g = job
    lat = Box(kap)
    u0, v0 = lat.packet()
    T = run_time(lat)
    if geom is None:
        u, v, drift = lat.run(u0, v0, T)
    else:
        _, Wc, Wx, _ = J.split_W(seed=1)
        W, Wm = links(lat, {"c": Wc, "x": Wx}[name], g, geom)
        u, v, drift = lat.run(u0, v0, T, W, Wm)
    return job, J.internal_state(lat, u, v), perp_leak(lat, u, v), drift


def predict():
    print("=" * 84)
    print("PREDICTIONS -- q = 3, packet kx = pi/2, k_perp = (pi, pi); stated before any run")
    print("=" * 84)
    print(f"  T_perp = {T_PERP:.1f}, kappa_req: slab (conserves k_perp) {kappa_req('slab'):.4f}, "
          f"stagger (does not) {kappa_req('stagger'):.4f}")
    print(f"  zone-wide floors: conserving 2c/sqrt(K+2c) = {2*C/math.sqrt(K+2*C):.6f}; "
          f"non-conserving 6c/sqrt(K+6c) = {6*C/math.sqrt(K+6*C):.6f}")
    print("     kappa    omega     slab       stagger")
    for kap in KAPPAS:
        s = "closed" if kap > kappa_req("slab") else "open"
        t = "closed" if kap > kappa_req("stagger") else "open"
        print(f"     {kap:4.2f}    {omega(kap):.4f}    {s:8s}   {t}")
    print("  closed: eff_x/eff_c falls x0.25 for g 0.04 -> 0.01, bar-chirality leak tiny;")
    print("  open: ratio does not fall in proportion, leak orders larger.")
    print("  slab: weight leaving k_perp = (pi, pi) at machine zero (translation invariance);")
    print("  stagger: that weight is where the open channel goes.")


def measure():
    predict()
    jobs = []
    for kap in KAPPAS:
        jobs.append((kap, None, None, None))
        for geom in ("slab", "stagger"):
            for g in GS:
                for name in ("c", "x"):
                    jobs.append((kap, geom, name, g))
    with Pool(4) as p:
        res = {r[0]: r[1:] for r in p.map(one, jobs)}
    print("\n" + "=" * 84)
    print("MEASURED")
    print("=" * 84)
    print("   kappa  geom     g      eff_c(deg)  eff_x(deg)  ratio     leak_x      off-k_perp(x)   verdict")
    for kap in KAPPAS:
        ref = res[(kap, None, None, None)][0]
        print(f"   {kap:4.2f}   reference drift {res[(kap, None, None, None)][2]:.1e}, "
              f"off-k_perp {res[(kap, None, None, None)][1]:.1e}")
        for geom in ("slab", "stagger"):
            rat = {}
            for g in GS:
                Rc, _, _ = res[(kap, geom, "c", g)]
                Rx, offx, _ = res[(kap, geom, "x", g)]
                ec, ex = J.bures_deg(ref, Rc), J.bures_deg(ref, Rx)
                leak = float(np.real(np.trace(Rx[2:, 2:])) - np.real(np.trace(ref[2:, 2:])))
                rat[g] = ex / ec
                print(f"   {kap:4.2f}   {geom:7s}  {g:5.3f}  {ec:10.5f}  {ex:10.5f}  {ex/ec:.5f}   "
                      f"{leak:+.2e}   {offx:.2e}", flush=True)
            fall = rat[0.01] / rat[0.04]
            pred = "closed" if kap > kappa_req(geom) else "open"
            got = "closed" if fall < 0.4 else ("open" if fall > 0.6 else "unclear")
            print(f"          ratio(0.01)/ratio(0.04) = {fall:.3f} -> {got}   [predicted {pred}]")


if __name__ == "__main__":
    {"predict": predict, "measure": measure}[sys.argv[1]]()
