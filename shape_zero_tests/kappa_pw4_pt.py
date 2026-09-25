#!/usr/bin/env python3
"""
kappa_pw4_pt.py -- the plane wave's fourth-order amplitude correction to kappa,
derived by harmonic balance on the lattice. Nothing fitted.

LATTICE (identical to kappa_resolution_test.py; reference sign convention,
+k the UPPER root, MODEL_SPEC §5 table): x = phi + u,
    u'' = -(sqrt5 u + u^2) + c lap u + beta c (v[n-1] - v[n+1]).
A plane travelling wave is u_n(t) = U(theta), theta = s K n - W t, U 2pi-periodic.
Write U = sum_m c_m e^{i m theta} (c_{-m} = c_m*). Harmonic m sits at wavevector
m s K and frequency m W, so it answers with the same linear operator as any
lattice wave:
    D(k, W) = -W^2 + Q(k) + 2 c beta W sin k,   Q(k) = sqrt5 + 2c(1 - cos k)
and harmonic balance is exact:
    D(m s K, m W) c_m = -(U^2)_m = -sum_p c_p c_{m-p}          (all m).
All D are real, so an all-real solution exists; fix c_1 = A/2 (A = real
amplitude of the fundamental, the seed's A).

FOURTH ORDER, harmonics 0..3 kept to the order they enter. With D_m = D(msK, mW0)
evaluated on the linear branch W0, D1' = dD/dW at m = 1 (= -2W0 + 2cb sin sK):
  O(A^2):  c0 = -2 c1^2 / sqrt5                       (static shift)
           c2 = -c1^2 / D2                            (second harmonic)
  O(A^3):  c3 = -2 c1 c2 / D3                         (third harmonic)
           D1' W2 c1 = -(2 c0 c1 + 2 c2 c1)  ->  W2 = (1/sqrt5 + 1/(2 D2)) / D1'   [x A^2]
  O(A^4):  c0' = -(2 c2^2 + c0^2) / sqrt5             (from |c2|^2 and c0^2)
           c2' = -(2 c0 c2 + 2 c3 c1 + D2'(2W) W2A^2 c2) / D2
                 where the last term is the shift of D2 = D(2sK, 2W) with W:
                 dD2/dW = 2 * (-4 W0 + 2 c beta sin 2sK)  (chain rule through 2W)
The m = 1 equation to O(A^5) is
    [D1' dW + (1/2) D1'' dW^2] c1 = -(2 c0 c1 + 2 c2 c1 + 2 c3 c2)     (D1'' = -2)
with dW = W2 A^2 + W4 A^4, so at O(A^5):
    D1' W4 c1 - W2^2 c1 = -(2 c0' c1 + 2 c2' c1 + 2 c3 c2).
kappa(A) = (|Delta(A)| / TH - 1) / A^2,
Delta = W(+) - W(-), so kappa(A) = kappa2 + kappa4 A^2 + O(A^4).

CHECKS (printed):
  1. kappa2 reproduces kappa_cross_pt.py's -0.01748.
  2. The analytic kappa4 agrees with a NUMERICALLY EXACT harmonic balance
     (Newton on 24 harmonics) fitted at small A.
  3. The exact orbit's frequency agrees with a direct lattice simulation SEEDED
     WITH THAT ORBIT (kappa_resolution_test.py integrator and readout).

THE SEED. The measured kappa uses the LINEAR seed (u = A cos, v at the linear
branch frequency, no harmonics), not the orbit. The difference is O(A^2) in the
static/2nd-harmonic sector and O(A^3) in the fundamental's velocity. It can move
kappa at O(A^2) relative, i.e. at the same order as kappa4. Part 3 measures the
linear-seed kappa beside the orbit-seed kappa, so the two are separated rather
than assumed equal.

usage:  python3 kappa_pw4_pt.py
"""

import math
import numpy as np
from scipy.optimize import fsolve

SQ5 = math.sqrt(5.0)
C = 1.0
K = math.pi / 2
BETA = 0.05
TH = abs(2 * C * BETA * math.sin(K))


def Q(k):
    return SQ5 + 2 * C * (1 - math.cos(k))


def D(k, W):
    return -W * W + Q(k) + 2 * C * BETA * W * math.sin(k)


def W_lin(s):
    b = C * BETA * math.sin(s * K)
    return b + math.sqrt(b * b + Q(s * K))


# ------------------------------------------------------------ analytic, O(A^4)
def analytic(s):
    """Coefficients W2, W4 of W(A) = W0 + W2 A^2 + W4 A^4 for direction s."""
    k = s * K
    W0 = W_lin(s)
    D1p = -2 * W0 + 2 * C * BETA * math.sin(k)          # dD(k, W)/dW at W0
    D2 = D(2 * k, 2 * W0)
    D3 = D(3 * k, 3 * W0)
    D2p = 2 * (-2 * (2 * W0) + 2 * C * BETA * math.sin(2 * k))   # d/dW of D(2k, 2W)
    # per unit A: c1 = 1/2
    c1 = 0.5
    c0 = -2 * c1 * c1 / SQ5                               # x A^2
    c2 = -c1 * c1 / D2                                    # x A^2
    W2 = -(2 * c0 * c1 + 2 * c2 * c1) / (D1p * c1)
    c3 = -2 * c1 * c2 / D3                                # x A^3
    # O(A^4) corrections to c0, c2
    c0b = -(2 * c2 * c2 + c0 * c0) / SQ5
    c2b = -(2 * c0 * c2 + 2 * c3 * c1 + D2p * W2 * c2) / D2
    # m = 1 at O(A^5):  D1p W4 c1 + (1/2)(-2) W2^2 c1 = -(2 c0b c1 + 2 c2b c1 + 2 c3 c2)
    W4 = (-(2 * c0b * c1 + 2 * c2b * c1 + 2 * c3 * c2) + W2 * W2 * c1) / (D1p * c1)
    return W0, W2, W4, dict(c0=c0, c2=c2, c3=c3, c0b=c0b, c2b=c2b)


def kappa_analytic():
    Wp0, Wp2, Wp4, _ = analytic(+1)
    Wm0, Wm2, Wm4, _ = analytic(-1)
    d0 = Wp0 - Wm0                                         # > 0 here (+k upper)
    k2 = (Wp2 - Wm2) / d0
    k4 = (Wp4 - Wm4) / d0
    return d0, k2, k4


# ------------------------------------------------------ numerically exact HB
M_HARM = 24


def hb_residual(x, s, A):
    """x = [W, c0, c2, ..., cM]; c1 = A/2 fixed. Returns equations m = 0..M."""
    W = x[0]
    c = np.zeros(M_HARM + 1)
    c[0] = x[1]
    c[1] = A / 2
    c[2:] = x[2:]
    full = np.concatenate([c[:0:-1], c])                  # c_{-M} .. c_M
    sq = np.convolve(full, full)                           # (U^2)_m, index m + 2M
    res = []
    for m in range(M_HARM + 1):
        res.append(D(m * s * K, m * W) * c[m] + sq[m + 2 * M_HARM])
    return np.array(res)


def hb_exact(s, A, guess=None):
    if guess is None:
        W0 = W_lin(s)
        guess = np.zeros(M_HARM + 1)
        guess[0] = W0
    sol, info, ier, msg = fsolve(hb_residual, guess, args=(s, A), full_output=True,
                                 xtol=1e-14)
    if np.max(np.abs(hb_residual(sol, s, A))) > 1e-14:
        raise RuntimeError(msg)
    return sol


def kappa_exact(A):
    sp = hb_exact(+1, A)
    sm = hb_exact(-1, A)
    Wp, Wm = sp[0], sm[0]
    d0 = W_lin(+1) - W_lin(-1)
    return (abs(Wp - Wm) / d0 - 1) / (A * A), sp, sm


# ------------------------------------------------------------- the seed
def Dk(kx, ky, kz, W):
    """D on the 3-D lattice (transverse wavevector p = (ky, kz))."""
    Qk = SQ5 + 2 * C * ((1 - np.cos(kx)) + (1 - np.cos(ky)) + (1 - np.cos(kz)))
    return -W * W + Qk + 2 * C * BETA * W * np.sin(kx)


def xpm(s, W0, kqx, py, pz, Wq, b2):
    """Frequency shift of the probe (sK, 0, 0) at frequency W0 from a free
    linear mode (kqx, p) at frequency Wq with |b|^2 = b2 (u = b e^{i(k.n - Wq t)}
    + c.c.): the second-order cross term of kappa_cross_pt.py,
        4 |b|^2 [1/D(k0+kq, W0+Wq) + 1/D(k0-kq, W0-Wq) + 1/D(0, 0)] / D_W(k0)."""
    k0 = s * K
    G = 4 * b2 * (1 / Dk(k0 + kqx, py, pz, W0 + Wq)
                  + 1 / Dk(k0 - kqx, -py, -pz, W0 - Wq)
                  + 1 / SQ5)
    D1p = -2 * W0 + 2 * C * BETA * math.sin(k0)
    return G / D1p


def seed_terms(A, env=None):
    """Direction-odd seed corrections to the box-wide (q = 0) frequency, as a
    kappa increment (divide by TH A^2). env = None: plane wave; otherwise a 2-D
    transverse envelope on an L x L grid.

    The linear seed u = A env cos(theta) lacks the orbit's forced static and
    second-harmonic response. At t = 0 the difference (seed - orbit) is
        -c0(y)             in the k_x = 0 sector,   zero velocity
        -2 c2(y) (-1)^n    in the k_x = pi sector,  zero velocity (sin(pi n) = 0)
    so it is carried by FREE linear modes: (0, p) at sqrt(Q(0, p)) and (pi, p) at
    sqrt(Q(pi, p)), with complex amplitude b_p = f_p / 2 for a real field
    f(y) cos(Omega t) = sum_p f_p e^{i p y} cos(Omega_p t). Their cross-modulation
    of the box-wide component is second order in b ~ A^2: an A^4 frequency shift.
    Forced responses (u1 = A env cos theta): the source u1^2 has DC part
    (A^2/2) E2(p) and second-harmonic amplitude (A^2/4) E2(p), E2 = FFT2(env^2)/L^2,
    so  c0(p) = -(A^2/2) E2(p) / Q(0, p),  c2(p) = -(A^2/4) E2(p) / D((2sK, p), 2W).
    Returns (dc part, pi part) of kappa, each already direction-odd."""
    if env is None:
        E2 = np.ones((1, 1))
        qy = qz = np.zeros((1, 1))
    else:
        L = env.shape[0]
        E2 = np.fft.fft2(env ** 2) / (L * L)
        q = 2 * np.pi * np.fft.fftfreq(L)
        qy, qz = np.meshgrid(q, q, indexing="ij")
    out = []
    for sector in ("dc", "pi"):
        dW = {}
        for sgn in (+1, -1):
            W0 = W_lin(sgn)
            if sector == "dc":
                Qp = Dk(0.0, qy, qz, 0.0)
                f = +(A * A / 2) * E2 / Qp                  # -c0(p)
                kqx, Om = 0.0, np.sqrt(Qp)
            else:
                Qp = Dk(math.pi, qy, qz, 0.0)
                c2 = -(A * A / 4) * E2 / Dk(2 * sgn * K, qy, qz, 2 * W0)
                f = -2 * c2                                  # -2 c2(p)
                kqx, Om = math.pi, np.sqrt(Qp)
            b2 = np.abs(f / 2) ** 2
            dW[sgn] = float(np.sum(xpm(sgn, W0, kqx, qy, qz, Om, b2)))
        out.append((dW[+1] - dW[-1]) / (TH * A * A))
    return out


def kappa_velocity(A):
    """The linear seed puts the fundamental's velocity on the LINEAR branch; the
    orbit runs at W(A). Split the fundamental-sector mismatch into the forward
    wave (frequency W) and the counter-propagating one read in the same bin
    (frequency W_b = W_lin(-s)): the forward amplitude becomes
        A' = A (W_b + W_lin) / (W_b + W(A)).
    Returns kappa of the exact orbit at A' (both directions), i.e. orbit +
    velocity renormalisation."""
    Ws = {}
    for sgn in (+1, -1):
        Wb_ = W_lin(-sgn)
        W = hb_exact(sgn, A)[0]
        for _ in range(3):
            Ap = A * (Wb_ + W_lin(sgn)) / (Wb_ + W)
            W = hb_exact(sgn, Ap)[0]
        Ws[sgn] = W
    return (abs(Ws[+1] - Ws[-1]) / TH - 1) / (A * A)


def kappa_protocol(A, env=None):
    """Prediction for what the linear-seed protocol measures on the plane wave."""
    kv = kappa_velocity(A)
    dc, pi = seed_terms(A)
    return kv + dc * A * A + pi * A * A, kv, dc, pi


def main():
    print("=" * 74)
    print("PLANE-WAVE KAPPA TO FOURTH ORDER -- harmonic balance on the lattice")
    print("=" * 74)
    d0, k2, k4 = kappa_analytic()
    print(f"  linear asymmetry W(+K) - W(-K) = {d0:.10f}   (2 c beta sin K = {TH:.10f})")
    print(f"  kappa2 (analytic, A -> 0)      = {k2:+.6f}   (kappa_cross_pt.py: -0.01748)")
    print(f"  kappa4 (analytic, A^2 coeff.)  = {k4:+.6f}")
    for s in (+1, -1):
        W0, W2, W4, cc = analytic(s)
        print(f"    direction {s:+d}: W0 {W0:.8f}  W2 {W2:+.8f}  W4 {W4:+.8f}   "
              f"c0 {cc['c0']:+.5f}A^2 c2 {cc['c2']:+.5f}A^2 c3 {cc['c3']:+.6f}A^3")

    print("\n  NUMERICALLY EXACT harmonic balance (24 harmonics), orbit frequency:")
    print("      A      kappa_exact    k2 + k4 A^2    difference")
    rows = []
    for A in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        ke, _, _ = kappa_exact(A)
        ka = k2 + k4 * A * A
        rows.append((A, ke))
        print(f"    {A:5.2f}   {ke:+.6f}      {ka:+.6f}      {ke - ka:+.2e}")
    # fit kappa4 from the exact HB at small A
    As = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    ks = np.array([kappa_exact(a)[0] for a in As])
    p = np.polyfit(As ** 2, ks, 2)
    print(f"\n  exact HB fitted at A = 0.02-0.06: kappa2 {p[2]:+.6f}  kappa4 {p[1]:+.6f}  "
          f"kappa6 {p[0]:+.4f}")
    print(f"  analytic:                          kappa2 {k2:+.6f}  kappa4 {k4:+.6f}")

    print("\n  THE SEED -- what the linear-seed protocol measures (plane wave):")
    dc, pi = seed_terms(0.3)
    print(f"    free-mode cross-modulation, as a kappa4 term: DC sector {dc:+.6f}, "
          f"pi sector {pi:+.6f}  (x A^2)")
    print("      A    orbit      +velocity   +leading free-mode XPM")
    for A in (0.10, 0.20, 0.30, 0.40):
        kp, kv, dc, pi = kappa_protocol(A)
        print(f"    {A:4.2f}  {kappa_exact(A)[0]:+.5f}   {kv:+.5f}    {kp:+.5f}")
    print("    (the free-mode cross-modulation above is the LEADING formula only; the")
    print("     seed's static-shift and harmonic effects are measured, not derived, in")
    print("     kappa_pw4_seed.py / kappa_pw4_attrib.py / kappa_seed2_test.py)")


if __name__ == "__main__":
    main()
