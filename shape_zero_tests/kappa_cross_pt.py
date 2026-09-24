#!/usr/bin/env python3
"""
kappa_cross_pt.py -- second-order perturbation theory for the cross kernel
K(q_perp) and the factor F of a localised beam, on the lattice. Nothing fitted.

LATTICE (identical to kappa_resolution_test.py): x = phi + u,
    u'' = -(sqrt5 u + u^2) + c lap u + beta c (v[x-1] - v[x+1])
For u ~ e^{i(k.n - W t)} the linear operator is
    D(k, W) = -W^2 + Q(k) + 2 c beta W sin kx,   Q(k) = sqrt5 + 2c sum_i (1 - cos k_i)
and D(k, W) u_k = -(u^2)_k. The branch seeded is W(k) = cb sin kx + sqrt((cb sin kx)^2 + Q),
so +k is the UPPER root (reference convention).

SECOND ORDER. Write the first-order field u1 = sum_j (a_j e^{i th_j} + c.c.). The
quadratic term drives sum modes (k_j + k_k, W_j + W_k), difference modes
(k_j - k_l, W_j - W_l) and the DC mode, each answering with 1/D at its own (k, W).
At third order the wave a_0 at k0 is driven by every triad (j, k, l*) with
k_j + k_k - k_l = k0:

    D_W(k0) i da0/dt = 2 sum_{(j,k) ordered} a_j a_k a_l* W_jkl e^{-i Delta t}
    W_jkl = 1/D(k_j+k_k, W_j+W_k) + 1/D(k_j-k_l, W_j-W_l) + 1/D(k_k-k_l, W_k-W_l)
    Delta = W_j + W_k - W_l - W_0,     D_W = dD/dW = -2W + 2cb sin kx.

Checks built in: the self term (j = k = l = 0) is |a|^2 (4/sqrt5 + 2/D(2k0, 2W0)),
the closure result (phi_gauge_delta.py: +0.0349 beta in the platform
convention); the cross term (j = 0, k = l = q) is 4|a_q|^2 [1/D_sum + 1/D_diff
+ 1/sqrt5], which is exactly twice the self term as q -> 0 (the smooth-space
factor 2 behind F = 2 - s).

DIRECTION-ODD PART. kappa measures (|W(+k) - W(-k)| / TH - 1) / A^2. For the
cross term the DC mode, the difference mode (kx = 0, frequency W0 - Wq, the same
in both directions) and D_W (= -2 sqrt(b^2 + Q) in both) are even; only the sum
mode at (2K = pi, q_perp), frequency W0 + Wq, carries the odd part. So

    R(q) = K(q) / K(0) = 2 [1/D_s(q)]_odd / [1/D_s(0)]_odd

DEFINITIONS. A beam psi = A env(y, z) e^{isKx} has transverse components
c_q = FFT2(env)/L^2; component q has real amplitude A|c_q|. K(q) is the odd
frequency shift of the box-wide component (q = 0) per unit (real amplitude)^2
of component q; K(0) = kappa_pw * TH. Then kappa_box = sum_q K(q)|c_q|^2 / TH and

    F = kappa_box / (kappa_pw fill) = sum_q R(q) |c_q|^2 / sum_q |c_q|^2

PREDICTIONS (two, both stated before any comparison):
  P3a  DIAGONAL: the formula above, summing the kernel over each beam's actual
       grid components -- the "sum K over components" model.
  P3b  FULL TRIADS: every triad driving the box-wide component, including the
       phase-coherent off-diagonal ones (a_{(a,0)} a_{(0,b)} a*_{(a,b)} and the
       like), each with its linear detuning Delta, read out as the experiment
       does: least-squares slope of the phase over t in [0, 300].
       Off-diagonal triads whose members sit on different transverse axes are
       nearly resonant on the lattice (Q is a sum over axes), so they are not
       averaged away; P3a omits them. Limitation stated in advance: detunings
       are linear; the O(A^2) nonlinear shifts of the components are not in them.

usage:  python3 kappa_cross_pt.py            (kernel table + P3a/P3b for every measured (w, L))
"""

import math
import numpy as np

SQ5 = math.sqrt(5.0)
C = 1.0
K = math.pi / 2
BETA = 0.05
TH = abs(2 * C * BETA * math.sin(K))
T_RUN = 300.0


def Q(kx, ky, kz):
    return SQ5 + 2 * C * ((1 - np.cos(kx)) + (1 - np.cos(ky)) + (1 - np.cos(kz)))


def Wb(kx, ky, kz):
    b = C * BETA * np.sin(kx)
    return b + np.sqrt(b * b + Q(kx, ky, kz))


def D(kx, ky, kz, W):
    return -W * W + Q(kx, ky, kz) + 2 * C * BETA * W * np.sin(kx)


def DW(kx, ky, kz):
    return -2 * Wb(kx, ky, kz) + 2 * C * BETA * np.sin(kx)


# ---------------------------------------------------------------- the kernel
def shift_self(s, A):
    """Frequency shift of a plane wave (s K, 0, 0), real amplitude A."""
    kx = s * K
    W0 = Wb(kx, 0.0, 0.0)
    a2 = A * A / 4
    G = a2 * (4 / SQ5 + 2 / D(2 * kx, 0.0, 0.0, 2 * W0))
    return G / DW(kx, 0.0, 0.0)


def shift_cross(s, A1, qy, qz):
    """Shift of the probe (s K, 0, 0) from a pump (s K, qy, qz) of real amplitude A1."""
    kx = s * K
    W0 = Wb(kx, 0.0, 0.0)
    Wq = Wb(kx, qy, qz)
    a2 = A1 * A1 / 4
    G = 4 * a2 * (1 / D(2 * kx, qy, qz, W0 + Wq)
                  + 1 / D(0.0, -qy, -qz, W0 - Wq)
                  + 1 / SQ5)
    return G / DW(kx, 0.0, 0.0)


def kappa_pw_pt(A=1e-3):
    return (shift_self(+1, A) - shift_self(-1, A)) / (TH * A * A)


def K_cross(qy, qz, A1=1e-3):
    """Odd shift per real amplitude^2 (units of frequency)."""
    return (shift_cross(+1, A1, qy, qz) - shift_cross(-1, A1, qy, qz)) / (A1 * A1)


def R(qy, qz):
    """K(q)/K(0), the kernel in units of the plane-wave self term."""
    k0 = kappa_pw_pt() * TH
    return K_cross(qy, qz) / k0


# ---------------------------------------------------------------- beams
def env2d(L, w):
    idx = np.indices((L, L)).astype(float)
    e = np.ones((L, L))
    for a in (0, 1):
        d = idx[a] - L / 2.0
        d = (d + L / 2) % L - L / 2
        e = e * np.exp(-0.5 * (d / w) ** 2)
    return e


def geometry(L, w):
    e = env2d(L, w)
    fill = float(np.mean(e ** 2))
    s = float(np.mean(e) ** 2 / np.mean(e ** 2))
    return fill, s


def F_diag(L, w):
    """P3a: sum of the kernel over the beam's grid components."""
    e = env2d(L, w)
    c = np.fft.fft2(e) / (L * L)
    q = 2 * np.pi * np.fft.fftfreq(L)
    qy, qz = np.meshgrid(q, q, indexing="ij")
    Rq = R(qy, qz)
    Rq[0, 0] = 1.0
    p = np.abs(c) ** 2
    return float(np.sum(Rq * p) / np.sum(p))


def _slope_coeffs(Delta, T=T_RUN):
    """OLS slopes over t in [0, T] of (cos(D t) - 1)/D and sin(D t)/D (continuous)."""
    Dl = np.where(np.abs(Delta) < 1e-12, 1e-12, Delta)
    # slope of f = 12/T^3 * int_0^T (t - T/2) f(t) dt
    x = Dl * T
    # int_0^T (t - T/2) cos(D t) dt  and  sin
    ic = (np.cos(x) - 1) / Dl ** 2 + T * np.sin(x) / Dl - (T / 2) * np.sin(x) / Dl
    isn = (np.sin(x) / Dl ** 2 - T * np.cos(x) / Dl) - (T / 2) * (1 - np.cos(x)) / Dl
    Sc = 12 / T ** 3 * (ic / Dl)                   # (cos-1)/D: the -1 term integrates to 0
    Ss = 12 / T ** 3 * (isn / Dl)
    small = np.abs(Delta) * T < 1e-6
    Sc = np.where(small, -Delta * T / 2 * 0.0, Sc)  # -> 0 to first order
    Ss = np.where(small, 1.0, Ss)
    return Sc, Ss


def odd_shift_triads(L, w, s, A=1e-3):
    """P3b: effective (read-out) frequency shift of the box-wide component,
    direction s, from every triad. Returns the shift (frequency units)."""
    e = env2d(L, w)
    c = np.fft.fft2(e) / (L * L)
    a = A * c / 2                                   # u = sum (a e^{i th} + c.c.)
    q = 2 * np.pi * np.fft.fftfreq(L)
    iy, iz = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    qy, qz = q[iy], q[iz]
    kx = s * K
    Wj = Wb(kx, qy, qz)
    W0 = Wj[0, 0]
    a0 = a[0, 0]
    dW = DW(kx, 0.0, 0.0)
    total = 0.0
    for jy in range(L):
        for jz in range(L):
            ly = (jy + iy) % L                      # l = j + k
            lz = (jz + iz) % L
            qjy, qjz = q[jy], q[jz]
            Wjj = Wj[jy, jz]
            Wk = Wj
            Wl = Wj[ly, lz]
            wt = (1 / D(2 * kx, qjy + qy, qjz + qz, Wjj + Wk)
                  + 1 / D(0.0, qjy - q[ly], qjz - q[lz], Wjj - Wl)
                  + 1 / D(0.0, qy - q[ly], qz - q[lz], Wk - Wl))
            T = 2 * a[jy, jz] * a * np.conj(a[ly, lz]) * wt
            Delta = Wjj + Wk - Wl - W0
            Z = (-1j / (dW * a0)) * T
            Sc, Ss = _slope_coeffs(Delta)
            # phase phi = Im(Z g), g = (e^{-iDt}-1)/(-iD): Im g = (cos-1)/D, Re g = sin/D
            slope = np.sum(Z.real * Sc + Z.imag * Ss)
            total += slope
    return -total                                   # frequency = W0 - d(phase)/dt


def F_triads(L, w, A=1e-3):
    fill, _ = geometry(L, w)
    kb = (odd_shift_triads(L, w, +1, A) - odd_shift_triads(L, w, -1, A)) / (TH * A * A)
    return kb / (kappa_pw_pt() * fill)


# measured (w, L) points: kappa_widthscan_gpu_colab_raw.txt, kappa_resolution_test_raw.txt,
# kappa_extended_gpu_colab_raw.txt
POINTS = [(1.0, 4), (2.0, 8), (3.0, 12), (4.0, 16), (5.0, 20), (6.0, 24),
          (1.5, 12), (2.0, 16), (3.0, 24), (4.0, 32),
          (2.0, 24), (3.0, 36), (4.0, 48),
          (1.5, 24), (2.0, 32), (3.0, 48), (4.0, 64),
          (2.0, 12), (2.0, 20), (2.0, 28), (2.0, 36), (2.0, 40), (2.0, 48),
          (2.0, 64), (2.0, 80)]


def main():
    print("=" * 74)
    print("CROSS KERNEL FROM SECOND-ORDER PT -- lattice, nothing fitted")
    print("=" * 74)
    kpw = kappa_pw_pt()
    print(f"  plane-wave kappa (PT, A -> 0): {kpw:+.5f}   "
          f"(closure in platform convention: +0.0349 beta -> {-0.0349/2:+.5f})")
    # the smooth limit
    print(f"  R(q -> 0) = {R(1e-4, 0.0):.5f}   (smooth-space value 2)")
    print("\n  KERNEL R(q) = K(q)/K(0), q along y (qz = 0) and along the diagonal:")
    print("     q/pi    R(q, 0)    R(q, q)")
    for m in range(0, 9):
        qq = m * math.pi / 8
        r0 = 1.0 if m == 0 else R(qq, 0.0)
        r1 = 1.0 if m == 0 else R(qq, qq)
        print(f"     {qq/math.pi:5.3f}   {r0:7.4f}    {r1:7.4f}")
    print("\n  PREDICTIONS for every measured (w, L):")
    print("     w     L    fill      s     2 - s    P3a diag    P3b triads")
    out = []
    for w, L in POINTS:
        fill, s = geometry(L, w)
        fa = F_diag(L, w)
        fb = F_triads(L, w)
        out.append((w, L, fill, s, fa, fb))
        print(f"   {w:4.1f}  {L:3d}  {fill:.5f}  {s:.3f}   {2-s:.3f}     {fa:.3f}       {fb:.3f}",
              flush=True)
    return out


if __name__ == "__main__":
    main()
