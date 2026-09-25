#!/usr/bin/env python3
"""
jcompat_effects.py -- closed-form quantities recorded at the gyroscopic ratio
kappa = 0.5 that move if kappa is raised to the J-compatibility bound
kappa* = 2c/sqrt(K + 2c) (jcompat_kappa.py), evaluated at the model's packet
wavenumber k = pi/2, c = 1, K = sqrt5.

  omega, v_g        : the packet branch w^2 + kappa w = Q(k), v_g = 2c sin k/(2w + kappa)
  Larmor split      : the two single-node frequencies differ by exactly kappa
  P-3 coefficient C : 04_scripts/platform/phi_gauge_precession.py's C_predicted()
                      (spinor self-precession, "simulation instance C = -0.0896",
                      01_source/shape_zero_predictions_v1.md P-3), same formula
  k_c               : the wavenumber below which the opposite-chirality channel is closed

usage:  python3 jcompat_effects.py
"""
import math

SQ5 = math.sqrt(5)
C = 1.0
K0 = math.pi / 2


def omega(k, kap):
    return 0.5 * (-kap + math.sqrt(kap * kap + 4 * (SQ5 + 2 * C * (1 - math.cos(k)))))


def P3(kap):
    w = omega(K0, kap)
    L = lambda K, W: -W * W - kap * W + SQ5 + 2 * C * (1 - math.cos(K))
    return (-1 / SQ5 - 1 / (4 * L(2 * K0, 2 * w)) - 1 / (4 * L(-2 * K0, -2 * w))) / (2 * w + kap)


def k_c(kap):
    lo, hi = 1e-9, math.pi
    for _ in range(100):
        m = 0.5 * (lo + hi)
        x = C * (1 - math.cos(m))
        lo, hi = (m, hi) if x / math.sqrt(SQ5 + x) < kap else (lo, m)
    return lo


def main():
    ks = 2 * C / math.sqrt(SQ5 + 2 * C)
    print("kappa     omega(pi/2)  v_g(pi/2)  Larmor split  P-3 C       k_c (rad)")
    for kap in (0.5, ks, 1.0):
        w = omega(K0, kap)
        kc = k_c(kap)
        kcs = f"{kc:.4f}" if kc < math.pi - 1e-6 else "pi (whole zone)"
        print(f"{kap:.6f}  {w:.6f}     {2 * C * math.sin(K0) / (2 * w + kap):.6f}   {kap:.6f}      "
              f"{P3(kap):+.6f}   {kcs}")
    print(f"\nkappa* = 2c/sqrt(K + 2c) = {ks:.9f};  2 phi^(-3/2) = {2 * ((1 + SQ5) / 2) ** -1.5:.9f}"
          "  (2 + sqrt5 = phi^3)")
    print(f"at kappa = 1, omega(pi/2) = {omega(K0, 1.0):.12f} = phi  (w^2 + w = 2 + sqrt5 = phi^3)")


if __name__ == "__main__":
    main()
