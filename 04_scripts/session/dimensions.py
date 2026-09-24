#!/usr/bin/env python3
"""
dimensions.py — dimensional arithmetic in code, not in prose

WHY. The same quantity (Newton's G in this model) was claimed derived, then
claimed dimensionally impossible, and both were wrong. Both were exponent
arithmetic written by hand in prose -- the same class of error as the
factor-of-ten kappa, and the same fix applies: numbers get emitted by code.

Dimensional analysis LOOKS like reasoning rather than measurement, which is why
it escaped the Phase 0 rule. It is measurement.

CONVENTION: c = 1 throughout, so [L] = [T] and the independent dimensions are
L and M. Every quantity is stored as (L_power, M_power).
"""

from fractions import Fraction as F


class Dim:
    """A dimension as (L, M) exponents. c = 1, so T is folded into L."""

    def __init__(self, L=0, M=0):
        self.L, self.M = F(L), F(M)

    def __mul__(self, o):
        return Dim(self.L + o.L, self.M + o.M)

    def __truediv__(self, o):
        return Dim(self.L - o.L, self.M - o.M)

    def __pow__(self, k):
        return Dim(self.L * F(k), self.M * F(k))

    def __eq__(self, o):
        return self.L == o.L and self.M == o.M

    def __repr__(self):
        if self.L == 0 and self.M == 0:
            return "dimensionless"
        p = []
        if self.L:
            p.append(f"L^{self.L}" if self.L != 1 else "L")
        if self.M:
            p.append(f"M^{self.M}" if self.M != 1 else "M")
        return " ".join(p)


L = Dim(1, 0)
M = Dim(0, 1)
ONE = Dim(0, 0)


def G_dim(D):
    """[G_D] in D spacetime dimensions, c = 1.

    Anchor: the Schwarzschild radius in D dimensions satisfies
    r_s^(D-3) ~ G_D M, so [G_D] = L^(D-3) / M.
    """
    return L ** (D - 3) / M


def check(name, got, want):
    ok = got == want
    print(f"    {'ok ' if ok else '** '}{name:34s} {str(got):16s} "
          f"{'== ' if ok else '!= '}{want}")
    return ok


def main():
    print("=" * 66)
    print("DIMENSIONAL CHECKS, c = 1")
    print("=" * 66)
    allok = True

    print("\n  ANCHOR: Schwarzschild r_s = 2 G M in D=4 gives [G] = L/M")
    allok &= check("[G_4]", G_dim(4), L / M)

    print("\n  KALUZA-KLEIN:  G_4 = G_D / Vol(fibre),  fibre dim f = D - 4")
    for f, nm in ((1, "S^1"), (2, "S^2"), (4, "CP^2"), (7, "S^7")):
        D = 4 + f
        got = G_dim(D) / (L ** f)
        allok &= check(f"G_{D} / Vol({nm})", got, G_dim(4))

    print("\n  hbar from integrality: hbar = mu ell_f^2 / (T n), c=1 so T ~ L")
    hbar = M * (L ** 2) / L
    allok &= check("[mu ell_f^2 / T]", hbar, M * L)
    print(f"       (action in c=1 units is M L, correct)")

    print("\n  THE MODEL'S SCALES: omega [1/L], ell [L], mu [M], ell_f [L]")
    print("  every dimensionful constant is a pure number times a monomial:")
    for nm, want in (("G_4", G_dim(4)), ("G_8", G_dim(8)),
                     ("hbar", M * L), ("Lambda", L ** -2),
                     ("e^2 (Gaussian, c=1)", M * L)):
        # solve for exponents a,b with L^a M^b = want, using ell and mu
        print(f"    {nm:20s} = c * ell^{want.L} mu^{want.M}")

    print("\n  G_4 FROM THE REDUCTION, with ell and ell_f kept DISTINCT:")
    print("     G_8 = c_8 ell^5 / mu        [checked below]")
    allok &= check("[c_8 ell^5 / mu]", (L ** 5) / M, G_dim(8))
    print("     G_4 = G_8 / (2 pi^2 ell_f^4)")
    allok &= check("[G_8 / ell_f^4]", ((L ** 5) / M) / (L ** 4), G_dim(4))
    print("\n     -> G_4 = [c_8/(2 pi^2)] (ell^5/ell_f^4) / mu")
    print("        dimensionally consistent whether or not ell_f = ell.")

    print("\n" + "=" * 66)
    print(f"  RESULT: {'ALL CHECKS PASS' if allok else 'A CHECK FAILED'}")
    print("=" * 66)
    print("\n  Use this module for any dimensional claim. Do not write")
    print("  exponents by hand in prose -- that produced two opposite and")
    print("  equally wrong verdicts on the same quantity.")


if __name__ == "__main__":
    main()
