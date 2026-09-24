#!/usr/bin/env python3
"""
octonionic_term_kernel.py — what is the term actually sensitive to?

The octonionic term is cubic and parity-odd. Before asking what it does to
solutions, ask what it can see at all. In momentum space, for phi^a with modes
A^a_k,

    S ~ sum_{k1+k2+k3=0}  c_{abc} k1^x k2^y |k3|^2  A^a_{k1} A^b_{k2} A^c_{k3}

and since c is antisymmetric in (a,b), only the part of k1^x k2^y antisymmetric
under (k1,a) <-> (k2,b) survives. Antisymmetrising gives

    M(k1,k2,k3)  =  (k1 x k2) |k3|^2 ,      k1 x k2 = k1^x k2^y - k1^y k2^x

the two-dimensional cross product. That factor vanishes whenever the momenta
are COLLINEAR, which kills whole classes of configuration outright.

WHY THIS MATTERS FOR THE CONE. A purely radial configuration on the cone has
gradients pointing along r everywhere, so d_1 phi and d_2 phi are everywhere
parallel as base vectors and the cross product vanishes pointwise. The term is
therefore blind to the entire radial sector -- and the cone's deficit is purely
ANGULAR. So the one structure the term can see is the one the cone actually
carries.

PREDICTIONS STATED BEFORE RUNNING
 K1 a configuration whose Fourier support lies on a single line through the
    origin in k-space (all momenta collinear) gives S = 0 to machine
    precision, for arbitrary target components.
 K2 a configuration depending on one base coordinate only gives S = 0.
    Weaker than K1 and expected to be exactly zero for the trivial reason
    that d_2 phi vanishes.
 K3 ANY purely radial configuration phi^a = f^a(r), with SEVEN INDEPENDENT
    radial profiles, gives S = 0 exactly. Reason: d_1 phi^a = f^a'(r) x/r and
    d_2 phi^b = f^b'(r) y/r, so c_{abc} d_1 phi^a d_2 phi^b carries
    f^a' f^b' (xy/r^2), symmetric in (a,b) against an antisymmetric c.
    This is the prediction that matters -- it is not trivial, since the seven
    profiles are unrelated.
 K4 a configuration with genuine angular structure gives S nonzero, far above
    the floor set by K1-K3.
 K5 for a three-mode configuration the sign of S tracks the sign of the cross
    product k1 x k2, and S passes through zero as the momenta become
    collinear.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations

N = 64
LBOX = 2.0 * np.pi
CELL = (LBOX / N) ** 2
KX = np.fft.fftfreq(N, d=LBOX / N) * 2.0 * np.pi
KXG, KYG = np.meshgrid(KX, KX, indexing='ij')
XS = np.linspace(0, LBOX, N, endpoint=False)
XG, YG = np.meshgrid(XS, XS, indexing='ij')


def d1(f):
    return np.real(np.fft.ifft2(1j * KXG * np.fft.fft2(f)))


def d2(f):
    return np.real(np.fft.ifft2(1j * KYG * np.fft.fft2(f)))


def lap(f):
    return np.real(np.fft.ifft2(-(KXG ** 2 + KYG ** 2) * np.fft.fft2(f)))


def oriented_lines():
    LINES = [tuple(sorted(((i + s - 1) % 7) + 1 for s in (0, 1, 3)))
             for i in range(1, 8)]
    used = {p: set() for p in range(1, 8)}
    assign = {}

    def bt(li):
        if li == len(LINES):
            return True
        Ln = LINES[li]
        for perm in permutations(range(3)):
            if all(perm[j] not in used[Ln[j]] for j in range(3)):
                for j in range(3):
                    used[Ln[j]].add(perm[j])
                assign[Ln] = perm
                if bt(li + 1):
                    return True
                for j in range(3):
                    used[Ln[j]].discard(perm[j])
                del assign[Ln]
        return False

    bt(0)
    return [tuple(Ln[assign[Ln].index(r)] for r in (0, 1, 2)) for Ln in LINES]


def imaginary_c(oriented):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in oriented:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


def action(phi, c):
    P1 = np.array([d1(phi[a]) for a in range(7)])
    P2 = np.array([d2(phi[a]) for a in range(7)])
    H = np.array([lap(phi[a]) for a in range(7)])
    return 2.0 * np.sum(np.einsum('abc,axy,bxy,cxy->xy', c, P1, P2, H)) * CELL


def modes_on_line(rng, direction, nmodes=5):
    """Field whose Fourier support lies on one line through the origin.

    Direction components must be INTEGERS and must NOT be normalised -- on a
    2*pi-periodic grid only integer wavevectors are periodic, and a normalised
    direction produces non-periodic modes that alias across the whole k-plane,
    destroying collinearity. That was the K1 miss on the first run.
    """
    dx, dy = direction
    phi = np.zeros((7, N, N))
    for a in range(7):
        for m in range(1, nmodes + 1):
            amp, ph = rng.normal(), rng.uniform(0, 2 * np.pi)
            phi[a] += amp * np.cos(m * (dx * XG + dy * YG) + ph)
    return phi


def radial_field(rng):
    """phi^a = f^a(r) with seven INDEPENDENT smooth radial profiles."""
    x0, y0 = LBOX / 2, LBOX / 2
    r = np.sqrt((XG - x0) ** 2 + (YG - y0) ** 2)
    phi = np.zeros((7, N, N))
    for a in range(7):
        for m in range(1, 4):
            phi[a] += rng.normal() * np.exp(-((r - 0.6 * m) ** 2) / 0.5)
    return phi


def angular_field(rng, mmax=3):
    """Angular structure with a genuinely 7-dimensional target image.

    The first run gave every component the same envelope and the same angular
    mode, so the image spanned only a 2-plane of the target and c_{abc} -- which
    needs THREE independent target directions -- annihilated it. Independent
    coefficients and phases per (component, mode) fix that.
    """
    x0, y0 = LBOX / 2, LBOX / 2
    r = np.sqrt((XG - x0) ** 2 + (YG - y0) ** 2) + 1e-9
    th = np.arctan2(YG - y0, XG - x0)
    phi = np.zeros((7, N, N))
    for a in range(7):
        for m in range(1, mmax + 1):
            env = np.exp(-(r - 0.7 * m) ** 2 / 0.5)
            phi[a] += env * rng.normal() * np.cos(m * th
                                                  + rng.uniform(0, 2 * np.pi))
    return phi


def three_mode(rng, k1, k2):
    k3 = (-k1[0] - k2[0], -k1[1] - k2[1])
    phi = np.zeros((7, N, N))
    for a in range(7):
        for k in (k1, k2, k3):
            amp, ph = rng.normal(), rng.uniform(0, 2 * np.pi)
            phi[a] += amp * np.cos(k[0] * XG + k[1] * YG + ph)
    return phi


def main():
    c = imaginary_c(oriented_lines())
    rng = np.random.default_rng(77)
    print("=" * 70)
    print("OCTONIONIC TERM :: WHAT CAN IT SEE?")
    print("=" * 70)

    print("\nK1  FOURIER SUPPORT ON ONE LINE THROUGH THE ORIGIN")
    print("-" * 70)
    vals = []
    for d in ((1, 0), (0, 1), (1, 1), (2, -1), (3, 1)):
        S = action(modes_on_line(rng, d), c)
        vals.append(abs(S))
        print(f"    direction {str(d):8s} : S = {S:+.6e}")
    floor = max(vals)
    print(f"    floor = {floor:.3e}"
          f"   [predicted 0 -> {'PASS' if floor < 1e-8 else 'MISS'}]")

    print("\nK2  DEPENDS ON ONE BASE COORDINATE ONLY")
    print("-" * 70)
    phi = np.zeros((7, N, N))
    for a in range(7):
        phi[a] = np.cos(3 * XG + rng.uniform(0, 6)) + 0.4 * np.cos(XG)
    S = action(phi, c)
    print(f"    S = {S:+.3e}   [{'PASS' if abs(S) < 1e-10 else 'MISS'}]")

    print("\nK3  PURELY RADIAL, SEVEN INDEPENDENT PROFILES")
    print("-" * 70)
    rvals = []
    for t in range(4):
        S = action(radial_field(rng), c)
        rvals.append(abs(S))
        print(f"    trial {t} : S = {S:+.6e}")
    print(f"    max |S| = {max(rvals):.3e}"
          f"   [predicted 0 -> {'PASS' if max(rvals) < 1e-8 else 'MISS'}]")
    print("    -> the term is BLIND to the entire radial sector")

    print("\nK4  GENUINE ANGULAR STRUCTURE")
    print("-" * 70)
    avals = []
    for m in (2, 3, 4, 5):
        S = action(angular_field(rng, m), c)
        avals.append(abs(S))
        print(f"    modes up to m = {m} : S = {S:+.6e}")
    big = max(avals)
    print(f"    max |S| = {big:.3e}  vs radial floor {max(rvals):.3e}"
          f"   [{'PASS' if big > 1e6 * max(max(rvals), 1e-30) else 'MISS'}]")

    print("\nK5  SIGN TRACKS THE CROSS PRODUCT k1 x k2")
    print("-" * 70)
    print("      angle      k1 x k2        S")
    k1 = (3.0, 0.0)
    okK5 = True
    prev = None
    for deg in (-60, -30, -5, 0, 5, 30, 60):
        th = np.deg2rad(deg)
        k2 = (2.0 * np.cos(th), 2.0 * np.sin(th))
        cross = k1[0] * k2[1] - k1[1] * k2[0]
        r2 = np.random.default_rng(5)
        S = action(three_mode(r2, k1, k2), c)
        print(f"    {deg:+4d} deg   {cross:+8.4f}   {S:+.6e}")
        if deg == 0:
            okK5 &= abs(S) < 1e-8
    print(f"    collinear case vanishes : {'PASS' if okK5 else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The kernel carries a two-dimensional cross product, so the term")
    print("  is blind to any configuration whose base gradients are everywhere")
    print("  parallel. That includes the ENTIRE radial sector, with all seven")
    print("  profiles free.")
    print()
    print("  On the cone this is the useful half of a restriction. The deficit")
    print("  is a purely angular defect, and angular structure is precisely")
    print("  what the term does see. The octonionic sector cannot couple to")
    print("  radial physics at all, and can only couple to the part of the")
    print("  cone that carries its deficit.")


if __name__ == "__main__":
    main()
