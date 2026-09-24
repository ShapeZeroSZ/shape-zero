#!/usr/bin/env python3
"""
octonionic_term_variation.py — is B.H a total derivative?

cone_coupling_d2.py showed the octonionic candidate

    L = c_{abc} eps^{mu nu} d_mu phi^a d_nu phi^b (box phi)^c  =  B . box phi

is POINTWISE independent of the non-octonionic scalars, but flagged that
pointwise independence is not the same as being an independent term in the
action. A total derivative is pointwise nonzero and still contributes nothing.

WHY INTEGRATION BY PARTS DOES NOT SETTLE IT. On a closed surface,
    int B.box phi = - int d_mu B . d_mu phi
and expanding the right side, the two surviving pieces are
    2[ c(d_1^2 phi, d_2 phi, d_1 phi) + c(d_1 phi, d_2^2 phi, d_2 phi) ]
Both reduce by total antisymmetry of c to -c(d_1 phi, d_2 phi, box phi) = -L,
so the identity closes on itself: int L = int L. IBP is uninformative here,
which is why this needs to be computed rather than argued.

METHOD. Put phi on a periodic torus with spectral (FFT) derivatives. On a
closed manifold the integral of any total derivative vanishes EXACTLY, for
every configuration. So a single configuration with nonzero integral settles
it. Maps T^2 -> R^7 are all contractible, so there is no topological invariant
that could make a nonzero integral non-dynamical.

PREDICTIONS STATED BEFORE RUNNING
 V1 INSTRUMENT CHECK: for a PLANAR configuration -- phi with only two nonzero
    components -- the integral vanishes to machine precision. B is orthogonal
    to both derivative directions, so it leaves the image plane, while box phi
    stays in it. A nonzero answer here would mean the machinery is broken.
 V2 INSTRUMENT CHECK: the IBP identity int B.box phi + int dB.dphi = 0 holds
    to spectral accuracy, confirming the derivatives and quadrature.
 V3 for generic configurations the integral is NONZERO, far above the noise
    floor set by V1. Predicted on the grounds that IBP closes on itself and no
    cancellation mechanism is visible.
 V4 the directional functional derivative is nonzero -- the term contributes
    to the equations of motion, so it is not a boundary term in disguise.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations

N = 64
LBOX = 2.0 * np.pi
KMAX = 6


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


# ---------------------------------------------------------------- spectral
KX = np.fft.fftfreq(N, d=LBOX / N) * 2.0 * np.pi
KY = np.fft.fftfreq(N, d=LBOX / N) * 2.0 * np.pi
KXG, KYG = np.meshgrid(KX, KY, indexing='ij')
CELL = (LBOX / N) ** 2


def d1(f):
    return np.real(np.fft.ifft2(1j * KXG * np.fft.fft2(f)))


def d2(f):
    return np.real(np.fft.ifft2(1j * KYG * np.fft.fft2(f)))


def lap(f):
    return np.real(np.fft.ifft2(-(KXG ** 2 + KYG ** 2) * np.fft.fft2(f)))


def random_field(rng, ncomp=7):
    """Band-limited random periodic field, shape (ncomp, N, N)."""
    out = np.zeros((ncomp, N, N))
    mask = (np.abs(KXG) <= KMAX) & (np.abs(KYG) <= KMAX)
    for a in range(ncomp):
        sp = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * mask
        f = np.real(np.fft.ifft2(sp))
        out[a] = f / np.std(f)
    return out


# ---------------------------------------------------------------- the term
def B_field(phi, c):
    """B^e = 2 c_{abe} d_1 phi^a d_2 phi^b."""
    P1 = np.array([d1(phi[a]) for a in range(7)])
    P2 = np.array([d2(phi[a]) for a in range(7)])
    return 2.0 * np.einsum('abe,axy,bxy->exy', c, P1, P2), P1, P2


def action(phi, c):
    B, _, _ = B_field(phi, c)
    H = np.array([lap(phi[a]) for a in range(7)])
    return np.sum(B * H) * CELL


def ibp_side(phi, c):
    """- integral of d_mu B . d_mu phi."""
    B, P1, P2 = B_field(phi, c)
    dB1 = np.array([d1(B[a]) for a in range(7)])
    dB2 = np.array([d2(B[a]) for a in range(7)])
    return -np.sum(dB1 * P1 + dB2 * P2) * CELL


def main():
    c = imaginary_c(oriented_lines())
    rng = np.random.default_rng(31)
    print("=" * 70)
    print("IS THE OCTONIONIC TERM A TOTAL DERIVATIVE?")
    print("=" * 70)

    # ---- V1 planar configurations -----------------------------------
    print("\nV1  INSTRUMENT :: planar configurations (2 active components)")
    print("-" * 70)
    planar = []
    for t in range(4):
        phi = np.zeros((7, N, N))
        two = random_field(rng, 2)
        phi[0], phi[3] = two[0], two[1]
        planar.append(abs(action(phi, c)))
        print(f"    trial {t}:  |S| = {planar[-1]:.3e}")
    floor = max(planar)
    print(f"    noise floor = {floor:.3e}"
          f"   [predicted ~0 -> {'PASS' if floor < 1e-8 else 'MISS'}]")

    # ---- V2 IBP identity --------------------------------------------
    print("\nV2  INSTRUMENT :: IBP identity  int B.box phi = - int dB.dphi")
    print("-" * 70)
    okV2 = True
    for t in range(3):
        phi = random_field(rng)
        a1, a2 = action(phi, c), ibp_side(phi, c)
        rel = abs(a1 - a2) / max(abs(a1), 1e-30)
        okV2 &= rel < 1e-8
        print(f"    trial {t}:  {a1:+.8e}  vs  {a2:+.8e}   rel {rel:.2e}")
    print(f"    {'PASS' if okV2 else 'MISS'} -- derivatives and quadrature sound")

    # ---- V3 generic configurations ----------------------------------
    print("\nV3  GENERIC CONFIGURATIONS")
    print("-" * 70)
    vals = []
    for t in range(6):
        phi = random_field(rng)
        S = action(phi, c)
        vals.append(S)
        print(f"    trial {t}:  S = {S:+.8e}")
    biggest = max(abs(v) for v in vals)
    print(f"\n    largest |S| = {biggest:.3e}   vs noise floor {floor:.3e}")
    print(f"    ratio = {biggest/max(floor,1e-30):.3e}"
          f"   [predicted >> 1 -> "
          f"{'PASS' if biggest > 1e6 * max(floor, 1e-30) else 'MISS'}]")

    # ---- V4 functional derivative -----------------------------------
    print("\nV4  DIRECTIONAL FUNCTIONAL DERIVATIVE")
    print("-" * 70)
    phi = random_field(rng)
    okV4 = True
    for t in range(3):
        eta = random_field(rng)
        h = 1e-5
        dS = (action(phi + h * eta, c) - action(phi - h * eta, c)) / (2 * h)
        okV4 &= abs(dS) > 1e-6
        print(f"    direction {t}:  dS/dt = {dS:+.8e}")
    print(f"    {'PASS' if okV4 else 'MISS'} -- nonzero variation")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    if biggest > 1e6 * max(floor, 1e-30) and okV4:
        print("  NOT a total derivative. The integral over a closed surface is")
        print("  nonzero and the variation does not vanish, so the term enters")
        print("  the equations of motion. Maps T^2 -> R^7 are contractible, so")
        print("  no topological invariant can be doing the work.")
        print()
        print("  This is the FIRST place in the construction where the")
        print("  octonions contribute something the target metric alone does")
        print("  not. The quartic sector lost its octonionic term to the")
        print("  composition identity; this order keeps it.")
    else:
        print("  Total derivative, or below resolution -- see the failing line.")


if __name__ == "__main__":
    main()
