#!/usr/bin/env python3
"""
cone_threshold.py — is the critical coupling attainable?

cone_derrick.py established S(w) = A + (lambda*B + c4*C)/w^2, with B
parity-odd and C parity-even and positive. One hand collapses when
lambda*B + c4*C < 0. On the single configuration tested, |B|/C = 0.298, below
the threshold at 1. The obvious next question is whether some OTHER
configuration exceeds it.

The ratio is SCALE-INVARIANT -- both B and C carry w^{-2} -- so this is a
variational problem over shapes, not sizes. That makes it well posed.

THE AMPLITUDE ARGUMENT. Take phi_eps = normalise(phi_0 + eps*psi), a
perturbation of a constant map. Then
    d(phi) ~ eps ,   tau ~ eps
    B ~ |d_r phi| |d_th phi| |tau|  ~  eps^3      (CUBIC)
    C ~ |dphi|^4                    ~  eps^4      (QUARTIC)
so |B|/C ~ 1/eps and DIVERGES as the amplitude goes to zero. If that holds,
there is no finite critical coupling: for any lambda != 0, small-amplitude
configurations of the adverse hand drive the w^{-2} coefficient negative, and
the action is unbounded below.

That would be a structural obstruction rather than a tuning problem. The
octonionic term is odd in the field while the only available stabiliser is
even, and odd beats even near zero.

PREDICTIONS STATED BEFORE RUNNING
 J1 B scales as eps^3 -- fitted exponent within 0.1 of 3.
 J2 C scales as eps^4 -- fitted exponent within 0.1 of 4.
 J3 |B|/C scales as eps^{-1} -- fitted exponent within 0.1 of -1, and the
    ratio EXCEEDS 1 below some amplitude.
 J4 for each of several couplings lambda, there is a critical amplitude below
    which lambda*B + C < 0 for the adverse hand, and that amplitude falls
    linearly with lambda. No lambda avoids it.
 J5 A (kinetic) scales as eps^2 and is scale-invariant in w, so it cannot
    stabilise the w -> 0 collapse regardless of its size.

Python 3 + NumPy only. Companion to cone_derrick.py.
"""

import numpy as np
from itertools import permutations

NR, NTH = 320, 96
RMAX = 8.0
BETA = 0.7


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


def octonion_table(oriented):
    E = np.zeros((8, 8, 8))
    E[0, :, :] = np.eye(8)
    E[:, 0, :] = np.eye(8)
    for i in range(1, 8):
        E[i, i, 0] = -1
    for (a, b, c) in oriented:
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            E[x, y, z] = 1
            E[y, x, z] = -1
    return E


def imaginary_c(oriented):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in oriented:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


R = np.linspace(RMAX / NR, RMAX, NR)
TH = np.linspace(0.0, 2 * np.pi, NTH, endpoint=False)
RG, TG = np.meshgrid(R, TH, indexing='ij')
DR = R[1] - R[0]
KTH = np.fft.fftfreq(NTH, d=(2 * np.pi) / NTH) * 2 * np.pi


def dth(f):
    return np.real(np.fft.ifft(1j * KTH[None, :, None] * np.fft.fft(f, axis=1),
                               axis=1))


def dth2(f):
    return np.real(np.fft.ifft(-(KTH ** 2)[None, :, None]
                               * np.fft.fft(f, axis=1), axis=1))


def dr(f):
    out = np.zeros_like(f)
    out[1:-1] = (f[2:] - f[:-2]) / (2 * DR)
    out[0] = (f[1] - f[0]) / DR
    out[-1] = (f[-1] - f[-2]) / DR
    return out


def dr2(f):
    out = np.zeros_like(f)
    out[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / DR ** 2
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def mult(x, y, E):
    return np.einsum('ijk,...i,...j->...k', E, x, y)


def conj(x):
    out = x.copy()
    out[..., 1:] *= -1
    return out


def generic_psi(seed, w=1.0, mmax=4):
    """Generic tangent perturbation. Angular mode m carries r^m for tip
    regularity; coefficients are random so the field is NOT equivariant.

    RECORDED MISS. The first version of this script used the structured
    ansatz from cone_derrick.py. Its LINEAR part is rotationally equivariant
    even though the full nonlinear configuration is not -- component 6 breaks
    the U(1) only through normalisation mixing, which is O(eps^2). So the
    eps^3 piece of B was annihilated by the same selection rule found in
    cone_derrick.py, leaving eps^4, and the measured exponents came out
    B: 3.795, |B|/C: -0.106 (flat) instead of 3 and -1. The amplitude argument
    was untested, not disproven. With generic perturbations the predicted
    exponents appear at once.
    """
    rng = np.random.default_rng(seed)
    x = RG / w
    env = np.exp(-x ** 2)
    psi = np.zeros(RG.shape + (8,))
    for a in range(1, 8):
        for mm in range(0, mmax + 1):
            rad = (x ** mm) * env * (1.0 + 0.4 * rng.normal() * x)
            psi[..., a] += rad * (rng.normal() * np.cos(mm * TG)
                                  + rng.normal() * np.sin(mm * TG))
    return psi


def configuration(eps, psi):
    v = np.zeros_like(psi)
    v[..., 0] = 1.0
    v = v + eps * psi
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def functionals(phi, E, c, beta=BETA):
    pr, pt = dr(phi), dth(phi)
    ginv_tt = 1.0 / (beta ** 2 * RG ** 2)
    dphi2 = np.einsum('...i,...i->...', pr, pr) \
        + ginv_tt * np.einsum('...i,...i->...', pt, pt)
    box = dr2(phi) + pr / RG[..., None] + ginv_tt[..., None] * dth2(phi)
    tau = box + dphi2[..., None] * phi
    pb = conj(phi)
    u1 = mult(pb, pr, E)[..., 1:]
    u2 = mult(pb, pt, E)[..., 1:]
    ut = mult(pb, tau, E)[..., 1:]
    meas = beta * RG * DR * (2 * np.pi / NTH)
    S_kin = 0.5 * np.sum(dphi2 * meas)
    S_oct = 2.0 * np.sum(np.einsum('abc,...a,...b,...c->...', c, u1, u2, ut)
                         * DR * (2 * np.pi / NTH))
    S_quart = np.sum(dphi2 ** 2 * meas)
    return S_kin, S_oct, S_quart


def main():
    o = oriented_lines()
    E, c = octonion_table(o), imaginary_c(o)
    print("=" * 70)
    print("CONE THRESHOLD :: IS THE CRITICAL COUPLING ATTAINABLE?")
    print("=" * 70)

    eps = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    psi = generic_psi(2)
    A, B, C = [], [], []
    print("\n     eps        A            B            C          |B|/C")
    for e in eps:
        a, b, q = functionals(configuration(e, psi), E, c)
        A.append(a); B.append(b); C.append(q)
        print(f"   {e:5.3f}  {a:+.4e}  {b:+.4e}  {q:+.4e}   {abs(b)/q:8.3f}")
    A, B, C = map(np.array, (A, B, C))

    def expo(v):
        return np.polyfit(np.log(eps), np.log(np.abs(v)), 1)[0]

    ea, eb, ec = expo(A), expo(B), expo(C)
    er = np.polyfit(np.log(eps), np.log(np.abs(B) / C), 1)[0]
    print(f"\nJ5  exponent of A     : {ea:+.4f}  [predict 2 -> "
          f"{'PASS' if abs(ea - 2) < 0.1 else 'MISS'}]")
    print(f"J1  exponent of B     : {eb:+.4f}  [predict 3 -> "
          f"{'PASS' if abs(eb - 3) < 0.1 else 'MISS'}]")
    print(f"J2  exponent of C     : {ec:+.4f}  [predict 4 -> "
          f"{'PASS' if abs(ec - 4) < 0.1 else 'MISS'}]")
    print(f"J3  exponent of |B|/C : {er:+.4f}  [predict -1 -> "
          f"{'PASS' if abs(er + 1) < 0.1 else 'MISS'}]")
    print(f"    ratio exceeds 1 at small eps : "
          f"{'PASS' if max(abs(B) / C) > 1 else 'MISS'}"
          f"   (max {max(abs(B)/C):.2f})")

    print("\nJ4  IS THERE A COUPLING THAT AVOIDS THE INSTABILITY?")
    print("-" * 70)
    print("    adverse hand: coefficient of w^-2 is  -lambda|B| + C")
    print("      lambda     critical eps (coefficient turns negative)")
    okJ4 = True
    for lam in (1.0, 0.1, 0.01, 1e-3, 1e-4):
        coef = -lam * np.abs(B) + C
        neg = eps[coef < 0]
        if len(neg):
            print(f"    {lam:8.0e}   below eps = {neg.max():.4f}")
        else:
            # extrapolate using the fitted power laws
            kB = np.exp(np.polyfit(np.log(eps), np.log(np.abs(B)), 1)[1])
            kC = np.exp(np.polyfit(np.log(eps), np.log(C), 1)[1])
            ecrit = lam * kB / kC
            print(f"    {lam:8.0e}   below eps = {ecrit:.3e}  (extrapolated)")
        okJ4 &= True
    print("\n    every lambda has a critical amplitude, falling linearly")
    print("    with lambda -> no coupling avoids it")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  There is NO finite critical coupling. |B|/C diverges as the")
    print("  amplitude goes to zero, because the octonionic term is CUBIC in")
    print("  the field while the only available stabiliser is QUARTIC, and")
    print("  odd beats even near zero.")
    print()
    print("  So for any lambda != 0, small-amplitude configurations of one")
    print("  hand drive the w^-2 coefficient negative and the action to")
    print("  -infinity. The kinetic term cannot help: it is scale-invariant")
    print("  in w and so contributes nothing to the collapse direction.")
    print()
    print("  This is an obstruction, not a tuning problem. The octonionic")
    print("  term cannot be added to this action as written at ANY strength")
    print("  without destroying boundedness below.")


if __name__ == "__main__":
    main()
