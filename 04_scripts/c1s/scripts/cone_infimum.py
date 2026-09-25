#!/usr/bin/env python3
"""
cone_infimum.py — is the critical coupling bound real or vacuous?

cone_competitors.py gave  lambda_crit[phi] = 2 sqrt(mu D[phi] c4 C[phi]) / |B[phi]|
on ONE configuration. The constraint that matters is the infimum over all
configurations. If that infimum is zero the rescue fails and lambda = 0
returns; if positive, the octonionic coupling is genuinely bounded rather than
forbidden.

Equivalently, maximise

    R[phi] = |B| / sqrt(C * D) ,        lambda_crit = 2 sqrt(mu c4) / R.

R IS INVARIANT UNDER BOTH SYMMETRIES. B ~ eps^3, C ~ eps^4, D ~ eps^2 so
sqrt(CD) ~ eps^3 and R is amplitude-invariant; all three carry w^-2 so R is
dilation-invariant. The search is over shapes modulo those two, which is what
makes it well posed.

THE ANALYTIC BOUND. In an orthonormal frame, with u1 = conj(phi) d_r phi,
u2 = conj(phi) d^_theta phi, ut = conj(phi) tau,

    B = 2 integral c(u1, u2, ut) dmu

and three elementary steps bound it:
    (1) |c(a,b,d)| = |(a x b).d| <= |a x b| |d| <= |a||b||d|
    (2) 2|u1||u2| <= |u1|^2 + |u2|^2 = |dphi|^2                    (AM-GM)
    (3) integral |dphi|^2 |tau| <= sqrt(integral |dphi|^4) sqrt(integral |tau|^2)
                                = sqrt(C) sqrt(D)                  (Cauchy-Schwarz)
Hence |B| <= sqrt(C D), so R <= 1 and

    inf lambda_crit  >=  2 sqrt(mu c4)  >  0.

The bound would then be REAL, not vacuous, and independent of configuration.
None of the three steps uses small amplitude, so this holds for every
configuration, not only perturbative ones.

Equality needs all three saturated at once: d_r phi orthogonal to
d^_theta phi in the target AND of equal length (a conformal map), tau
everywhere parallel to the octonionic normal u1 x u2, and |tau| proportional
to |dphi|^2 pointwise. That is restrictive, so the achieved supremum is
expected well below 1.

PREDICTIONS STATED BEFORE RUNNING
 P1 the pointwise inequality |c(a,b,d)| <= |a||b||d| holds for random vectors,
    with the ratio reaching 1 only when a is orthogonal to b and d lies along
    a x b. Instrument check on the first step of the chain.
 P2 R <= 1 for EVERY configuration sampled -- random and optimised. A single
    violation would refute the analytic bound and is the thing to watch for.
 P3 random search over shapes finds max R well below 1; predict < 0.5, since
    the three equality conditions are simultaneous and restrictive.
 P4 hill-climbing improves on the random best but still returns R < 1.
 P5 therefore inf lambda_crit = 2 sqrt(mu c4) / sup R is strictly positive,
    and with the achieved sup R it is 2 sqrt(mu c4) / R*.

Python 3 + NumPy only.

(Cone deficit renamed beta -> zeta on 2026-09-25, to free beta for the lattice
gyroscopic coupling; the code variable keeps the name beta/BETA.)
"""

import numpy as np
from itertools import permutations

NR, NTH = 160, 64
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


R_ = np.linspace(RMAX / NR, RMAX, NR)
TH = np.linspace(0.0, 2 * np.pi, NTH, endpoint=False)
RG, TG = np.meshgrid(R_, TH, indexing='ij')
DR = R_[1] - R_[0]
KTH = np.fft.fftfreq(NTH, d=(2 * np.pi) / NTH) * 2 * np.pi
MEAS = BETA * RG * DR * (2 * np.pi / NTH)
GINV = 1.0 / (BETA ** 2 * RG ** 2)


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


# ---------------------------------------------------------------- shapes
MMAX, NPROF = 3, 2
NPAR = 7 * (MMAX + 1) * 2 * NPROF


def build(coef, eps=0.3, w=1.0):
    x = RG / w
    env = np.exp(-x ** 2)
    psi = np.zeros(RG.shape + (8,))
    i = 0
    for a in range(1, 8):
        for m in range(MMAX + 1):
            for k in range(NPROF):
                rad = (x ** m) * env * (x ** k)
                psi[..., a] += rad * (coef[i] * np.cos(m * TG)
                                      + coef[i + 1] * np.sin(m * TG))
                i += 2
    v = np.zeros_like(psi)
    v[..., 0] = 1.0
    v = v + eps * psi / max(np.max(np.abs(psi)), 1e-12)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def BCD(phi, E, c):
    pr, pt = dr(phi), dth(phi)
    dphi2 = np.einsum('...i,...i->...', pr, pr) \
        + GINV * np.einsum('...i,...i->...', pt, pt)
    box = dr2(phi) + pr / RG[..., None] + GINV[..., None] * dth2(phi)
    tau = box + dphi2[..., None] * phi
    pb = conj(phi)
    u1 = mult(pb, pr, E)[..., 1:]
    u2 = mult(pb, pt, E)[..., 1:]
    ut = mult(pb, tau, E)[..., 1:]
    B = 2.0 * np.sum(np.einsum('abc,...a,...b,...c->...', c, u1, u2, ut)
                     * DR * (2 * np.pi / NTH))
    C = np.sum(dphi2 ** 2 * MEAS)
    D = np.sum(np.einsum('...i,...i->...', tau, tau) * MEAS)
    return B, C, D


def ratio(coef, E, c):
    B, C, D = BCD(build(coef), E, c)
    if C <= 0 or D <= 0:
        return 0.0
    return abs(B) / np.sqrt(C * D)


def main():
    o = oriented_lines()
    E, c = octonion_table(o), imaginary_c(o)
    rng = np.random.default_rng(9)
    print("=" * 70)
    print("INFIMUM OF THE CRITICAL COUPLING")
    print("=" * 70)

    # ---- P1 the pointwise step -------------------------------------
    print("\nP1  |c(a,b,d)| <= |a||b||d|   [first step of the chain]")
    print("-" * 70)
    worst = 0.0
    for _ in range(4000):
        a, b, d = (rng.normal(size=7) for _ in range(3))
        lhs = abs(np.einsum('abc,a,b,c->', c, a, b, d))
        worst = max(worst, lhs / (np.linalg.norm(a) * np.linalg.norm(b)
                                  * np.linalg.norm(d)))
    a = rng.normal(size=7)
    b = rng.normal(size=7)
    b = b - (b @ a) / (a @ a) * a
    d = np.einsum('abc,a,b->c', c, a, b)
    sat = abs(np.einsum('abc,a,b,c->', c, a, b, d)) / (
        np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(d))
    print(f"    max ratio over random triples : {worst:.6f}"
          f"   [<= 1 -> {'PASS' if worst <= 1 + 1e-9 else 'MISS'}]")
    print(f"    at the saturating configuration : {sat:.6f}"
          f"   [-> 1 -> {'PASS' if abs(sat - 1) < 1e-9 else 'MISS'}]")

    # ---- P2/P3 random search ----------------------------------------
    print("\nP2/P3  RANDOM SEARCH OVER SHAPES")
    print("-" * 70)
    best, bestc = 0.0, None
    viol = 0
    for t in range(500):
        coef = rng.normal(size=NPAR)
        r = ratio(coef, E, c)
        if r > 1 + 1e-9:
            viol += 1
        if r > best:
            best, bestc = r, coef.copy()
    print(f"    configurations sampled : 500  ({NPAR} shape parameters)")
    print(f"    violations of R <= 1   : {viol}"
          f"   [predicted 0 -> {'PASS' if viol == 0 else 'MISS'}]")
    print(f"    best random R          : {best:.6f}"
          f"   [predicted < 0.5 -> {'PASS' if best < 0.5 else 'MISS'}]")

    # ---- P4 hill climb ----------------------------------------------
    print("\nP4  HILL CLIMB FROM THE BEST RANDOM START")
    print("-" * 70)
    cur, curr = bestc.copy(), best
    step = 0.6
    for it in range(1200):
        cand = cur.copy()
        k = rng.integers(1, 8)
        idx = rng.choice(NPAR, size=k, replace=False)
        cand[idx] += step * rng.normal(size=k)
        r = ratio(cand, E, c)
        if r > curr:
            cur, curr = cand, r
        if (it + 1) % 300 == 0:
            print(f"    iter {it+1:4d} : R = {curr:.6f}   (step {step:.3f})")
            step *= 0.6
    print(f"\n    achieved sup R : {curr:.6f}"
          f"   [< 1 -> {'PASS' if curr < 1 else 'MISS'}]")

    # ---- P5 the infimum ---------------------------------------------
    print("\nP5  THE RESULTING BOUND")
    print("-" * 70)
    print("    lambda_crit = 2 sqrt(mu c4) / R")
    print(f"    analytic   : R <= 1        ->  inf lambda_crit >= "
          f"2 sqrt(mu c4)")
    print(f"    achieved   : R* = {curr:.6f}  ->  inf lambda_crit <= "
          f"{2/curr:.4f} sqrt(mu c4)")
    for mu, c4 in ((1.0, 1.0), (0.1, 1.0), (1.0, 0.1)):
        print(f"      mu={mu:4.2f}, c4={c4:4.2f} :  lambda_crit in "
              f"[{2*np.sqrt(mu*c4):.4f}, {2*np.sqrt(mu*c4)/curr:.4f}]")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The bound is REAL, not vacuous. R = |B|/sqrt(CD) is bounded by 1")
    print("  analytically -- cross product, AM-GM, Cauchy-Schwarz -- so")
    print("  inf lambda_crit >= 2 sqrt(mu c4) > 0 for EVERY configuration, with")
    print("  no smallness assumption anywhere in the chain.")
    print()
    print("  So lambda = 0 is NOT forced. The octonionic coupling is bounded")
    print("  above by a quantity fixed entirely by the two metric-only")
    print("  couplings, and the bound does not degenerate over shapes.")


if __name__ == "__main__":
    main()
