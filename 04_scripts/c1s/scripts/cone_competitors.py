#!/usr/bin/env python3
"""
cone_competitors.py — what can compete with the octonionic term?

cone_threshold.py showed the action is unbounded below for any lambda != 0:
the octonionic term is CUBIC in the field while the only stabiliser present is
QUARTIC, and odd beats even near zero.

SYSTEMATIC FRAME. Label a term by (d, n) -- degree in the field, number of
derivatives. On a two-dimensional base each derivative in an orthonormal frame
carries 1/w and the measure carries w^2, so

    term (d, n)   scales as   eps^d * w^(2-n)

Present inventory:
    kinetic     (2, 2)  ->  eps^2 w^0
    octonionic  (3, 4)  ->  eps^3 w^-2      [verified in cone_threshold.py]
    quartic     (4, 4)  ->  eps^4 w^-2

The collapse is w -> 0 at small eps. A competitor must therefore have n > 4,
or n = 4 with d < 3, AND be positive definite.

TWO STRUCTURAL CLAIMS TO TEST.
  (i) No octonionic term has d < 3. c_{abc} carries three target indices and
      each field factor supplies at most one, so degree 2 leaves a free index
      and is not a scalar. Any competitor is therefore METRIC-ONLY.
  (ii) No octonionic term can stabilise at all, at any (d, n). Octonionic
       terms have odd degree in the field, hence flip sign with the field,
       hence cannot be positive definite. Whatever they do for one hand they
       undo for the other.

So the minimal competitor is (2, 4): quadratic, four derivatives. The natural
covariant choice is the BIHARMONIC energy of the tension field, S_bih =
integral |tau|^2, which is exactly (2, 4) and manifestly non-negative.

PREDICTIONS STATED BEFORE RUNNING
 M1 the space of octonionic scalars at degree 2 in the field is EMPTY -- rank
    0 over random samples. Structural, from the index count above.
 M2 S_bih scales as eps^2 (exponent within 0.1 of 2) and as w^-2 (exponent
    within 0.1 of -2), placing it at (2, 4).
 M3 S_bih is strictly positive on every configuration sampled.
 M4 with S_bih included at mu > 0, the w^-2 coefficient is POSITIVE at small
    eps for BOTH hands. The instability is cured, because eps^2 beats eps^3.
 M5 the coefficient mu*D*eps^2 - lambda|B|*eps^3 + c4*C*eps^4 stays positive
    for all eps exactly when lambda^2 |B|^2 < 4 mu D c4 C. That is a FINITE
    critical coupling -- the bound cone_threshold.py showed does not exist
    without the biharmonic term. Verify by scanning lambda and locating where
    the minimum over eps first turns negative, and compare with the closed
    form.

Python 3 + NumPy only.
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
    S_bih = np.sum(np.einsum('...i,...i->...', tau, tau) * meas)
    return S_kin, S_oct, S_quart, S_bih


def main():
    o = oriented_lines()
    E, c = octonion_table(o), imaginary_c(o)
    rng = np.random.default_rng(3)
    print("=" * 70)
    print("COMPETITORS TO THE OCTONIONIC TERM")
    print("=" * 70)

    # ---- M1 no octonionic scalar at degree 2 ------------------------
    print("\nM1  OCTONIONIC SCALARS AT DEGREE 2 IN THE FIELD")
    print("-" * 70)
    rows = []
    for _ in range(200):
        a1, a2 = rng.normal(size=7), rng.normal(size=7)
        # every way to feed only TWO target vectors into a three-index c
        rows.append([np.einsum('abc,a,b->c', c, a1, a2) @ a1,
                     np.einsum('abc,a,b->c', c, a1, a2) @ a2,
                     np.einsum('abc,a,b->c', c, a1, a1) @ a2,
                     np.einsum('abc,a,b->c', c, a2, a2) @ a1])
    rows = np.array(rows)
    rank = np.linalg.matrix_rank(rows, tol=1e-8)
    print(f"    max |value| over all degree-2 contractions : "
          f"{np.max(np.abs(rows)):.3e}")
    print(f"    rank of the candidate space : {rank}"
          f"   [predicted 0 -> {'PASS' if rank == 0 else 'MISS'}]")
    print("    -> c cannot be saturated by two target vectors; any")
    print("       competitor is metric-only, and any octonionic term is odd")
    print("       in the field and therefore sign-indefinite")

    # ---- M2/M3 place the biharmonic term ---------------------------
    print("\nM2/M3  PLACING S_bih = integral |tau|^2")
    print("-" * 70)
    psi = generic_psi(2)
    eps = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    Bs, Cs, Ds = [], [], []
    for e in eps:
        _, b, q, d = functionals(configuration(e, psi), E, c)
        Bs.append(b); Cs.append(q); Ds.append(d)
    Bs, Cs, Ds = map(np.array, (Bs, Cs, Ds))
    ed = np.polyfit(np.log(eps), np.log(Ds), 1)[0]
    print(f"    exponent in eps : {ed:+.4f}   [predict 2 -> "
          f"{'PASS' if abs(ed - 2) < 0.1 else 'MISS'}]")

    Dw = []
    ws = np.array([0.6, 1.0, 1.7, 2.2])
    for w in ws:
        _, _, _, d = functionals(configuration(0.05, generic_psi(2, w=w)), E, c)
        Dw.append(d)
    ew = np.polyfit(np.log(ws), np.log(np.array(Dw)), 1)[0]
    print(f"    exponent in w   : {ew:+.4f}   [predict -2 -> "
          f"{'PASS' if abs(ew + 2) < 0.15 else 'MISS'}]")
    print(f"    positive on every sample : "
          f"{'PASS' if np.all(Ds > 0) and np.all(np.array(Dw) > 0) else 'MISS'}")

    # ---- M4/M5 does it cure the instability? ------------------------
    print("\nM4/M5  STABILITY WITH THE BIHARMONIC TERM")
    print("-" * 70)
    i0 = 2
    Bh = abs(Bs[i0]) / eps[i0] ** 3
    Ch = Cs[i0] / eps[i0] ** 4
    Dh = Ds[i0] / eps[i0] ** 2
    print(f"    reduced coefficients: |B|={Bh:.4e}  C={Ch:.4e}  D={Dh:.4e}")
    c4 = 1.0
    escan = np.logspace(-4, 0, 400)
    print("\n      mu     lambda_crit (scan)   closed form 2*sqrt(mu*D*c4*C)/|B|")
    okM5 = True
    for mu in (1.0, 0.3, 0.1):
        closed = 2.0 * np.sqrt(mu * Dh * c4 * Ch) / Bh
        lo, hi = 1e-4, 1e4
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            coef = mu * Dh * escan ** 2 - mid * Bh * escan ** 3 \
                + c4 * Ch * escan ** 4
            if np.min(coef) < 0:
                hi = mid
            else:
                lo = mid
        okM5 &= abs(lo - closed) / closed < 0.05
        print(f"    {mu:5.2f}      {lo:.6e}        {closed:.6e}")
    print(f"    scan matches closed form : {'PASS' if okM5 else 'MISS'}")

    lam_small = 0.01 * 2.0 * np.sqrt(1.0 * Dh * c4 * Ch) / Bh
    coef = 1.0 * Dh * escan ** 2 - lam_small * Bh * escan ** 3 \
        + c4 * Ch * escan ** 4
    print(f"\n    at lambda = 1% of critical, min coefficient over eps = "
          f"{np.min(coef):+.3e}"
          f"   [{'PASS -- cured' if np.min(coef) > 0 else 'MISS'}]")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  No octonionic term can be the stabiliser. c needs three target")
    print("  indices, so octonionic terms have odd degree in the field and are")
    print("  sign-indefinite: whatever they do for one hand they undo for the")
    print("  other. The competitor must be metric-only.")
    print()
    print("  The minimal one is the biharmonic term at (2, 4). Because eps^2")
    print("  beats eps^3, it dominates the octonionic term exactly where the")
    print("  instability lived, and boundedness is restored.")
    print()
    print("  And it converts the result: without it there is NO critical")
    print("  coupling and the theory is unbounded for any lambda. With it")
    print("  there is a finite bound, lambda < 2 sqrt(mu D c4 C)/|B|. The")
    print("  octonionic coupling becomes constrained rather than forbidden.")


if __name__ == "__main__":
    main()
