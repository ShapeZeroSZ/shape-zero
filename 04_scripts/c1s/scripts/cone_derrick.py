#!/usr/bin/env python3
"""
cone_derrick.py — does the theory have a stable size, and does it pick a hand?

Before solving the field equations on the cone, ask whether a solution can
exist. The octonionic term is cubic with a second derivative, so in two
dimensions it carries conformal weight e^{-2s} while the kinetic term is
scale-invariant. Under a dilation of the configuration by width w:

    S_kin   ~  w^0     (2D conformal invariance)
    S_oct   ~  w^{-2}  (see the recorded miss below -- NOT w^{-1})
    S_quart ~  w^{-2}  (four derivatives)
    S_apex  ~  w^{-2}  (curvature delta times |dphi|^2 at the tip)

So  S(w) = A + (B + C)/w^2, with B parity-odd and C parity-even.

RECORDED MISS, AND IT CHANGES THE CONCLUSION. This script first predicted
S_oct ~ w^{-1}, counting the term as three derivatives. Measured exponent:
-2.0016, a clean power law, not noise. Two errors: the TENSION FIELD carries
two derivatives, not one, so the term is quartic in derivatives; and under
radial dilation on a cone the ANGULAR derivative does not scale at all. In
conformal-weight terms the octonionic term has two inverse metrics -- one from
eps^{mu nu} = eps~/sqrt(g), one from the Laplacian in tau -- exactly like the
quartic. The two are DEGENERATE under dilation.

Consequence: there is no Derrick minimum. dS/dw = -2(B+C)/w^3 vanishes only in
the marginal case B + C = 0. Otherwise the configuration either expands
without bound (B + C > 0) or collapses (B + C < 0). The parity content
survives in a different and sharper form: the two hands have B + C and
-B + C, so when |B| > C exactly ONE hand collapses while the other expands.
That is a parity-selective instability THRESHOLD at |B| = C, not a stable
soliton.

CONSEQUENCE. The octonionic term ALONE is unbounded below: if B < 0, shrinking
the configuration drives S to -infinity. There is no minimiser, and any PDE
solve would be chasing a collapse. With C > 0 the quartic and apex terms
stabilise it, and dS/dw = 0 gives a finite equilibrium size

    w* = -2C/B ,   which is positive only when B < 0,

and the second derivative there is -B/w*^3 > 0, a genuine minimum.

THE PARITY CONSEQUENCE, which is the point. B is parity-odd (verified in
octonionic_term_parity.py: the term flips sign under theta -> -theta) while A
and C are parity-even. So for any configuration, its mirror image has the same
A and C and the opposite B. Exactly ONE of the two hands has B < 0 and admits
a stable size; the other has no stationary point and unwinds. The theory does
not merely violate parity in its equations -- it selects a handedness for
stable structure.

SETUP. Target is S^7 = unit octonions, which is Spin(7)/G2 as established in
sigma_topology.py. Tangent vectors at phi are identified with Im(O) by left
multiplication: u |-> conj(phi) * u, an isometry taking T_phi S^7 to Im(O)
since |phi| = 1. Tension field for a sphere target is tau = box phi +
|dphi|^2 phi. Cone metric ds^2 = dr^2 + zeta^2 r^2 dtheta^2.

PREDICTIONS STATED BEFORE RUNNING
 H1 S_kin is independent of w -- 2D conformal invariance, fitted exponent
    within 0.05 of 0.
 H2 S_oct scales as w^{-1}, fitted exponent within 0.05 of -1.
    [MISSED -- see above. True exponent -2, degenerate with the quartic.]
 H3 S_quart scales as w^{-2}, fitted exponent within 0.05 of -2.
 H4 under theta -> -theta, S_oct flips sign exactly while S_kin and S_quart
    are unchanged.
 H5 [SUPERSEDED by the corrected scaling] no interior minimum exists; the
    combined coefficient B + C decides expansion versus collapse.
 H6 the two hands carry B + C and -B + C, so they differ in fate only when
    |B| > C. Report the measured ratio |B|/C, which is the control parameter.

Python 3 + NumPy only.

(Cone deficit renamed beta -> zeta on 2026-09-25, to free beta for the lattice
gyroscopic coupling; the code variable keeps the name beta/BETA.)
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


# ---------------------------------------------------------------- grid
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


# ---------------------------------------------------------------- field
def configuration(w, mirror=False):
    """Smooth S^7-valued configuration of width w with angular structure.

    REGULARITY AT THE TIP. On a cone the angular gradient enters as
    |d_theta phi|^2 / (zeta^2 r^2), so a component carrying angular mode m must
    vanish like r^m at the origin or the kinetic energy diverges there. The
    first version of this function used cos(theta) components that survived to
    r = 0; the resulting cutoff-dependent divergence dominated every functional
    and destroyed scale invariance. Each mode now carries its required r^m.

    All radial dependence is a function of r/w, so varying w is a genuine
    dilation of the cone, which is itself scale-invariant.
    """
    x = RG / w
    env = np.exp(-x ** 2)
    sgn = -1.0 if mirror else 1.0
    v = np.zeros(RG.shape + (8,))
    v[..., 0] = 1.0
    v[..., 1] = 0.9 * x * env * np.cos(TG)                    # m = 1, ~ r
    v[..., 2] = 0.9 * x * env * np.sin(sgn * TG)              # m = 1, ~ r
    v[..., 3] = 0.7 * env                                     # m = 0
    v[..., 4] = 0.6 * x ** 2 * env * np.cos(2 * TG)           # m = 2, ~ r^2
    v[..., 5] = 0.5 * x ** 2 * env * np.sin(sgn * 2 * TG)     # m = 2, ~ r^2
    v[..., 6] = 0.55 * x ** 2 * env ** 2                      # m = 0, see below
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

# NOTE on component 6. Without it the configuration is rotationally
# equivariant -- theta -> theta + alpha is undone by a target rotation acting
# on the (1,2) and (4,5) planes -- and the octonionic density then lands in a
# nontrivial U(1) sector, oscillating in theta with zero mean at EVERY radius.
# The pointwise density was 0.54 while the integral was 3e-16. Component 6
# breaks the equivariance and opens the trivial-sector channel. This was a
# property of the test configuration, not of the term.


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
    # eps^{r theta} = 1/sqrt(g) cancels the sqrt(g) in the measure
    # eps^{r theta} = 1/sqrt(g), which cancels the sqrt(g) in the measure
    oct_density = np.einsum('abc,...a,...b,...c->...', c, u1, u2, ut)
    S_oct = 2.0 * np.sum(oct_density * DR * (2 * np.pi / NTH))
    S_quart = np.sum(dphi2 ** 2 * meas)          # (tr M)^2, positive definite
    return S_kin, S_oct, S_quart


def fit_exponent(ws, vals):
    m = np.abs(vals) > 0
    return np.polyfit(np.log(ws[m]), np.log(np.abs(vals[m])), 1)[0]


def main():
    oriented = oriented_lines()
    E, c = octonion_table(oriented), imaginary_c(oriented)
    print("=" * 70)
    print("CONE DERRICK :: STABLE SIZE AND CHIRALITY SELECTION")
    print("=" * 70)
    print(f"  cone deficit parameter zeta = {BETA}")

    ws = np.array([0.6, 0.8, 1.0, 1.3, 1.7, 2.2])
    K, O, Q = [], [], []
    print("\n     w       S_kin         S_oct         S_quart")
    for w in ws:
        k, o, q = functionals(configuration(w), E, c)
        K.append(k); O.append(o); Q.append(q)
        print(f"   {w:4.2f}   {k:+.5e}  {o:+.5e}  {q:+.5e}")
    K, O, Q = map(np.array, (K, O, Q))

    ek, eo, eq = fit_exponent(ws, K), fit_exponent(ws, O), fit_exponent(ws, Q)
    print(f"\nH1  exponent of S_kin   : {ek:+.4f}   [predict  0 -> "
          f"{'PASS' if abs(ek) < 0.05 else 'MISS'}]")
    print(f"H2  exponent of S_oct   : {eo:+.4f}   [predict -1 -> "
          f"{'PASS' if abs(eo + 1) < 0.05 else 'MISS'}]")
    print(f"H3  exponent of S_quart : {eq:+.4f}   [predict -2 -> "
          f"{'PASS' if abs(eq + 2) < 0.05 else 'MISS'}]")

    print("\nH4  PARITY OF EACH TERM  (theta -> -theta)")
    print("-" * 70)
    k0, o0, q0 = functionals(configuration(1.0), E, c)
    k1, o1, q1 = functionals(configuration(1.0, mirror=True), E, c)
    print(f"    S_kin   : {k0:+.6e} -> {k1:+.6e}   even: "
          f"{abs(k1-k0) < 1e-8*abs(k0)}")
    print(f"    S_oct   : {o0:+.6e} -> {o1:+.6e}   odd : "
          f"{abs(o1+o0) < 1e-8*max(abs(o0),1e-30)}")
    print(f"    S_quart : {q0:+.6e} -> {q1:+.6e}   even: "
          f"{abs(q1-q0) < 1e-8*abs(q0)}")

    # ---- H5/H6 the two hands ---------------------------------------
    print("\nH5/H6  STABLE SIZE FOR EACH HAND")
    print("-" * 70)
    A = K[2]
    Bc = O[2] * ws[2] ** 2     # S_oct = B / w^2  (corrected exponent)
    Cc = Q[2] * ws[2] ** 2     # S_quart = C / w^2
    print(f"    A = {A:+.5e}   B = {Bc:+.5e}   C = {Cc:+.5e}")
    print(f"    control parameter |B| / C = {abs(Bc)/Cc:.4f}"
          f"   (threshold at 1)")
    for label, B in (("as built ", Bc), ("mirrored ", -Bc)):
        tot = B + Cc
        fate = "expands without bound" if tot > 0 else "collapses"
        print(f"    {label}: B + C = {tot:+.5e}  ->  {fate}")
    print(f"\n    hands differ in fate : "
          f"{'YES' if abs(Bc) > Cc else 'NO -- below threshold'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The octonionic term is DEGENERATE with the quartic under")
    print("  dilation -- both carry two inverse metrics. So there is no")
    print("  Derrick minimum and no soliton scale at this order. The theory")
    print("  is scale-marginal, as 2D sigma models generally are.")
    print()
    print("  What survives is sharper than a stable size. B is parity-odd and")
    print("  C parity-even, so the two hands carry B + C and -B + C. Below")
    print("  |B| = C both expand and parity has no consequence for fate.")
    print("  ABOVE it, one hand collapses while its mirror expands. The")
    print("  octonionic coupling has a critical value at which handedness")
    print("  starts to decide stability.")
    print()
    print("  Neither regime is a solution. Derrick constrains scale only;")
    print("  existence still needs the PDE.")


if __name__ == "__main__":
    main()
