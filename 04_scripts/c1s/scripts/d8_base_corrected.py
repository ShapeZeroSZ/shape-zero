#!/usr/bin/env python3
"""
d8_base_corrected.py — redoing the arena question with base = Im(O) = R^7

CORRECTION THIS SUPERSEDES. Sessions of work used a 2-dimensional base, taken
from phi_gauge_wilson.py. That is a D2-level script -- the synthetic U(1) and
gauge-emergence work -- not the D8 arena. The C1 rung scripts show the arena
advancing with the ladder without exception:

    z1_d1_rung.py  scalar          z1_d4_rung.py  4x4 structures, u in R^4
    z1_d2_rung.py  z in R^2        z1_d8_attempt.py  np.zeros(8), R^8

So the D8 base is O = R^8, and Im(O) = R^7 after the D3-style deletion. G2 acts
on the base as well as the fibre. Everything computed on a 2D base is about a
theory the program does not have.

TWO CONSEQUENCES TESTED HERE.

(1) SCALING. Under x -> mu x with the profile fixed, a term with n derivatives
in d base dimensions scales as mu^(d-n): the measure gives mu^d and each
derivative gives 1/mu. In d = 2 the kinetic term (n = 2) is MARGINAL, which is
why nothing could stabilise the collapse and why the whole instability analysis
went the way it did. In d = 7 the kinetic term scales as mu^5 and dominates
every four-derivative term at large mu, so the action is bounded below with no
coupling bound required.

(2) A TERM THAT WAS FORBIDDEN. On a 2D base the natural cubic octonionic term
phi^{mu nu rho} c_{abc} d_mu psi^a d_nu psi^b d_rho psi^c does not exist:
Lambda^3 of a two-dimensional space is zero, which is exactly why the surviving
2D term had to carry a second derivative and land at (3, 4). With G2 acting on
the base, phi_{mu nu rho} is available and the term exists at (3, 3) -- purely
first-derivative.

PREDICTIONS STATED BEFORE RUNNING
 E1 the exponent law mu^(d-n) verified numerically: in d = 2 the kinetic term
    has exponent 0 and the quartic -2; in d = 3, +1 and -1.
 E2 the 7D octonionic term is nonzero, and on the IDENTITY map (base index
    equal to target index) equals exactly sum phi^2 = 42.
 E3 the same contraction is identically zero on a 2-dimensional base, for
    every configuration -- the term really is forbidden there.
 E4 with A > 0 the 7D action S(mu) = A mu^5 + O mu^4 + K mu^3 is bounded below
    for ANY O and K. The 2D unboundedness does not occur, so the coupling
    bound lambda < 2 sqrt(mu c4) is an artefact of the wrong base.
 E5 S has a stationary point at mu > 0 whenever the discriminant
    16 O^2 - 60 A K > 0 and the relevant root is positive; report when it is a
    minimum.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations


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


def imaginary_c(orient):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in orient:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


# ---------------------------------------------------------------- E1
def functionals_dD(d, N, w, rng):
    """Kinetic and quartic functionals of a width-w profile on a d-dim torus."""
    L = 2 * np.pi
    ax = np.linspace(0, L, N, endpoint=False)
    grids = np.meshgrid(*([ax] * d), indexing='ij')
    ctr = L / 2
    r2 = sum((g - ctr) ** 2 for g in grids)
    ncomp = 4
    psi = np.zeros((ncomp,) + grids[0].shape)
    for a in range(ncomp):
        env = np.exp(-r2 / (2 * w ** 2))
        ph = sum(rng.normal() * (g - ctr) / w for g in grids)
        psi[a] = env * np.cos(ph + rng.uniform(0, 2 * np.pi))
    h = L / N
    dphi2 = np.zeros(grids[0].shape)
    for mu in range(d):
        g = np.gradient(psi, h, axis=mu + 1)
        dphi2 += np.einsum('a...,a...->...', g, g)
    cell = h ** d
    return np.sum(dphi2) * cell, np.sum(dphi2 ** 2) * cell


def main():
    c = imaginary_c(oriented_lines())
    print("=" * 70)
    print("D8 ARENA, BASE CORRECTED TO Im(O) = R^7")
    print("=" * 70)

    print("\nE1  EXPONENT LAW  mu^(d-n)")
    print("-" * 70)
    print("      d   term        fitted        predicted")
    ok1 = True
    for d, N in ((2, 96), (3, 40)):
        ws = np.array([0.35, 0.5, 0.7, 1.0])
        Ks, Qs = [], []
        for w in ws:
            rng = np.random.default_rng(5)
            k, q = functionals_dD(d, N, w, rng)
            Ks.append(k); Qs.append(q)
        ek = np.polyfit(np.log(ws), np.log(Ks), 1)[0]
        eq = np.polyfit(np.log(ws), np.log(Qs), 1)[0]
        for nm, fit, pred in (("kinetic", ek, d - 2), ("quartic", eq, d - 4)):
            good = abs(fit - pred) < 0.15
            ok1 &= good
            print(f"      {d}   {nm:8s}   {fit:+.4f}      {pred:+d}"
                  f"    {'PASS' if good else 'MISS'}")
    print(f"\n    E1 {'PASS' if ok1 else 'MISS'}  -- kinetic is MARGINAL only "
          f"at d = 2")

    # ---- E2/E3 the forbidden term ----------------------------------
    print("\nE2/E3  THE TERM THAT 2D FORBIDS")
    print("-" * 70)
    print("      T[J] = phi^{mu nu rho} c_{abc} J_mu^a J_nu^b J_rho^c")
    ident = np.eye(7)
    Tid = np.einsum('mnr,abc,ma,nb,rc->', c, c, ident, ident, ident)
    print(f"      on the identity map, d = 7 : {Tid:.1f}"
          f"   [predict 42 -> {'PASS' if abs(Tid - 42) < 1e-9 else 'MISS'}]")
    rng = np.random.default_rng(3)
    vals = [abs(np.einsum('mnr,abc,ma,nb,rc->', c, c,
                          *([rng.normal(size=(7, 7))] * 1 * 3))) for _ in range(3)]
    gen = [abs(np.einsum('mnr,abc,ma,nb,rc->', c, c, J, J, J))
           for J in (rng.normal(size=(7, 7)) for _ in range(200))]
    print(f"      generic J, d = 7 : max |T| = {max(gen):.3f}"
          f"   [nonzero -> {'PASS' if max(gen) > 1e-6 else 'MISS'}]")
    # 2D base: only two base directions exist, phi restricted to them
    worst2 = 0.0
    for _ in range(200):
        J2 = rng.normal(size=(2, 7))
        Jp = np.zeros((7, 7))
        Jp[:2, :] = J2                     # only two base directions active
        worst2 = max(worst2, abs(np.einsum('mnr,abc,ma,nb,rc->',
                                           c, c, Jp, Jp, Jp)))
    print(f"      2D base (only 2 base directions) : max |T| = {worst2:.3e}"
          f"   [predict 0 -> {'PASS' if worst2 < 1e-9 else 'MISS'}]")
    print("      -> Lambda^3 of a 2-dimensional space vanishes; the term is")
    print("         genuinely unavailable there, which is why the 2D survivor")
    print("         had to carry a second derivative and sit at (3, 4)")

    # ---- E4/E5 Derrick in 7 dimensions ------------------------------
    print("\nE4/E5  DERRICK STRUCTURE IN d = 7")
    print("-" * 70)
    print("      S(mu) = A mu^5 + O mu^4 + K mu^3     (n = 2, 3, 4 terms)")
    A = 1.0
    print("        O        K      bounded below   stationary mu>0   type")
    okE4 = True
    for O, K in ((0.0, 1.0), (0.0, -1.0), (-3.0, -1.0), (2.0, -5.0), (-6.0, 2.0)):
        mus = np.logspace(-3, 3, 4000)
        S = A * mus ** 5 + O * mus ** 4 + K * mus ** 3
        bounded = np.min(S) > -1e12 and S[-1] > 0
        okE4 &= bounded
        roots = np.roots([5 * A, 4 * O, 3 * K])
        pos = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
        if pos:
            m = max(pos)
            d2 = 20 * A * m ** 3 + 12 * O * m ** 2 + 6 * K * m
            typ = "MINIMUM" if d2 > 0 else "maximum"
            print(f"      {O:+5.1f}   {K:+5.1f}      {str(bounded):5s}"
                  f"        {m:8.4f}       {typ}")
        else:
            print(f"      {O:+5.1f}   {K:+5.1f}      {str(bounded):5s}"
                  f"        none            --")
    print(f"\n    E4 bounded below in every case : "
          f"{'PASS' if okE4 else 'MISS'}")
    print("    -> the 2D unboundedness does not occur. The coupling bound")
    print("       lambda < 2 sqrt(mu c4) was an artefact of the marginal")
    print("       kinetic term at d = 2, and does not apply at d = 7.")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  Correcting the base changes the conclusions, not just the")
    print("  arithmetic. At d = 7 the kinetic term is no longer marginal, so")
    print("  it bounds the action by itself and no coupling bound is needed;")
    print("  and the cubic three-derivative octonionic term, structurally")
    print("  forbidden on a 2D base, exists. The instability, the bound, and")
    print("  the 'no window' result were all properties of the wrong arena.")


if __name__ == "__main__":
    main()
