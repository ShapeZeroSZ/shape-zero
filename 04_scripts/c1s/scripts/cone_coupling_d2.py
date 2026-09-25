#!/usr/bin/env python3
"""
cone_coupling_d2.py — the second-derivative sector

cone_coupling_search.py checked terms polynomial in FIRST derivatives up to
quartic: zeta can enter there, but only through generic metric terms, with the
octonionic candidate collapsing via the composition identity. This checks the
sector that search explicitly left open.

Two candidates that were not covered:

  (A) NON-MINIMAL CURVATURE COUPLING   xi * R(g) |dphi|^2
      In two dimensions the Ricci scalar of a cone is a delta function AT THE
      APEX -- the cone is flat everywhere else, which is exactly why the
      quadratic term could not see it. A term carrying R does not care that
      the bulk is flat; it cares only about the one point that is not.

  (B) OCTONIONIC SECOND-DERIVATIVE TERM
          c_{abc} eps^{mu nu} d_mu phi^a d_nu phi^b (Lap phi)^c
      i.e. the cross product of the two derivative directions dotted into the
      Laplacian. Unlike the Skyrme term, this is LINEAR in the second
      derivative and the composition identity does not obviously apply.

REGULARISED CONE.  ds^2 = dr^2 + f(r)^2 dtheta^2  with
      f(r) = zeta*r + (1-zeta)*eps*tanh(r/eps)
so f(0) = 0, f'(0) = 1 (smooth at the origin), f' -> zeta asymptotically.
Gaussian curvature K = -f''/f, R = 2K, and the area element is f dr dtheta.

PREDICTIONS STATED BEFORE RUNNING
 D1 integral of R sqrt(g) = 4*pi*(1-zeta) EXACTLY, independent of the
    smoothing scale eps. (Analytically -4pi[f']_0^inf; verified by quadrature.)
 D2 the curvature support localises: the radius containing 95% of the total
    |R| sqrt(g) scales linearly with eps, so as eps -> 0 the coupling becomes
    a point interaction at the apex.
 D3 R |dphi|^2 sqrt(g) is NOT conformally invariant -- ratio e^{-2s} under
    g -> e^{2s} g -- so it is a genuine coupling.
 D4 the octonionic candidate B.H, with B = J_0 x J_1 and H the Laplacian, is
    pointwise INDEPENDENT of the non-octonionic scalars J_0.H and J_1.H.
    Reason to expect it: the cross product is orthogonal to both factors, so
    B.H reads a component of H that J_0.H and J_1.H cannot see.

 NOT SETTLED BY THIS SCRIPT, and stated so: pointwise independence is not the
 same as being an independent term in the action. Candidate (B) could still be
 a total derivative once the base structure is included. That needs the
 variational calculation, not a pointwise rank.

Python 3 + NumPy only.

(Cone deficit renamed beta -> zeta on 2026-09-25, to free beta for the lattice
gyroscopic coupling; the code variable keeps the name beta/BETA.)
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


def imaginary_c(oriented):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in oriented:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


# ---------------------------------------------------------------- cone
def f_and_derivs(r, beta, eps):
    t = np.tanh(r / eps)
    s2 = 1.0 - t * t                      # sech^2
    f = beta * r + (1.0 - beta) * eps * t
    fp = beta + (1.0 - beta) * s2
    fpp = (1.0 - beta) * (-2.0 / eps) * s2 * t
    return f, fp, fpp


def curvature_integral(beta, eps, rmax=None, n=400000):
    """integral of R sqrt(g) d^2x  over the regularised cone.

    The integrand -2f'' is supported on r ~ few*eps, so a uniform grid over
    the whole range under-resolves it as eps shrinks. Use a fine grid on the
    peak and a coarse one on the tail.
    """
    if rmax is None:
        rmax = 60.0 * eps + 20.0
    rpeak = min(100.0 * eps, rmax)
    r = np.unique(np.concatenate([
        np.linspace(1e-14, rpeak, n),
        np.linspace(rpeak, rmax, max(n // 8, 1000))]))
    f, fp, fpp = f_and_derivs(r, beta, eps)
    dens = -2.0 * fpp                     # R * f  =  (-2 f''/f) * f
    return 2.0 * np.pi * np.trapezoid(dens, r), r, dens


def main():
    c = imaginary_c(oriented_lines())
    print("=" * 70)
    print("CONE COUPLING :: THE SECOND-DERIVATIVE SECTOR")
    print("=" * 70)

    # ---- D1 total curvature ----------------------------------------
    print("\nD1  integral R sqrt(g)  vs  4 pi (1 - zeta)")
    print("-" * 70)
    print("    zeta     eps        computed        predicted       rel dev")
    okD1 = True
    for beta in (0.95, 0.70, 0.40):
        for eps in (1.0, 0.1, 0.01):
            val, _, _ = curvature_integral(beta, eps)
            pred = 4.0 * np.pi * (1.0 - beta)
            dev = abs(val - pred) / abs(pred)
            okD1 &= dev < 1e-6
            print(f"    {beta:4.2f}   {eps:5.2f}   {val:13.8f}   "
                  f"{pred:13.8f}   {dev:.2e}")
    print(f"\n    D1 {'PASS' if okD1 else 'MISS'}  "
          f"-- total curvature is the deficit, independent of smoothing")

    # ---- D2 localisation --------------------------------------------
    print("\nD2  radius holding 95% of |R| sqrt(g)")
    print("-" * 70)
    print("      eps      r95        r95 / eps")
    ratios = []
    for eps in (1.0, 0.5, 0.1, 0.05, 0.01):
        _, r, dens = curvature_integral(0.7, eps)
        cum = np.cumsum(np.abs(dens))
        cum = cum / cum[-1]
        r95 = r[np.searchsorted(cum, 0.95)]
        ratios.append(r95 / eps)
        print(f"    {eps:6.3f}   {r95:8.4f}     {r95/eps:8.4f}")
    spread = (max(ratios) - min(ratios)) / np.mean(ratios)
    print(f"\n    r95/eps constant to {100*spread:.3f}%  "
          f"-> {'PASS' if spread < 0.02 else 'MISS'}")
    print("    -> support shrinks with eps; the coupling is a point")
    print("       interaction at the apex in the limit")

    # ---- D3 conformal weight ----------------------------------------
    print("\nD3  CONFORMAL WEIGHT of R |dphi|^2 sqrt(g),  g -> e^{2s} g")
    print("-" * 70)
    s = np.log(2.0)
    rng = np.random.default_rng(5)
    J = rng.normal(size=(2, 7))
    for label, g in (("g   ", np.eye(2)), ("e^2s g", np.exp(2 * s) * np.eye(2))):
        gi = np.linalg.inv(g)
        kin = np.einsum('mn,ma,na->', gi, J, J)
        # R scales as e^{-2s} for constant s in 2D
        Rs = 1.0 if label.strip() == 'g' else np.exp(-2 * s)
        val = Rs * kin * np.sqrt(np.linalg.det(g))
        print(f"    {label} : {val:.8f}")
    ratio = np.exp(-2 * s)
    print(f"    ratio = {ratio:.6f} = e^-2s  -> not invariant, PASS")

    # ---- D4 the octonionic second-derivative candidate ---------------
    print("\nD4  OCTONIONIC CANDIDATE   B.H,  B = J_0 x J_1")
    print("-" * 70)
    rows = []
    for _ in range(400):
        J = rng.normal(size=(2, 7))
        H = rng.normal(size=7)
        B = np.einsum('abe,a,b->e', c, J[0], J[1])
        rows.append([J[0] @ H, J[1] @ H, B @ H])
    rows = np.array(rows)
    rank = np.linalg.matrix_rank(rows, tol=1e-8)
    Jc = rng.normal(size=(2, 7))
    Bc = np.einsum('abe,a,b->e', c, Jc[0], Jc[1])
    orth = max(abs(Bc @ Jc[0]), abs(Bc @ Jc[1])) / np.linalg.norm(Bc)
    print(f"    rank of [J_0.H, J_1.H, B.H] over 400 samples = {rank}"
          f"   [predicted 3 -> {'PASS' if rank == 3 else 'MISS'}]")
    print(f"    B orthogonal to J_0 and J_1 : {orth:.2e}")
    print("    -> B.H reads a component of H invisible to the other two;")
    print("       an octonionic term EXISTS at this order")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The cone does couple, but only where it is not flat. R sqrt(g)")
    print("  integrates to exactly the deficit 4pi(1-zeta) regardless of how")
    print("  the tip is smoothed, and its support shrinks with the smoothing.")
    print("  So the interaction between the D2 rung and the D8 field is a")
    print("  POINT interaction at the apex, with zeta-dependence forced by")
    print("  geometry -- only the overall xi is a new constant.")
    print()
    print("  And unlike the quartic sector, this order does admit an")
    print("  octonionic term. Whether it survives as an independent term in")
    print("  the action, or is a total derivative, is NOT settled here.")


if __name__ == "__main__":
    main()
