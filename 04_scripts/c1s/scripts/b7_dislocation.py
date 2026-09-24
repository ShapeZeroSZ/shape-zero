#!/usr/bin/env python3
"""
b7_dislocation.py — B-7 first build: the torsion counterpart
(Shape Zero open threads item B-7; contact test against Penrose E_G)

The conical deficit already built is a disclination. This is its dictionary
counterpart: a DISLOCATION, imposed as a closure failure (Burgers vector b)
on a 2D lattice, with the defect energy measured as a function of b and of
system size.

Why this and not B-3: Penrose's E_G measures the failure of a comparison
between two superposed geometries to close. A Burgers vector is that object
made buildable. The Einstein-Cartan caveat (torsion algebraic,
non-propagating, vacuum-vanishing) forbids deriving a long-range FORCE, but
E_G is not a force -- it is a local energy attached to a structural
incompatibility, evaluated instantaneously. The caveat that closes B-3
leaves this open.

MODEL. Scalar field u on an L x L square lattice, Dirichlet u = 0 on the
boundary. The dislocation is imposed as bond frustration: vertical bonds
crossing a cut running from the core to the right edge carry an offset a = b.
The physical strain on a bond is (delta_u - a). Single-valued u plus
frustrated bonds is exactly a closure failure: any circuit enclosing the core
accumulates b.

PREDICTIONS STATED BEFORE RUNNING
 P1 topological: |circulation of (delta_u - a) around the core| = b to
    machine precision, independent of loop radius and of L.
 P2 harmonic quadratic: E(b)/E(1) = b^2 to <= 1e-10. NOTE this is FORCED by
    linearity of the harmonic problem -- it is an instrument check, not a
    result. Recorded as such.
 P3 harmonic logarithm: E vs ln(L) linear with slope b^2/(4*pi) = 0.0796 at
    b = 1 (continuum screw-dislocation result, K = 1). This is the
    non-trivial test. Expect a few percent core/lattice correction, with the
    local slope improving as L grows.
 P4 DESIGNED EXPERIMENT, not a prediction: with a saturating bond potential
    -- E_bond = s^2 [1 - exp(-x^2 / 2 s^2)], which is x^2/2 at small x and
    plateaus at s^2 -- does E(b) leave quadratic and approach a plateau, the
    way E_G(d) does? No outcome is predicted.

E_G REFERENCE (from the Layer-1 Gaussian-smeared form, R0 = smearing width):
    E_G(d) = G m^2 [ 1/(R0 sqrt(pi)) - erf(d / 2 R0) / d ]
    small d : E_G -> G m^2 d^2 / (12 sqrt(pi) R0^3)      [quadratic]
    large d : E_G -> G m^2 / (R0 sqrt(pi))               [SATURATES]
Both energies are quadratic in their closure failure at small argument. The
question this script asks is what happens past the core scale.

Python 3 + NumPy only. Runtime ~1 min.
"""

import numpy as np
from math import erf as _erf

erf = np.vectorize(_erf)

# ---------------------------------------------------------------- lattice ops
def bond_offsets(L, b):
    """Frustration arrays. Cut runs right from the core along a row boundary.

    a_h : (L, L-1) horizontal bonds u[i,j+1]-u[i,j]   -- unfrustrated
    a_v : (L-1, L) vertical bonds   u[i+1,j]-u[i,j]   -- cut carries b
    """
    a_h = np.zeros((L, L - 1))
    a_v = np.zeros((L - 1, L))
    ic, jc = L // 2, L // 2
    a_v[ic - 1, jc:] = b
    return a_h, a_v


def bond_offsets_dipole(L, b, sep):
    """Two opposite dislocations separated by `sep` along the cut row.

    The cut is a FINITE segment, so the far field of the pair cancels and the
    outer boundary condition is irrelevant. This is the well-posed version of
    the same measurement.
    """
    a_h = np.zeros((L, L - 1))
    a_v = np.zeros((L - 1, L))
    ic = L // 2
    j0 = (L - sep) // 2
    a_v[ic - 1, j0:j0 + sep] = b
    return a_h, a_v


def interior_mask(L):
    m = np.zeros((L, L), dtype=bool)
    m[1:-1, 1:-1] = True
    return m


def grad_harmonic(u, a_h, a_v, mask):
    """Gradient of 0.5*sum((du - a)^2) w.r.t. u, boundary pinned."""
    gh = (u[:, 1:] - u[:, :-1]) - a_h
    gv = (u[1:, :] - u[:-1, :]) - a_v
    g = np.zeros_like(u)
    g[:, :-1] -= gh
    g[:, 1:] += gh
    g[:-1, :] -= gv
    g[1:, :] += gv
    return g * mask


def energy_harmonic(u, a_h, a_v):
    gh = (u[:, 1:] - u[:, :-1]) - a_h
    gv = (u[1:, :] - u[:-1, :]) - a_v
    return 0.5 * (np.sum(gh**2) + np.sum(gv**2))


def solve_harmonic(L, b, tol=1e-12, itmax=20000, offsets=None):
    """Conjugate gradient on the (quadratic) harmonic problem. Matrix-free."""
    a_h, a_v = bond_offsets(L, b) if offsets is None else offsets
    mask = interior_mask(L)
    zero = np.zeros((L, L))

    def A(v):
        return grad_harmonic(v, np.zeros_like(a_h), np.zeros_like(a_v), mask)

    f = -grad_harmonic(zero, a_h, a_v, mask)     # since grad = A u - f
    u = np.zeros((L, L))
    r = f - A(u)
    p = r.copy()
    rs = np.sum(r * r)
    for _ in range(itmax):
        Ap = A(p)
        denom = np.sum(p * Ap)
        if denom == 0.0:
            break
        alpha = rs / denom
        u += alpha * p
        r -= alpha * Ap
        rs_new = np.sum(r * r)
        if np.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return u, a_h, a_v, energy_harmonic(u, a_h, a_v)


def circulation(u, a_h, a_v, L, rad):
    """Sum of (delta_u - a) around a square loop of half-width rad about core.

    Traversed counterclockwise. For single-valued u the delta_u parts cancel,
    leaving minus the enclosed frustration.
    """
    ic, jc = L // 2, L // 2
    i0, i1 = ic - rad, ic + rad
    j0, j1 = jc - rad, jc + rad
    tot = 0.0
    for j in range(j0, j1):                       # bottom edge, +x
        tot += (u[i1, j + 1] - u[i1, j]) - a_h[i1, j]
    for i in range(i1, i0, -1):                   # right edge, -y
        tot -= (u[i, j1] - u[i - 1, j1]) - a_v[i - 1, j1]
    for j in range(j1, j0, -1):                   # top edge, -x
        tot -= (u[i0, j] - u[i0, j - 1]) - a_h[i0, j - 1]
    for i in range(i0, i1):                       # left edge, +y
        tot += (u[i + 1, j0] - u[i, j0]) - a_v[i, j0]
    return tot


# ---------------------------------------------------------------- saturating
def energy_sat(u, a_h, a_v, s):
    xh = (u[:, 1:] - u[:, :-1]) - a_h
    xv = (u[1:, :] - u[:-1, :]) - a_v
    e = s * s * (1.0 - np.exp(-xh**2 / (2 * s * s)))
    f = s * s * (1.0 - np.exp(-xv**2 / (2 * s * s)))
    return np.sum(e) + np.sum(f)


def grad_sat(u, a_h, a_v, s, mask):
    xh = (u[:, 1:] - u[:, :-1]) - a_h
    xv = (u[1:, :] - u[:-1, :]) - a_v
    gh = xh * np.exp(-xh**2 / (2 * s * s))
    gv = xv * np.exp(-xv**2 / (2 * s * s))
    g = np.zeros_like(u)
    g[:, :-1] -= gh
    g[:, 1:] += gh
    g[:-1, :] -= gv
    g[1:, :] += gv
    return g * mask


def solve_sat(L, b, s, steps=40000, step=0.15):
    u, a_h, a_v, _ = solve_harmonic(L, b)          # warm start
    mask = interior_mask(L)
    for _ in range(steps):
        g = grad_sat(u, a_h, a_v, s, mask)
        u -= step * g
    gnorm = np.max(np.abs(grad_sat(u, a_h, a_v, s, mask)))
    return energy_sat(u, a_h, a_v, s), gnorm


# ---------------------------------------------------------------- E_G form
def E_G_shape(d, R0):
    """Gaussian-smeared Penrose self-energy, in units of G m^2."""
    return 1.0 / (R0 * np.sqrt(np.pi)) - erf(d / (2.0 * R0)) / d


def main():
    print("=" * 70)
    print("B-7 :: DISLOCATION AS CLOSURE FAILURE")
    print("=" * 70)

    # ---- P1 topological check ---------------------------------------
    print("\nP1  TOPOLOGICAL CLOSURE")
    print("-" * 70)
    L = 81
    for b in [1.0, 2.0, 0.5]:
        u, a_h, a_v, _ = solve_harmonic(L, b)
        for rad in [5, 12, 25]:
            c = circulation(u, a_h, a_v, L, rad)
            print(f"  b={b:4.1f}  loop rad={rad:3d}   circulation={c:+.12f}"
                  f"   |c|-b = {abs(c)-b:+.2e}")

    # ---- P2 quadratic in b (instrument check) ------------------------
    print("\nP2  HARMONIC ENERGY vs b   [forced by linearity -- instrument]")
    print("-" * 70)
    L = 81
    _, _, _, E1 = solve_harmonic(L, 1.0)
    print(f"    b        E(b)          E(b)/E(1)      b^2        dev")
    for b in [0.5, 1.0, 2.0, 3.0, 5.0]:
        _, _, _, E = solve_harmonic(L, b)
        r = E / E1
        print(f"  {b:4.1f}   {E:12.6f}   {r:12.8f}  {b*b:9.4f}   {r-b*b:+.2e}")

    # ---- P3 logarithm in system size --------------------------------
    print("\nP3  HARMONIC ENERGY vs ln(L)   [the non-trivial test]")
    print("-" * 70)
    Ls = np.array([25, 35, 51, 71, 101, 141])
    Es = []
    for Lv in Ls:
        _, _, _, E = solve_harmonic(int(Lv), 1.0)
        Es.append(E)
    Es = np.array(Es)
    x = np.log(Ls.astype(float))
    print("     L        E          local slope dE/dln(L)")
    for k in range(len(Ls)):
        if k == 0:
            print(f"  {Ls[k]:4d}  {Es[k]:10.6f}        --")
        else:
            sl = (Es[k] - Es[k - 1]) / (x[k] - x[k - 1])
            print(f"  {Ls[k]:4d}  {Es[k]:10.6f}    {sl:10.6f}")
    A = np.vstack([x, np.ones_like(x)]).T
    slope, icept = np.linalg.lstsq(A, Es, rcond=None)[0]
    pred = 1.0 / (4.0 * np.pi)
    print(f"\n  global fit slope : {slope:.6f}")
    print(f"  predicted b^2/4pi: {pred:.6f}")
    print(f"  deviation        : {100*(slope-pred)/pred:+.2f}%")
    tail = (Es[-1] - Es[-2]) / (x[-1] - x[-2])
    print(f"  largest-L local  : {tail:.6f}  ({100*(tail-pred)/pred:+.2f}%)")

    # ---- P3b diagnostic: dipole, boundary-independent ----------------
    print("\nP3b DIAGNOSTIC :: dislocation DIPOLE vs separation")
    print("-" * 70)
    print("  Finite cut -> far field cancels -> outer BC irrelevant.")
    print("  Continuum pair energy: (b^2/2pi) ln(d/r_c), slope = 0.159155")
    Lp = 161
    seps = np.array([4, 6, 9, 13, 19, 28, 41])
    Ed = []
    for sp in seps:
        off = bond_offsets_dipole(Lp, 1.0, int(sp))
        _, _, _, E = solve_harmonic(Lp, 1.0, offsets=off)
        Ed.append(E)
    Ed = np.array(Ed)
    xd = np.log(seps.astype(float))
    print("\n     sep      E          local slope dE/dln(d)")
    for k in range(len(seps)):
        if k == 0:
            print(f"  {seps[k]:5d}  {Ed[k]:10.6f}        --")
        else:
            sl = (Ed[k] - Ed[k - 1]) / (xd[k] - xd[k - 1])
            print(f"  {seps[k]:5d}  {Ed[k]:10.6f}    {sl:10.6f}")
    pred_d = 1.0 / (2.0 * np.pi)
    tail_d = (Ed[-1] - Ed[-2]) / (xd[-1] - xd[-2])
    print(f"\n  predicted slope  : {pred_d:.6f}")
    print(f"  largest-sep local: {tail_d:.6f}  ({100*(tail_d-pred_d)/pred_d:+.2f}%)")

    # ---- P4 designed experiment: saturating bonds --------------------
    print("\nP4  DESIGNED EXPERIMENT :: saturating bond potential")
    print("-" * 70)
    L, s = 61, 0.6
    print(f"  s = {s} (bond energy plateaus at s^2 = {s*s:.3f})")
    print("     b      E_sat        E_sat/b^2     max|grad|")
    base = None
    for b in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        E, gn = solve_sat(L, b, s)
        if base is None:
            base = E / (b * b)
        print(f"  {b:5.2f}  {E:11.5f}   {E/(b*b):11.5f}    {gn:.2e}")

    # ---- E_G comparison ---------------------------------------------
    print("\nE_G FORM, same axis   [units of G m^2, R0 = 1]")
    print("-" * 70)
    print("     d      E_G(d)      E_G/d^2      frac of plateau")
    plateau = 1.0 / np.sqrt(np.pi)
    for d in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        e = float(E_G_shape(np.array([d]), 1.0)[0])
        print(f"  {d:5.2f}  {e:10.6f}   {e/(d*d):10.6f}     {e/plateau:8.4f}")
    print(f"\n  plateau = 1/sqrt(pi) = {plateau:.6f}")


if __name__ == "__main__":
    main()
