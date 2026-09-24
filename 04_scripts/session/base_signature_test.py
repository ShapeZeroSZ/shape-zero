#!/usr/bin/env python3
"""
base_signature_test.py — does anything forbid a LORENTZIAN BASE?

The ladder's persistence principle was shown to force the definite branch of
Hurwitz: definite norms give compact level sets and bounded orbits, indefinite
norms give hyperboloids and escape. An earlier write-up over-extended that to
"the ladder cannot describe propagation," which the referee corrected: the rungs
are fibre algebras with NO BASE AT ALL, so persistence constrains the fibre and
says nothing about base signature.

That leaves a decidable question, and it decides whether the absence of gravity
is a scope limit or a refutation:

    does the construction FORBID a Lorentzian base, or merely not produce one?

TEST. A field phi mapping a 1+1 Minkowski base into a 2-dimensional fibre, with

    S = integral  1/2 eta^{mu nu} G_ab d_mu phi^a d_nu phi^b  -  V(phi)

    eta = diag(+1,-1)   LORENTZIAN BASE, both cases
    G   = diag(+1,+1)   Euclidean fibre      -- what the ladder forces
    G   = diag(+1,-1)   indefinite fibre     -- what persistence rejects

Energy density is T_00 = 1/2 G_ab (d_t phi^a d_t phi^b + d_x phi^a d_x phi^b).
With G positive definite this is non-negative pointwise regardless of the base
signature. With G indefinite it is not bounded below, and an interaction lets
one mode grow against the other at fixed total energy -- the standard ghost
instability.

So the prediction is that the obstruction sits entirely in the fibre. If it does,
a Lorentzian base is permitted, gravity is absent rather than excluded, and the
correct ledger entry is "not yet connected."

PREDICTIONS STATED BEFORE RUNNING
 P1 free case: both fibres give conserved energy and bounded amplitudes, since
    the equations of motion are the same wave equation either way. The fibre
    signature does not change the free dynamics -- only what the energy MEANS.
 P2 interacting, Euclidean fibre over a Lorentzian base: energy conserved,
    strictly positive, amplitudes bounded over long evolution.
 P3 interacting, indefinite fibre over the same base: energy conserved but NOT
    positive, and amplitudes grow without bound -- one mode feeding the other.
 P4 therefore nothing in the construction forbids a Lorentzian base. The
    obstruction is the fibre, which is exactly what persistence constrains.

Python 3 + NumPy only.
"""

import numpy as np


def evolve(G, lam, T=400.0, n=160000, N=192, seed=3, amp=0.35):
    """Leapfrog for  G_ab box phi^b = -dV/dphi^a  on a periodic 1+1 lattice."""
    rng = np.random.default_rng(seed)
    dx = 1.0
    dt = T / n
    x = np.arange(N)
    phi = np.zeros((2, N))
    for a in range(2):
        for k in (1, 2, 3):
            phi[a] += amp * rng.normal() * np.sin(2 * np.pi * k * x / N)
    pi = np.zeros((2, N))
    Ginv = np.linalg.inv(G)

    def lap(f):
        return (np.roll(f, 1, axis=-1) - 2 * f + np.roll(f, -1, axis=-1)) / dx ** 2

    def force(f):
        # V = lam/2 * (phi^1)^2 (phi^2)^2  ->  dV/dphi^1 = lam phi^1 (phi^2)^2
        dV = np.zeros_like(f)
        dV[0] = lam * f[0] * f[1] ** 2
        dV[1] = lam * f[1] * f[0] ** 2
        return lap(f) - Ginv @ dV

    def energy(f, p):
        dxf = (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dx)
        kin = 0.5 * np.einsum('ab,ax,bx->', G, p, p)
        grad = 0.5 * np.einsum('ab,ax,bx->', G, dxf, dxf)
        pot = np.sum(0.5 * lam * f[0] ** 2 * f[1] ** 2)
        return kin + grad + pot

    a = force(phi)
    E0 = energy(phi, pi)
    peak = np.max(np.abs(phi))
    for step in range(n):
        pi = pi + 0.5 * dt * a
        phi = phi + dt * pi
        a = force(phi)
        pi = pi + 0.5 * dt * a
        peak = max(peak, np.max(np.abs(phi)))
        if not np.isfinite(peak) or peak > 1e8:
            return np.inf, E0, energy(phi, pi), step * dt
    return peak, E0, energy(phi, pi), T


def main():
    EUC = np.diag([1.0, 1.0])
    IND = np.diag([1.0, -1.0])
    print("=" * 70)
    print("DOES ANYTHING FORBID A LORENTZIAN BASE?")
    print("=" * 70)
    print("\n  base is 1+1 Minkowski in EVERY run; only the fibre metric changes")

    print("\nP1  FREE FIELD (lambda = 0)")
    print("-" * 70)
    print("      fibre         max |phi|      E initial     E final    drift")
    for nm, G in (("Euclidean  ", EUC), ("indefinite ", IND)):
        pk, E0, E1, t = evolve(G, 0.0)
        d = abs(E1 - E0) / max(abs(E0), 1e-30)
        print(f"      {nm}   {pk:10.4f}    {E0:10.4f}  {E1:10.4f}   {d:.2e}")

    print("\nP2/P3  INTERACTING (lambda = 0.6)")
    print("-" * 70)
    print("      fibre         max |phi|      E initial     E final    survived")
    res = {}
    for nm, G in (("Euclidean  ", EUC), ("indefinite ", IND)):
        pk, E0, E1, t = evolve(G, 0.6)
        res[nm.strip()] = (pk, E0, t)
        s = f"{t:.0f}/400" if np.isfinite(pk) else f"BLEW UP at t={t:.1f}"
        print(f"      {nm}   {pk:10.4f}    {E0:10.4f}  {E1:10.4f}   {s}")

    euc_ok = np.isfinite(res['Euclidean'][0]) and res['Euclidean'][1] > 0
    ind_bad = (not np.isfinite(res['indefinite'][0])) or res['indefinite'][1] < 0
    print(f"\n    P2 Euclidean fibre bounded and positive-energy : "
          f"{'PASS' if euc_ok else 'MISS'}")
    print(f"    P3 indefinite fibre unbounded or negative-energy: "
          f"{'PASS' if ind_bad else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    if euc_ok:
        print("  A Euclidean fibre over a LORENTZIAN base is stable: energy")
        print("  density is non-negative pointwise because the fibre metric is")
        print("  positive definite, and that is true whatever the base signature")
        print("  is. Persistence is satisfied.")
        print()
        print("  The obstruction sits entirely in the FIBRE, which is exactly")
        print("  what persistence constrains and exactly what the rungs are.")
        print()
        print("  So the construction does NOT forbid a Lorentzian base. Gravity")
        print("  is ABSENT from the ladder, not REJECTED by it. The correct")
        print("  ledger entry is 'not yet connected', which is recoverable;")
        print("  'excluded by principle' would not have been.")
    else:
        print("  The Euclidean fibre also fails over a Lorentzian base. That")
        print("  would make the base signature genuinely obstructed and the")
        print("  failure verdict correct.")


if __name__ == "__main__":
    main()
