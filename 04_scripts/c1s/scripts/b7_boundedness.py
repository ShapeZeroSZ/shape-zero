#!/usr/bin/env python3
"""
b7_boundedness.py — is the E_G shape a property of boundedness alone?

Follow-up to b7_dislocation.py. There the saturating bond potential was a
CHOSEN: one functional form (Gaussian well), one scale s. If the E_G-shaped
plateau is an artifact of that choice, the correspondence is decoration. If
it follows from boundedness alone, then exactly one thing has to be earned in
the ladder -- that the bond coupling is bounded -- and the functional form is
free.

Three unrelated bounded families, all satisfying V(x) -> x^2/2 as x -> 0 and
V(x) -> s^2 as |x| -> inf:

  gauss   V = s^2 [1 - exp(-x^2 / 2s^2)]
  alg     V = s^2 x^2 / (2 s^2 + x^2)
  tanh    V = s^2 tanh(x^2 / 2 s^2)

They agree only in those two limits; in between they differ substantially.

PREDICTIONS STATED BEFORE RUNNING
 Q1 deep-saturation plateau = N_cut * s^2 for ALL THREE families, to <= 1e-6
    relative. (N_cut = number of frustrated bonds; s^2 = per-bond ceiling.)
 Q2 small-b energies agree across families to <= 1%: all three reduce to the
    same harmonic problem there.
 Q3 mid-range (b ~ s .. 10s) energies DIFFER between families by >= 5%: this
    is where the form matters and where a discriminator would live.

Python 3 + NumPy only.
"""

import numpy as np

# ---------------------------------------------------------------- lattice
def bond_offsets(L, b):
    a_h = np.zeros((L, L - 1))
    a_v = np.zeros((L - 1, L))
    ic, jc = L // 2, L // 2
    a_v[ic - 1, jc:] = b
    return a_h, a_v, int(L - jc)


def interior_mask(L):
    m = np.zeros((L, L), dtype=bool)
    m[1:-1, 1:-1] = True
    return m


# ---------------------------------------------------------------- families
def V_and_dV(x, s, kind):
    s2 = s * s
    if kind == 'gauss':
        e = np.exp(-x * x / (2 * s2))
        return s2 * (1.0 - e), x * e
    if kind == 'alg':
        den = 2 * s2 + x * x
        return s2 * x * x / den, 4 * s2 * s2 * x / (den * den)
    if kind == 'tanh':
        t = np.tanh(x * x / (2 * s2))
        return s2 * t, x * (1.0 - t * t)
    raise ValueError(kind)


def energy(u, a_h, a_v, s, kind):
    xh = (u[:, 1:] - u[:, :-1]) - a_h
    xv = (u[1:, :] - u[:-1, :]) - a_v
    return np.sum(V_and_dV(xh, s, kind)[0]) + np.sum(V_and_dV(xv, s, kind)[0])


def gradient(u, a_h, a_v, s, kind, mask):
    xh = (u[:, 1:] - u[:, :-1]) - a_h
    xv = (u[1:, :] - u[:-1, :]) - a_v
    gh = V_and_dV(xh, s, kind)[1]
    gv = V_and_dV(xv, s, kind)[1]
    g = np.zeros_like(u)
    g[:, :-1] -= gh
    g[:, 1:] += gh
    g[:-1, :] -= gv
    g[1:, :] += gv
    return g * mask


def relax(L, b, s, kind, steps=60000, step=0.15):
    a_h, a_v, ncut = bond_offsets(L, b)
    mask = interior_mask(L)
    u = np.zeros((L, L))
    for _ in range(steps):
        u -= step * gradient(u, a_h, a_v, s, kind, mask)
    gn = np.max(np.abs(gradient(u, a_h, a_v, s, kind, mask)))
    return energy(u, a_h, a_v, s, kind), gn, ncut


def main():
    L, s = 61, 0.6
    kinds = ['gauss', 'alg', 'tanh']
    _, _, ncut = bond_offsets(L, 1.0)
    ceiling = ncut * s * s
    print("=" * 68)
    print("B-7 FOLLOW-UP :: IS THE E_G SHAPE JUST BOUNDEDNESS?")
    print("=" * 68)
    print(f"  L = {L}, s = {s}, frustrated bonds N_cut = {ncut}")
    print(f"  predicted plateau N_cut * s^2 = {ceiling:.6f}\n")

    print("Q1  DEEP SATURATION (b = 20)")
    print("-" * 68)
    for k in kinds:
        E, gn, _ = relax(L, 20.0, s, k)
        print(f"  {k:6s}  E = {E:11.6f}   rel dev = {(E-ceiling)/ceiling:+.3e}"
              f"   max|grad| = {gn:.1e}")

    print("\nQ2  SMALL b  [expect agreement: all reduce to harmonic]")
    print("-" * 68)
    print("      b        gauss         alg          tanh      max spread")
    for b in [0.05, 0.1, 0.2]:
        row = [relax(L, b, s, k)[0] for k in kinds]
        spread = (max(row) - min(row)) / np.mean(row)
        print(f"  {b:6.2f}  {row[0]:11.7f}  {row[1]:11.7f}  {row[2]:11.7f}"
              f"   {100*spread:7.3f}%")

    print("\nQ3  MID-RANGE  [expect divergence: the form matters here]")
    print("-" * 68)
    print("      b        gauss         alg          tanh      max spread")
    for b in [0.6, 1.2, 2.5, 5.0]:
        row = [relax(L, b, s, k)[0] for k in kinds]
        spread = (max(row) - min(row)) / np.mean(row)
        print(f"  {b:6.2f}  {row[0]:11.6f}  {row[1]:11.6f}  {row[2]:11.6f}"
              f"   {100*spread:7.3f}%")

    print("\n  Reading: agreement at both ends with divergence between means")
    print("  the E_G-shaped envelope is fixed by boundedness; only the")
    print("  crossover profile carries the choice of form.")


if __name__ == "__main__":
    main()
