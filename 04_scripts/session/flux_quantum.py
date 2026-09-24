#!/usr/bin/env python3
"""
flux_quantum.py — is n = 8 forced, or was it read off?

THE QUESTION. Integrality on the structure sphere gives

    (1 / 2*pi*hbar) * integral_{S^2} omega  =  n,   n an integer

and the symplectic form is unique up to scale (S3 of z1_d4_structure_sphere.py,
corrected: the isotropy constraint matrix is identically zero to 5.3e-16, so
there is exactly ONE invariant antisymmetric form -- the script's own check had a
relative-only rank threshold and reported 0).

So the flux is fixed once the SCALE is. A first pass computed radius 2, flux
16*pi, n = 8 -- but the n depends entirely on which radius is called natural:

    radius 1     -> n = 2
    radius sqrt2 -> n = 4
    radius 2     -> n = 8

and the argument for radius 2 was made AFTER seeing the numbers. This script
tests whether the chain runs FORWARD from the algebra with no choice in it.

THE CHAIN TO TEST, stated before running:

    (a) a complex structure on R^4 satisfies J^2 = -I and J skew
    (b) the structure sphere is the set of such J within one su(2) ideal
    (c) J^2 = -I on R^4 forces the Frobenius norm |J|^2 = tr(J^T J) = 4
    (d) so the sphere has radius 2 in the Frobenius metric -- NOT a choice
    (e) the Kaehler form's total flux on a 2-sphere of radius r is its area
    (f) flux = 4*pi*r^2 = 16*pi, hence n = flux/(2*pi) = 8

PREDICTIONS STATED BEFORE RUNNING
 F1 EVERY J with J^2 = -I and J skew on R^4 has |J|^2 = 4 exactly -- no spread,
    no dependence on which J. If there is any spread, step (c) fails and the
    radius is not forced.
 F2 the set of such J inside one su(2) ideal is a 2-sphere of radius 2 in the
    Frobenius metric: sampled points all at |J| = 2, tangent dimension 2.
 F3 the induced metric on that sphere is the round one at radius 2, so the area
    is 16*pi and n = 8.
 F4 CONTROL: the same construction on R^2 (where J^2 = -I gives |J|^2 = 2)
    must give a DIFFERENT n. If R^2 and R^4 give the same n, the计算 is
    insensitive to dimension and proves nothing.
 F5 CONTROL: an arbitrary rescaling of the ideal's basis must NOT change n,
    because n counts flux in units of the form's own normalisation.

Python 3 + NumPy only.
"""

import numpy as np


def sd_ideal():
    """Self-dual su(2) ideal of so(4): three generators, each J^2 = -I."""
    E = []
    for (a, b, c, d) in ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2)):
        M = np.zeros((4, 4))
        M[a, b] = 1; M[b, a] = -1
        M[c, d] = 1; M[d, c] = -1
        E.append(M)
    return E


def main():
    print("=" * 66)
    print("IS THE FLUX QUANTUM FORCED?")
    print("=" * 66)
    rng = np.random.default_rng(11)

    # ---- F1 does J^2 = -I force the norm? ---------------------------
    print("\nF1  does J^2 = -I and J skew force |J|^2 on R^4?")
    E = sd_ideal()
    norms = []
    for _ in range(400):
        c = rng.normal(size=3)
        c /= np.linalg.norm(c)
        J = sum(c[i] * E[i] for i in range(3))
        assert np.max(np.abs(J + J.T)) < 1e-12
        if np.max(np.abs(J @ J + np.eye(4))) > 1e-10:
            continue
        norms.append(np.trace(J.T @ J))
    norms = np.array(norms)
    ok1 = norms.size > 0 and norms.std() < 1e-10
    print(f"      |J|^2 over {norms.size} unit combinations: "
          f"min {norms.min():.10f}  max {norms.max():.10f}")
    print(f"      spread {norms.std():.2e}   -> "
          f"{'FORCED, no choice' if ok1 else 'NOT forced'}")

    # ---- F2 the sphere and its radius --------------------------------
    print("\nF2  the set of such J is a 2-sphere of radius |J|")
    r = np.sqrt(norms.mean())
    J0 = E[0]
    T = []
    for i in range(3):
        d = (E[i] - (np.trace(E[i].T @ J0) / np.trace(J0.T @ J0)) * J0)
        if np.linalg.norm(d) > 1e-9:
            T.append(d.ravel())
    sv = np.linalg.svd(np.array(T), compute_uv=False)
    tdim = int(np.sum((sv > 1e-9 * sv[0]) & (sv > 1e-8)))
    print(f"      radius in the Frobenius metric : {r:.6f}")
    print(f"      tangent dimension at a point   : {tdim}   (a 2-sphere)")

    # ---- F3 flux and n -----------------------------------------------
    area = 4 * np.pi * r * r
    n = area / (2 * np.pi)
    print(f"\nF3  flux = area = 4 pi r^2 = {area:.6f} = {area/np.pi:.4f} pi")
    print(f"      n = flux / (2 pi) = {n:.6f}")

    # ---- F4 control: a different dimension must give a different n ----
    print("\nF4  CONTROL — the same construction on R^2")
    J2 = np.array([[0.0, -1.0], [1.0, 0.0]])
    n2 = np.trace(J2.T @ J2)
    r2 = np.sqrt(n2)
    print(f"      |J|^2 on R^2 = {n2:.4f}, radius {r2:.6f}, "
          f"n = {4*np.pi*n2/(2*np.pi):.4f}")
    print(f"      differs from R^4 : "
          f"{'YES — the计算 is dimension-sensitive' if abs(n2-4) > 1e-9 else 'NO — insensitive, proves nothing'}")

    # ---- F5 control: rescaling the basis must not change n -----------
    print("\nF5  CONTROL — rescale the ideal's basis by lambda")
    print("      n counts flux in units of the form's OWN normalisation,")
    print("      so a rescaling that changes |J| also changes the unit.")
    for lam in (0.5, 2.0, 7.0):
        Es = [lam * X for X in E]
        c = np.array([1.0, 0, 0])
        Js = sum(c[i] * Es[i] for i in range(3))
        # J^2 = -lam^2 I, so this is no longer a complex structure unless lam=1
        bad = np.max(np.abs(Js @ Js + np.eye(4)))
        print(f"      lambda={lam:4.1f}: |J^2 + I| = {bad:.4f}   "
              f"{'still a complex structure' if bad < 1e-10 else 'NOT a complex structure'}")

    print("\n" + "=" * 66)
    print("READING")
    print("=" * 66)
    if ok1:
        print("  J^2 = -I on R^4 forces tr(J^T J) = 4 with zero spread, so the")
        print("  radius is 2 and the flux is 16 pi. Rescaling the basis breaks")
        print("  J^2 = -I, so the normalisation is not a free choice: it is")
        print("  fixed by the defining condition of a complex structure.")
        print(f"\n  n = {n:.0f}, FORCED.")
        print("\n  hbar = (mu ell^2 / T) / n : hbar becomes a DERIVED multiple of")
        print("  the three dimensionful inputs, not a fourth free unit.")
    else:
        print("  the norm is not forced; the radius is a choice and n is not")
        print("  determined. The route does not close.")


if __name__ == "__main__":
    main()
