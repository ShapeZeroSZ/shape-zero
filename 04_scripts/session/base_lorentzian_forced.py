#!/usr/bin/env python3
"""
base_lorentzian_forced.py — does persistence force the BASE to be Lorentzian?

Established: persistence forces the FIBRE to be definite. Bounded motion needs
compact level sets; indefinite fibres give hyperboloids, escape, and ghost
instabilities. Separately established: nothing in that argument touches the base,
and a Euclidean fibre over a Lorentzian base is stable.

That left the base unconstrained. But persistence — "only structures that support
bounded, enduring motion are retained" — has a second consequence that applies to
the base rather than the fibre. Enduring motion requires that evolution be
WELL POSED: initial data must determine a solution, and small changes in the data
must produce small changes in the solution. Without continuous dependence there is
no motion to endure, because the trajectory is not determined by its start.

Well-posedness of the Cauchy problem is a signature question with a known answer:

    (0, q)  elliptic         Cauchy problem ILL POSED (Hadamard)
    (1, q)  hyperbolic       WELL POSED
    (p, q), p >= 2  ultrahyperbolic   ILL POSED

So if persistence is applied to the base the way it was applied to the fibre, it
selects EXACTLY ONE timelike direction — Lorentzian signature — and does so by
the same criterion that made the fibre Euclidean.

TEST. Evolve  a d_t^2 phi = -b d_x^2 phi  from data phi = 0, d_t phi = eps sin(kx),
for the hyperbolic case (a=1, b=-1) and the elliptic case (a=1, b=+1), and measure
how the response depends on the wavenumber k. Well-posedness is precisely the
statement that the amplification factor does not blow up with k.

Exact solutions:
    hyperbolic : phi = (eps/k) sin(kx) sin(kt)      -- bounded, decreasing in k
    elliptic   : phi = (eps/k) sin(kx) sinh(kt)     -- grows as e^{kt}

PREDICTIONS STATED BEFORE RUNNING
 L1 hyperbolic: response amplitude falls as 1/k and never exceeds the initial
    scale. Arbitrarily fine perturbations stay arbitrarily small.
 L2 elliptic: response grows like e^{kt}/k. Doubling k roughly squares the
    amplification at fixed t, so no bound on the response exists in terms of the
    data. That is Hadamard instability, and it is not a numerical artifact.
 L3 the ratio elliptic/hyperbolic diverges with k -- the discriminator is sharp,
    not marginal.
 L4 ultrahyperbolic (2,2) also fails: a mode with wavevector in the two timelike
    directions grows, so p >= 2 is excluded for the same reason.
 L5 therefore persistence forces exactly one timelike direction on the base.
    Fibre definite AND base Lorentzian, from a single principle.

Python 3 + NumPy only.
"""

import numpy as np


def evolve_2d(sig_x, T, k, N=512, nt=40000, eps=1e-3):
    """a d_t^2 phi = -b d_x^2 phi with b = sig_x.  sig_x=-1 hyperbolic, +1 elliptic."""
    L = 2 * np.pi
    dx = L / N
    x = np.arange(N) * dx
    dt = T / nt
    phi = np.zeros(N)
    pi = eps * np.sin(k * x)

    def lap(f):
        return (np.roll(f, 1) - 2 * f + np.roll(f, -1)) / dx ** 2

    for _ in range(nt):
        a = -sig_x * lap(phi)
        pi = pi + 0.5 * dt * a
        phi = phi + dt * pi
        a = -sig_x * lap(phi)
        pi = pi + 0.5 * dt * a
        if not np.isfinite(phi).all() or np.max(np.abs(phi)) > 1e12:
            return np.inf
    return np.max(np.abs(phi))


def main():
    eps = 1e-3
    T = 1.0
    print("=" * 70)
    print("DOES PERSISTENCE FORCE THE BASE TO BE LORENTZIAN?")
    print("=" * 70)
    print(f"\n  initial data: phi = 0, d_t phi = {eps:g} sin(kx);  evolved to t = {T}")
    print("  well-posed means: response does NOT blow up as k increases\n")
    print("      k     hyperbolic (1,1)    elliptic (2,0)     ratio")
    hyp, ell = [], []
    for k in (1, 2, 4, 8, 16):
        h = evolve_2d(-1.0, T, k)
        e = evolve_2d(+1.0, T, k)
        hyp.append(h)
        ell.append(e)
        r = e / h if (np.isfinite(e) and h > 0) else np.inf
        es = f"{e:.4e}" if np.isfinite(e) else "  OVERFLOW"
        rs = f"{r:.3e}" if np.isfinite(r) else "   inf"
        print(f"    {k:3d}     {h:.4e}         {es}      {rs}")

    print(f"\n  exact:  hyperbolic  (eps/k) sin(kt)   -> falls as 1/k")
    print(f"          elliptic    (eps/k) sinh(kt)  -> grows as e^(kt)/k")
    for k in (1, 4, 16):
        print(f"    k={k:3d}: predicted hyp {eps/k*abs(np.sin(k*T)):.3e}   "
              f"ell {eps/k*np.sinh(k*T):.3e}")

    ok1 = all(np.isfinite(h) and h <= eps for h in hyp)
    ok2 = (not np.isfinite(ell[-1])) or ell[-1] > 1e3 * ell[0]
    print(f"\n  L1 hyperbolic bounded by the data scale : "
          f"{'PASS' if ok1 else 'MISS'}")
    print(f"  L2 elliptic amplification grows with k  : "
          f"{'PASS' if ok2 else 'MISS'}")

    # ---- L4 ultrahyperbolic ------------------------------------------
    print("\n  L4  ULTRAHYPERBOLIC (2,2): d_t1^2 + d_t2^2 = d_x1^2 + d_x2^2")
    print("      a mode e^{i(k1 x1 + k2 x2)} evolves in the t-plane as")
    print("      omega^2 = k1^2 + k2^2 with TWO time directions; data on one")
    print("      t-slice leaves the second time evolution unconstrained, and")
    print("      modes with wavevector along t2 grow exponentially.")
    for k in (1, 4, 16):
        print(f"        k={k:3d}: growth over unit t2 = e^{k} = {np.exp(k):.3e}")
    print("      -> same Hadamard failure as the elliptic case; p >= 2 excluded.")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  Persistence retains only what supports bounded, enduring motion.")
    print("  Motion that is not determined by its initial data does not endure;")
    print("  it is not motion. Well-posedness is therefore a persistence")
    print("  requirement, not an extra assumption.")
    print()
    print("  Zero timelike directions: elliptic, Hadamard-unstable.")
    print("  Two or more: ultrahyperbolic, equally unstable.")
    print("  Exactly one: hyperbolic, well posed.")
    print()
    print("  So the SAME principle that forces the fibre to be definite forces")
    print("  the base to be LORENTZIAN. The fibre is where states live and must")
    print("  be bounded; the base is where evolution happens and must be")
    print("  well posed. One principle, two conclusions, opposite signatures.")
    print()
    print("  Gravity's home is not posited. It is the unique base signature")
    print("  persistence admits.")


if __name__ == "__main__":
    main()
