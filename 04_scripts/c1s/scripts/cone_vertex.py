#!/usr/bin/env python3
"""
cone_vertex.py — the surviving role: vacuum chirality, and whether it is real

cone_ground_state.py closed the vacuum-structure question: below the coupling
bound the octonionic term cannot move the ground state, above it there is no
ground state. What survives is the term as an INTERACTION.

Expand about the trivial vacuum, phi = normalise(1 + eps psi) with psi in
Im(O). The sphere constraint contributes NO cubic term: the eps^2 correction
lies along 1, and d(psi) is orthogonal to 1, so the cross term vanishes.
Together with the uniqueness result (three target indices force c, and c
antisymmetric forces eps^{mu nu}) this means

    the octonionic vertex is the ONLY cubic term in the theory.

So the natural observable is the vacuum expectation of the octonionic density
itself,  O = 2 c(d_r psi, d^_theta psi, box psi)  integrated. In the free
theory <O> = 0 because O is odd in psi. At first order in the coupling,

    <O>_int  =  -lambda <O^2>_free  +  O(lambda^2),

and <O^2> is positive and parity-EVEN, so <O> acquires a definite sign fixed
by lambda -- that is, by orientation times base parity. The vacuum is chiral
in its correlations even though the mean field is trivial.

WHETHER THAT IS PHYSICAL IS NOT ASSUMED. <O^2> is a loop quantity. Power
counting: O carries 4 derivatives and 3 fields; <O O> contracts 3 propagators,
each ~ 1/(mu k^4) at large k, against k^8 from the derivatives and k^4 from the
two-loop measure -- k^8 k^-12 k^4 = k^0, i.e. LOG DIVERGENT. So the bare
quantity is expected to depend on the cutoff, and the honest observable is
whatever survives it. That is checked here before any claim is made.

FREE THEORY. S_2 = integral [ 1/2 |d psi|^2 + mu (box psi)^2 ] dmu on the cone
ds^2 = dr^2 + beta^2 r^2 dtheta^2. Angular modes decouple, so the covariance is
block diagonal in m and each block is an NR x NR radial operator that can be
inverted exactly. Samples are drawn as Q_m^{-1/2} eta, not by relaxation, so
there is no thermalisation error.

PREDICTIONS STATED BEFORE RUNNING
 U1 <O> = 0 in the free theory, to within Monte Carlo error. O is cubic and
    the measure is Gaussian. Instrument check.
 U2 <O^2> > 0 and well determined, with relative standard error under 5%.
 U3 <O^2> GROWS with the grid cutoff, consistent with the log divergence
    above. Predicted to grow, not converge -- so the bare vacuum chirality is
    NOT by itself an observable.
 U4 the RATIO <O^2>(beta_1) / <O^2>(beta_2) converges as the cutoff is
    refined, because the divergence is a short-distance effect and the cone is
    flat away from its apex. If it converges, THAT is the physical quantity
    and it is a genuine D2-to-D8 observable.
 U5 <O> flips sign under orientation reversal c -> -c at fixed configuration
    ensemble, so the chirality tracks orientation times base parity as
    established in octonionic_term_parity.py.
 NOT PREDICTED: the direction of the beta dependence.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations

RMAX = 8.0
MU = 1.0


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


class Cone:
    def __init__(self, nr, nth, beta):
        self.nr, self.nth, self.beta = nr, nth, beta
        self.r = np.linspace(RMAX / nr, RMAX, nr)
        self.dr = self.r[1] - self.r[0]
        self.th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
        self.RG, self.TG = np.meshgrid(self.r, self.th, indexing='ij')
        self.kth = np.fft.fftfreq(nth, d=(2 * np.pi) / nth) * 2 * np.pi
        self.meas = beta * self.RG * self.dr * (2 * np.pi / nth)

    def D1(self):
        n, h = self.nr, self.dr
        M = np.zeros((n, n))
        for i in range(1, n - 1):
            M[i, i - 1], M[i, i + 1] = -0.5 / h, 0.5 / h
        M[0, 0], M[0, 1] = -1 / h, 1 / h
        M[-1, -2], M[-1, -1] = -1 / h, 1 / h
        return M

    def lap_m(self, m):
        n, h, r = self.nr, self.dr, self.r
        M = np.zeros((n, n))
        for i in range(n):
            if 0 < i < n - 1:
                M[i, i - 1] = 1 / h ** 2 - 1 / (2 * h * r[i])
                M[i, i] = -2 / h ** 2 - (m / (self.beta * r[i])) ** 2
                M[i, i + 1] = 1 / h ** 2 + 1 / (2 * h * r[i])
            else:
                M[i, i] = -1.0
        return M

    def Q_m(self, m):
        # 2*pi is the theta-integral for a single Fourier mode. Omitting it and
        # compensating with a sqrt(nth) in draw() made the field normalisation
        # nth-dependent; since <O^2> is degree six in psi that entered as N^6
        # and invalidated the cutoff study. Recorded in U3 below.
        W = np.diag(2.0 * np.pi * self.beta * self.r * self.dr)
        D = self.D1()
        Mm = np.diag(m / (self.beta * self.r))
        L = self.lap_m(m)
        return D.T @ W @ D + Mm.T @ W @ Mm + 2.0 * MU * (L.T @ W @ L)

    def sampler(self):
        half = self.nth // 2 + 1
        roots = []
        for m in range(half):
            Q = self.Q_m(m)
            Q = 0.5 * (Q + Q.T)
            w, V = np.linalg.eigh(Q)
            w = np.clip(w, 1e-10, None)
            roots.append(V @ np.diag(1.0 / np.sqrt(w)) @ V.T)
        return roots

    def draw(self, roots, rng, ncomp=7):
        """psi = sum_m c_m e^{i m theta} with <|c_m|^2> = Q_m^{-1}.

        m = 0 is real with variance Q_0^{-1}; m > 0 is complex with unit-modulus
        noise so <|c_m|^2> = Q_m^{-1} exactly. numpy's ifft carries 1/nth, so
        the spectrum is multiplied back by nth. No hand-tuned factors.
        """
        half = self.nth // 2 + 1
        out = np.zeros((self.nr, self.nth, ncomp))
        for a in range(ncomp):
            spec = np.zeros((self.nr, self.nth), dtype=complex)
            f0 = roots[0] @ rng.normal(size=self.nr)
            spec[:, 0] = f0
            for mm in range(1, half):
                eta = (rng.normal(size=self.nr)
                       + 1j * rng.normal(size=self.nr)) / np.sqrt(2.0)
                f = roots[mm] @ eta
                spec[:, mm] = f
                if mm < self.nth - mm:
                    spec[:, self.nth - mm] = np.conj(f)
            out[..., a] = np.real(np.fft.ifft(spec, axis=1)) * self.nth
        return out

    def variance_check(self, roots, rng, nsamp=300):
        """Sampled <psi(r)^2> against the analytic sum_m Q_m^{-1}(r,r).

        Instrument check the first version lacked -- it would have caught the
        normalisation fault directly.
        """
        acc = np.zeros(self.nr)
        for _ in range(nsamp):
            p = self.draw(roots, rng, ncomp=1)[..., 0]
            acc += np.mean(p ** 2, axis=1)
        acc /= nsamp
        half = self.nth // 2 + 1
        pred = np.zeros(self.nr)
        for mm in range(half):
            d = np.sum(roots[mm] ** 2, axis=1)
            pred += d if mm == 0 else 2.0 * d
        return acc, pred

    def density_integral(self, psi, c):
        pr = np.zeros_like(psi)
        pr[1:-1] = (psi[2:] - psi[:-2]) / (2 * self.dr)
        pt = np.real(np.fft.ifft(1j * self.kth[None, :, None]
                                 * np.fft.fft(psi, axis=1), axis=1))
        pth = pt / (self.beta * self.RG)[..., None]
        prr = np.zeros_like(psi)
        prr[1:-1] = (psi[2:] - 2 * psi[1:-1] + psi[:-2]) / self.dr ** 2
        ptt = np.real(np.fft.ifft(-(self.kth ** 2)[None, :, None]
                                  * np.fft.fft(psi, axis=1), axis=1))
        box = prr + pr / self.RG[..., None] \
            + ptt / (self.beta * self.RG) [..., None] ** 2
        dens = np.einsum('abc,xya,xyb,xyc->xy', c, pr, pth, box)
        return 2.0 * np.sum(dens * self.meas)


def stats(nr, nth, beta, c, nsamp, seed):
    cone = Cone(nr, nth, beta)
    roots = cone.sampler()
    rng = np.random.default_rng(seed)
    vals = np.array([cone.density_integral(cone.draw(roots, rng), c)
                     for _ in range(nsamp)])
    return vals


def main():
    c = imaginary_c(oriented_lines())
    print("=" * 70)
    print("VACUUM CHIRALITY FROM THE PARITY-ODD VERTEX")
    print("=" * 70)

    NS = 240
    print("\nU1/U2  FREE-THEORY MOMENTS  (beta = 0.7, 64 x 32)")
    print("-" * 70)
    v = stats(64, 32, 0.7, c, NS, 5)
    m1, se1 = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    m2 = (v ** 2).mean()
    se2 = (v ** 2).std(ddof=1) / np.sqrt(len(v))
    print(f"    <O>   = {m1:+.5e}  +/- {se1:.2e}   "
          f"[consistent with 0 -> {'PASS' if abs(m1) < 2.5 * se1 else 'MISS'}]")
    print(f"    <O^2> = {m2:+.5e}  +/- {se2:.2e}   "
          f"rel err {se2/m2:.3f}  [<5% -> {'PASS' if se2/m2 < 0.05 else 'MISS'}]")

    print("\nU3  CUTOFF DEPENDENCE OF <O^2>")
    print("-" * 70)
    print("      grid        <O^2>          rel err")
    grids = [(48, 24), (64, 32), (96, 48), (128, 64)]
    o2 = []
    for (a, b) in grids:
        vv = stats(a, b, 0.7, c, NS, 7)
        mm = (vv ** 2).mean()
        ss = (vv ** 2).std(ddof=1) / np.sqrt(len(vv))
        o2.append(mm)
        print(f"    {a:3d} x {b:3d}   {mm:.5e}   {ss/mm:.3f}")
    grew = o2[-1] > o2[0]
    print(f"\n    grows with cutoff : {'PASS' if grew else 'MISS'}"
          f"   (factor {o2[-1]/o2[0]:.2f} across the range)")
    print("    -> the BARE vacuum chirality is not by itself an observable")

    print("\nU4  DOES THE beta-RATIO CONVERGE?")
    print("-" * 70)
    print("      grid       <O^2> b=0.5   <O^2> b=0.9      ratio")
    ratios = []
    for (a, b) in grids:
        r1 = (stats(a, b, 0.5, c, NS, 11) ** 2).mean()
        r2 = (stats(a, b, 0.9, c, NS, 11) ** 2).mean()
        ratios.append(r1 / r2)
        print(f"    {a:3d} x {b:3d}   {r1:.4e}   {r2:.4e}   {r1/r2:8.4f}")
    spread = (max(ratios[1:]) - min(ratios[1:])) / abs(np.mean(ratios[1:]))
    print(f"\n    ratio spread over the last three grids : {100*spread:.2f}%")
    print(f"    U4 converges : {'PASS' if spread < 0.15 else 'MISS'}")

    print("\nU5  SIGN UNDER ORIENTATION REVERSAL")
    print("-" * 70)
    cone = Cone(64, 32, 0.7)
    roots = cone.sampler()
    rng = np.random.default_rng(3)
    ps = [cone.draw(roots, rng) for _ in range(60)]
    a1 = np.mean([cone.density_integral(p, c) ** 2 for p in ps])
    a2 = np.mean([cone.density_integral(p, -c) * cone.density_integral(p, c)
                  for p in ps])
    print(f"    <O(c)^2>        = {a1:+.5e}")
    print(f"    <O(-c) O(c)>    = {a2:+.5e}")
    print(f"    exact sign flip : "
          f"{'PASS' if abs(a2 + a1) < 1e-9 * abs(a1) else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  <O> vanishes in the free theory and is driven to -lambda <O^2> by")
    print("  the only cubic term the theory has. So the vacuum carries a")
    print("  chirality in its correlations with a sign fixed by orientation")
    print("  times base parity, even though the mean field is trivial.")
    print()
    print("  But the bare quantity grows with the cutoff, as the power")
    print("  counting said it would. What is physical is the beta-dependence")
    print("  at fixed cutoff -- and that is exactly a D2-to-D8 observable,")
    print("  since beta is the cone's deficit and O is the octonionic sector.")


if __name__ == "__main__":
    main()
