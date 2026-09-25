#!/usr/bin/env python3
"""
kappa_launch_pt.py -- derivation of the launch pieces of the plane-wave kappa.

A launch that is not the exact travelling wave X(A) (kappa_pw4_pt.py) is the wave
plus a deviation delta. What the readout measures is the frequency of the
fundamental on the motion that follows.

FIRST ORDER in delta -- energy projection. The linearised flow about the periodic
wave has two neutral directions (phase d/dtheta X and amplitude d/dA X); every
other Floquet mode has multiplier != 1. The energy E is conserved, so its
differential dE is invariant under the linearised flow: dE(xi) = 0 for any Floquet
mode xi with multiplier != 1, and dE(d/dtheta X) = 0 because E is constant along
the wave. Hence the deviation's amplitude component is exactly
        dA = dE(delta) / E'(A),        dW = W'(A) dE(delta) / E'(A).
  * uniform (static-shift) deviation: dE = -sum_n u''_n = 0 identically
    (sum_n u_n is constant on a travelling wave; the gyroscopic term sums to 0),
    so the static-shift piece is second order;
  * staggered (k = pi) deviation beta (-1)^n: dE = 8 N W^2 c2 beta; with
    E'(A) = N A W sqrt(b^2 + Q) (b = c beta sin sK) and the launch's
    beta = -2 c2 this gives, to leading order,
        dW = -32 W W2 c2^2 / sqrt(b^2 + Q)     (W2, c2 the A^2 coefficients; x A^4)
  * velocity at the linear frequency:
        dW = -W2^2 A^4 / sqrt(b^2 + Q).
    sqrt(b^2 + Q) is the same in both directions, so the odd part comes from W2^2.
The script evaluates dE, E'(A) and W'(A) exactly (harmonic balance, 24 harmonics)
and prints the leading closed forms beside them.

SECOND ORDER -- the wave with free modes, by harmonic balance on a torus. The
launch leaves free oscillations of the uniform mode (k = 0, near 5^(1/4)) and the
staggered mode (k = pi, near sqrt(Q(pi))). Write the motion as a three-frequency
torus
    u_n(t) = sum_{m,j,l} c_mjl exp i(m theta_n + j phi0 + l phipi),
    theta_n = sKn - W t,  phi0 = -Omega0 t,  phipi = pi n - Omegapi t,
term (m,j,l) at wavevector msK + l pi and frequency mW + j Omega0 + l Omegapi,
each answering with the lattice operator D(k, w) = -w^2 + Q(k) + 2 c beta w sin k,
    D c_mjl + (u^2)_mjl = 0         (exact harmonic balance; all c real).
Fix c_100 = A*/2, c_010 = b0, c_001 = bpi and solve for W, Omega0, Omegapi.
Match a launch by three conditions at t = 0: its uniform (k = 0) displacement, its
staggered (k = pi) displacement, and its energy. Both free modes are always kept:
the wave's second harmonic (k = pi, 2W) lies near Omega0 + Omegapi -- the
(0, pi) parametric channel of the P-1 investigation -- with detuning 0.24 for +k
and 0.04 for -k at A = 0.3, so a free uniform mode drives a staggered one (about
seven times more strongly for -k). That direction asymmetry is where the
static-shift piece's odd part comes from. The inner harmonic balance is solved by
Newton's method with an analytic Jacobian. (The launch's small
fundamental-sector velocity mismatch goes into the counter-propagating wave,
whose energy is O(A^6) and is neglected; its first-order effect is in the energy.)
The torus frequency W for each direction gives kappa for that launch.

CHECKS (printed): the torus with b0 = bpi = 0 reproduces the wave; the first-order
closed forms against the exact projection; every launch piece against the
measured attribution (kappa_launch_attrib_output.txt), at every amplitude.

usage:  python3 kappa_launch_pt.py
"""

import math
import numpy as np
from scipy.optimize import fsolve
from scipy.signal import fftconvolve

import kappa_pw4_pt as P

SQ5 = P.SQ5
C, BETA, K, TH = P.C, P.BETA, P.K, P.TH
N = 8
PHI = (1 + math.sqrt(5)) / 2
OM0 = 5 ** 0.25
OMPI = math.sqrt(SQ5 + 4 * C)


# ------------------------------------------------------------ lattice energy
def energy(u, v):
    x = PHI + u
    V = x ** 3 / 3 - x ** 2 / 2 - x
    el = 0.5 * C * (np.roll(u, -1) - u) ** 2
    return float(np.sum(0.5 * v * v + V + el))


def orbit_state(s, A):
    sol = P.hb_exact(s, A)
    W = sol[0]
    c = np.zeros(P.M_HARM + 1)
    c[0], c[1], c[2:] = sol[1], A / 2, sol[2:]
    th = s * K * np.arange(N)
    u = c[0] + sum(2 * c[m] * np.cos(m * th) for m in range(1, P.M_HARM + 1))
    v = sum(2 * m * W * c[m] * np.sin(m * th) for m in range(1, P.M_HARM + 1))
    return u, v, W, c


def launch_state(kind, s, A):
    """(u, v) at t = 0 for a launch; the linear launch is the plain cosine."""
    if kind == "linear":
        n = np.arange(N)
        return A * np.cos(s * K * n), A * P.W_lin(s) * np.sin(s * K * n)
    u, v, W, c = orbit_state(s, A)
    th = s * K * np.arange(N)
    if kind in ("no c0", "fund only"):
        u = u - c[0]
    if kind in ("no c2", "fund only"):
        drop = [2] if kind == "no c2" else range(2, P.M_HARM + 1)
        for m in drop:
            u = u - 2 * c[m] * np.cos(m * th)
            v = v - 2 * m * W * c[m] * np.sin(m * th)
    return u, v


# ------------------------------------------------------- first order: dE
def first_order(kind, s, A, h=1e-5):
    u0, v0, W, _ = orbit_state(s, A)
    ul, vl = launch_state(kind, s, A)
    du, dv = ul - u0, vl - v0
    dE = (energy(u0 + h * du, v0 + h * dv) - energy(u0 - h * du, v0 - h * dv)) / (2 * h)
    Ep = (energy(*orbit_state(s, A + h)[:2]) - energy(*orbit_state(s, A - h)[:2])) / (2 * h)
    Wp = (P.hb_exact(s, A + h)[0] - P.hb_exact(s, A - h)[0]) / (2 * h)
    return Wp * dE / Ep


# --------------------------------------------------- second order: torus HB
MT, JMAX = 8, 3


class Torus:
    """Harmonic balance on the torus (wave + free uniform mode + free staggered
    mode). A mode the launch does not excite is left out (its index range is 0)."""

    def __init__(self, s, use0=True, usepi=True):
        self.s = s
        self.M, self.J, self.L = MT, (JMAX if use0 else 0), (JMAX if usepi else 0)
        self.use0, self.usepi = use0, usepi
        M, J, L = self.M, self.J, self.L
        m = np.arange(-M, M + 1)[:, None, None]
        j = np.arange(-J, J + 1)[None, :, None]
        l = np.arange(-L, L + 1)[None, None, :]
        self.m, self.j, self.l = m, j, l
        self.shape = (2 * M + 1, 2 * J + 1, 2 * L + 1)
        self.k = (m * s * K + l * math.pi) * np.ones(self.shape)
        idx = np.array(np.unravel_index(np.arange(np.prod(self.shape)), self.shape)).T - [M, J, L]
        keep = []
        for t in idx:
            nz = t[np.nonzero(t)[0]]
            if len(nz) == 0 or nz[0] > 0:
                keep.append(tuple(int(v) for v in t))
        self.fixed = [(1, 0, 0)] + ([(0, 1, 0)] if use0 else []) + ([(0, 0, 1)] if usepi else [])
        self.free = [t for t in keep if t not in self.fixed]
        self.eqs = keep
        self.nf = 1 + use0 + usepi                      # frequencies solved for
        self.guess = None

    def _amps(self, A, b0, bpi):
        a = {(1, 0, 0): A / 2}
        if self.use0:
            a[(0, 1, 0)] = b0
        if self.usepi:
            a[(0, 0, 1)] = bpi
        return a

    def full(self, x, A, b0, bpi):
        M, J, L = self.M, self.J, self.L
        c = np.zeros(self.shape)
        vals = dict(zip(self.free, x[self.nf:]))
        vals.update(self._amps(A, b0, bpi))
        for (a, b, d), v in vals.items():
            c[a + M, b + J, d + L] = v
            c[-a + M, -b + J, -d + L] = v
        return c

    def freqs(self, x):
        W = x[0]
        O0 = x[1] if self.use0 else 0.0
        Op = x[1 + self.use0] if self.usepi else 0.0
        return W, O0, Op

    def omega(self, x):
        W, O0, Op = self.freqs(x)
        return self.m * W + self.j * O0 + self.l * Op

    def resid(self, x, A, b0, bpi):
        M, J, L = self.M, self.J, self.L
        c = self.full(x, A, b0, bpi)
        w = self.omega(x)
        Dm = -w * w + (SQ5 + 2 * C * (1 - np.cos(self.k))) + 2 * C * BETA * w * np.sin(self.k)
        sq = fftconvolve(c, c)[M:3 * M + 1, J:3 * J + 1, L:3 * L + 1]
        R = Dm * c + sq
        amp = self._amps(A, b0, bpi)
        return np.array([R[a + M, b + J, d + L] / amp.get((a, b, d), 1.0)
                         for (a, b, d) in self.eqs])

    def jacobian(self, x, A, b0, bpi):
        """Analytic Jacobian of resid: d(D c)/dc and d(c*c)/dc = 2 c(e - f) + 2 c(e + f)
        for each free unknown f (which sits at f and -f); d/dW etc. through D."""
        M, J, L = self.M, self.J, self.L
        c = self.full(x, A, b0, bpi)
        w = self.omega(x)
        Dm = -w * w + (SQ5 + 2 * C * (1 - np.cos(self.k))) + 2 * C * BETA * w * np.sin(self.k)
        dDdw = -2 * w + 2 * C * BETA * np.sin(self.k)
        amp = self._amps(A, b0, bpi)
        E = np.array(self.eqs)
        F = np.array(self.free)
        box = np.array([M, J, L])

        def lookup(idx):
            ok = np.all(np.abs(idx) <= box, axis=-1)
            ii = np.where(ok[..., None], idx + box, 0)
            return np.where(ok, c[ii[..., 0], ii[..., 1], ii[..., 2]], 0.0)

        Jm = np.zeros((len(E), self.nf + len(F)))
        conv = 2 * lookup(E[:, None, :] - F[None, :, :]) + 2 * lookup(E[:, None, :] + F[None, :, :])
        same = np.all(E[:, None, :] == F[None, :, :], axis=-1) | \
               np.all(E[:, None, :] == -F[None, :, :], axis=-1)
        De = Dm[E[:, 0] + M, E[:, 1] + J, E[:, 2] + L]
        Jm[:, self.nf:] = conv + same * De[:, None]
        ce = c[E[:, 0] + M, E[:, 1] + J, E[:, 2] + L]
        de = dDdw[E[:, 0] + M, E[:, 1] + J, E[:, 2] + L]
        cols = [E[:, 0]] + ([E[:, 1]] if self.use0 else []) + ([E[:, 2]] if self.usepi else [])
        for i, mult in enumerate(cols):
            Jm[:, i] = de * mult * ce
        scale = np.array([amp.get(tuple(e), 1.0) for e in self.eqs])
        return Jm / scale[:, None]

    def solve(self, A, b0, bpi):
        x = self._start()
        for _ in range(60):
            r = self.resid(x, A, b0, bpi)
            if np.max(np.abs(r)) < 1e-15:
                break
            dx = np.linalg.solve(self.jacobian(x, A, b0, bpi), -r)
            x = x + dx
            if np.max(np.abs(dx)) < 1e-16:
                break
        r = np.max(np.abs(self.resid(x, A, b0, bpi)))
        if r > 1e-12:
            raise RuntimeError(f"torus residual {r:.1e}")
        self.guess = x
        return x

    def _start(self):
        if self.guess is not None:
            return self.guess.copy()
        x0 = np.zeros(self.nf + len(self.free))
        x0[:self.nf] = [P.W_lin(self.s)] + ([OM0] if self.use0 else []) + \
                       ([OMPI] if self.usepi else [])
        return x0

    def solve_hybrd(self, A, b0, bpi):
        if self.guess is None:
            x0 = np.zeros(self.nf + len(self.free))
            x0[:self.nf] = [P.W_lin(self.s)] + ([OM0] if self.use0 else []) + \
                           ([OMPI] if self.usepi else [])
        else:
            x0 = self.guess
        x = fsolve(self.resid, x0, args=(A, b0, bpi), xtol=1e-14)
        r = np.max(np.abs(self.resid(x, A, b0, bpi)))
        if r > 1e-8:
            raise RuntimeError(f"torus residual {r:.1e}")
        self.guess = x
        return x

    def state0(self, x, A, b0, bpi):
        """(u, v) at t = 0 on the N-site ring."""
        c = self.full(x, A, b0, bpi)
        w = self.omega(x)
        n = np.arange(N)
        ph = np.exp(1j * self.k[..., None] * n)
        u = np.real(np.sum(c[..., None] * ph, axis=(0, 1, 2)))
        v = np.real(np.sum((c * (-1j) * w)[..., None] * ph, axis=(0, 1, 2)))
        return u, v


def sector(u):
    """(uniform, staggered) components of a ring field."""
    n = np.arange(N)
    return float(np.mean(u)), float(np.mean(u * (-1.0) ** n))


def torus_launch(kind, s, A):
    """Frequency of the fundamental after a launch, from the matched torus.
    Both free modes are always included and all three conditions matched: near
    the resonance 2W ~ Omega0 + Omegapi (the P-1 (0, pi) channel) a free uniform
    mode drives the staggered sector through the second harmonic, so a launch
    that leaves only one sector off the wave still starts both free modes."""
    use0 = usepi = True
    T = Torus(s, use0, usepi)
    ul, vl = launch_state(kind, s, A)
    E_target = energy(ul, vl)
    dc_t, pi_t = sector(ul)
    _, _, _, c = orbit_state(s, A)

    def unpack(p):
        As = p[0]
        b0 = p[1] if use0 else 0.0
        bpi = p[1 + use0] if usepi else 0.0
        return As, b0, bpi

    def match(p):
        As, b0, bpi = unpack(p)
        x = T.solve(As, b0, bpi)
        u, v = T.state0(x, As, b0, bpi)
        dc, pi = sector(u)
        out = [(energy(u, v) - E_target) / (A * A)]
        if use0:
            out.append(dc - dc_t)
        if usepi:
            out.append(pi - pi_t)
        return out

    p0 = [A, (dc_t - c[0]) / 2 + 1e-5, (pi_t - 2 * c[2]) / 2 + 1e-5]
    p = fsolve(match, p0, xtol=1e-13)
    x = T.solve(*unpack(p))
    return x[0], unpack(p)


def main():
    print("=" * 84)
    print("LAUNCH PIECES OF THE PLANE-WAVE KAPPA -- derived")
    print("=" * 84)
    # check: the torus with no free modes is the wave
    for s in (+1, -1):
        T = Torus(s, False, False)
        x = T.solve(0.3, 0.0, 0.0)
        print(f"  check, direction {s:+d}: torus W {x[0]:.10f}  wave W {P.hb_exact(s, 0.3)[0]:.10f}")

    print("\n  FIRST ORDER (energy projection), kappa increment over the wave:")
    print("     A     velocity: exact   closed form   |  2nd harmonic: exact   closed form   |  static shift")
    for A in (0.15, 0.20, 0.30, 0.40):
        inc = {}
        for kind in ("fund only", "no c2", "no c0"):
            d = {s: first_order(kind, s, A) for s in (+1, -1)}
            inc[kind] = (d[+1] - d[-1]) / (TH * A * A)
        # velocity-only first-order: the linear launch minus fund-only (fund-only
        # has no velocity mismatch) -- evaluated directly:
        dv = {s: first_order("linear", s, A) - first_order("fund only", s, A) for s in (+1, -1)}
        vel = (dv[+1] - dv[-1]) / (TH * A * A)
        cf_vel, cf_pi = {}, {}
        for s in (+1, -1):
            W0, W2, _, cc = P.analytic(s)
            c2h = cc["c2"]
            root = W0 - C * BETA * math.sin(s * K)             # sqrt(b^2 + Q), even
            cf_vel[s] = -W2 * W2 * A ** 4 / root
            cf_pi[s] = -32 * W0 * W2 * c2h * c2h * A ** 4 / root
        cvel = (cf_vel[+1] - cf_vel[-1]) / (TH * A * A)
        cpi = (cf_pi[+1] - cf_pi[-1]) / (TH * A * A)
        print(f"   {A:4.2f}   {vel:+.6f}      {cvel:+.6f}     |   {inc['no c2']:+.6f}          "
              f"{cpi:+.6f}      |   {inc['no c0']:+.2e}")

    print("\n  FULL (torus harmonic balance, energy-matched), kappa for each launch:")
    print("     A     orbit      no c0      no c2      fund only   linear")
    rows = {}
    for A in (0.15, 0.20, 0.30, 0.40):
        ks = {}
        for kind in ("orbit", "no c0", "no c2", "fund only", "linear"):
            if kind == "orbit":
                W = {s: P.hb_exact(s, A)[0] for s in (+1, -1)}
            else:
                W = {s: torus_launch(kind, s, A)[0] for s in (+1, -1)}
            ks[kind] = (abs(W[+1] - W[-1]) / TH - 1) / (A * A)
        rows[A] = ks
        print(f"   {A:4.2f}  " + "  ".join(f"{ks[k]:+.6f}" for k in
                                        ("orbit", "no c0", "no c2", "fund only", "linear")), flush=True)
    print("\n  increments over the orbit (compare kappa_launch_attrib_output.txt):")
    print("     A     no c0      no c2      fund only   linear")
    for A, ks in rows.items():
        print(f"   {A:4.2f}  " + "  ".join(f"{ks[k] - ks['orbit']:+.6f}" for k in
                                        ("no c0", "no c2", "fund only", "linear")))


if __name__ == "__main__":
    main()
