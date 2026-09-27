#!/usr/bin/env python3
"""
gravity_scope_checks.py -- checks behind natural/GRAVITY_SCOPE.md (target 2, scoping only;
nothing here extends the model). Hypotheses and predictions: natural/GRAVITY_HYPOTHESES.md
(e3365d4), committed before this file was written.

G1a  H1: at fixed k, beta's frequency shift vs a u(1) Peierls phase's, at three stiffnesses K.
G1b  H1: the beta force is the Euler-Lagrange force of L_beta = -beta c P with
         P = -sum u'_n (u_{n+1} - u_{n-1})/2; and whether P is conserved at beta = 0
         (linear lattice vs the phi-well).
G2   H3: free fall of a packet at rest under a lapse potential Phi = g x, H = sum (1+Phi) h:
         scalar sector at K = sqrt5, 4 sqrt5, 16 sqrt5; gyroscopic sector (kappa*) a and b branches.
         Predicted a/(-c g): 1.000 (scalar, all K); 0.691 (a), 1.309 (b).
     A diagnostic (added after the first run) varies g and the packet width for the heaviest
     scalar packet: its ~2% shortfall scales as g^2 -- the k the packet reaches -- not with width.
G4   H5: Coleman's criterion for softening node forms; a self-gravitating Gaussian lump under
         node form A' (energy minimum exists for every G > 0?).
"""

import math
import numpy as np

SQ5 = math.sqrt(5.0)
C = 1.0
BETA = 0.05
KSTAR = 2 * C / math.sqrt(SQ5 + 2 * C)
rng = np.random.default_rng(1)


# ------------------------------------------------------------------ G1
def g1a():
    k = math.pi / 2
    th = 0.05
    print("  G1a  first-order frequency shift at k = pi/2, beta = 0.05 vs u(1) phase theta = 0.05")
    print("        K         beta: d(omega)      u(1): d(omega)     group velocity")
    for K in (SQ5, 4 * SQ5, 16 * SQ5):
        Q = lambda kk: K + 2 * C * (1 - math.cos(kk))
        w0 = math.sqrt(Q(k))
        b = C * BETA * math.sin(k)
        wb = b + math.sqrt(b * b + Q(k))            # reference-convention upper root
        h = 1e-6
        dbeta = ((C * (BETA + h) * math.sin(k) + math.sqrt((C * (BETA + h) * math.sin(k)) ** 2 + Q(k)))
                 - (C * (BETA - h) * math.sin(k) + math.sqrt((C * (BETA - h) * math.sin(k)) ** 2 + Q(k)))) / (2 * h)
        du1 = math.sqrt(Q(k + th)) - w0
        vg = C * math.sin(k) / w0
        print(f"      {K:7.3f}     {wb - w0:+.5f} (d/dbeta {dbeta:.4f})   {du1:+.5f}          {vg:.4f}")


def g1b():
    N = 16
    u = rng.standard_normal(N); ud = rng.standard_normal(N)
    # Euler-Lagrange force of L_beta = (beta c /2) sum ud_n (u_{n+1} - u_{n-1})
    dL_du = (BETA * C / 2) * (np.roll(ud, 1) - np.roll(ud, -1))
    ddt_dL_dud = (BETA * C / 2) * (np.roll(ud, -1) - np.roll(ud, 1))
    F_el = dL_du - ddt_dL_dud
    F_model = BETA * C * (np.roll(ud, 1) - np.roll(ud, -1))
    print(f"  G1b  EL force of -beta c P vs the model's beta force: max diff {np.max(np.abs(F_el - F_model)):.1e}")

    def run(nonlin):
        n = 64
        x = 0.3 * np.exp(-0.5 * ((np.arange(n) - 32) / 4.0) ** 2) * np.cos(math.pi / 2 * np.arange(n))
        v = 0.3 * 2.0 * np.exp(-0.5 * ((np.arange(n) - 32) / 4.0) ** 2) * np.sin(math.pi / 2 * np.arange(n))

        def f(x):
            return -SQ5 * x - (x * x if nonlin else 0) + C * (np.roll(x, 1) + np.roll(x, -1) - 2 * x)

        def P(x, v):
            return -0.5 * np.sum(v * (np.roll(x, -1) - np.roll(x, 1)))
        P0 = P(x, v); dt = 0.01; worst = 0.0
        for _ in range(20000):
            k1v = f(x); k1x = v
            k2v = f(x + .5 * dt * k1x); k2x = v + .5 * dt * k1v
            k3v = f(x + .5 * dt * k2x); k3x = v + .5 * dt * k2v
            k4v = f(x + dt * k3x); k4x = v + dt * k3v
            x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            worst = max(worst, abs(P(x, v) - P0))
        return P0, worst
    for nl in (False, True):
        P0, w = run(nl)
        print(f"  G1b  P conservation at beta = 0, {'phi-well (u^2 term)' if nl else 'linear lattice':20s}: "
              f"P0 = {P0:+.4f}, max |P - P0| over T = 200: {w:.1e}")


# ------------------------------------------------------------------ G2
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def fall(K, kappa, branch, g=2e-4, N=1400, sigma=30.0, T=300.0, dt=0.05):
    """Linear lattice, 2 components per site, lapse Phi_n = g (n - n0) on sites and bonds.
    Returns fitted acceleration of the energy centroid and the relative energy drift."""
    n = np.arange(N); n0 = N // 2
    Phi = g * (n - n0); Phib = g * (n + 0.5 - n0)           # bond n -> n+1
    env = np.exp(-0.5 * ((n - n0) / sigma) ** 2)
    if kappa == 0:
        w = math.sqrt(K); eps = np.array([1.0, 0.0])
        u = np.outer(env, eps); q = np.zeros_like(u)          # at rest, standing
    else:
        # J e = +i e  -> w^2 - kappa w - K = 0 (b); J e = -i e -> w^2 + kappa w - K = 0 (a)
        R = math.sqrt(kappa ** 2 + 4 * K)
        if branch == "a":
            w = (-kappa + R) / 2; e = np.array([1.0, 1j]) / math.sqrt(2)
        else:
            w = (kappa + R) / 2; e = np.array([1.0, -1j]) / math.sqrt(2)
        assert np.allclose(J @ e, (1j if branch == "b" else -1j) * e)
        psi = np.outer(env, e)
        u = psi.real.copy(); q = (-1j * w * psi).real.copy()  # q = u'
    lap = 1 + Phi
    p = q / lap[:, None] + (kappa / 2) * (u @ J.T)

    def rhs(u, p):
        qv = p - (kappa / 2) * (u @ J.T)
        ud = lap[:, None] * qv
        d = np.zeros_like(u)
        d[:-1] += (1 + Phib[:-1])[:, None] * (u[1:] - u[:-1])
        d[1:] -= (1 + Phib[:-1])[:, None] * (u[1:] - u[:-1])
        pd = -lap[:, None] * ((kappa / 2) * (qv @ J.T) + K * u) + C * d
        return ud, pd

    def energy_density(u, p):
        qv = p - (kappa / 2) * (u @ J.T)
        e = 0.5 * np.sum(qv ** 2, 1) + 0.5 * K * np.sum(u ** 2, 1)
        b = 0.5 * C * np.sum((u[1:] - u[:-1]) ** 2, 1)
        e[:-1] += 0.5 * b; e[1:] += 0.5 * b
        return e

    def H(u, p):
        qv = p - (kappa / 2) * (u @ J.T)
        e = lap * (0.5 * np.sum(qv ** 2, 1) + 0.5 * K * np.sum(u ** 2, 1))
        return e.sum() + 0.5 * C * np.sum((1 + Phib[:-1]) * np.sum((u[1:] - u[:-1]) ** 2, 1))
    E0 = H(u, p)
    ts, xs = [], []
    steps = int(T / dt)
    for s in range(steps + 1):
        if s % 20 == 0:
            e = energy_density(u, p)
            ts.append(s * dt); xs.append(np.sum(n * e) / np.sum(e))
        if s == steps:
            break
        k1 = rhs(u, p)
        k2 = rhs(u + .5 * dt * k1[0], p + .5 * dt * k1[1])
        k3 = rhs(u + .5 * dt * k2[0], p + .5 * dt * k2[1])
        k4 = rhs(u + dt * k3[0], p + dt * k3[1])
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        p = p + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    ts, xs = np.array(ts), np.array(xs)
    a = 2 * np.polyfit(ts, xs, 2)[0]
    return a, abs(H(u, p) - E0) / abs(E0), w


def g2():
    g = 2e-4
    print(f"  G2   free fall under a lapse potential Phi = g x, g = {g}: a / (-c g), predicted in brackets")
    for K in (SQ5, 4 * SQ5, 16 * SQ5):
        a, dr, w = fall(K, 0.0, None, g, dt=0.1 / math.sqrt(K + 4 * C))   # dt * omega_max = 0.1
        print(f"        scalar      K = {K:7.3f}  omega = {w:.4f}:  {a/(-C*g):.4f}  [1.000]   energy drift {dr:.1e}")
    for br, s in (("a", +1), ("b", -1)):
        a, dr, w = fall(SQ5, KSTAR, br, g, dt=0.1 / math.sqrt(SQ5 + 4 * C))
        pred = w / (w + s * KSTAR / 2)
        print(f"        gyroscopic  {br}-branch, kappa* = {KSTAR:.4f}, omega = {w:.4f}:  {a/(-C*g):.4f}  "
              f"[{pred:.3f}]   energy drift {dr:.1e}")
    # diagnostic added after the first run: the heavy scalar's shortfall scales as g^2, not with
    # packet width -- the packet reaches k = g w T (0.36 at K = 16 sqrt5), where v = c sin k / w
    # departs from ck/w. Universality is a small-velocity (Newtonian) statement.
    K = 16 * SQ5
    for gg, sig in ((1e-4, 30.0), (2e-4, 60.0)):
        a, dr, w = fall(K, 0.0, None, gg, sigma=sig, dt=0.1 / math.sqrt(K + 4 * C))
        print(f"        diagnostic  K = {K:7.3f}, g = {gg:g}, sigma = {sig:g}:  {a/(-C*gg):.4f}   "
              f"(k reached ~ g w T = {gg*w*300:.2f})")


# ------------------------------------------------------------------ G4
def g4():
    K = SQ5
    r = np.linspace(1e-4, 12, 120001)
    forms = {
        "A' (main): K r^2/2 + r^3/3": K * r ** 2 / 2 + r ** 3 / 3,
        "pendulum: K (1 - cos r)": K * (1 - np.cos(r)),
        "smooth |psi|^2 psi softening + sextic: K r^2/2 - r^4/4 + r^6/6 (lam = sig = 1)": K * r ** 2 / 2 - r ** 4 / 4 + r ** 6 / 6,
        "smooth |psi|^2 psi hardening: K r^2/2 + r^4/4": K * r ** 2 / 2 + r ** 4 / 4,
    }
    for name, V in forms.items():
        m = np.min(2 * V / r ** 2)
        print(f"  G4   {name:78s} min 2V/r^2 - K = {m - K:+.4f}  -> "
              f"{'Q-balls admitted' if m < K - 1e-9 else 'no Q-balls'}")
    # self-gravitating Gaussian lump under A' (q = 3), norm N: E(R) = a N/R^2 + b N^1.5 R^-1.5 - G N^2/R
    a = 1.5 / 2                                           # (1/2) int |grad psi|^2 = 3N/(4R^2)
    b = (1 / 3) * math.pi ** (-9 / 4) * (2 * math.pi / 3) ** 1.5
    Nn = 1.0
    R = np.logspace(-2, 16, 400001)
    for G in (1.0, 1e-2, 1e-4, 1e-6):
        E = a * Nn / R ** 2 + b * Nn ** 1.5 * R ** -1.5 - G * Nn ** 2 / (math.sqrt(2 * math.pi) * R)
        i = int(np.argmin(E))
        print(f"  G4   self-gravitating Gaussian lump under A', G = {G:g}: E_min = {E[i]:+.3e} at R = {R[i]:.3e}  "
              f"({'bound (E < 0, interior minimum)' if E[i] < 0 and 0 < i < len(R) - 1 else 'NOT bound'})")


if __name__ == "__main__":
    print("GRAVITY SCOPING CHECKS (natural/GRAVITY_HYPOTHESES.md)")
    g1a(); g1b(); g2(); g4()
