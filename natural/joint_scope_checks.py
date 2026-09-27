#!/usr/bin/env python3
"""
joint_scope_checks.py -- checks behind natural/JOINT_SCOPE.md (target 3, scoping only; nothing
here extends the model). Hypotheses and predictions: natural/JOINT_HYPOTHESES.md (d054844).

J1  H2: light and matter in a lapse gradient Phi = g y on a 2-D lattice,
        H = sum (1+Phi) (p^2/2 + K u^2/2) + sum_bonds (1+Phi_b) c_w (du)^2/2.
        Matter: K = sqrt5, c_w = c = 1. Light (one lattice-Maxwell polarisation): K = 0,
        c_w = c_gamma^2. Transverse acceleration of the energy centroid in units of -c g.
        Predicted: 1.000 (matter at rest), 1.000 (matter moving), 1.000 (light, c_gamma^2 = c),
        1.500 (light, c_gamma^2 = 1.5 c).
    A diagnostic (added after the first run) varies g and the packet width: light's ~2%
    shortfall is a finite-width (diffraction) effect, not a g effect.
J2  H2: the kappa* branches with the lapse coupled to the GAUGE-INVARIANT (rotating-frame)
        energy, H = sum (1+Phi)(|p|^2/2 + K'|u|^2/2) - (kappa/2) sum p.J u + bonds,
        K' = K + kappa^2/4; the charge term (A0 times the Gauss constraint) is not
        lapse-weighted. Predicted: both branches 1.000. Contrast: lapse on lab energy
        (gravity_scope_checks.fall): 0.69 / 1.30.
J4  L1 (late hypothesis, committed at 8a40790 before this check was written): gauge dependence.
        In a frame rotating at nu -- a gauge transformation once kappa is eA0 -- the system has
        gyroscopic kappa_nu = kappa - 2 nu and stiffness K_nu = K + nu kappa - nu^2. A lapse on
        that frame's energy ("lab-type") should give a nu-dependent fall,
        (w_rot -+ (kappa/2 - nu))/w_rot; the gauge-invariant lapse 1.000 at every nu.
J3  H3: self-gravitating charged Gaussian lump under node form A' (q = 3):
        E(R) = a N/R^2 + b N^1.5 R^-1.5 + (lam - 1) G w^2 N^2/(sqrt(2 pi) R),
        lam = e^2/(4 pi G w^2). Predicted: bound for lam = 0.5, 0.9; unbound for 1.1, 2.
"""

import math
import numpy as np
import gravity_scope_checks as GS

SQ5 = math.sqrt(5.0)
C = 1.0
KSTAR = GS.KSTAR
J = GS.J


def fall2d(K, cw, kx, g=4e-4, Nx=256, Ny=160, sigma=12.0, T=150.0):
    x = np.arange(Nx)[:, None]; y = np.arange(Ny)[None, :]
    y0 = Ny // 2
    Phi = g * (y - y0) * np.ones((Nx, 1))
    Phiy = g * (y + 0.5 - y0) * np.ones((Nx, 1))           # bond y -> y+1
    env = np.exp(-0.5 * (((x - Nx // 2) / sigma) ** 2 + ((y - y0) / sigma) ** 2))
    w = math.sqrt(K + 2 * cw * (1 - math.cos(kx)))
    u = env * np.cos(kx * x)
    q = env * w * np.sin(kx * x) if kx else np.zeros_like(u)   # q = u' (travelling in +x)
    lap = 1 + Phi
    p = q / lap
    dt = 0.1 / math.sqrt(K + 8 * cw)

    def rhs(u, p):
        ud = lap * p
        fx = (np.roll(u, -1, 0) - u)                        # bond x -> x+1 (periodic), lapse = site's
        lapx = 1 + 0.5 * (Phi + np.roll(Phi, -1, 0))
        f = cw * (lapx * fx - np.roll(lapx * fx, 1, 0))
        dy = np.zeros_like(u)
        by = (1 + Phiy[:, :-1]) * (u[:, 1:] - u[:, :-1])
        dy[:, :-1] += by; dy[:, 1:] -= by
        pd = -lap * K * u + f + cw * dy
        return ud, pd

    def dens(u, p):
        e = 0.5 * (lap * p) ** 2 + 0.5 * K * u ** 2
        bx = 0.5 * cw * (np.roll(u, -1, 0) - u) ** 2
        e += 0.5 * (bx + np.roll(bx, 1, 0))
        by = 0.5 * cw * (u[:, 1:] - u[:, :-1]) ** 2
        e[:, :-1] += 0.5 * by; e[:, 1:] += 0.5 * by
        return e
    ts, ys = [], []
    steps = int(T / dt)
    for s in range(steps + 1):
        if s % 10 == 0:
            e = dens(u, p)
            ts.append(s * dt); ys.append(np.sum(y * e) / np.sum(e))
        if s == steps:
            break
        k1 = rhs(u, p)
        k2 = rhs(u + .5 * dt * k1[0], p + .5 * dt * k1[1])
        k3 = rhs(u + .5 * dt * k2[0], p + .5 * dt * k2[1])
        k4 = rhs(u + dt * k3[0], p + dt * k3[1])
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        p = p + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    a = 2 * np.polyfit(np.array(ts), np.array(ys), 2)[0]
    return a


def j1():
    g = 4e-4
    print(f"  J1   transverse fall in a lapse gradient, g = {g}: a / (-c g), predicted in brackets")
    for name, K, cw, kx, pred in (("matter at rest     ", SQ5, C, 0.0, 1.0),
                                  ("matter moving kx=0.6", SQ5, C, 0.6, 1.0),
                                  ("light c_g^2 = c   ", 0.0, C, 0.6, 1.0),
                                  ("light c_g^2 = 1.5 c", 0.0, 1.5 * C, 0.6, 1.5)):
        a = fall2d(K, cw, kx, g)
        print(f"        {name}:  {a/(-C*g):.4f}  [{pred:.3f}]", flush=True)
    # diagnostic added after the first run: light's ~2% shortfall shrinks with packet width
    # (diffraction of the massless packet), barely with g -- light/matter -> 1 as sigma grows.
    for gg, sig in ((2e-4, 12.0), (4e-4, 20.0), (2e-4, 20.0)):
        al = fall2d(0.0, C, 0.6, gg, sigma=sig); am = fall2d(SQ5, C, 0.6, gg, sigma=sig)
        print(f"        diagnostic g = {gg:g}, sigma = {sig:g}: light {al/(-C*gg):.4f}, "
              f"matter moving {am/(-C*gg):.4f}, light/matter {al/am:.4f}", flush=True)


def fall_invariant(K, kappa, branch, g=2e-4, N=1400, sigma=30.0, T=300.0):
    """As gravity_scope_checks.fall, but the lapse weights the gauge-invariant energy."""
    n = np.arange(N); n0 = N // 2
    Phi = g * (n - n0); Phib = g * (n + 0.5 - n0)
    env = np.exp(-0.5 * ((n - n0) / sigma) ** 2)
    R = math.sqrt(kappa ** 2 + 4 * K)
    if branch == "a":
        w = (-kappa + R) / 2; e = np.array([1.0, 1j]) / math.sqrt(2)
    else:
        w = (kappa + R) / 2; e = np.array([1.0, -1j]) / math.sqrt(2)
    psi = np.outer(env, e)
    u = psi.real.copy(); ud0 = (-1j * w * psi).real.copy()
    Kp = K + kappa ** 2 / 4
    lap = 1 + Phi
    p = (ud0 + (kappa / 2) * (u @ J.T)) / lap[:, None]
    dt = 0.1 / math.sqrt(Kp + 4 * C)

    def rhs(u, p):
        ud = lap[:, None] * p - (kappa / 2) * (u @ J.T)
        d = np.zeros_like(u)
        b = (1 + Phib[:-1])[:, None] * (u[1:] - u[:-1])
        d[:-1] += b; d[1:] -= b
        pd = -lap[:, None] * Kp * u - (kappa / 2) * (p @ J.T) + C * d
        return ud, pd

    def dens(u, p):
        e = 0.5 * np.sum(p ** 2, 1) + 0.5 * Kp * np.sum(u ** 2, 1)
        bb = 0.5 * C * np.sum((u[1:] - u[:-1]) ** 2, 1)
        e[:-1] += 0.5 * bb; e[1:] += 0.5 * bb
        return e

    def H(u, p):
        return (np.sum(lap * (0.5 * np.sum(p ** 2, 1) + 0.5 * Kp * np.sum(u ** 2, 1)))
                - (kappa / 2) * np.sum(p * (u @ J.T))
                + 0.5 * C * np.sum((1 + Phib[:-1]) * np.sum((u[1:] - u[:-1]) ** 2, 1)))
    E0 = H(u, p)
    ts, xs = [], []
    steps = int(T / dt)
    for s in range(steps + 1):
        if s % 20 == 0:
            e = dens(u, p)
            ts.append(s * dt); xs.append(np.sum(n * e) / np.sum(e))
        if s == steps:
            break
        k1 = rhs(u, p)
        k2 = rhs(u + .5 * dt * k1[0], p + .5 * dt * k1[1])
        k3 = rhs(u + .5 * dt * k2[0], p + .5 * dt * k2[1])
        k4 = rhs(u + dt * k3[0], p + dt * k3[1])
        u = u + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        p = p + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    a = 2 * np.polyfit(np.array(ts), np.array(xs), 2)[0]
    return a, abs(H(u, p) - E0) / abs(E0), w


def j2():
    g = 2e-4
    print(f"  J2   kappa* branches, lapse on gauge-invariant vs lab energy, g = {g}: a / (-c g)")
    for br in ("a", "b"):
        ai, dr, w = fall_invariant(SQ5, KSTAR, br, g)
        al, _, _ = GS.fall(SQ5, KSTAR, br, g, dt=0.1 / math.sqrt(SQ5 + 4 * C))
        print(f"        {br}-branch (lab omega {w:.4f}):  invariant-energy lapse {ai/(-C*g):.4f} [1.000]   "
              f"lab-energy lapse {al/(-C*g):.4f}   energy drift {dr:.1e}", flush=True)


def j3():
    a = 0.75
    b = (1 / 3) * math.pi ** (-9 / 4) * (2 * math.pi / 3) ** 1.5
    Gw2 = 1e-2
    R = np.logspace(-2, 14, 400001)
    print(f"  J3   charged self-gravitating Gaussian lump under A' (G w^2 = {Gw2}):")
    for lam in (0.5, 0.9, 1.1, 2.0):
        E = a / R ** 2 + b * R ** -1.5 + (lam - 1) * Gw2 / (math.sqrt(2 * math.pi) * R)
        i = int(np.argmin(E))
        bound = E[i] < 0 and 0 < i < len(R) - 1
        print(f"        lam = e^2/(4 pi G w^2) = {lam:3.1f}:  E_min = {E[i]:+.3e} at R = {R[i]:.3e}  "
              f"-> {'bound' if bound else 'NOT bound'}")


def j4():
    g = 2e-4
    k = KSTAR
    wr = math.sqrt(SQ5 + k * k / 4)
    print(f"  J4   fall in a frame rotating at nu (a gauge transformation), g = {g}: a / (-c g), "
          f"predicted in brackets")
    print("          nu        lab-type a      lab-type b      invariant a     invariant b")
    for nu in (-k / 2, 0.0, k / 4, k / 2):
        kn, Kn = k - 2 * nu, SQ5 + nu * k - nu * nu
        dt = 0.1 / math.sqrt(Kn + kn * kn / 4 + 4 * C)
        cells = []
        for br, s in (("a", +1), ("b", -1)):
            al, _, _ = GS.fall(Kn, kn, br, g, dt=dt)
            cells.append(f"{al/(-C*g):.4f} [{(wr - s*(k/2 - nu))/wr:.3f}]")
        for br in ("a", "b"):
            ai, _, _ = fall_invariant(Kn, kn, br, g)
            cells.append(f"{ai/(-C*g):.4f} [1.000]")
        print(f"        {nu:+.4f}   " + "   ".join(cells), flush=True)


if __name__ == "__main__":
    print("JOINT SCOPING CHECKS (natural/JOINT_HYPOTHESES.md)")
    j1(); j2(); j3(); j4()
