#!/usr/bin/env python3
"""
phi_gauge_chiral.py — A1: dynamical generation of non-Abelian transport
(Shape Zero action log item A1; upgrades v5.2 Result 4)

Each node carries u in R^4: two phi-attractor dimers. Structure:

  on-site   : per-component phi restoring force  -(sqrt5*u + u^2)
              + intra-node gyroscopic term  kappa*JJ*u'   (JJ = J2 (+) J2)
  inter-node: diffusive position coupling  c*(u_{n+1}+u_{n-1}-2u_n)
              + directional velocity coupling  c*(W_n v_{n+1} - W_{n-1} v_{n-1})

NO link/transport matrices are imposed. The W_n are constant coupling matrices
in the equations of motion, same family as the scalar beta of the U(1) branch.

Theory being tested:
 1. kappa*JJ dynamically selects a complex structure: the node splits into
    chiral modes; a chirality-pure packet is a C^2 spinor field.
 2. Energy conservation of the directional velocity coupling requires W
    SYMMETRIC (matrix case flips the scalar case's antisymmetry rule), and
    real-symmetric W commuting with JJ = complex HERMITIAN 2x2 = exactly the
    u(2) gauge-potential class. The conservative class and the gauge class
    coincide.
 3. A packet crossing a region with W ~ rho(g*sigma_a) precesses its spinor
    about axis a by theta = sum_links (k_+ - k_-), computed independently
    from Bloch dispersion at the packet's conserved frequency.
 4. Two regions with non-parallel axes (sigma_x, sigma_y) traversed in both
    orders give DIFFERENT final spinors (non-commuting, dynamically generated
    transport); parallel axes (control) give identical finals.

Failure criteria: chirality purity loss, Bloch-vector disagreement with the
independent prediction, or null ordering splitting.
"""

import numpy as np

SQ5 = np.sqrt(5)
N = 200
C = 1.0
KAPPA = 0.5
DT = 0.02
K0 = np.pi / 2
AMP = 1e-3
W_ENV = 8.0
N0 = 20
SEG_START = (60, 120)          # first link index of each segment
RAMP = [0.25, 0.5, 0.75] + [1.0] * 6 + [0.75, 0.5, 0.25]   # 12 links

# free-region dispersion (chirality-+ sector): w^2 + kappa*w - (sqrt5 + 2c(1-cos k)) = 0
OMEGA = 0.5 * (-KAPPA + np.sqrt(KAPPA**2 + 4 * (SQ5 + 2 * C * (1 - np.cos(K0)))))
VG = 2 * C * np.sin(K0) / (2 * OMEGA + KAPPA)

# ---------- complex <-> real-4 representation ----------
def rho(A):
    """Complex 2x2 -> real 4x4 block representation."""
    M = np.zeros((4, 4))
    for j in range(2):
        for l in range(2):
            a = A[j, l]
            M[2*j:2*j+2, 2*l:2*l+2] = [[a.real, -a.imag], [a.imag, a.real]]
    return M

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SIG = (SX, SY, SZ)
JJ = rho(1j * np.eye(2))

# structural checks: Hermitian <-> symmetric, commutation with JJ, homomorphism
for S in SIG:
    R = rho(S)
    assert np.allclose(R, R.T), 'Hermitian generator must map to symmetric real matrix'
    assert np.allclose(R @ JJ, JJ @ R), 'generator must commute with complex structure'
assert np.allclose(rho(SX @ SY), rho(SX) @ rho(SY)), 'rho must be a homomorphism'

# ---------- Bloch-theory prediction (independent of the simulation) ----------
def k_branch(g, s, w=OMEGA):
    """Solve 2c(1-cos k) - 2c*g*s*w*sin k = w^2 + kappa*w - sqrt5 for k near K0."""
    Q = w * w + KAPPA * w - SQ5
    f = lambda k: 2*C*(1 - np.cos(k)) - 2*C*g*s*w*np.sin(k) - Q
    lo, hi = 0.2, np.pi - 0.2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def U_segment(axis, g_max):
    """Per-link product of branch phases over the ramp profile."""
    S = SIG[axis]
    Pp, Pm = (np.eye(2) + S) / 2, (np.eye(2) - S) / 2
    U = np.eye(2, dtype=complex)
    for wgt in RAMP:
        g = g_max * wgt
        U = (np.exp(1j * k_branch(g, +1)) * Pp + np.exp(1j * k_branch(g, -1)) * Pm) @ U
    return U

def bloch(spinor):
    s = spinor / np.linalg.norm(spinor)
    return np.array([np.real(np.conj(s) @ (S @ s)) for S in SIG])

# segment strengths: target ~90 deg and ~60 deg rotations
def theta_of(g_max):
    return sum(abs(k_branch(g_max*w, +1) - k_branch(g_max*w, -1)) for w in RAMP)
G_X = 0.0479   # axis sigma_x  (~90 deg at corrected omega)
G_Y = 0.0319   # axis sigma_y  (~60 deg at corrected omega)
G_C1, G_C2 = 0.0479, 0.0319   # Abelian control: both sigma_x

# ---------- simulation ----------
def make_links(spec):
    """spec: list of (start, axis, g_max). Returns per-link W array (N,4,4)."""
    W = np.zeros((N, 4, 4))
    for start, axis, g in spec:
        R = rho(SIG[axis])
        for j, wgt in enumerate(RAMP):
            W[start + j] = g * wgt * R
    return W

def force(u, v, W, Wm):
    up, um = np.roll(u, -1, axis=0), np.roll(u, 1, axis=0)
    vp, vm = np.roll(v, -1, axis=0), np.roll(v, 1, axis=0)
    f = -(SQ5 * u + u * u) + KAPPA * (v @ JJ.T) + C * (up + um - 2 * u)
    f += C * (np.einsum('nab,nb->na', W, vp) - np.einsum('nab,nb->na', Wm, vm))
    return f

def energy(u, v):
    up = np.roll(u, -1, axis=0)
    return (0.5*(v*v).sum() + 0.5*SQ5*(u*u).sum() + (u**3).sum()/3
            + 0.5*C*((up-u)**2).sum())

def run(spec, t_snapshots):
    W = make_links(spec)
    Wm = np.roll(W, 1, axis=0)
    n = np.arange(N)
    env = np.exp(-(((n - N0 + N//2) % N - N//2)**2) / (2 * W_ENV**2))
    psi0 = AMP * env * np.exp(1j * K0 * (n - N0))          # spinor (1,0)
    u = np.zeros((N, 4)); v = np.zeros((N, 4))
    u[:, 0], u[:, 1] = psi0.real, psi0.imag
    dpsi = -1j * OMEGA * psi0
    v[:, 0], v[:, 1] = dpsi.real, dpsi.imag
    E0 = energy(u, v)
    out, snaps = {}, sorted(t_snapshots)
    steps = int(snaps[-1] / DT) + 1
    for s in range(steps):
        t = s * DT
        for ts in snaps:
            if abs(t - ts) < DT / 2:
                out[ts] = readout(u, v)
        k1v = force(u, v, W, Wm);                          k1u = v
        k2v = force(u + 0.5*DT*k1u, v + 0.5*DT*k1v, W, Wm); k2u = v + 0.5*DT*k1v
        k3v = force(u + 0.5*DT*k2u, v + 0.5*DT*k2v, W, Wm); k3u = v + 0.5*DT*k2v
        k4v = force(u + DT*k3u, v + DT*k3v, W, Wm);         k4u = v + DT*k3v
        u = u + DT/6*(k1u + 2*k2u + 2*k3u + k4u)
        v = v + DT/6*(k1v + 2*k2v + 2*k3v + k4v)
    out['E_drift'] = abs(energy(u, v) - E0) / E0
    return out

def readout(u, v):
    """Bloch vector + chirality purity via positive-frequency projection."""
    psi = u[:, 0::2] + 1j * u[:, 1::2]                     # (N,2)
    dps = v[:, 0::2] + 1j * v[:, 1::2]
    chi = psi + (1j / OMEGA) * dps                         # e^{-iwt} sector
    bar = psi - (1j / OMEGA) * dps                         # e^{+iwt} sector
    rho_s = chi.T @ chi.conj()          # sum_n chi_n chi_n^dagger (NOT its transpose)
    nvec = np.array([np.real(np.trace(S @ rho_s)) for S in SIG]) / np.real(np.trace(rho_s))
    purity = 1 - np.linalg.norm(bar) / np.linalg.norm(chi)
    return nvec, purity

# ---------- experiment ----------
if __name__ == '__main__':
    print(f'omega={OMEGA:.4f}  v_g={VG:.3f}  predicted rotation angles: '
          f'X={np.degrees(theta_of(G_X)):.1f} deg, Y={np.degrees(theta_of(G_Y)):.1f} deg')
    T1, T2 = 150.0, 280.0                                  # after seg1; after both
    s0 = np.array([1.0, 0.0], dtype=complex)
    UX, UY = U_segment(0, G_X), U_segment(1, G_Y)
    UC1, UC2 = U_segment(0, G_C1), U_segment(0, G_C2)
    cases = [
        ('free',      [],                                          np.eye(2), np.eye(2)),
        ('X then Y',  [(SEG_START[0], 0, G_X), (SEG_START[1], 1, G_Y)], UX, UY),
        ('Y then X',  [(SEG_START[0], 1, G_Y), (SEG_START[1], 0, G_X)], UY, UX),
        ('ctrl x,x',  [(SEG_START[0], 0, G_C1), (SEG_START[1], 0, G_C2)], UC1, UC2),
        ('ctrl x,x r',[(SEG_START[0], 0, G_C2), (SEG_START[1], 0, G_C1)], UC2, UC1),
    ]
    print(f'{"case":>10} {"stage":>7} {"measured Bloch (x,y,z)":>26} '
          f'{"predicted":>26} {"err deg":>8} {"purity":>7}')
    finals_m, finals_p = {}, {}
    for name, spec, U1, U2 in cases:
        res = run(spec, (T1, T2))
        for stage, tt, Upred in (('mid', T1, U1), ('final', T2, U2 @ U1)):
            nm, pur = res[tt]
            npred = bloch(Upred @ s0)
            err = np.degrees(np.arccos(np.clip(nm @ npred / np.linalg.norm(nm), -1, 1)))
            fm = np.array2string(nm, precision=3, suppress_small=True)
            fp = np.array2string(npred, precision=3, suppress_small=True)
            print(f'{name:>10} {stage:>7} {fm:>26} {fp:>26} {err:>8.2f} {pur:>7.4f}')
            if stage == 'final':
                finals_m[name], finals_p[name] = nm, npred
        if name == 'free':
            print(f'{"":>10} energy drift over run: {res["E_drift"]:.2e}')

    ang = lambda a, b: np.degrees(np.arccos(np.clip(
        a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
    print(f'\nordering splitting (non-parallel axes): measured '
          f'{ang(finals_m["X then Y"], finals_m["Y then X"]):.2f} deg,  predicted '
          f'{ang(finals_p["X then Y"], finals_p["Y then X"]):.2f} deg')
    print(f'ordering splitting (Abelian control)  : measured '
          f'{ang(finals_m["ctrl x,x"], finals_m["ctrl x,x r"]):.2f} deg,  predicted 0.00 deg')
