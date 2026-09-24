#!/usr/bin/env python3
"""
phi_gauge_u3.py — A1 at Rank 3: does the platform dynamically realise u(3)?

STATUS: algebra VERIFIED (U1-U3 pass). Dynamics NOT MEASURED -- the first
version failed its own control. Four defects are marked FIX-1 .. FIX-4 below
with the exact change required and the order to do them in.

WHY THIS EXISTS. The spec (v5.3, "Gauge emergence mechanisms", line 113) derives
SU(3) from TRIADIC coupling at Rank 3 -- trimer sub-lattices giving three
internal non-commutative modes per node -- and flags it "less explored than
SU(2)". That is the programme's OWN route, distinct from the G2-stabiliser
construction (Gunaydin-Gursey 1973) that the earlier reference note documented
and wrongly called "not derived".

The A1 passivity theorem generalises exactly: for n dimers per node,
{W : W symmetric, [W,JJ] = 0} has dimension n^2 = dim u(n), verified n = 1..5.
ONE theorem, three node sizes: u(1), u(2) = u(1)+su(2), u(3) = u(1)+su(3).

WHAT IS VERIFIED (runs below)
 U1  all eight Gell-Mann generators map to SYMMETRIC real 6x6 matrices
 U2  all eight commute with the complex structure JJ
 U3  rho is a homomorphism; admissible class has dimension 9 = dim u(3)

WHAT IS NOT
 U4/U5  the ordering measurement. The first attempt gave 44.891 deg for
        non-commuting axes, but its CONTROL (same axis, both orders) gave
        14.987 deg where it must give ~0, with energy drift 7.43e-01 against
        4.62e-07 in the working u(2) script. That number is uninterpretable and
        is not quoted anywhere.

============================================================================
THE FOUR DEFECTS
============================================================================

FIX-1  INTEGRATOR.  The first version used velocity-Verlet. The force here is
       VELOCITY-DEPENDENT (the gyroscopic KAPPA*JJ*v term and the directional
       coupling W*v), and velocity-Verlet is not symplectic for such forces --
       it drifts. phi_gauge_chiral.py uses RK4 and holds 4.62e-07.

           k1v = force(u, v, W, Wm);                      k1u = v
           k2v = force(u+.5*DT*k1u, v+.5*DT*k1v, W, Wm);  k2u = v+.5*DT*k1v
           k3v = force(u+.5*DT*k2u, v+.5*DT*k2v, W, Wm);  k3u = v+.5*DT*k2v
           k4v = force(u+DT*k3u, v+DT*k3v, W, Wm);        k4u = v+DT*k3v
           u = u + DT/6*(k1u + 2*k2u + 2*k3u + k4u)
           v = v + DT/6*(k1v + 2*k2v + 2*k3v + k4v)

FIX-2  READOUT.  The first version extracted the state at a SINGLE SITE via
       argmax of the amplitude, letting packet spreading and ramp position leak
       into the result -- the likely direct cause of the non-zero control.
       phi_gauge_chiral.py projects onto the positive-frequency sector and forms
       a density matrix over the WHOLE lattice:

           psi   = u[:, 0::2] + 1j*u[:, 1::2]        # (N, n) complex
           dps   = v[:, 0::2] + 1j*v[:, 1::2]
           chi   = psi + (1j/OMEGA)*dps              # e^{-iwt} sector
           bar   = psi - (1j/OMEGA)*dps              # counter-rotating sector
           rho_s = chi.T @ chi.conj()                # (n, n) density matrix
           coords = [Re tr(LAM_a @ rho_s)] / Re tr(rho_s)      # 8 coords
           purity = 1 - |bar|/|chi|

       Port that with SIG -> LAM, 3 Bloch coords -> 8 Gell-Mann coords. The
       purity is the diagnostic for whether the packet stayed in one frequency
       sector; the u(2) run holds 0.9835 throughout.

FIX-3  NO INDEPENDENT PREDICTION.  The u(2) script computes U_segment() from
       Bloch branch phases and compares the simulation against it -- that is
       what makes it a verification rather than an observation. Generalising is
       the real work: su(3) generators have THREE eigenvalues, not two.
       lambda_1 and lambda_2 have spectrum {+1, -1, 0}; lambda_8 has
       {1/sqrt3, 1/sqrt3, -2/sqrt3}. Use the spectral decomposition:

           eigs, vecs = np.linalg.eigh(LAM[axis])
           P_j = np.outer(vecs[:, j], vecs[:, j].conj())
           U = sum_j exp(1j*k_branch(g*wgt*eigs[j])) * P_j   @ U   over RAMP

       The u(2) case is the degenerate instance with eigenvalues +-1 and
       P_+- = (I +- S)/2, which is what U_segment() hard-codes.

FIX-4  MISSING FREE CONTROL.  The u(2) script runs a 'free' case with NO
       segments and confirms the state is unchanged, validating the readout
       before any gauge effect is measured. If the free case does not return
       the initial state to ~0 deg, the readout is wrong and nothing downstream
       means anything.

ORDER OF WORK: FIX-1, FIX-4, FIX-2, FIX-3.
FIX-1 with FIX-4 alone will tell you whether drift was the whole problem. Only
after BOTH the free case and the control read ~0 should the non-commuting
number be quoted.

SCOPE, unchanged by any of this: u(3) on a lattice fibre is a SYNTHETIC gauge
structure, not colour SU(3) acting on quark representations. The honest word is
"consistency", not "derivation of QCD". And what fixes n = 3 remains open --
the spec says Rank 3, but the articulation theorem concerns vertices rather
than internal modes per node, so that link is a step nobody has proven.

Python 3 + NumPy only.
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
SEG_START = (60, 120)
RAMP = [0.25, 0.5, 0.75] + [1.0] * 6 + [0.75, 0.5, 0.25]
NDIM = 3
D = 2 * NDIM

OMEGA = 0.5 * (-KAPPA + np.sqrt(KAPPA ** 2 + 4 * (SQ5 + 2 * C * (1 - np.cos(K0)))))
VG = 2 * C * np.sin(K0) / (2 * OMEGA + KAPPA)


def rho(A):
    """Complex n x n -> real 2n x 2n block representation. Dimension-agnostic."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    for j in range(n):
        for l in range(n):
            a = A[j, l]
            M[2 * j:2 * j + 2, 2 * l:2 * l + 2] = [[a.real, -a.imag],
                                                   [a.imag, a.real]]
    return M


def gell_mann():
    E = lambda i, j: (np.eye(3, dtype=complex)[:, [i]]
                      @ np.eye(3, dtype=complex)[[j], :])
    return [E(0, 1) + E(1, 0),
            -1j * (E(0, 1) - E(1, 0)),
            E(0, 0) - E(1, 1),
            E(0, 2) + E(2, 0),
            -1j * (E(0, 2) - E(2, 0)),
            E(1, 2) + E(2, 1),
            -1j * (E(1, 2) - E(2, 1)),
            (E(0, 0) + E(1, 1) - 2 * E(2, 2)) / np.sqrt(3)]


LAM = gell_mann()
JJ = rho(1j * np.eye(NDIM))


# =================================================== U1-U3: VERIFIED, unchanged
def structural_checks():
    print("U1/U2  STRUCTURAL")
    sym = sum(np.allclose(rho(S), rho(S).T) for S in LAM)
    herm = sum(np.allclose(rho(S) @ JJ, JJ @ rho(S)) for S in LAM)
    print(f"      Hermitian -> symmetric real        : {sym}/8")
    print(f"      commutes with the complex structure: {herm}/8")
    hom = np.allclose(rho(LAM[0] @ LAM[1]), rho(LAM[0]) @ rho(LAM[1]))
    print(f"      rho is a homomorphism              : {hom}")
    basis = []
    for i in range(D):
        for j in range(i, D):
            M = np.zeros((D, D)); M[i, j] = 1.0; M[j, i] = 1.0
            basis.append(M)
    s = np.linalg.svd(np.array([(M @ JJ - JJ @ M).ravel() for M in basis]),
                      compute_uv=False)
    dim = len(basis) - int(np.sum(s > 1e-9 * s[0]))
    print(f"U3    admissible coupling class dimension: {dim}   (u(3) = 9)  "
          f"{'MATCH' if dim == 9 else 'MISMATCH'}")
    return sym == 8 and herm == 8 and hom and dim == 9


# ============================================================ shared machinery
def k_branch(g_eig, w=OMEGA):
    """Branch wavenumber for (coupling strength x eigenvalue)."""
    Q = w * w + KAPPA * w - SQ5
    f = lambda k: 2 * C * (1 - np.cos(k)) - 2 * C * g_eig * w * np.sin(k) - Q
    lo, hi = 0.2, np.pi - 0.2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def make_links(spec):
    W = np.zeros((N, D, D))
    for start, axis, g in spec:
        R = rho(LAM[axis])
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
    return (0.5 * (v * v).sum() + 0.5 * SQ5 * (u * u).sum() + (u ** 3).sum() / 3
            + 0.5 * C * ((up - u) ** 2).sum())


# ==================================================================== FIX-3
def U_segment(axis, g_max):
    """Independent Bloch-theory prediction. NOT YET GENERALISED — see FIX-3."""
    raise NotImplementedError(
        "FIX-3: use the spectral decomposition of LAM[axis] (three eigenvalues, "
        "three projectors), not the two-projector (I +- S)/2 form")


# ==================================================================== FIX-2
def readout(u, v):
    """Density-matrix readout over the whole lattice. NOT YET PORTED — FIX-2."""
    raise NotImplementedError(
        "FIX-2: port the chi/bar positive-frequency projection and the "
        "rho_s = chi.T @ chi.conj() density matrix; return 8 Gell-Mann "
        "coordinates plus chirality purity")


# ============================================================== FIX-1 and FIX-4
def run(spec, t_snapshots):
    """RK4 evolution with snapshots. NOT YET REPLACED — see FIX-1, FIX-4."""
    raise NotImplementedError(
        "FIX-1: replace velocity-Verlet with the RK4 stepper. "
        "FIX-4: add a spec=[] free case and confirm ~0 before measuring")


if __name__ == "__main__":
    print("=" * 70)
    print("u(3) AT RANK 3")
    print("=" * 70)
    print()
    ok = structural_checks()
    print()
    if ok:
        print("  ALGEBRA VERIFIED. Dynamics blocked pending FIX-1 .. FIX-4;")
        print("  see the module docstring for the exact changes and their order.")
        print()
        print("  reference values from the working u(2) case (phi_gauge_chiral.py):")
        print("      energy drift          4.62e-07")
        print("      ordering splitting    59.86 deg measured, 59.84 predicted")
        print("      Abelian control        0.00 deg measured,  0.00 predicted")
        print("      chirality purity      0.9835 held throughout")
        print()
        print("  the u(3) run must reproduce a ~0 free case AND a ~0 control")
        print("  before its non-commuting number is quoted.")
    else:
        print("  structural checks failed — fix those first")
