#!/usr/bin/env python3
"""
joint5_cq.py -- the Joint #5 coefficient C_q(N): delta(Delta omega) = C S^2, rebuilt as a saved
instrument (MODEL_SPEC sec 5b.6, 5b.6a; PROVENANCE sec 6i). No earlier script reproduced these.

THE SYSTEM (linear scalar beta sector, as kappa_side_gpu.py / kappa_extended_gpu.py):
    u'' = -K(x) u + c lap(u) + beta c (u'[x0-1] - u'[x0+1])        (beta along axis 0)
on a periodic q-dimensional lattice of side L, c = 1, beta = 0.05, K(x) = s (1 + S eta(x)),
s = sqrt5 (the phi-well stiffness), eta a mean-zero, unit-rms Gaussian blob centred in the box.
Reconstructed from the records: at q = 1, N = 128 this gives C = -0.06274 (sigma 8) and -0.06258
(sigma 16) against the recorded -0.06268 and -0.06256.
PROBE: the branch continued from the plane wave k = (2 pi m / L, 0, ...) at S = 0; +k is the upper
root, -k the lower; Delta omega = omega(+k) - omega(-k) = 2 beta c sin k at S = 0.

ESTIMATOR (the one that survived sec 5b.4): epsilon-continuation. The problem is written as the
Hermitian-definite pencil  (i S_op) z = omega B z,  S_op = [[0, Kmat], [-Kmat, G]] (skew),
B = diag(Kmat, I) (positive definite), z = (u, v); the probe eigenvector is followed stepwise
in S by maximum B-overlap. Dense (numpy.linalg.eigh on the B-whitened matrix) for small
systems; sparse shift-invert Lanczos (scipy eigsh, sigma at the previous omega) otherwise, with
no de-duplication (Hermitian solver; degenerate symmetric partners are real). The 'gap' printed is
to the nearest eigenvalue of any symmetry, coupled or not; resolution is judged by the kernel
(joint5_kernel.py), which knows which modes the blob couples.

C AND ITS ERROR BAR (the policy of kappa_extended_gpu.py): C from a fit delta(Delta omega) =
C S^2 + D S^4 over S = 0.0025 ... 0.02 (8 points); the error bar is the larger of the fit's
standard error and |C - C'|, C' the S^2-only fit over S <= 0.01. The probe must stay resolved:
max |delta omega| < 0.1 x the gap to its nearest neighbour.

usage:  python3 joint5_cq.py validate           (q = 1 records; q = 2, 3 reproduction attempts)
        python3 joint5_cq.py measure q L [m] [frac]
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

C, BETA, S0 = 1.0, 0.05, np.sqrt(5.0)
SGRID = np.linspace(0.0025, 0.02, 8)


def grid(q, L):
    return np.indices((L,) * q).reshape(q, -1).T


def eta_blob(q, L, sigma):
    X = grid(q, L).astype(float)
    r2 = ((X - L / 2.0) ** 2).sum(1)
    e = np.exp(-0.5 * r2 / sigma ** 2)
    e -= e.mean()
    return e / np.sqrt((e ** 2).mean())


def operators(q, L):
    n = L ** q
    idx = np.arange(n).reshape((L,) * q)
    lap = sp.lil_matrix((n, n))
    rows, cols, vals = [], [], []
    for a in range(q):
        for s in (1, -1):
            nb = np.roll(idx, -s, axis=a).ravel()
            rows += list(range(n)); cols += list(nb); vals += [1.0] * n
    lap = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) - 2 * q * sp.identity(n)
    xm = np.roll(idx, 1, axis=0).ravel()    # x0 - 1
    xp = np.roll(idx, -1, axis=0).ravel()   # x0 + 1
    G = BETA * C * (sp.csr_matrix((np.ones(n), (np.arange(n), xm)), shape=(n, n))
                    - sp.csr_matrix((np.ones(n), (np.arange(n), xp)), shape=(n, n)))
    return lap.tocsr(), G.tocsr()


def pencil(lap, G, Kx):
    n = len(Kx)
    Km = sp.diags(Kx) - C * lap
    Sop = sp.bmat([[None, Km], [-Km, G]]).tocsc()
    B = sp.block_diag([Km, sp.identity(n)]).tocsc()
    return (1j * Sop).tocsc(), B


def plane_wave(q, L, m, sign, Kx_uniform):
    X = grid(q, L)
    k = 2 * np.pi * m / L
    w0sq = S0 + 2 * C * (1 - np.cos(k))
    om = sign * BETA * C * np.sin(k) + np.sqrt(BETA ** 2 * C ** 2 * np.sin(k) ** 2 + w0sq)
    u = np.exp(1j * sign * k * X[:, 0])
    return np.concatenate([u, -1j * om * u]), om


def eig_near(H, B, z_prev, om_prev, dense_limit=1600):
    n2 = H.shape[0]
    if n2 <= 2 * dense_limit:
        Bd = B.toarray(); Hd = H.toarray()
        Lc = np.linalg.cholesky(Bd); Li = np.linalg.inv(Lc)
        w, Y = np.linalg.eigh(Li @ Hd @ Li.conj().T)
        Z = Li.conj().T @ Y
    else:
        w, Z = spl.eigsh(H, k=10, M=B, sigma=om_prev, which="LM", tol=1e-13)
    # no de-duplication: the pencil is Hermitian-definite, and genuinely degenerate partners
    # (cubic symmetry) are real; the probe is picked by B-overlap, which separates them.
    norms = np.sqrt(np.real(np.einsum("ij,ij->j", Z.conj(), B @ Z)))
    ov = np.abs(z_prev.conj() @ (B @ Z)) / norms / np.sqrt(np.real(z_prev.conj() @ (B @ z_prev)))
    i = int(np.argmax(ov))
    srt = np.sort(np.abs(w - w[i]))
    gap = srt[1] if len(srt) > 1 else np.inf
    return w[i], Z[:, i] / norms[i], ov[i], gap


def branch(q, L, m, sign, eta, lap, G, S_list):
    z, om = plane_wave(q, L, m, sign, None)
    om0 = om
    out, gaps, ovs = [], [], []
    S_prev = 0.0
    for S in S_list:
        # continuation in sub-steps of at most 0.0025
        nsub = max(1, int(np.ceil((S - S_prev) / 0.0025 - 1e-9)))
        for j in range(1, nsub + 1):
            Sj = S_prev + (S - S_prev) * j / nsub
            H, B = pencil(lap, G, S0 * (1 + Sj * eta))
            om, z, ov, gap = eig_near(H, B, z, om)
        S_prev = S
        out.append(om); gaps.append(gap); ovs.append(ov)
    return om0, np.array(out), np.array(gaps), np.array(ovs)


def measure(q, L, m=None, frac=1 / 16, sigma=None, verbose=True):
    m = m if m is not None else L // 4
    sigma = sigma if sigma is not None else frac * L
    eta = eta_blob(q, L, sigma)
    lap, G = operators(q, L)
    t0 = time.time()
    p0, wp, gp, op = branch(q, L, m, +1, eta, lap, G, SGRID)
    m0, wm, gm, om_ = branch(q, L, m, -1, eta, lap, G, SGRID)
    D0 = p0 - m0
    dD = (wp - wm) - D0
    A = np.vstack([SGRID ** 2, SGRID ** 4]).T
    coef, *_ = np.linalg.lstsq(A, dD, rcond=None)
    r = dD - A @ coef
    cov = (r @ r) / max(len(dD) - 2, 1) * np.linalg.inv(A.T @ A)
    Cfit, se = coef[0], np.sqrt(cov[0, 0])
    small = SGRID <= 0.01 + 1e-12
    C2 = np.linalg.lstsq(SGRID[small, None] ** 2, dD[small], rcond=None)[0][0]
    err = max(se, abs(Cfit - C2))
    shift = max(np.abs(wp - p0).max(), np.abs(wm - m0).max())
    gap = min(gp.min(), gm.min())
    res = dict(q=q, L=L, m=m, sigma=sigma, C=Cfit, err=err, C_S2only=C2, D=coef[1], resolved=bool(shift < 0.1 * gap),
               shift=shift, gap=gap, overlap=float(min(op.min(), om_.min())), D0=D0, secs=time.time() - t0)
    if verbose:
        print(f"  q={q} L={L:3d} m={m} sigma={sigma:.3g}: C = {Cfit:+.5f} +- {err:.5f} (S^2-only {C2:+.5f}); "
              f"shift {shift:.1e} vs gap {gap:.1e} ({'resolved' if res['resolved'] else 'NOT resolved'}); "
              f"min overlap {res['overlap']:.4f}; {res['secs']:.1f} s", flush=True)
    return res


def validate():
    print("V1  q = 1 records (MODEL_SPEC 5b.6): N = 128, sigma 8 -> -0.06268, sigma 16 -> -0.06256")
    for sg in (8, 16):
        measure(1, 128, sigma=sg)
    print("V2  q = 1 sides 48, 64 (5b.6a: -0.06343, -0.06303) -- box fraction unrecorded; tried sigma = L/16, L/8")
    for fr in (1 / 16, 1 / 8):
        for L in (48, 64):
            measure(1, L, frac=fr)
    print("V3  q = 3 sides 8, 12, 16 (5b.6a: -0.188, +1.458, -1.429) and q = 3 side 6 (+1.928), q = 2 sides 10, 12"
          " (-0.2905, +2.9336) -- m and the box fraction unrecorded; tried m = round(L/4), frac = 1/8, 1/4")
    for fr in (1 / 8, 1 / 4):
        for q, L in ((3, 8), (3, 12), (3, 16), (3, 6), (2, 10), (2, 12)):
            measure(q, L, m=max(1, int(round(L / 4))), frac=fr)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate()
    else:
        a = sys.argv[2:]
        measure(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else None, float(a[3]) if len(a) > 3 else 1 / 16)
