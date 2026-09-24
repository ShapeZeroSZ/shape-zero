#!/usr/bin/env python3
"""
j_compat_test.py — do the dynamics suppress the J-breaking part of a coupling?

n = 2, q = 1, model.py's force law. One random symmetric W is split into
    Wc = (W - J W J)/2   commutes with J (J = model.JJ = rho(i I))
    Wx = (W + J W J)/2   anticommutes with J
each rescaled to Frobenius norm 2 (that of a Pauli generator, so g means what it
means in model.py's gate 7). One ramped gauge segment is built from each, and the
same colour-0 packet is sent through both and through an empty lattice.

Internal state after exit: the lattice-summed chirality density matrices
    rho_chi = sum chi chi^H,  rho_bar = sum bar bar^H   (chi = psi + i psi'/omega)
stacked block-diagonally and normalised. Effect of a segment = Bures angle
(degrees) between that state and the no-segment reference at the same time.
Linear in amplitude for both a chi-rotation and leakage into bar.

On-site stiffness K is varied by subclassing Lattice (model.py fixes K = sqrt 5).
"""
import os, sys, time
import numpy as np

# model.py pinned to the archive version this script was run with (c49da46f);
# MODEL_DIR overrides it.
sys.path.insert(0, os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_versions", "c49da46f")))
import model as M

N_SITES, SEG0, N0, WIDTH = 200, 60, 20, 8.0


class KLattice(M.Lattice):
    """model.Lattice with the linear on-site stiffness K as a parameter."""

    def __init__(self, n, K, **kw):
        super().__init__(n=n, **kw)
        self.K = K
        self.omega = 0.5 * (-self.kappa + np.sqrt(self.kappa ** 2 + 4 *
                            (K + 2 * M.C * (1 - np.cos(M.K0)))))

    def force(self, u, v, W=None, Wm=None):
        f = super().force(u, v, W, Wm)
        return f + (M.SQ5 - self.K) * u          # replace sqrt5*u by K*u

    def energy(self, u, v):
        return super().energy(u, v) + 0.5 * (self.K - M.SQ5) * (u * u).sum()


def k_branch_K(lat, g_eig):
    w = lat.omega
    Q = w * w + lat.kappa * w - lat.K
    f = lambda k: 2 * M.C * (1 - np.cos(k)) - 2 * M.C * g_eig * w * np.sin(k) - Q
    lo, hi = 0.2, np.pi - 0.2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def U_segment_H(lat, H, g):
    """model.U_segment generalised to an arbitrary Hermitian H."""
    eigs, vecs = np.linalg.eigh(H)
    P = [np.outer(vecs[:, j], vecs[:, j].conj()) for j in range(lat.n)]
    U = np.eye(lat.n, dtype=complex)
    for wgt in M.RAMP:
        U = sum(np.exp(1j * k_branch_K(lat, g * wgt * eigs[j])) * P[j]
                for j in range(lat.n)) @ U
    return U


def links(lat, Wmat, g):
    W = np.zeros((lat.N, lat.D, lat.D))
    for j, wgt in enumerate(M.RAMP):
        W[(SEG0 + j) % lat.N] = g * wgt * Wmat
    return W, np.roll(W, 1, axis=0)


def to_complex(Wc):
    """Inverse of model.rho for a J-commuting real matrix."""
    n = Wc.shape[0] // 2
    return np.array([[Wc[2 * j, 2 * l] + 1j * Wc[2 * j + 1, 2 * l]
                      for l in range(n)] for j in range(n)])


def internal_state(lat, u, v):
    psi = u[:, 0::2] + 1j * u[:, 1::2]
    dps = v[:, 0::2] + 1j * v[:, 1::2]
    chi = psi + (1j / lat.omega) * dps
    bar = psi - (1j / lat.omega) * dps
    R = np.zeros((2 * lat.n, 2 * lat.n), dtype=complex)
    R[:lat.n, :lat.n] = chi.T @ chi.conj()
    R[lat.n:, lat.n:] = bar.T @ bar.conj()
    return R / np.real(np.trace(R))


def psd_sqrt(A):
    w, V = np.linalg.eigh((A + A.conj().T) / 2)
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def bures_deg(A, B):
    sA = psd_sqrt(A)
    F = np.real(np.trace(psd_sqrt(sA @ B @ sA)))
    return np.degrees(np.arccos(np.clip(F, -1, 1)))


def centroid(lat, u):
    w = np.linalg.norm(u, axis=1) ** 2
    ang = 2 * np.pi * np.arange(lat.N) / lat.N
    return (np.angle((w * np.exp(1j * ang)).sum()) % (2 * np.pi)) * lat.N / (2 * np.pi)


def run(lat, T, W=None, Wm=None):
    u, v = lat.packet(amp=1e-3, n0=N0, width=WIDTH, colour=0)
    u, v, drift = lat.run(u, v, T, W, Wm)
    return u, v, drift


def split_W(seed=1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((4, 4)); W = A + A.T
    J = M.rho(1j * np.eye(2))
    Wc = (W - J @ W @ J) / 2
    Wx = (W + J @ W @ J) / 2
    Wc *= 2.0 / np.linalg.norm(Wc); Wx *= 2.0 / np.linalg.norm(Wx)
    assert np.allclose(Wc, Wc.T) and np.allclose(Wx, Wx.T)
    assert np.allclose(Wc @ J, J @ Wc) and np.allclose(Wx @ J, -J @ Wx)
    return W, Wc, Wx, J


def exit_time(lat):
    vg = 2 * M.C * np.sin(M.K0) / (2 * lat.omega + lat.kappa)   # d omega / dk
    dist = (SEG0 + len(M.RAMP) + 3 * WIDTH) - N0                # tail clears segment
    return dist / vg


def measure(K, g, Wc, Wx, H):
    lat = KLattice(n=2, K=K, N=N_SITES)
    T = exit_time(lat)
    u0, v0, d0 = run(lat, T)
    ref = internal_state(lat, u0, v0)
    out = {"K": K, "g": g, "omega": lat.omega, "T": T}
    for name, Wm_ in (("c", Wc), ("x", Wx)):
        W, Wm = links(lat, Wm_, g)
        u, v, dr = run(lat, T, W, Wm)
        out["eff_" + name] = bures_deg(ref, internal_state(lat, u, v))
        out["drift_" + name] = dr
        st = internal_state(lat, u, v)
        out["leak_" + name] = float(np.real(np.trace(st[2:, 2:])) - np.real(np.trace(ref[2:, 2:])))
        if name == "c":
            co, _ = lat.readout(u, v)
            out["co_c"] = co
    # centroid after the run must be past the segment and not wrapped back
    out["centroid_ref"] = centroid(lat, u0)
    psi0 = np.array([1, 0], dtype=complex)
    out["co_pred"] = M.coords_of_state(lat, U_segment_H(lat, H, g) @ psi0)
    out["co_ref"], _ = lat.readout(u0, v0)
    return out
