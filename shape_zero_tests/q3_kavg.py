#!/usr/bin/env python3
"""
q3_kavg.py — carrier vs spectrum-averaged segment prediction at q = 3.

No new dynamics. The packet's initial fields are regenerated exactly (packet3),
Fourier-transformed on the 320 x 12 x 12 lattice, and each wavevector component
k = (kx, ky, kz) is propagated through the segments with its OWN frequency
omega(k) and transverse term Qt(ky, kz): inside a segment with generator
eigenvalue lam, kx solves
    2c(1 - cos kx) - 2c g lam omega sin kx = omega^2 + kappa omega - K - Qt.
Frequency and transverse k are conserved by a transversely uniform, static slab,
so each component gets its own unitary U(k) = sum_j exp(i sum_ramp kx_j) P_j.
Predicted readout: rho = sum_k P(k) U(k) rho0 U(k)^H, P(k) = |chi_hat(k)|^2 of the
initial field (chi = psi + i psi'/omega0, the readout's chirality).
Compared with the saved CLEARING-readout Bloch vectors.
"""
import os, sys, json
import numpy as np
# model.py pinned to the archive version this script was run with (948b09e8);
# MODEL_DIR overrides it.
sys.path.insert(0, os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_versions", "948b09e8")))
import model as M
import q3_readout as Q

K = M.SQ5


def spectrum(n):
    lat = Q.L3(n)
    u, v = lat.packet3()
    psi = (u[:, 0] + 1j * u[:, 1]).reshape(lat.shape)
    dps = (v[:, 0] + 1j * v[:, 1]).reshape(lat.shape)
    chi = psi + (1j / lat.omega) * dps
    P = np.abs(np.fft.fftn(chi)) ** 2
    ks = [2 * np.pi * np.fft.fftfreq(m) for m in lat.shape]
    KX, KY, KZ = np.meshgrid(*ks, indexing="ij")
    P = P / P.sum()
    return lat, P, KX, KY, KZ


def kx_segment_vec(kx0, om, Qt, geig, kappa):
    """Vectorised root of 2c(1-cos k) - 2c geig om sin k = om^2 + kappa om - K - Qt,
    bracketed around each mode's free kx0 (monotone there for these small g)."""
    rhs = om * om + kappa * om - K - Qt
    f = lambda k: 2 * M.C * (1 - np.cos(k)) - 2 * M.C * geig * om * np.sin(k) - rhs
    lo = np.maximum(1e-6, kx0 - 0.5); hi = np.minimum(np.pi - 1e-6, kx0 + 0.5)
    bad = f(lo) * f(hi) > 0
    lo[bad], hi[bad] = 1e-6, np.pi - 1e-6
    flo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi); fm = f(mid)
        left = flo * fm <= 0
        hi = np.where(left, mid, hi); lo = np.where(left, lo, mid); flo = np.where(left, flo, fm)
    return 0.5 * (lo + hi), int(bad.sum())


def averaged_coords(lat, P, KX, KY, KZ, segs, cut=1e-12):
    """segs: list of (axis, g) in traversal order. Power-weighted over modes."""
    n = lat.n
    psi0 = np.zeros(n, complex); psi0[0] = 1
    mask = (P > cut * P.max()) & (KX > 0) & (KX < np.pi)
    p, kx, ky, kz = P[mask], KX[mask], KY[mask], KZ[mask]
    Qt = 2 * M.C * ((1 - np.cos(ky)) + (1 - np.cos(kz)))
    om = 0.5 * (-lat.kappa + np.sqrt(lat.kappa ** 2 + 4 * (K + 2 * M.C * (1 - np.cos(kx)) + Qt)))
    state = np.tile(psi0, (len(p), 1))                       # (modes, n)
    nbad = 0
    for axis, g in segs:
        eigs, vecs = np.linalg.eigh(lat.G[axis])
        ph = np.zeros((len(p), n))
        for wgt in M.RAMP:
            for jj in range(n):
                r, b = kx_segment_vec(kx, om, Qt, g * wgt * eigs[jj], lat.kappa)
                ph[:, jj] += r; nbad += b
        coef = state @ vecs.conj()                           # components in eigenbasis
        state = (coef * np.exp(1j * ph)) @ vecs.T
    rho = np.einsum("m,mi,mj->ij", p, state, state.conj())
    co = np.array([np.real(np.trace(S @ rho)) / np.real(np.trace(rho)) for S in lat.G])
    return co, float(p.sum()), int(mask.sum()), nbad


def carrier_coords(n, segs, Qt):
    lat = M.Lattice(n=n, N=200)
    psi0 = np.zeros(n, complex); psi0[0] = 1
    U = np.eye(n, dtype=complex)
    for axis, g in segs:
        U = M.U_segment(lat, axis, g, Qt) @ U
    return M.coords_of_state(lat, U @ psi0)


if __name__ == "__main__":
    out = []
    for n in (2, 3):
        lat, P, KX, KY, KZ = spectrum(n)
        gA, gB = Q.G[n]
        cases = [("AB", [(0, gA), (1, gB)]), ("BA", [(1, gB), (0, gA)])]
        if n == 2:
            cases = [("single", [(0, 0.12)])] + cases
        meas = {}
        for name, segs in cases:
            r = json.load(open(f"q3/{n}_{name}.json"))["clear"]
            co_m = np.array(r["co"])
            co_c = carrier_coords(n, segs, r["Qt"])
            co_a, kept, nk, nbad = averaged_coords(lat, P, KX, KY, KZ, segs)
            meas[name] = (co_m, co_c, co_a)
            rec = dict(n=n, case=name, carrier_err=M.angle(co_m, co_c),
                       avg_err=M.angle(co_m, co_a), kept_weight=kept, n_modes=nk, rebracketed=nbad)
            print(json.dumps(rec), flush=True)
            out.append(rec)
        if "AB" in meas:
            (mA, cA, aA), (mB, cB, aB) = meas["AB"], meas["BA"]
            split_m = M.angle(mA, mB)
            rec = dict(n=n, case="split", measured=split_m,
                       carrier=M.angle(cA, cB), averaged=M.angle(aA, aB),
                       carrier_err=abs(split_m - M.angle(cA, cB)),
                       avg_err=abs(split_m - M.angle(aA, aB)))
            print(json.dumps(rec), flush=True)
            out.append(rec)
    json.dump(out, open("q3_kavg.json", "w"), indent=1)
