#!/usr/bin/env python3
"""
cpg_pilot2_checks.py -- POST-HOC checks on cpg_pilot2.py (predictions 99131c0). Written after the run;
nothing here re-scores a prediction. Same model, same background, same integrator (imported).

K1  the 1c occupation offset: occupation change at t = 10, 100, 500, 1000, 2000 (seed 0, A_U = 0.005 and
    0.02), with the linear a/b split and with a split at the populated frequencies (K + dK(1 + 1/8)).
    An offset present by t = 10 is a decomposition effect, not collisions.
K2  the 2b far field: the CRN far-field background energy change at F0 = 1e-3 and 2e-3 (seed 0), in the
    windows [400, 1200] and [1200, 2000]. A static response keeps its sign and scales as F0^2; noise does not.
K3  identical a- and b-lump background profiles: a symmetric real envelope gives |psi_b(x,t)| = |psi_a(x,t)|
    in free evolution, and the background sees only |u|. Measured: max |dE_B(a) - dE_B(b)| / max |dE_B|, the
    lump radius profiles, the absolute energy gains, and a CONTROL lump (a-branch, k0 = pi/4) whose
    background profile must differ.
K4  the charge-drift denominator: per background component, Q(0), |Q(T) - Q(0)|, and the drift relative to
    N_a + N_b (the component's total quanta, sum over k of (w_a + kappa/2)|a|^2 + (w_b - kappa/2)|b|^2).
"""

import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cpg_pilot2 as P  # noqa: E402

M = P.M
N, T, DTS = P.N1D, P.T1D, P.DTS
TSNAP = (0.0, 10.0, 100.0, 500.0, 1000.0, 2000.0)


def ab(lat, u, v, j, K):
    k = 2 * np.pi * np.fft.fftfreq(lat.N)
    Q = K + 2 * M.C * (1 - np.cos(k))
    wa = 0.5 * (-lat.kappa + np.sqrt(lat.kappa ** 2 + 4 * Q)); wb = wa + lat.kappa
    Pk = np.fft.fft(u[:, 2 * j] + 1j * u[:, 2 * j + 1]); D = np.fft.fft(v[:, 2 * j] + 1j * v[:, 2 * j + 1])
    return (wb * Pk + 1j * D) / (wa + wb), (wa * Pk - 1j * D) / (wa + wb), wa, wb


def run(job):
    t0 = time.time()
    lat = M.Lattice(n=8, N=N, well="node")
    u, v = P.incoherent_upper(lat, job["amp_u"], job["seed"])
    if "lamp" in job:
        lu, lv = P.lump(lat, job["lamp"], job["branch"])
        if job.get("k0"):
            ph = np.exp(1j * job["k0"] * (np.arange(N) - P.X0))
            psi = (lu[:, 0] + 1j * lu[:, 1]) * ph
            wa = lat.branch_omega()
            dps = np.fft.ifft(-1j * wa * np.fft.fft(psi))
            lu[:, 0], lu[:, 1], lv[:, 0], lv[:, 1] = psi.real, psi.imag, dps.real, dps.imag
        u, v = u + lu, v + lv
    S = None
    if job.get("F0"):
        S = np.zeros((lat.N, lat.D)); S[0, 0] = job["F0"]
    out = dict(job=job, snaps={}, win={"w1": np.zeros(N), "w2": np.zeros(N)}, n={"w1": 0, "w2": 0}, series=[])
    t, steps = 0.0, int(round(DTS / M.DT))
    for i in range(int(round(T / DTS)) + 1):
        if i:
            ext = S * P.ramp(t, 200.0) if S is not None else None
            for _ in range(steps):
                u, v = P.rk4(lat, u, v, ext)
            t += DTS
        if any(abs(t - s) < 1e-9 for s in TSNAP):
            out["snaps"][t] = (u.copy(), v.copy())
        eB = P.e_density(u, v, P.UP)
        for w, lo, hi in (("w1", 400, 1200), ("w2", 1200, 2000)):
            if lo - 1e-9 <= t < hi - 1e-9 or (w == "w2" and abs(t - hi) < 1e-9):
                out["win"][w] += eB; out["n"][w] += 1
        if "lamp" in job:
            out["series"].append((t, P.e_density(u, v, P.LOW).sum(), float(np.max(np.linalg.norm(u[:, :8], axis=1)))))
    for w in out["win"]:
        out["win"][w] /= out["n"][w]
    out["late_eB"] = P.e_density(u, v, P.UP)
    out["r_low"] = np.linalg.norm(u[:, :8], axis=1)
    out["secs"] = time.time() - t0
    return out


def main():
    jobs = [dict(tag="bg0", seed=0, amp_u=0.005), dict(tag="bg0_02", seed=0, amp_u=0.02),
            dict(tag="src1", seed=0, amp_u=0.005, F0=1e-3), dict(tag="src2", seed=0, amp_u=0.005, F0=2e-3),
            dict(tag="la", seed=0, amp_u=0.005, lamp=0.05, branch="a"),
            dict(tag="lb", seed=0, amp_u=0.005, lamp=0.05, branch="b"),
            dict(tag="lc", seed=0, amp_u=0.005, lamp=0.05, branch="a", k0=math.pi / 4),
            dict(tag="bg3", seed=3, amp_u=0.005)]
    with Pool(4) as pool:
        res = {r["job"]["tag"]: r for r in pool.map(run, jobs, chunksize=1)}
    lat = M.Lattice(n=8, N=N, well="node")
    K = M.SQ5
    print("=" * 96)
    print("CPG PILOT 2 -- POST-HOC CHECKS (not re-scoring any prediction of 99131c0)")
    print("=" * 96)

    print("\nK1  occupation change (16-mode bins, rms relative to t = 0), seed 0")
    for tag, amp in (("bg0", 0.005), ("bg0_02", 0.02)):
        rB = float(np.linalg.norm(res[tag]["snaps"][0.0][0], axis=1).mean())
        for lab, KK in (("linear split", K), ("populated split", K + rB * 9 / 8)):
            occ = {}
            for ts, (uu, vv) in res[tag]["snaps"].items():
                rows = []
                for j in P.BG:
                    a, b, _, _ = ab(lat, uu, vv, j, KK)
                    rows.append(np.concatenate([(np.abs(a) ** 2).reshape(-1, 16).sum(1),
                                                (np.abs(b) ** 2).reshape(-1, 16).sum(1)]))
                occ[ts] = np.array(rows)
            ch = [float(np.sqrt(np.mean((occ[ts] / occ[0.0] - 1) ** 2))) for ts in TSNAP[1:]]
            print(f"    A_U = {amp:<6} {lab:16s} t = 10, 100, 500, 1000, 2000: " + " ".join(f"{c:.2e}" for c in ch))

    print("\nK2  far-field background energy change, CRN against the background-only run (seed 0)")
    xx = np.abs(((np.arange(N) + N // 2) % N) - N // 2)
    sel = (xx >= 5) & (xx <= 250)
    for tag, F0 in (("src1", 1e-3), ("src2", 2e-3)):
        Es = 0.5 * F0 * F0 * 0.26701
        row = []
        for w in ("w1", "w2"):
            de = res[tag]["win"][w] - res["bg0"]["win"][w]
            row.append((de[sel].sum(), np.abs(de[sel]).sum(), de[xx < 5].sum()))
        print(f"    F0 = {F0:.0e} (E_s = {Es:.2e}): window [400,1200] far signed {row[0][0]:+.2e}, far |.| {row[0][1]:.2e} "
              f"({row[0][1]/Es:.2f} E_s), near {row[0][2]:+.2e}; [1200,2000] far signed {row[1][0]:+.2e}, "
              f"far |.| {row[1][1]:.2e} ({row[1][1]/Es:.2f} E_s), near {row[1][2]:+.2e}")
    d1 = res["src1"]["win"]["w2"] - res["bg0"]["win"]["w2"]
    d2 = res["src2"]["win"]["w2"] - res["bg0"]["win"]["w2"]
    c = np.corrcoef(d1[sel], d2[sel])[0, 1]
    ratio = np.abs(d2[sel]).sum() / np.abs(d1[sel]).sum()
    dw = res["src1"]["win"]["w1"] - res["bg0"]["win"]["w1"]
    cw = np.corrcoef(dw[sel], d1[sel])[0, 1]
    print(f"    far profile F0 2e-3 vs 1e-3: correlation {c:+.3f}, |.| ratio {ratio:.2f} (F0^2 scaling: 4; F0: 2)")
    print(f"    far profile window 1 vs window 2 (F0 = 1e-3): correlation {cw:+.3f} (a static response: ~ +1)")

    print("\nK3  a- vs b-lump background profiles (seed 0)")
    la, lb, lc, bg = res["la"], res["lb"], res["lc"], res["bg0"]
    dA, dB, dC = la["late_eB"] - bg["late_eB"], lb["late_eB"] - bg["late_eB"], lc["late_eB"] - bg["late_eB"]
    print(f"    max |dE_B(a) - dE_B(b)| / max |dE_B(a)| = {np.max(np.abs(dA-dB))/np.max(np.abs(dA)):.2e}")
    print(f"    control (a-branch, k0 = pi/4): max |dE_B(ctrl) - dE_B(a)| / max |dE_B(a)| = "
          f"{np.max(np.abs(dC-dA))/np.max(np.abs(dA)):.2e}  (must be O(1))")
    print(f"    lump radius profile at T: max | |u_a| - |u_b| | / max |u_a| = "
          f"{np.max(np.abs(la['r_low']-lb['r_low']))/np.max(la['r_low']):.2e}")
    sa, sb = np.array(la["series"]), np.array(lb["series"])
    print(f"    lump self-energy: E_a(0) = {sa[0,1]:.5e}, E_b(0) = {sb[0,1]:.5e}, ratio {sb[0,1]/sa[0,1]:.4f} "
          f"(w_b/w_a at k = 0 = {(1.086434+M.KAPPA)/1.086434:.4f})")
    print(f"    absolute gain over T (not reference-subtracted): a {sa[-1,1]-sa[0,1]:+.4e}, b {sb[-1,1]-sb[0,1]:+.4e}")

    print("\nK4  charge drift per background component: Q(0), |dQ|, |dQ|/|Q(0)|, |dQ|/(N_a + N_b)")
    for tag in ("bg0", "bg3"):
        u0, v0 = res[tag]["snaps"][0.0]; uT, vT = res[tag]["snaps"][T]
        for j in P.BG:
            Q0 = P.rho(lat, u0, v0, j).sum(); QT = P.rho(lat, uT, vT, j).sum()
            a, b, wa, wb = ab(lat, u0, v0, j, K)
            Ntot = ((wa + lat.kappa / 2) * np.abs(a) ** 2 + (wb - lat.kappa / 2) * np.abs(b) ** 2).sum() / lat.N
            print(f"    {tag} comp {j}: Q(0) {Q0:+.3e}  |dQ| {abs(QT-Q0):.2e}  rel {abs(QT-Q0)/abs(Q0):.1e}  "
                  f"rel to N_a+N_b ({Ntot:.3e}) {abs(QT-Q0)/Ntot:.1e}")
    print("\nrun time %.0f s (sum)" % sum(r["secs"] for r in res.values()))


if __name__ == "__main__":
    main()
