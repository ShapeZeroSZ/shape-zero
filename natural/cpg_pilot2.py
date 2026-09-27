#!/usr/bin/env python3
"""
cpg_pilot2.py -- CP-G second pilot: the POPULATED background (main's P0), not the empty vacuum.
Predictions committed first: natural/CPG_PILOT2_PREDICTIONS.md (99131c0).

Background = main's tower background, unchanged: A', kappa*, D16 (n = 8), q = 1 ring N = 512, components
4-7 populated incoherently on both branches at rms A_U = 0.005 by main's incoherent_upper
(shape_zero_tests/tower_populated_test.py on main, reproduced verbatim below for q = 1); components 0-3
empty unless a source, probe or lump is placed. Seeds 0-3, T = 2000. Force: main's model.py
(04_scripts/session, identical to main), integrated by the same RK4 at model.DT; an external static force
is added where stated. Every source / lump run is subtracted against the background-only run of the same
seed (common random numbers).

N1 census   spectra of the field (k = 0 line) and of the conserved densities (energy e, charges rho_4..7)
N2 response static point force (q = 1 and q = 3, side 16); a lump as a sink (q = 1)
N3 correlations  equal-time connected C(r) of e and rho_j
N4 universality  sink strength of a- vs b-lumps and of two amplitudes
usage: python3 cpg_pilot2.py
"""

import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M  # noqa: E402

N1D, T1D, DTS = 512, 2000.0, 0.5
A_U = 0.005
SEEDS = (0, 1, 2, 3)
BG = range(4, 8)                  # populated complex components (main's upper part of D16)
MMAX = 40                         # density Fourier modes kept for the census
KSTRIDE = 8                       # field modes kept (every 8th, incl. k = 0)
F0 = 1e-3
LUMP_W, X0 = 8.0, N1D // 2


# ---------------------------------------------------------------- main's background, reproduced
def incoherent_upper(lat, amp, seed):
    """main's tower_populated_test.incoherent_upper (q = 1), generalised to q = 3 by fftn."""
    rng = np.random.default_rng(1000 + seed)
    wa = lat.branch_omega()
    wb = wa + lat.kappa
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    for c in range(4, lat.n):
        if lat.q == 1:
            a = rng.normal(size=lat.N) + 1j * rng.normal(size=lat.N)
            b = rng.normal(size=lat.N) + 1j * rng.normal(size=lat.N)
            psi = np.fft.ifft(a + b)
            dps = np.fft.ifft(-1j * wa * a + 1j * wb * b)
        else:
            a = rng.normal(size=lat.shape) + 1j * rng.normal(size=lat.shape)
            b = rng.normal(size=lat.shape) + 1j * rng.normal(size=lat.shape)
            psi = np.fft.ifftn(a + b).reshape(-1)
            dps = np.fft.ifftn(-1j * wa * a + 1j * wb * b).reshape(-1)
        s = amp / np.sqrt(np.mean(np.abs(psi) ** 2))
        psi, dps = s * psi, s * dps
        u[:, 2 * c], u[:, 2 * c + 1] = psi.real, psi.imag
        v[:, 2 * c], v[:, 2 * c + 1] = dps.real, dps.imag
    return u, v


def lump(lat, amp, branch):
    """Stationary lump (k0 = 0, width LUMP_W) in component 0 at X0, per-mode launch on one branch."""
    x = np.arange(lat.N)
    psi = amp * np.exp(-0.5 * ((x - X0) / LUMP_W) ** 2) + 0j
    wa = lat.branch_omega()
    om = -1j * wa if branch == "a" else 1j * (wa + lat.kappa)
    dps = np.fft.ifft(om * np.fft.fft(psi))
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    u[:, 0], u[:, 1], v[:, 0], v[:, 1] = psi.real, psi.imag, dps.real, dps.imag
    return u, v


# ---------------------------------------------------------------- densities (q = 1)
def e_density(u, v, sl):
    uu, vv = u[:, sl], v[:, sl]
    g = ((np.roll(uu, -1, 0) - uu) ** 2).sum(1)
    r = np.linalg.norm(uu, axis=1)
    return (0.5 * (vv * vv).sum(1) + 0.5 * M.SQ5 * (uu * uu).sum(1) + r ** 3 / 3
            + 0.25 * M.C * (g + np.roll(g, 1)))


def rho(lat, u, v, j):
    return (-v[:, 2 * j] * u[:, 2 * j + 1] + v[:, 2 * j + 1] * u[:, 2 * j]
            - 0.5 * lat.kappa * (u[:, 2 * j] ** 2 + u[:, 2 * j + 1] ** 2))


ALL, LOW, UP = slice(0, 16), slice(0, 8), slice(8, 16)


def rk4(lat, u, v, ext):
    dt = M.DT
    f = (lambda uu, vv: lat.force(uu, vv) + ext) if ext is not None else (lambda uu, vv: lat.force(uu, vv))
    k1v, k1u = f(u, v), v
    k2v, k2u = f(u + .5 * dt * k1u, v + .5 * dt * k1v), v + .5 * dt * k1v
    k3v, k3u = f(u + .5 * dt * k2u, v + .5 * dt * k2v), v + .5 * dt * k2v
    k4v, k4u = f(u + dt * k3u, v + dt * k3v), v + dt * k3v
    return (u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u), v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))


def ramp(t, tr):
    return 1.0 if t >= tr else 0.5 * (1 - math.cos(math.pi * t / tr))


# ---------------------------------------------------------------- one run
def run(job):
    kind, seed, amp_u = job["kind"], job["seed"], job["amp_u"]
    t0 = time.time()
    q3 = job.get("q3", False)
    if q3:
        side = 16
        lat = M.Lattice(n=8, N=side ** 3, q=3, shape=side, well="node")
        T, tr, avg0 = 300.0, 100.0, 150.0
    else:
        lat = M.Lattice(n=8, N=N1D, well="node")
        T, tr, avg0 = T1D, 200.0, 400.0
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    if amp_u > 0:
        u, v = incoherent_upper(lat, amp_u, seed)
    if kind.startswith("lump"):
        lu, lv = lump(lat, job["lamp"], job["branch"])
        u, v = u + lu, v + lv
    S = None
    if kind == "src":
        S = np.zeros((lat.N, lat.D)); S[0, 0] = F0
    E0 = lat.energy(u, v)
    out = dict(job=job)
    out["Q0"] = [float(rho(lat, u, v, j).sum()) for j in BG] if not q3 else None
    nsamp = int(round(T / DTS))
    steps = int(round(DTS / M.DT))
    census = kind == "bg" and not q3
    if census:
        dens = np.zeros((nsamp + 1, 5, MMAX + 1), complex)       # e, rho4..rho7
        psik = np.zeros((nsamp + 1, 4, N1D // KSTRIDE), complex)
        cwin = {w: np.zeros((5, N1D)) for w in ("early", "late", "all")}
        cnt = {w: 0 for w in cwin}
        occ0 = modes_occ(lat, u, v)
    if kind in ("src", "bg") or kind.startswith("lump"):
        uavg = np.zeros((lat.N, lat.D)); eBavg = np.zeros(lat.N) if not q3 else None; navg = 0
    snaps = {}
    series = []
    t = 0.0
    for i in range(nsamp + 1):
        if i > 0:
            ext = S * ramp(t, tr) if S is not None else None
            for _ in range(steps):
                u, v = rk4(lat, u, v, ext)
            t += DTS
        if census:
            fields = [e_density(u, v, ALL)] + [rho(lat, u, v, j) for j in BG]
            for a, X in enumerate(fields):
                Xk = np.fft.fft(X - X.mean())
                dens[i, a] = Xk[:MMAX + 1]
                p = np.abs(Xk) ** 2
                for w, ok in (("early", t <= 200), ("late", t >= T - 200), ("all", True)):
                    if ok:
                        cwin[w][a] += p
            for w, ok in (("early", t <= 200), ("late", t >= T - 200), ("all", True)):
                cnt[w] += ok
            for c, j in enumerate(BG):
                psik[i, c] = np.fft.fft(u[:, 2 * j] + 1j * u[:, 2 * j + 1])[::KSTRIDE]
        if t >= avg0 - 1e-9:
            uavg += u; navg += 1
            if not q3:
                eBavg += e_density(u, v, UP)
        if not q3:
            if any(abs(t - s) < 1e-9 for s in (200.0, 400.0)) or t >= T - 50 - 1e-9:
                key = "late" if t >= T - 50 - 1e-9 else f"t{int(t)}"
                snaps.setdefault(key, []).append(e_density(u, v, UP))
            if kind.startswith("lump") or i % 20 == 0:
                El = e_density(u, v, LOW).sum(); Eb = e_density(u, v, UP).sum()
                series.append((t, El, Eb, lat.energy(u, v) - El - Eb))
    out["uavg"] = uavg / max(navg, 1)
    if not q3:
        out["eBavg"] = eBavg / max(navg, 1)
        out["snaps"] = {k: np.mean(vv, axis=0) for k, vv in snaps.items()}
        out["series"] = np.array(series)
    if census:
        out.update(dens=dens, psik=psik, occ0=occ0, occT=modes_occ(lat, u, v),
                   corr={w: cwin[w] / max(cnt[w], 1) for w in cwin})
    out["drift"] = abs(lat.energy(u, v) - E0) / abs(E0) if E0 else 0.0
    out["Q"] = [float(rho(lat, u, v, j).sum()) for j in BG] if not q3 else None
    out["secs"] = time.time() - t0
    return out


def modes_occ(lat, u, v):
    """|a_k|^2 + |b_k|^2 per background component, smoothed over 16-mode bins."""
    wa = lat.branch_omega(); wb = wa + lat.kappa
    res = []
    for j in BG:
        P = np.fft.fft(u[:, 2 * j] + 1j * u[:, 2 * j + 1]); D = np.fft.fft(v[:, 2 * j] + 1j * v[:, 2 * j + 1])
        a = (wb * P + 1j * D) / (wa + wb); b = (wa * P - 1j * D) / (wa + wb)
        res.append(np.concatenate([(np.abs(a) ** 2).reshape(-1, 16).sum(1), (np.abs(b) ** 2).reshape(-1, 16).sum(1)]))
    return np.array(res)


# ---------------------------------------------------------------- analysis helpers
def green(K, shape):
    k = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m) for m in shape], indexing="ij")
    return np.fft.ifftn(1.0 / (K + 2 * M.C * sum(1 - np.cos(x) for x in k))).real


def spectrum(series):
    """Power vs angular frequency of a (nt,) complex series, Hann window."""
    x = series - series.mean()
    x = x * np.hanning(len(x))
    P = np.abs(np.fft.fft(x)) ** 2
    w = 2 * np.pi * np.fft.fftfreq(len(x), d=DTS)
    return w, P


def line_peak(w, P, lo, hi):
    aw = np.abs(w)
    idx = np.where((aw > lo) & (aw < hi))[0]
    i = idx[np.argmax(P[idx])]
    j = [np.where(np.isclose(w, w[i] + s * (w[1] - w[0])))[0] for s in (-1, 1)]
    if all(len(x) for x in j):
        a, b, c = np.log(P[j[0][0]]), np.log(P[i]), np.log(P[j[1][0]])
        d = 0.5 * (a - c) / (a - 2 * b + c)
        return abs(w[i] + d * (w[1] - w[0]))
    return abs(w[i])


def main():
    K = M.SQ5
    ka = M.KAPPA
    kk = np.linspace(-np.pi, np.pi, 200001)
    wa_k = 0.5 * (-ka + np.sqrt(ka ** 2 + 4 * (K + 2 * M.C * (1 - np.cos(kk)))))
    vmax = float(np.max(2 * M.C * np.sin(kk) / (2 * wa_k + ka)))

    jobs = []
    for s in SEEDS:
        jobs.append(dict(kind="bg", seed=s, amp_u=A_U))
        jobs.append(dict(kind="src", seed=s, amp_u=A_U))
        for tag, la, br in (("a05", 0.05, "a"), ("b05", 0.05, "b"), ("a02", 0.02, "a")):
            jobs.append(dict(kind="lump_" + tag, seed=s, amp_u=A_U, lamp=la, branch=br))
    for s in (0, 1):
        jobs.append(dict(kind="bg", seed=s, amp_u=0.02))
    jobs.append(dict(kind="bg", seed=0, amp_u=1e-5))                    # vacuum line, same method
    jobs.append(dict(kind="src", seed=0, amp_u=0.0))                    # vacuum static response
    for tag, la, br in (("a05", 0.05, "a"), ("b05", 0.05, "b"), ("a02", 0.02, "a")):
        jobs.append(dict(kind="lump_" + tag, seed=0, amp_u=0.0, lamp=la, branch=br))  # isolated lumps
    for s in (0, 1):
        jobs.append(dict(kind="src", seed=s, amp_u=A_U, q3=True))
        jobs.append(dict(kind="bg", seed=s, amp_u=A_U, q3=True))
    jobs.append(dict(kind="src", seed=0, amp_u=0.0, q3=True))
    jobs.sort(key=lambda j: -j.get("q3", False))                         # long ones first
    with Pool(4) as pool:
        res = pool.map(run, jobs, chunksize=1)

    def get(kind, seed, amp_u, q3=False):
        for r in res:
            j = r["job"]
            if j["kind"] == kind and j["seed"] == seed and j["amp_u"] == amp_u and j.get("q3", False) == q3:
                return r
        raise KeyError((kind, seed, amp_u, q3))

    print("=" * 96)
    print("CP-G SECOND PILOT -- the populated background (predictions: CPG_PILOT2_PREDICTIONS.md, 99131c0)")
    print("=" * 96)
    print(f"background: A', kappa* = {ka:.6f}, D16, ring N = {N1D}, comps 4-7 incoherent (main's incoherent_upper), "
          f"T = {T1D:.0f}; v_max = {vmax:.4f}")
    print("validation: energy drift max over runs = %.1e; background charge drift max = see below"
          % max(r["drift"] for r in res))
    rB = []
    for s in SEEDS:
        lat = M.Lattice(n=8, N=N1D, well="node")
        uu, _ = incoherent_upper(lat, A_U, s)
        rB.append(np.linalg.norm(uu, axis=1).mean())
    dK = float(np.mean(rB))
    print(f"measured <|psi_B|> at t = 0 = {dK:.5f} (predicted 0.00969); tangent-term shift for a bg component "
          f"{dK*9/8:.5f}")

    # ---------------- N1
    print("\nN1 CENSUS")
    lines = []
    for s in SEEDS:
        r = get("bg", s, A_U)
        for c in range(4):
            w, P = spectrum(r["psik"][:, c, 0])
            lines.append(line_peak(w, P, 1.0, 1.2))
    rv = get("bg", 0, 1e-5)
    vac = [line_peak(*spectrum(rv["psik"][:, c, 0]), 1.0, 1.2) for c in range(4)]
    pred = (-ka + math.sqrt(ka ** 2 + 4 * (K + dK * 9 / 8))) / 2
    print(f" 1a k = 0 a-branch line: populated {np.mean(lines):.5f} +- {np.std(lines)/2:.5f} (16 lines); "
          f"vacuum (A_U = 1e-5, same method) {np.mean(vac):.5f}; predicted {pred:.5f} (vacuum 1.08643)")
    print(f"    shift populated - vacuum = {np.mean(lines)-np.mean(vac):+.5f} (predicted {pred-1.086434:+.5f})")
    low = tot = 0.0
    for s in SEEDS:
        r = get("bg", s, A_U)
        for c in range(4):
            for m in range(r["psik"].shape[2]):
                w, P = spectrum(r["psik"][:, c, m])
                low += P[np.abs(w) < 1.0].sum(); tot += P.sum()
    print(f"    field power below |w| = 1.0: fraction {low/tot:.2e} (predicted < 1e-3) -> field GAPPED")

    names = ["e", "rho4", "rho5", "rho6", "rho7"]
    for amp in (A_U, 0.02):
        seeds = SEEDS if amp == A_U else (0, 1)
        print(f" 1b density spectra, A_U = {amp}  (m: q = 2 pi m / {N1D}; low band |w| < 0.7)")
        print("    field   m   w95/(vmax q)  w_rms/q   peak-frac(+-10%)  mid(0.7-2.9)/total   S(q,0)/S(q,peak)")
        pfits = {}
        for a, nm in enumerate(names):
            rms = []
            for m in range(1, MMAX + 1):
                Pm = None
                for s in seeds:
                    w, P = spectrum(get("bg", s, amp)["dens"][:, a, m])
                    Pm = P if Pm is None else Pm + P
                q = 2 * np.pi * m / N1D
                aw = np.abs(w)
                lb = aw < 0.7
                wl, Pl = aw[lb], Pm[lb]
                o = np.argsort(wl); cw = np.cumsum(Pl[o]) / Pl.sum()
                w95 = wl[o][np.searchsorted(cw, 0.95)]
                wr = math.sqrt((Pl * wl ** 2).sum() / Pl.sum())
                rms.append(wr)
                ip = np.argmax(Pl); wp = wl[ip]
                pf = Pl[(wl > 0.9 * wp) & (wl < 1.1 * wp)].sum() / Pl.sum() if wp > 0 else 1.0
                mid = Pm[(aw > 0.7) & (aw < 2.9)].sum() / Pm.sum()
                s0 = Pm[aw < 0.5 * (w[1] - w[0])].sum() / Pl.max()
                if m in (4, 8, 16, 32) and (a == 0 or m == 16):
                    print(f"    {nm:6s} {m:3d}   {w95/(vmax*q):.3f}        {wr/q:.3f}     {pf:.3f}             "
                          f"{mid:.2e}             {s0:.3f}")
            ms = np.arange(4, 33)
            pfits[nm] = np.polyfit(np.log(2 * np.pi * ms / N1D), np.log(np.array(rms)[ms - 1]), 1)[0]
        print("    exponent p (w_rms ~ q^p, m = 4..32): " + ", ".join(f"{k} {v:.3f}" for k, v in pfits.items()))
    for amp in (A_U, 0.02):
        seeds = SEEDS if amp == A_U else (0, 1)
        ch = []
        for s in seeds:
            r = get("bg", s, amp)
            ch.append(np.sqrt(np.mean((r["occT"] / r["occ0"] - 1) ** 2)))
        print(f" 1c occupation change over T (16-mode bins, rms relative): A_U = {amp}: {np.mean(ch):.2e} "
              f"(predicted < 2e-2 at 0.005; < 0.3 at 0.02)")
    qd = max(max(abs(a - b) / abs(b) for a, b in zip(r["Q"], r["Q0"])) for r in res if r["Q"] and r["job"]["amp_u"] >= A_U)
    print(f" validation: background charge drift (per component, relative) max over populated runs = {qd:.1e}")

    # ---------------- N2
    print("\nN2 RESPONSE")
    Gv = green(K, (N1D,)); Gp = green(K + dK, (N1D,))
    rvac = get("src", 0, 0.0)
    gvac = rvac["uavg"][:, 0] / F0
    print(f" 2a vacuum check: max |<u0>/F0 - G_vac| = {np.max(np.abs(gvac - Gv)):.1e} (G_vac(0) = {Gv[0]:.5f})")
    diffs = []
    for s in SEEDS:
        d = (get("src", s, A_U)["uavg"] - get("bg", s, A_U)["uavg"]) / F0
        diffs.append(d)
    d = np.mean(diffs, axis=0)
    g0 = [x[0, 0] / Gv[0] for x in diffs]
    print(f"    populated: G(0)/G_vac(0) = {np.mean(g0):.5f} +- {np.std(g0)/2:.5f} (predicted {Gp[0]/Gv[0]:.5f})")
    x = np.arange(1, 6)
    xi_m = -1 / np.polyfit(x, np.log(np.abs(d[x, 0])), 1)[0]
    xi_v = -1 / np.polyfit(x, np.log(Gv[x]), 1)[0]; xi_p = -1 / np.polyfit(x, np.log(Gp[x]), 1)[0]
    print(f"    decay length (x = 1..5): measured {xi_m:.4f}, predicted populated {xi_p:.4f}, vacuum {xi_v:.4f}")
    print("    <du0>/F0 at x = 0..6: " + " ".join(f"{d[i,0]:.3e}" for i in range(7)))
    print("    G_pop       x = 0..6: " + " ".join(f"{Gp[i]:.3e}" for i in range(7)))
    oth = max(np.max(np.abs(d[:, c])) for c in [1] + list(range(8, 16)))
    print(f"    other components max |<du>|/du0(0) = {oth/abs(d[0,0]):.2e} (predicted < 1e-2)")
    Es = 0.5 * F0 * F0 * d[0, 0]
    far = []
    for s in SEEDS:
        de = get("src", s, A_U)["eBavg"] - get("bg", s, A_U)["eBavg"]
        xx = np.abs(((np.arange(N1D) + N1D // 2) % N1D) - N1D // 2)
        sel = (xx >= 5) & (xx <= 250)
        far.append((de[sel].sum(), np.abs(de[sel]).sum(), de[xx < 5].sum()))
    far = np.array(far)
    print(f" 2b source static field energy {Es:.3e}; background energy change (time-avg), seed mean:")
    print(f"    near (|x| < 5) {far[:,2].mean():+.3e}; far (5..250) signed {far[:,0].mean():+.3e} +- "
          f"{far[:,0].std()/2:.1e}; far sum|.| {far[:,1].mean():.3e} = {far[:,1].mean()/Es:.2e} of the source energy "
          f"(predicted < 1e-2)")

    side = 16
    Gv3 = green(K, (side,) * 3); Gp3 = green(K + dK, (side,) * 3)
    v3 = get("src", 0, 0.0, True)["uavg"][:, 0].reshape((side,) * 3) / F0
    print(f" 2c q = 3, side 16: vacuum check max |<u0>/F0 - G_vac| = {np.max(np.abs(v3 - Gv3)):.1e}")
    d3 = [((get("src", s, A_U, True)["uavg"] - get("bg", s, A_U, True)["uavg"])[:, 0]).reshape((side,) * 3) / F0
          for s in (0, 1)]
    dm = np.mean(d3, axis=0)
    ax = np.array([dm[i, 0, 0] for i in range(9)])
    print("    <du0>/F0 along axis r = 0..8: " + " ".join(f"{x:.3e}" for x in ax))
    print("    G_pop (side 16)       r = 0..8: " + " ".join(f"{Gp3[i,0,0]:.3e}" for i in range(9)))
    print(f"    G(0)/G_vac(0) = {ax[0]/Gv3[0,0,0]:.5f} (seeds {d3[0][0,0,0]/Gv3[0,0,0]:.5f}, "
          f"{d3[1][0,0,0]/Gv3[0,0,0]:.5f}; predicted {Gp3[0,0,0]/Gv3[0,0,0]:.5f})")
    print("    ratio to G_pop r = 0..3: " + " ".join(f"{ax[i]/Gp3[i,0,0]:.4f}" for i in range(4)))
    r_ = np.arange(1, 5)
    xi3 = -1 / np.polyfit(r_, np.log(r_ * np.abs(ax[1:5])), 1)[0]
    print(f"    decay length (ln rG, r = 1..4): {xi3:.4f} (side-16 G_pop: "
          f"{-1/np.polyfit(r_, np.log(r_*np.array([Gp3[i,0,0] for i in r_])), 1)[0]:.4f})")
    print(f"    max |<du0>| for r >= 5 (any direction) / du0(0) = "
          f"{np.max(np.abs(dm[far3mask(side)]))/abs(ax[0]):.2e} (predicted < 1e-4)")

    print(" 2d lump as a sink (component 0, k0 = 0, width 8, at x0 = %d)" % X0)
    fr = {}
    for tag in ("a05", "b05", "a02"):
        iso = get("lump_" + tag, 0, 0.0)["series"]
        fs, rows = [], []
        for s in SEEDS:
            r = get("lump_" + tag, s, A_U)
            ref = get("bg", s, A_U)
            ser = r["series"]
            El0 = iso[0, 1]
            f = (ser[-1, 1] - iso[-1, 1]) / El0
            fs.append(f)
            xx = np.abs(np.arange(N1D) - X0)
            dEB = {k: r["snaps"][k] - ref["snaps"][k] for k in r["snaps"]}
            late = dEB["late"]
            tot = late.sum()
            farE = late[xx > 30].sum()
            cone = {}
            for k, tt in (("t200", 200.0), ("t400", 400.0)):
                rc = vmax * tt + 30
                inside = dEB[k][(xx > 30) & (xx <= rc)].sum()
                outside = dEB[k][xx > rc].sum() if np.any(xx > rc) else 0.0
                cone[k] = (inside, outside, rc)
            inner = late[(xx > 30) & (xx <= 143)].mean(); outer = late[xx > 143].mean()
            bal = (ser[-1, 1] - ser[0, 1]) + (ser[-1, 2] - ser[0, 2]) + (ser[-1, 3] - ser[0, 3])
            rows.append((f, tot, farE, cone, inner / outer if outer else np.nan, bal / (ser[0, 1] + ser[0, 2])))
        fr[tag] = np.array(fs)
        print(f"    {tag}: f(T) seeds " + " ".join(f"{x:+.2e}" for x in fs) + f"; mean {np.mean(fs):+.3e}")
        for s, (f, tot, farE, cone, io, bal) in zip(SEEDS, rows):
            print(f"      seed {s}: dE_B total(T) {tot:+.3e}, far (|x-x0|>30) {farE:+.3e} ({farE/tot if tot else 0:.2f}); "
                  f"t=200 in-cone {cone['t200'][0]:+.2e} / beyond {cone['t200'][1]:+.2e}; "
                  f"t=400 in {cone['t400'][0]:+.2e} / beyond {cone['t400'][1]:+.2e}; inner/outer {io:.2f}; "
                  f"balance {bal:+.1e}")
    print("\nN4 UNIVERSALITY (source side)")
    ra = fr["a05"].mean() / fr["b05"].mean()
    print(f"    f(a-lump 0.05) / f(b-lump 0.05) = {ra:.3f} (predicted outside [0.8, 1.25]; central 1.9)")
    print(f"    f(a-lump 0.02) / f(a-lump 0.05) = {fr['a02'].mean()/fr['a05'].mean():.3f} (predicted [0.6, 1.6])")

    # ---------------- N3
    print("\nN3 EQUAL-TIME CONNECTED CORRELATIONS, A_U = 0.005")
    for a, nm in enumerate(names):
        cs = {w: [] for w in ("early", "late", "all")}
        for s in SEEDS:
            for w in cs:
                C = np.fft.ifft(get("bg", s, A_U)["corr"][w][a]).real
                cs[w].append(C / C[0])
        allc = np.array(cs["all"])
        mean_far = allc[:, 5:256].mean(1)
        print(f"    {nm:5s} C(r)/C(0) r=1..4: " + " ".join(f"{allc.mean(0)[i]:+.4f}" for i in range(1, 5))
              + f"; max|.| r>=3 {np.max(np.abs(allc.mean(0)[3:256])):.2e}; mean r in [5,255] "
              f"{mean_far.mean():+.2e} +- {mean_far.std()/2:.1e}; early r=1 {np.mean(cs['early'],0)[1]:+.4f} "
              f"late r=1 {np.mean(cs['late'],0)[1]:+.4f}")
    print("\nrun time %.0f s (sum over jobs)" % sum(r["secs"] for r in res))


def far3mask(side):
    c = np.indices((side,) * 3)
    d = np.minimum(c, side - c)
    return np.sqrt((d ** 2).sum(0)) >= 5


if __name__ == "__main__":
    main()
