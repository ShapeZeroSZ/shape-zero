#!/usr/bin/env python3
"""
irrev_pilot.py -- target 5, the GENERIC completion of main ("G"), q = 1 pilot.
Predictions committed first: natural/IRREV_HYPOTHESES.md (1c9e5bf, 2026-09-28T13:56:35Z).

Model G. State per node: main's (u, v) plus an internal energy eps_i (T_i = eps_i / C, s_i = C ln eps_i).
  Hamiltonian part: main's force (04_scripts/session/model.py), RK4 at model.DT, unchanged.
  Dissipative part, applied after each RK4 step (Lie splitting), exact energy booking:
   D1 absorption at matter: each tower component's RADIAL velocity w = v_c . u_c/|u_c| at node i gets an exact
      Ornstein-Uhlenbeck update with rate gamma_i and temperature T_i (noise off in the deterministic limit);
      the kinetic energy removed, sum (w^2 - w'^2)/2, is added to eps_i. Radial kicks leave v . JJ u_c unchanged,
      so every component's U(1) charge is conserved exactly (J-compatibility, GENERIC degeneracy).
      gamma_i = Gamma * e_low,i (variant G) or Gamma_R * |u_low,i|^2 (variant G-R), plus a uniform gamma0.
   D2 heat conduction: eps_i += dt K_th (T_{i+1} + T_{i-1} - 2 T_i).
Tests: I2 (sink, arrow), I3 (parity, relaxation), I4 at q = 1 (depletion, sink ratios, drag). q = 3: irrev_q3.py.
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
import cpg_pilot2 as P  # noqa: E402  (main's incoherent_upper, densities, rk4 with model.py's force)

M = P.M
N, X0, A_U = 512, 256, 0.005
TOWER = range(4, 8)
DT = M.DT
VMAX = 0.4859


def make_lump(lat, amp, branch, k0=0.0):
    x = np.arange(lat.N)
    psi = amp * np.exp(-0.5 * ((x - X0) / 8.0) ** 2) * np.exp(1j * k0 * (x - X0))
    wa = lat.branch_omega()
    om = -1j * wa if branch == "a" else 1j * (wa + lat.kappa)
    dps = np.fft.ifft(om * np.fft.fft(psi))
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    u[:, 0], u[:, 1], v[:, 0], v[:, 1] = psi.real, psi.imag, dps.real, dps.imag
    return u, v


def radial_theta(u, v):
    w = []
    for c in TOWER:
        uc, vc = u[:, 2 * c:2 * c + 2], v[:, 2 * c:2 * c + 2]
        r = np.linalg.norm(uc, axis=1) + 1e-300
        w.append(((vc * uc).sum(1) / r) ** 2)
    return float(np.mean(w))


def dissipate(u, v, eps, gam, C, K_th, noise, rng):
    """One dissipative step of length DT; returns number of eps clamps."""
    T = eps / C
    a = np.exp(-gam * DT)
    s = np.sqrt(np.maximum(T, 0.0) * (1 - a * a)) if noise else None
    dE = np.zeros(len(eps))
    for c in TOWER:
        uc, vc = u[:, 2 * c:2 * c + 2], v[:, 2 * c:2 * c + 2]
        r = np.linalg.norm(uc, axis=1)
        ok = r > 0
        rh = np.zeros_like(uc); rh[ok] = uc[ok] / r[ok, None]
        w = (vc * rh).sum(1)
        w2 = a * w + (s * rng.standard_normal(len(w)) if noise else 0.0)
        w2 = np.where(ok, w2, w)
        vc += (w2 - w)[:, None] * rh
        dE += 0.5 * (w * w - w2 * w2)
    eps += dE
    if K_th:
        T = eps / C
        eps += DT * K_th * (np.roll(T, 1) + np.roll(T, -1) - 2 * T)
    bad = eps < 0
    nb = int(bad.sum())
    if nb:
        eps[bad] = 0.0
    return nb


def run(job):
    t0 = time.time()
    rng = np.random.default_rng(10_000 + 97 * job.get("seed", 0) + job.get("salt", 0))
    lat = M.Lattice(n=8, N=N, well="node")
    u = np.zeros((N, lat.D)); v = np.zeros((N, lat.D))
    if job.get("tower", True):
        u, v = P.incoherent_upper(lat, A_U, job["seed"])
    if job.get("lump"):
        lu, lv = make_lump(lat, 0.05, job["lump"], job.get("k0", 0.0))
        u, v = u + lu, v + lv
    C, K_th = job["C"], job["K_th"]
    Gam, var, g0 = job.get("Gamma", 0.0), job.get("variant", "E"), job.get("gamma0", 0.0)
    noise = job.get("noise", True)
    th0 = radial_theta(u, v)
    T0 = job["T0"] if "T0" in job else th0 * (1 - job["delta"])
    eps = np.full(N, C * T0)
    Em0 = lat.energy(u, v)
    E0 = Em0 + eps.sum()
    Qc0 = [float(P.rho(lat, u, v, c).sum()) for c in TOWER]
    rec = dict(t=[], Em=[], Eh=[], S=[], th=[], Tm=[], xc=[])
    snaps = {}
    T, dts = job["T"], job.get("dts", 1.0)
    steps = int(round(dts / DT))
    clamps = 0
    t = 0.0
    xs = np.arange(N)

    def sample():
        rec["t"].append(t); rec["Em"].append(lat.energy(u, v)); rec["Eh"].append(float(eps.sum()))
        rec["S"].append(float(C * np.log(np.maximum(eps, 1e-300)).sum()))
        rec["th"].append(radial_theta(u, v)); rec["Tm"].append(float(eps.mean() / C))
        el = P.e_density(u, v, P.LOW)
        if el.sum() > 0:
            ang = 2 * np.pi * xs / N
            rec["xc"].append(float(np.angle((el * np.exp(1j * ang)).sum()) * N / (2 * np.pi)))
        else:
            rec["xc"].append(0.0)
        for ts in job.get("snaps", ()):
            if abs(t - ts) < 1e-9:
                snaps[ts] = P.e_density(u, v, P.UP)

    sample()
    nsteps = int(round(T / dts))
    for _ in range(nsteps):
        for _ in range(steps):
            u, v = P.rk4(lat, u, v, None)
            if Gam or g0:
                if Gam:
                    base = P.e_density(u, v, P.LOW) if var == "E" else (u[:, :8] ** 2).sum(1)
                    gam = Gam * base + g0
                else:
                    gam = np.full(N, g0)
                clamps += dissipate(u, v, eps, gam, C, K_th, noise, rng)
        t += dts
        sample()
    E1 = lat.energy(u, v) + eps.sum()
    Qc1 = [float(P.rho(lat, u, v, c).sum()) for c in TOWER]
    Ntot = []
    wa = lat.branch_omega(); wb = wa + lat.kappa
    for c in TOWER:
        Pk = np.fft.fft(u[:, 2 * c] + 1j * u[:, 2 * c + 1]); Dk = np.fft.fft(v[:, 2 * c] + 1j * v[:, 2 * c + 1])
        a = (wb * Pk + 1j * Dk) / (wa + wb); b = (wa * Pk - 1j * Dk) / (wa + wb)
        Ntot.append(float(((wa + lat.kappa / 2) * abs(a) ** 2 + (wb - lat.kappa / 2) * abs(b) ** 2).sum() / N))
    out = {k: np.array(val) for k, val in rec.items()}
    out.update(job=job, snaps=snaps, dE=abs(E1 - E0) / abs(Em0), clamps=clamps, th0=th0, T0=T0,
               dQ=max(abs(q1 - q0) / nt for q0, q1, nt in zip(Qc0, Qc1, Ntot)), secs=time.time() - t0)
    return out


def heat_only():
    """I3(b): pure conduction on a 512-ring, halves at T0 +- dT/2, C = 1, K_th = 1."""
    res = []
    lam_pred = 1.0 * (2 - 2 * math.cos(2 * math.pi / N)) / 1.0
    for dT in (1e-1, 1e-2, 1e-3, 1e-4):
        T = np.full(N, 1.0); T[:N // 2] += dT / 2; T[N // 2:] -= dT / 2
        J0 = (T[N // 2 - 1] - T[N // 2]) * 1.0
        dt = 0.1; amps = []; ts = []
        for i in range(200001):
            if i % 5000 == 0:
                amps.append(abs(np.fft.fft(T)[1]) * 2 / N); ts.append(i * dt)
            T = T + dt * (np.roll(T, 1) + np.roll(T, -1) - 2 * T)
        lam = -np.polyfit(np.array(ts[4:]), np.log(np.array(amps[4:])), 1)[0]
        res.append((dT, J0, lam))
    return res, lam_pred


def main():
    base = dict(C=1000.0, K_th=1.0, Gamma=10.0, T0=1e-5)
    jobs = []
    for s in range(4):
        jobs.append(dict(base, tag=f"sink{s}", seed=s, lump="a", T=2000.0, snaps=(200.0, 400.0, 1000.0, 2000.0)))
    jobs.append(dict(base, tag="det", seed=0, lump="a", T=2000.0, noise=False, snaps=(200.0, 400.0, 1000.0, 2000.0)))
    jobs.append(dict(base, tag="ref", seed=0, lump="a", T=2000.0, Gamma=0.0, snaps=(200.0, 400.0, 1000.0, 2000.0)))
    for s in (0, 1):
        for lump in ("a", "b"):
            jobs.append(dict(base, tag=f"E_{lump}{s}", seed=s, lump=lump, T=50.0, dts=0.5))
            jobs.append(dict(base, tag=f"R_{lump}{s}", seed=s, lump=lump, T=50.0, dts=0.5, variant="R",
                             Gamma=10.0 * 0.5 * (1.0864345 ** 2 + M.SQ5)))
        for d in (0.5, 0.2, 0.1, 0.05):
            jobs.append(dict(tag=f"par{d}_{s}", seed=s, C=4.0, K_th=0.0, gamma0=0.05, delta=d, T=60.0, dts=0.5))
    jobs.append(dict(base, tag="dragG", seed=0, lump="a", k0=math.pi / 4, T=500.0, noise=False))
    jobs.append(dict(base, tag="dragR", seed=0, lump="a", k0=math.pi / 4, T=500.0, noise=False, Gamma=0.0))
    jobs.sort(key=lambda j: -j["T"])
    with Pool(4) as pool:
        R = {r["job"]["tag"]: r for r in pool.map(run, jobs, chunksize=1)}

    print("=" * 92)
    print("IRREVERSIBILITY PILOT, q = 1 (predictions: IRREV_HYPOTHESES.md, 1c9e5bf)")
    print("=" * 92)
    print(f"validation: max total-energy drift {max(r['dE'] for r in R.values()):.1e} of the mechanical energy (<= 5e-5); max charge drift "
          f"{max(r['dQ'] for r in R.values()):.1e} of N_a+N_b (<= 5e-5); eps clamps {sum(r['clamps'] for r in R.values())}")
    th = [R[f"sink{s}"]["th0"] for s in range(4)]
    print(f"theta_r at t = 0: {np.mean(th):.3e} (predicted 5.9e-5 +- 15%)")

    print("\nI2  SINK (fluctuating G, a-lump, C = 1000, Gamma = 10, K_th = 1)")
    for s in range(4):
        r = R[f"sink{s}"]; Eh = r["Eh"] - r["Eh"][0]
        win = [(Eh[k + 100] - Eh[k]) / 100 for k in range(0, 2000, 100)]
        print(f"    seed {s}: P0 (first 10) {Eh[10]/10:.3e}; cumulative at T {Eh[-1]:.4f}; 100-unit window rates min "
              f"{min(win):+.2e} max {max(win):+.2e}; windows > 0: {sum(w > 0 for w in win)}/20")
    print(f"    predicted P0 1.45e-4 (x2), cumulative [0.03, 0.2], every window > 0")
    rd = R["det"]
    dS = np.diff(rd["S"])
    print(f"    ARROW deterministic: S decreases at {int((dS < 0).sum())} of {len(dS)} samples (predicted 0); "
          f"S(T) - S(0) = {rd['S'][-1]-rd['S'][0]:+.3e}")
    Sm = np.mean([R[f"sink{s}"]["S"] for s in range(4)], axis=0)
    wS = [Sm[k] for k in range(0, 2001, 100)]
    print(f"    ARROW fluctuating: seed-mean S at 100-unit marks nondecreasing: {all(b >= a for a, b in zip(wS, wS[1:]))}; "
          f"increments min {min(np.diff(wS)):+.2e}")
    sdev = [np.std(np.diff(R[f'sink{s}']['S'])) for s in range(4)]
    worst = min(min(np.diff(R[f"sink{s}"]["S"])) / sd for s, sd in zip(range(4), sdev))
    print(f"    single-run S decreases: worst step = {worst:+.2f} sigma of step fluctuation (predicted > -3)")

    print("\nI3a PARITY (uniform gamma0 = 0.05, C = 4, K_th = 0): J over [0, 5] and relaxation rate of theta_r - T")
    Js, ds, rates = [], [], []
    for d in (0.5, 0.2, 0.1, 0.05):
        jj, rr = [], []
        for s in (0, 1):
            r = R[f"par{d}_{s}"]
            jj.append((r["Eh"][10] - r["Eh"][0]) / 5.0)
            x = r["th"] - r["Tm"]; t = r["t"]
            sel = (x > 0.2 * x[0]) & (t <= 40)
            k = np.where(~sel)[0]; last = k[0] if len(k) else len(x)
            rr.append(-np.polyfit(t[:last], np.log(x[:last]), 1)[0] if last > 4 else float("nan"))
        Js.append(np.mean(jj)); ds.append(d); rates.append(np.nanmean(rr))
        print(f"    delta = {d:<5}: J = {np.mean(jj):+.3e} (seeds {jj[0]:+.2e}, {jj[1]:+.2e}); relaxation rate "
              f"{np.nanmean(rr):.4f} (seeds {rr[0]:.4f}, {rr[1]:.4f})")
    p = np.polyfit(np.log(ds), np.log(np.abs(Js)), 1)[0]
    print(f"    J ~ delta^p: p = {p:.3f} (predicted 1.0 +- 0.1); rate spread (max-min)/mean = "
          f"{(max(rates)-min(rates))/np.mean(rates):.3f} (predicted < 0.10; rate in [0.05, 0.4])")
    ho, lam_pred = heat_only()
    print("I3b PURE CONDUCTION: dT, J0, slowest-mode decay rate (predicted J ~ dT, rate = %.4e independent of dT)" % lam_pred)
    for dT, J0, lam in ho:
        print(f"    dT = {dT:.0e}: J0 = {J0:.3e}  J0/dT = {J0/dT:.4f}  rate = {lam:.4e} ({lam/lam_pred-1:+.2%})")

    print("\nI4  q = 1 DEPLETION (deterministic G minus Gamma = 0, seed 0): tower energy change by region")
    xx = np.abs(np.arange(N) - X0)
    for ts in (200.0, 400.0, 1000.0, 2000.0):
        d = R["det"]["snaps"][ts] - R["ref"]["snaps"][ts]
        cone = VMAX * ts + 30
        near = d[xx <= 30].sum(); mid = d[(xx > 30) & (xx <= cone)].sum(); far = d[xx > cone].sum() if cone < 256 else 0.0
        print(f"    t = {ts:6.0f}: |x-x0| <= 30 {near:+.3e}; 30 < |x-x0| <= cone({min(cone,256):.0f}) {mid:+.3e}; "
              f"beyond cone {far:+.3e}")
    print("    SINK RATIO over [0, 50] (fluctuating): absorbed(b) / absorbed(a)")
    for var in ("E", "R"):
        ra = [R[f"{var}_a{s}"]["Eh"][-1] - R[f"{var}_a{s}"]["Eh"][0] for s in (0, 1)]
        rb = [R[f"{var}_b{s}"]["Eh"][-1] - R[f"{var}_b{s}"]["Eh"][0] for s in (0, 1)]
        pred = "1.885 +- 10% (by construction)" if var == "E" else "1.00 +- 10%"
        print(f"    G{'' if var == 'E' else '-R'}: a {np.mean(ra):.3e}, b {np.mean(rb):.3e}, ratio {np.mean(rb)/np.mean(ra):.3f}  (predicted {pred})")
    g, r0 = R["dragG"], R["dragR"]
    dx = np.unwrap(g["xc"] * 2 * np.pi / N) * N / (2 * np.pi); dx0 = np.unwrap(r0["xc"] * 2 * np.pi / N) * N / (2 * np.pi)
    vel = (dx0[-1] - dx0[0]) / 500.0
    print(f"    F-DRAG (k0 = pi/4, deterministic): lump displacement over 500: with sinks {dx[-1]-dx[0]:.4f}, without "
          f"{dx0[-1]-dx0[0]:.4f}; dv/v = {((dx[-1]-dx[0])-(dx0[-1]-dx0[0]))/(dx0[-1]-dx0[0]):+.2e} (predicted < 0, |.| < 1e-2); v = {vel:.4f}")
    print("\nrun time %.0f s (sum)" % sum(r["secs"] for r in R.values()))


if __name__ == "__main__":
    main()
