#!/usr/bin/env python3
"""tower_populated_test.py -- the tower as an environment under node form (A'), with EVERY level
populated from the start (non-isolation, PREMISE_LEDGER P0). Predictions in
tower_populated_predictions.txt, committed with this script before any run.

Same model as tower_env_test.py (E1-E3b): A' unchanged, C_r = 0, no new coupling, no new
parameter. Node n = 8, 16, 32 (D16, D32, D64); the D <= 8 part = components 0-7 (the octonion
half, 4 complex components); upper = the rest (M = n - 4 = 4, 12, 28 complex components).
q = 1 ring, N = 128, kappa*, T = 2000.

Initial state:
  D <= 8 part: the coherent packet of tower_env_test.py (amplitude 0.05 in the first dimer, width 8,
    k0 = pi/2, per-mode launch).
  upper part: INCOHERENT, low amplitude -- in every upper complex component, every Fourier mode of the
    ring on BOTH chirality branches (w_a^2 + kappa w_a = Q(k), w_b = w_a + kappa) with independent
    complex-Gaussian amplitudes (random phases), normalised so the rms of |psi_c| over sites is A_U
    in each component. A_U = 0.005 (10% of the packet's peak). Realisations: seeds 0-3.
  control F (fixed total): D32 and D64 with A_U * sqrt(4 / M), so the upper part's total norm
    equals D16's (separates "more levels" from "more energy"); seed 0.
  validation V: D16 with A_U = 0 must reproduce the isolated reference (D8 lattice) exactly.

Per part (at every DTS): self-energy E_self = kinetic + sqrt5 |u|^2/2 + c grad^2/2 + sum_x |u_part(x)|^3/3
  -- the energy the part would conserve if isolated, so every change of E_l,self is caused by the
  upper levels; E_int = E_total - E_l,self - E_u,self. Charge Q = sum v.JJ u - (kappa/2)|u|^2 per part.
Coherence of the D <= 8 part: chi = psi + i dpsi / omega over its 4 complex components; overlap with the
  isolated reference O_ref = <chi_ref, chi> / (|chi_ref||chi|); overlap between realisations
  O_ss' = <chi_s, chi_s'> / (|chi_s||chi_s'|); chirality purity 1 - |bar|/|chi| as model.readout.
usage: python3 tower_populated_test.py            (validation + all runs, prints the summary)
"""
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from multiprocessing import Pool  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M  # noqa: E402

N, AMP, T, DTS = 128, 0.05, 2000.0, 2.0
A_U = 0.005
SEEDS = (0, 1, 2, 3)
CHI_EVERY = 5            # store chi every CHI_EVERY samples (every 10 time units)


def self_energy(lat, u, v, sl):
    uu, vv = u[:, sl], v[:, sl]
    grad = ((np.roll(uu, -1, 0) - uu) ** 2).sum()
    return float(0.5 * (vv * vv).sum() + 0.5 * M.SQ5 * (uu * uu).sum() + 0.5 * M.C * grad
                 + (np.linalg.norm(uu, axis=1) ** 3).sum() / 3)


def charge(lat, u, v, sl):
    uu, vv = u[:, sl], v[:, sl]
    return float(np.einsum("na,ab,nb->", vv, lat.JJ[sl, sl], uu) - 0.5 * lat.kappa * (uu * uu).sum())


def chi_low(lat, u, v):
    psi = u[:, 0:8:2] + 1j * u[:, 1:8:2]
    dps = v[:, 0:8:2] + 1j * v[:, 1:8:2]
    return psi + (1j / lat.omega) * dps, psi - (1j / lat.omega) * dps


def incoherent_upper(lat, amp, seed):
    """Random-phase field on both branches in every upper complex component, rms |psi_c| = amp."""
    rng = np.random.default_rng(1000 + seed)
    wa = lat.branch_omega()
    wb = wa + lat.kappa
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    for c in range(4, lat.n):
        a = rng.normal(size=lat.N) + 1j * rng.normal(size=lat.N)
        b = rng.normal(size=lat.N) + 1j * rng.normal(size=lat.N)
        psi = np.fft.ifft(a + b)
        dps = np.fft.ifft(-1j * wa * a + 1j * wb * b)
        s = amp / np.sqrt(np.mean(np.abs(psi) ** 2))
        psi, dps = s * psi, s * dps
        u[:, 2 * c], u[:, 2 * c + 1] = psi.real, psi.imag
        v[:, 2 * c], v[:, 2 * c + 1] = dps.real, dps.imag
    return u, v


def run(job):
    n, amp_u, seed = job
    t0 = time.time()
    lat = M.Lattice(n=n, N=N, well="node")
    ref = M.Lattice(n=4, N=N, well="node")
    u, v = lat.packet(amp=AMP, n0=N // 2, width=8.0, per_mode=True)
    if n > 4 and amp_u > 0:
        uu, vv = incoherent_upper(lat, amp_u, seed)
        u[:, 8:], v[:, 8:] = uu[:, 8:], vv[:, 8:]
    lo, up = slice(0, 8), slice(8, lat.D)
    E0 = lat.energy(u, v)
    rec = dict(t=[], El=[], Eu=[], Eint=[], Ql=[], Qu=[], pur=[], chi=[], chi_t=[])
    urms0 = float(np.sqrt(np.mean(np.sum(u[:, 8:] ** 2, axis=1)))) if n > 4 else 0.0

    def sample(t, u, v, k):
        El = self_energy(lat, u, v, lo)
        Eu = self_energy(lat, u, v, up) if n > 4 else 0.0
        rec["t"].append(t); rec["El"].append(El); rec["Eu"].append(Eu)
        rec["Eint"].append(lat.energy(u, v) - El - Eu)
        rec["Ql"].append(charge(lat, u, v, lo)); rec["Qu"].append(charge(lat, u, v, up) if n > 4 else 0.0)
        chi, bar = chi_low(lat, u, v)
        rec["pur"].append(float(1 - np.linalg.norm(bar) / np.linalg.norm(chi)))
        if k % CHI_EVERY == 0:
            rec["chi"].append(chi.copy()); rec["chi_t"].append(t)

    sample(0.0, u, v, 0)
    t, k = 0.0, 0
    while t < T - 1e-9:
        u, v, _ = lat.run(u, v, DTS); t += DTS; k += 1
        sample(t, u, v, k)
    out = {key: np.array(val) for key, val in rec.items()}
    out.update(n=n, amp_u=amp_u, seed=seed, urms0=urms0,
               dE_total=abs(lat.energy(u, v) - E0) / abs(E0), secs=time.time() - t0)
    return out


def near_return(ts, dev):
    """tower_env_test.py's definition: after |dev| first exceeds half its max, the first time
    |dev| < 5% of its max."""
    m = np.abs(dev).max()
    if m == 0:
        return None
    k = int(np.argmax(np.abs(dev) > 0.5 * m))
    back = np.where(np.abs(dev[k:]) < 0.05 * m)[0]
    return float(ts[k + back[0]]) if len(back) else None


def overlap(a, b):
    return np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def summarise(r, ref):
    ts = r["t"]; El0 = r["El"][0]
    f = (r["El"] - El0) / El0                          # fractional change of the D<=8 self-energy
    q1, q4 = ts <= T / 4, ts >= 3 * T / 4
    nz = f[1:][np.abs(f[1:]) > 1e-12]
    sc = int(np.sum(np.diff(np.sign(nz)) != 0)) if len(nz) else 0
    slope = np.polyfit(ts, f, 1)[0]
    O = np.array([overlap(a, b) for a, b in zip(ref["chi"], r["chi"])])
    return dict(f=f, max_f=np.abs(f).max(), f_late=f[q4].mean(), sd_early=f[q1].std(), sd_late=f[q4].std(),
                sign_changes=sc, t_ret=near_return(ts, f), slope=slope,
                dEu=(r["Eu"][-1] - r["Eu"][0]) / El0, dEint=(r["Eint"][-1] - r["Eint"][0]) / El0,
                dQl=r["Ql"][-1] / r["Ql"][0] - 1,
                dQu=(r["Qu"][-1] / r["Qu"][0] - 1) if r["Qu"][0] else r["Qu"][-1],
                maxdQl=np.abs(r["Ql"] / r["Ql"][0] - 1).max(),
                maxdQu=np.abs(r["Qu"] / r["Qu"][0] - 1).max() if r["Qu"][0] else np.abs(r["Qu"]).max(),
                O_abs_T=abs(O[-1]), O_abs_min=np.abs(O).min(), O_ph_T=float(np.unwrap(np.angle(O))[-1]),
                pur0=r["pur"][0], purT=r["pur"][-1], purmin=r["pur"].min())


def fmt(x):
    return "None" if x is None else f"{x:.1f}"


def main():
    jobs = [(4, 0.0, 0), (8, 0.0, 0)]
    jobs += [(n, A_U, s) for n in (8, 16, 32) for s in SEEDS]
    jobs += [(n, A_U * np.sqrt(4.0 / (n - 4)), 0) for n in (16, 32)]
    jobs.sort(key=lambda j: -j[0])                     # longest first
    with Pool(4) as p:
        res = p.map(run, jobs)
    R = {(r["n"], round(r["amp_u"], 6), r["seed"]): r for r in res}
    ref = R[(4, 0.0, 0)]
    print(f"TOWER POPULATED (non-isolation) under A' -- N = {N}, T = {T:g}, packet 0.05, A_U = {A_U} per upper "
          f"complex component, seeds {SEEDS}")
    print(f"reference (D8, isolated): D<=8 self-energy drift {abs(ref['El'][-1] / ref['El'][0] - 1):.1e}, "
          f"charge drift {abs(ref['Ql'][-1] / ref['Ql'][0] - 1):.1e}, purity {ref['pur'][0]:.4f} -> {ref['pur'][-1]:.4f}")
    v = R[(8, 0.0, 0)]
    print(f"V  (D16, upper = 0): max |E_l - E_l,ref| / E_l = {np.abs(v['El'] - ref['El']).max() / ref['El'][0]:.1e}; "
          f"max |chi - chi_ref| = {max(np.abs(a - b).max() for a, b in zip(v['chi'], ref['chi'])):.1e}")
    print()
    rows = {}
    for label, keys in (("MAIN (A_U per component)", [(n, round(A_U, 6), s) for n in (8, 16, 32) for s in SEEDS]),
                        ("CONTROL F (fixed total upper norm = D16's)",
                         [(n, round(A_U * np.sqrt(4.0 / (n - 4)), 6), 0) for n in (16, 32)])):
        print(label)
        for key in keys:
            r = R[key]; s = summarise(r, ref); rows[key] = s
            print(f"  D{2 * r['n']:<3d} seed {r['seed']}: upper rms |u| per site {r['urms0']:.4f}; "
                  f"E_tot drift {r['dE_total']:.1e}; ({r['secs']:.0f} s)")
            print(f"      energy: max|dE_l|/E_l {s['max_f']:.2e}; late mean {s['f_late']:+.2e} "
                  f"(sd early {s['sd_early']:.1e}, late {s['sd_late']:.1e}); slope {s['slope']:+.1e}/t; "
                  f"sign changes {s['sign_changes']}; first near-return t = {fmt(s['t_ret'])}; "
                  f"at T: dE_u {s['dEu']:+.2e}, dE_int {s['dEint']:+.2e} (of E_l)")
            print(f"      charge: max drift D<=8 {s['maxdQl']:.1e}, upper {s['maxdQu']:.1e}")
            print(f"      coherence: |O_ref| at T {s['O_abs_T']:.4f} (min {s['O_abs_min']:.4f}), arg O_ref at T "
                  f"{s['O_ph_T']:+.3f} rad; purity {s['pur0']:.4f} -> {s['purT']:.4f} (min {s['purmin']:.4f})")
    print()
    print("ENSEMBLE (main runs, across seeds)")
    for n in (8, 16, 32):
        rr = [R[(n, round(A_U, 6), s)] for s in SEEDS]
        ph = np.array([rows[(n, round(A_U, 6), s)]["O_ph_T"] for s in SEEDS])
        pair = [abs(overlap(rr[i]["chi"][-1], rr[j]["chi"][-1])) for i in range(len(rr)) for j in range(i + 1, len(rr))]
        pair_min_t = min(min(abs(overlap(a, b)) for a, b in zip(rr[i]["chi"], rr[j]["chi"]))
                         for i in range(len(rr)) for j in range(i + 1, len(rr)))
        fl = np.array([rows[(n, round(A_U, 6), s)]["f_late"] for s in SEEDS])
        mx = np.array([rows[(n, round(A_U, 6), s)]["max_f"] for s in SEEDS])
        tr = [rows[(n, round(A_U, 6), s)]["t_ret"] for s in SEEDS]
        print(f"  D{2 * n:<3d}: inter-seed |O| at T mean {np.mean(pair):.4f} (min {np.min(pair):.4f}; min over t "
              f"{pair_min_t:.4f}); arg O_ref spread (sd) {ph.std():.3f} rad, mean {ph.mean():+.3f}; "
              f"late dE_l/E_l mean {fl.mean():+.2e} (sd {fl.std():.1e}); max|dE_l|/E_l mean {mx.mean():.2e}; "
              f"near-return t {[fmt(x) for x in tr]}")


if __name__ == "__main__":
    main()
