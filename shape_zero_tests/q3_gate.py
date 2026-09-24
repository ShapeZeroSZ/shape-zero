#!/usr/bin/env python3
"""
q3_gate.py — q = 3 ordering gate: u(2) and u(3), clearing readout,
spectrum-averaged prediction.

Runs on the working model.py, 04_scripts/session/model.py (found relative to
this file, so it runs from any directory). Eight evolutions (u(2) and u(3):
AB, BA, and the two Abelian-floor orders) on an L0 x S x S slab with a full
transverse gauge slab (MODEL_SPEC 4d), a width-3 isotropic packet at x0 = 30,
k0 = pi/2, and segments at 50 and 70. Each run is read out only when every
segment window [start - 10, start + len(RAMP) + 10) holds < 1e-6 of the packet
weight. The run refuses to report if the lattice is too short for that to be
trustworthy (see no_wrap_check).

Prediction (default 'averaged'): the packet's exact wavenumber spectrum on this
lattice; each component crosses the segments with its own omega(k) and
transverse term, and the readout is the power-weighted sum of the rotated states.
'carrier' uses model.U_segment at the single carrier wavenumber (the old
prediction) and is expected to FAIL.

PASS iff every per-order error < 1 deg, both split errors < 1 deg, and both
Abelian floors < 0.5 deg.

usage:
  python3 q3_gate.py [--L0 260] [--S 8] [--workers 4] [--predictor averaged|carrier]
  python3 q3_gate.py --from-saved q3_gate_runs_260x8.json --predictor carrier

Default 260 x 8 x 8 is the smallest lattice verified to clear and pass
(11.6 min wall on 4 workers). 240 fails to clear: the backward stray
re-enters the second window at t ~ 350.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys, json, time, argparse
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
# the WORKING model.py -- the single copy, in 04_scripts/session/
MODEL = os.path.join(HERE, "..", "04_scripts", "session")
sys.path.insert(0, MODEL)
import model as M

WIDTH, X0, SEGS = 3.0, 30, (50, 70)
THR, CHECK, TMAX = 1e-6, 1.0, 3000.0
G = {2: (0.12, 0.08), 3: (0.15, 0.15)}          # MODEL_SPEC 4d gate-7 strengths
GFLOOR = {2: (0.12, 0.08), 3: (0.15, 0.10)}     # floors need UNEQUAL strengths
TOL_ORDER, TOL_SPLIT, TOL_FLOOR = 1.0, 1.0, 0.5
JOBS = [(n, j) for n in (2, 3) for j in ("AB", "BA", "fAB", "fBA")]


# --------------------------------------------------------------- lattice
class Slab(M.Lattice):
    def __init__(self, n, L0, S):
        super().__init__(n=n, N=S ** 3, q=3, shape=S)
        self.shape = (L0, S, S); self.N = L0 * S * S

    def packet3(self, amp=1e-3):
        c = np.indices(self.shape).astype(float)
        r2 = np.zeros(self.shape)
        for a, ctr in zip(range(3), (X0, self.shape[1] / 2.0, self.shape[2] / 2.0)):
            d = c[a] - ctr
            d = (d + self.shape[a] / 2) % self.shape[a] - self.shape[a] / 2
            r2 += d ** 2
        env = np.exp(-0.5 * r2 / WIDTH ** 2).reshape(-1)
        ph = (M.K0 * (c[0] - X0)).reshape(-1)
        u = np.zeros((self.N, self.D)); v = np.zeros((self.N, self.D))
        u[:, 0], u[:, 1] = amp * env * np.cos(ph), amp * env * np.sin(ph)
        v[:, 0], v[:, 1] = self.omega * u[:, 1], -self.omega * u[:, 0]
        return u, v


def spec_for(n, job):
    gA, gB = G[n]; fA, fB = GFLOOR[n]; s0, s1 = SEGS
    return {"AB": [(s0, 0, gA), (s1, 1, gB)], "BA": [(s0, 1, gB), (s1, 0, gA)],
            "fAB": [(s0, 0, fA), (s1, 0, fB)], "fBA": [(s0, 0, fB), (s1, 0, fA)]}[job]


def windows(lat, u, v):
    w = (u * u + (v * v) / lat.omega ** 2).sum(axis=1).reshape(lat.shape).sum(axis=(1, 2))
    x = np.arange(lat.shape[0])
    return [float(w[(x >= s - 10) & (x < s + len(M.RAMP) + 10)].sum() / w.sum()) for s in SEGS]


def v_max(kappa):
    """Largest axis-0 group velocity of the chi branch over the whole band."""
    kx = np.linspace(1e-4, np.pi - 1e-4, 20001)
    om = 0.5 * (-kappa + np.sqrt(kappa ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(kx)))))
    return float((2 * M.C * np.sin(kx) / (2 * om + kappa)).max())


def no_wrap_check(L0, t, kappa):
    """Nothing may re-enter a segment window before readout. Both fronts are the
    packet centre plus 4 widths of leading tail, moving at v_max:
    forward front must not wrap round to the first window's near edge;
    backward stray must not wrap round to the last window's far edge.
    (An earlier version omitted the tail on the backward side and accepted a
    240-site lattice on which the backward stray re-entered the second window
    at t ~ 350, just before it could clear.)"""
    vt = v_max(kappa) * t
    first_win, last_win = SEGS[0] - 10, SEGS[1] + len(M.RAMP) + 10
    fwd_margin = L0 + first_win - (X0 + vt + 4 * WIDTH)
    bwd_margin = X0 + L0 - last_win - (vt + 4 * WIDTH)
    return fwd_margin > 0 and bwd_margin > 0, dict(v_max=v_max(kappa), travel=vt,
                                                   fwd_margin=fwd_margin, bwd_margin=bwd_margin)


# --------------------------------------------------------------- one run
def run_one(args):
    n, job, L0, S = args
    t0 = time.time()
    lat = Slab(n, L0, S)
    spec = spec_for(n, job)
    W, Wm = M.make_links(lat, spec)
    u, v = lat.packet3()
    E0 = lat.energy(u, v)
    t, seen = 0.0, False
    while t < TMAX:
        u, v, _ = lat.run(u, v, CHECK, W, Wm); t += CHECK
        wins = windows(lat, u, v)
        seen = seen or max(wins) > 1e-3
        if seen and max(wins) < THR:
            co, pur = lat.readout(u, v)
            return dict(n=n, job=job, t=t, windows=wins, co=co.tolist(), pur=pur,
                        Qt=M.transverse_Q(lat, u, v),
                        drift=abs(lat.energy(u, v) - E0) / abs(E0),
                        wall=time.time() - t0)
    return dict(n=n, job=job, t=t, windows=windows(lat, u, v), error="never cleared",
                wall=time.time() - t0)


# --------------------------------------------------------------- predictions
def spectrum(n, L0, S):
    lat = Slab(n, L0, S)
    u, v = lat.packet3()
    psi = (u[:, 0] + 1j * u[:, 1]).reshape(lat.shape)
    dps = (v[:, 0] + 1j * v[:, 1]).reshape(lat.shape)
    P = np.abs(np.fft.fftn(psi + (1j / lat.omega) * dps)) ** 2
    KX, KY, KZ = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m) for m in lat.shape], indexing="ij")
    return lat, P / P.sum(), KX, KY, KZ


def _kx_in_segment(kx0, om, Qt, geig, kappa):
    rhs = om * om + kappa * om - M.SQ5 - Qt
    f = lambda k: 2 * M.C * (1 - np.cos(k)) - 2 * M.C * geig * om * np.sin(k) - rhs
    lo = np.maximum(1e-6, kx0 - 0.5); hi = np.minimum(np.pi - 1e-6, kx0 + 0.5)
    bad = f(lo) * f(hi) > 0
    lo[bad], hi[bad] = 1e-6, np.pi - 1e-6
    flo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi); fm = f(mid)
        left = flo * fm <= 0
        hi = np.where(left, mid, hi); lo = np.where(left, lo, mid); flo = np.where(left, flo, fm)
    return 0.5 * (lo + hi)


def predict_averaged(spec_cache, n, segs, cut=1e-12):
    lat, P, KX, KY, KZ = spec_cache[n]
    mask = (P > cut * P.max()) & (KX > 0) & (KX < np.pi)
    p, kx, ky, kz = P[mask], KX[mask], KY[mask], KZ[mask]
    Qt = 2 * M.C * ((1 - np.cos(ky)) + (1 - np.cos(kz)))
    om = 0.5 * (-lat.kappa + np.sqrt(lat.kappa ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(kx)) + Qt)))
    state = np.zeros((len(p), n), complex); state[:, 0] = 1
    for axis, g in segs:
        eigs, vecs = np.linalg.eigh(lat.G[axis])
        ph = np.zeros((len(p), n))
        for wgt in M.RAMP:
            for j in range(n):
                ph[:, j] += _kx_in_segment(kx, om, Qt, g * wgt * eigs[j], lat.kappa)
        state = ((state @ vecs.conj()) * np.exp(1j * ph)) @ vecs.T
    rho = np.einsum("m,mi,mj->ij", p, state, state.conj())
    return np.array([np.real(np.trace(Sg @ rho)) / np.real(np.trace(rho)) for Sg in lat.G])


def predict_carrier(n, segs, Qt):
    lat = M.Lattice(n=n, N=200)
    psi0 = np.zeros(n, complex); psi0[0] = 1
    U = np.eye(n, dtype=complex)
    for axis, g in segs:
        U = M.U_segment(lat, axis, g, Qt) @ U
    return M.coords_of_state(lat, U @ psi0)


def evaluate(runs, predictor, L0, S):
    by = {(r["n"], r["job"]): r for r in runs}
    cache = {n: spectrum(n, L0, S) for n in (2, 3)} if predictor == "averaged" else None
    rows, ok = [], True
    for n in (2, 3):
        gA, gB = G[n]
        segs = {"AB": [(0, gA), (1, gB)], "BA": [(1, gB), (0, gA)]}
        meas, pred = {}, {}
        for j in ("AB", "BA"):
            r = by[(n, j)]
            meas[j] = np.array(r["co"])
            pred[j] = (predict_averaged(cache, n, segs[j]) if predictor == "averaged"
                       else predict_carrier(n, segs[j], r["Qt"]))
        e_AB, e_BA = M.angle(meas["AB"], pred["AB"]), M.angle(meas["BA"], pred["BA"])
        split_m, split_p = M.angle(meas["AB"], meas["BA"]), M.angle(pred["AB"], pred["BA"])
        floor = M.angle(np.array(by[(n, "fAB")]["co"]), np.array(by[(n, "fBA")]["co"]))
        checks = {f"u({n}) AB order": (e_AB, TOL_ORDER), f"u({n}) BA order": (e_BA, TOL_ORDER),
                  f"u({n}) split": (abs(split_m - split_p), TOL_SPLIT),
                  f"u({n}) Abelian floor": (floor, TOL_FLOOR)}
        for name, (val, tol) in checks.items():
            passed = bool(val < tol)
            ok &= passed
            rows.append(dict(check=name, value=val, tol=tol, passed=passed))
        rows.append(dict(check=f"u({n}) split measured / predicted", value=split_m,
                         predicted=split_p))
    return ok, rows


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L0", type=int, default=260)
    ap.add_argument("--S", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--predictor", choices=("averaged", "carrier"), default="averaged")
    ap.add_argument("--from-saved", default=None)
    a = ap.parse_args()

    t0 = time.time()
    if a.from_saved:
        saved = json.load(open(os.path.join(HERE, a.from_saved)))
        runs, L0, S, sim_wall = saved["runs"], saved["L0"], saved["S"], saved["sim_wall"]
    else:
        L0, S = a.L0, a.S
        with Pool(a.workers) as pool:
            runs = pool.map(run_one, [(n, j, L0, S) for n, j in JOBS])
        sim_wall = time.time() - t0
        json.dump(dict(L0=L0, S=S, sim_wall=sim_wall, workers=a.workers, runs=runs),
                  open(os.path.join(HERE, f"q3_gate_runs_{L0}x{S}.json"), "w"), indent=1)

    print(f"q = 3 ORDERING GATE   lattice {L0} x {S} x {S}   predictor: {a.predictor}")
    errs = [r for r in runs if "error" in r]
    if errs:
        for r in errs:
            print(f"  [ERROR] u({r['n']}) {r['job']}: {r['error']} "
                  f"(windows {r['windows']}, t = {r['t']:.0f})")
        print("  GATE: ERROR — readout not certified, no verdict")
        sys.exit(2)
    t_read = max(r["t"] for r in runs)
    wrap_ok, wrap = no_wrap_check(L0, t_read, M.Lattice(n=2).kappa)
    print(f"  clearing: all windows < {THR:.0e}; readout t = "
          f"{min(r['t'] for r in runs):.0f}-{t_read:.0f}; max drift "
          f"{max(r['drift'] for r in runs):.1e}")
    print(f"  no-wrap: v_max {wrap['v_max']:.3f}, travel {wrap['travel']:.0f} sites, "
          f"margins fwd {wrap['fwd_margin']:.0f} / bwd {wrap['bwd_margin']:.0f}")
    if not wrap_ok:
        print("  GATE: ERROR — lattice too short: a wave could re-enter a segment window "
              "before readout, no verdict")
        sys.exit(2)

    ok, rows = evaluate(runs, a.predictor, L0, S)
    for r in rows:
        if "tol" in r:
            print(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['check']:<22} "
                  f"{r['value']:8.3f} deg   (need < {r['tol']:.1f})")
        else:
            print(f"         {r['check']:<34} {r['value']:.2f} / {r['predicted']:.2f} deg")
    print(f"  GATE: {'PASS' if ok else 'FAIL'}")
    print(f"  runtime: simulation {sim_wall / 60:.1f} min wall "
          f"({sum(r['wall'] for r in runs) / 60:.1f} min CPU over {len(runs)} runs); "
          f"evaluation {time.time() - t0 - (0 if a.from_saved else sim_wall):.0f} s")
    out = dict(L0=L0, S=S, predictor=a.predictor, passed=ok, rows=rows,
               t_read=t_read, no_wrap=wrap, sim_wall=sim_wall)
    json.dump(out, open(os.path.join(HERE, f"q3_gate_result_{L0}x{S}_{a.predictor}.json"), "w"),
              indent=1, default=float)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
