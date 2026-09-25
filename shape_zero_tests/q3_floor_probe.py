#!/usr/bin/env python3
"""
q3_floor_probe.py -- why does q3_gate.py's Abelian floor read 0.066 deg (u(2)) and
0.030 deg (u(3)) at kappa = kappa* when it read 0.000 deg at kappa = 0.5?

Commuting segments must give no split (algebra). Hypotheses, stated before any run:

  H1 incomplete clearing (MODEL_SPEC §4d.1 trap 6). Windows at readout were
     1.5e-7 to 1.0e-6 in every floor run at both kappa. Predicted NOT the cause:
     reading later, with emptier windows, will not remove the floor.
  H2 chirality impurity. The launch (q3_gate.Slab.packet3) gives every Fourier mode
     the carrier frequency, so off-carrier modes carry opposite-chirality weight
     (purity ~0.95) and the carrier-omega readout mixes the two. Alone it cannot
     break commutation; it acts through H3.
  H3 readout-time mismatch (leading). The mixing makes the whole-lattice readout
     oscillate in time (~2 omega). fAB and fBA were read at their own clearing times:
     equal at kappa = 0.5 (346/346, 341/341), unequal at kappa* (357/363, 359/361).
     Predicted: floor ~0 at equal times; co(t) of one run fluctuates ~0.05 deg; a
     kappa = 0.5 pair read at unequal times shows a floor too; a pure launch (each
     mode at its own omega(k)) gives a constant readout and no floor at any times.
  H4 order-dependent reflection. Unequal strengths reflect differently by order.
     Test: floor from the forward region only (x >= last window end) vs whole
     lattice, at equal times.

Runs (q3_gate geometry, 260 x 8 x 8, GFLOOR strengths, segments 50/70):
  carrier launch at kappa*: u(2) fAB, fBA; u(3) fAB, fBA
  pure launch    at kappa*: the same four
  carrier launch at kappa = 0.5: u(2) fAB, fBA (control for H3)
Each run is recorded at every t = 330, 331, ..., 372: both window weights, the
whole-lattice Bloch vector and purity (model.Lattice.readout), the forward-region
Bloch vector, and the weight fraction behind the first window (x < 40).

usage:  python3 q3_floor_probe.py run      -> q3_floor_probe_runs.json
        python3 q3_floor_probe.py report   (reads the json)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import q3_gate as Q

M = Q.M
L0, S = 260, 8
T_REC = list(range(330, 373))
FWD0 = Q.SEGS[1] + len(M.RAMP) + 10          # forward region starts past the last window
BACK1 = Q.SEGS[0] - 10                       # behind the first window
OUT = os.path.join(Q.HERE, "q3_floor_probe_runs.json")


class Box(Q.Slab):
    def __init__(self, n, kappa):
        super().__init__(n, L0, S)
        self.kappa = kappa
        self.omega = 0.5 * (-kappa + np.sqrt(kappa ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(M.K0)))))

    def packet_carrier(self, amp=1e-3):
        """q3_gate's launch as it was when this probe ran: every mode at the carrier
        omega (q3_gate.packet3 has since become the per-mode launch)."""
        u, v = self.packet3(amp)
        v[:, 0], v[:, 1] = self.omega * u[:, 1], -self.omega * u[:, 0]
        return u, v

    def packet_pure(self, amp=1e-3):
        """packet3's field, but each Fourier mode moving at its own chi-branch omega(k)."""
        u, v = self.packet3(amp)
        psi = (u[:, 0] + 1j * u[:, 1]).reshape(self.shape)
        k = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m) for m in self.shape], indexing="ij")
        Qk = M.SQ5 + 2 * M.C * sum(1 - np.cos(ka) for ka in k)
        om = 0.5 * (-self.kappa + np.sqrt(self.kappa ** 2 + 4 * Qk))
        dpsi = np.fft.ifftn(-1j * om * np.fft.fftn(psi)).reshape(-1)
        v = np.zeros_like(v)
        v[:, 0], v[:, 1] = dpsi.real, dpsi.imag
        return u, v


def readout_mask(lat, u, v, mask):
    psi = u[mask][:, 0::2] + 1j * u[mask][:, 1::2]
    dps = v[mask][:, 0::2] + 1j * v[mask][:, 1::2]
    chi = psi + (1j / lat.omega) * dps
    rs = chi.T @ chi.conj()
    tr = np.real(np.trace(rs)) + 1e-30
    return [float(np.real(np.trace(G @ rs)) / tr) for G in lat.G]


def one(job):
    n, which, launch, kappa = job
    t0 = time.time()
    lat = Box(n, kappa)
    W, Wm = M.make_links(lat, Q.spec_for(n, which))
    u, v = lat.packet_carrier() if launch == "carrier" else lat.packet_pure()
    x = np.repeat(np.arange(L0), S * S)
    fwd, back = x >= FWD0, x < BACK1
    rec, t = [], 0.0
    for tt in T_REC:
        u, v, _ = lat.run(u, v, tt - t, W, Wm)
        t = tt
        co, pur = lat.readout(u, v)
        w = (u * u + (v * v) / lat.omega ** 2).sum(axis=1)
        rec.append(dict(t=tt, windows=Q.windows(lat, u, v), co=co.tolist(), pur=pur,
                        co_fwd=readout_mask(lat, u, v, fwd),
                        w_back=float(w[back].sum() / w.sum())))
    return dict(n=n, job=which, launch=launch, kappa=kappa, rec=rec, wall=time.time() - t0)


def run():
    ks = float(M.KAPPA)
    jobs = ([(n, j, "carrier", ks) for n in (2, 3) for j in ("fAB", "fBA")]
            + [(n, j, "pure", ks) for n in (2, 3) for j in ("fAB", "fBA")]
            + [(2, j, "carrier", 0.5) for j in ("fAB", "fBA")])
    with Pool(4) as p:
        res = p.map(one, jobs)
    json.dump(res, open(OUT, "w"), indent=1)


def ang(a, b):
    return float(M.angle(np.array(a), np.array(b)))


def report():
    res = json.load(open(OUT))
    by = {(r["n"], r["job"], r["launch"], round(r["kappa"], 4)): r for r in res}
    ks = round(float(M.KAPPA), 4)
    print("=" * 92)
    print("q = 3 ABELIAN FLOOR PROBE -- q3_gate geometry 260 x 8 x 8")
    print("=" * 92)
    for (n, launch, kap) in ((2, "carrier", ks), (3, "carrier", ks), (2, "pure", ks),
                             (3, "pure", ks), (2, "carrier", 0.5)):
        a, b = by[(n, "fAB", launch, kap)]["rec"], by[(n, "fBA", launch, kap)]["rec"]
        ta = {r["t"]: r for r in a}
        tb = {r["t"]: r for r in b}
        print(f"\n  u({n})  launch {launch}  kappa {kap}")
        eq = [ang(ta[t]["co"], tb[t]["co"]) for t in T_REC]
        eqf = [ang(ta[t]["co_fwd"], tb[t]["co_fwd"]) for t in T_REC]
        # within-run fluctuation: angle from each run's own time-mean state
        for name, rr in (("fAB", a), ("fBA", b)):
            m = np.mean([r["co"] for r in rr], axis=0)
            fl = [ang(r["co"], m) for r in rr]
            pu = [r["pur"] for r in rr]
            print(f"     {name}: co(t) spread about its mean: max {max(fl):.4f} deg, rms "
                  f"{np.sqrt(np.mean(np.square(fl))):.4f}; purity {min(pu):.5f}-{max(pu):.5f}")
        print(f"     floor at EQUAL times, whole lattice: max {max(eq):.5f}  median {np.median(eq):.5f} deg")
        print(f"     floor at EQUAL times, forward region: max {max(eqf):.5f}  median {np.median(eqf):.5f} deg")
        un = [ang(ta[t1]["co"], tb[t2]["co"]) for t1 in T_REC for t2 in T_REC if t1 != t2]
        print(f"     floor at UNEQUAL times (all pairs): median {np.median(un):.4f}  max {max(un):.4f} deg")
        print("        t   windows fAB            windows fBA            w_back fAB/fBA       floor(t,t)  fwd-only")
        for t in (340, 345, 350, 355, 357, 360, 363, 366, 370):
            ra, rb = ta[t], tb[t]
            print(f"      {t:4d}   {ra['windows'][0]:.1e} {ra['windows'][1]:.1e}   "
                  f"{rb['windows'][0]:.1e} {rb['windows'][1]:.1e}   {ra['w_back']:.1e} {rb['w_back']:.1e}   "
                  f"{ang(ra['co'], rb['co']):.5f}    {ang(ra['co_fwd'], rb['co_fwd']):.5f}")
        if launch == "carrier" and kap == ks:
            t1, t2 = {2: (357, 363), 3: (359, 361)}[n]
            print(f"     q3_gate's readout pair (fAB t = {t1}, fBA t = {t2}): "
                  f"{ang(ta[t1]['co'], tb[t2]['co']):.4f} deg (gate reported "
                  f"{ {2: 0.066, 3: 0.030}[n] })")
        if kap == 0.5:
            print(f"     kappa = 0.5 at the gate's equal-time pair (346, 346): "
                  f"{ang(ta[346]['co'], tb[346]['co']):.5f}; at (346, 352): {ang(ta[346]['co'], tb[352]['co']):.4f} deg")


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
