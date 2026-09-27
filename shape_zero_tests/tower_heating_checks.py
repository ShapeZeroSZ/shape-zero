#!/usr/bin/env python3
"""tower_heating_checks.py -- POST HOC checks on tower_heating_test.py, written after its output was seen
(predictions 4b82759). Not part of the pre-registered test.
 1. U2's modulation depth. U2 gave the golden-angle phases 2.39996 c to its four components; the beat
    terms of s = sum_c |alpha_c + beta_c|^2 then carry phases 2 theta_c and nearly cancel. Print the
    depth (amplitude of the s oscillation / mean s) as built, and rerun with ALIGNED phases (theta_c = 0,
    full modulation: s oscillates between 0 and 2 <s>).
 2. Criterion (iv) needs min f over [2000, 20000]; the test printed min f over the whole run only.
    Rerun the two D16 long runs and print it (D64's whole-run min was 0, so it needs no rerun).
usage: python3 tower_heating_checks.py"""
from multiprocessing import Pool

import numpy as np

import tower_heating_test as TH


def aligned(lat, u, v, comps, amp):
    wa0 = lat.branch_omega()[0]; wb0 = wa0 + lat.kappa
    for c in comps:
        al = amp / np.sqrt(2) * np.ones(lat.N) + 0j; be = al.copy()
        TH.set_comp(u, v, c, al + be, -1j * wa0 * al + 1j * wb0 * be)


def job(j):
    if j == "U2aligned":
        TH.uniform_modulated = aligned
        r = TH.run(("U2 aligned", 8, "umod", TH.A0, 0, 2000.0, 2.0))
        ref = TH.run(("ref", 4, "none", 0, 0, 2000.0, 2.0))
        s = TH.summary(r, ref)
        ms = r["ms"]
        return (f"U2 aligned phases: s(t) at site-mean ranges {ms.min():.2e} - {ms.max():.2e} (mean {ms.mean():.2e}); "
                f"<var s> {s['vs']:.2e}; gain rate {s['g']:+.3e} (+- {s['ge']:.1e}); f(T) {s['fT']:+.3e}; phase at T "
                f"{s['ph']:+.3f} rad; dE_a {s['dEa']:+.2e}, dE_b {s['dEb']:+.2e}; E_tot {r['dE_total']:.1e}")
    seed = j
    r = TH.run(("L D16", 8, "incoh", TH.A0, seed, 20000.0, 10.0))
    f = (r["El"] - r["El"][0]) / r["El"][0]; m = r["t"] >= 2000
    return (f"L D16 seed {seed}: min f over [2000, 20000] {f[m].min():+.3e} at t = {r['t'][m][np.argmin(f[m])]:.0f}; "
            f"f(2000) {f[np.argmin(np.abs(r['t'] - 2000))]:+.3e}; f(20000) {f[-1]:+.4f}")


if __name__ == "__main__":
    th = np.array([2.39996 * c for c in range(4, 8)])
    depth = abs(np.exp(2j * th).sum()) / 4
    print(f"U2 as built: modulation depth of s (beat amplitude / mean) = {depth:.3f} (1.0 = full)", flush=True)
    with Pool(3) as p:
        for line in p.map(job, ["U2aligned", 0, 1], chunksize=1):
            print(line, flush=True)
