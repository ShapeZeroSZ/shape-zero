#!/usr/bin/env python3
"""irrev_ratio_check.py -- POST-HOC (written after irrev_pilot_output.txt): why the energy-weighted sink ratio b/a came out
1.51, not the 1.885 predicted "by construction" in IRREV_HYPOTHESES.md (1c9e5bf).
Hypothesis (post-hoc): the lump is optically thick at Gamma = 10 -- a tower quantum crossing it is absorbed almost fully
(integrated rate ~ Gamma e_low,peak x width / v ~ 3) -- so absorption is limited by the tower's incoming supply, not by the
lump's energy. Test: sweep Gamma (deterministic limit, seeds 0-1, window [0, 50]); the ratio should rise toward
E_b/E_a = 1.885 as Gamma -> 0 (optically thin), and absorbed/Gamma should be ~constant at small Gamma.
Also: the early-time ratio (window [0, 5]) at Gamma = 10, before the local tower is depleted."""
from multiprocessing import Pool
import numpy as np
import irrev_pilot as I

GAMMAS = (0.1, 0.3, 1.0, 3.0, 10.0)


def job(g, lump, s):
    return dict(C=1000.0, K_th=1.0, Gamma=g, T0=1e-5, tag=f"{g}_{lump}{s}", seed=s, lump=lump, T=50.0, dts=0.5, noise=False)


if __name__ == "__main__":
    jobs = [job(g, l, s) for g in GAMMAS for l in ("a", "b") for s in (0, 1)]
    with Pool(4) as p:
        R = {r["job"]["tag"]: r for r in p.map(I.run, jobs, chunksize=1)}
    print("POST-HOC: energy-weighted sink ratio b/a against Gamma (deterministic G, seeds 0-1)")
    print("  Gamma   absorbed a [0,50]   absorbed b   ratio [0,50]   ratio [0,5]   a absorbed / Gamma")
    for g in GAMMAS:
        a = [R[f"{g}_a{s}"]["Eh"] - R[f"{g}_a{s}"]["Eh"][0] for s in (0, 1)]
        b = [R[f"{g}_b{s}"]["Eh"] - R[f"{g}_b{s}"]["Eh"][0] for s in (0, 1)]
        A, B = np.mean([x[-1] for x in a]), np.mean([x[-1] for x in b])
        A5, B5 = np.mean([x[10] for x in a]), np.mean([x[10] for x in b])
        print(f"  {g:5.1f}   {A:.4e}          {B:.4e}   {B/A:.3f}          {B5/A5:.3f}         {A/g:.4e}")
    print("  E_b / E_a = 1.8854 (optically thin limit)")
