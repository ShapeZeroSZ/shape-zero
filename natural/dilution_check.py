#!/usr/bin/env python3
"""dilution_check.py -- D1b (DILUTION_HYPOTHESES.md, 45c30b4): the tower's effect on D <= 8 matter through A''s shared
radius grows with depth. <|u_tower|> (= the stiffness shift dK sensed by a weak D <= 8 probe) for main's incoherent tower,
per-level rms A_U = 0.005, M = n - 4 populated components, ring N = 512, seeds 0-3; predicted 0.00969 sqrt(M/4)
(sqrt<s> = A_U sqrt(M) times <|psi|>/sqrt<s>, which tends to 1 as M grows)."""
import math
import numpy as np
import cpg_pilot2 as P

M_ = P.M
print("D1b  dK = <|u_tower|> against depth (A_U = 0.005 per level)")
for n in (8, 16, 32, 64, 128):
    lat = M_.Lattice(n=n, N=512, well="node")
    r = [np.linalg.norm(P.incoherent_upper(lat, 0.005, s)[0], axis=1).mean() for s in range(4)]
    Mc = n - 4
    print(f"    n = {n:4d} (M = {Mc:4d} levels): dK = {np.mean(r):.5f}   predicted ~ {0.00969*math.sqrt(Mc/4):.5f}"
          f"   ratio to sqrt(M) A_U: {np.mean(r)/(0.005*math.sqrt(Mc)):.4f}")
