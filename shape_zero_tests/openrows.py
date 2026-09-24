import json, sys, numpy as np
import resid as R          # 1200-site geometry, clearing readout (THR 1e-6), cached clear times
T, M = R.T, R.M
R.MODE['readout'] = 'clear'
k0 = {'pi2': np.pi/2, '3pi4': 3*np.pi/4}[sys.argv[1]]
M.K0 = k0
for g in (0.0025, 0.005, 0.01, 0.02, 0.04):
    r = R.one(8, 12, g); r['k0'] = sys.argv[1]; r['seed'] = 1; print(json.dumps(r), flush=True)
for seed in (2, 3):
    for g in (0.0025, 0.01):
        r = R.one(8, 12, g, seed); r['k0'] = sys.argv[1]; r['seed'] = seed; print(json.dumps(r), flush=True)
