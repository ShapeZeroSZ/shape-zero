#!/usr/bin/env python3
"""anomA_smooth.py -- anomaly A, the SMOOTH-FORCE DIAGNOSTIC (not a change of premise): inside this process only, the
node-form on-site nonlinearity |psi| psi is replaced by |psi|^2 psi (energy |psi|^4 / 4). Runs main's own gate-7 protocol
(model.ordering_test, clearing readout; single-carrier linear prediction as amp_scaling.py) at A = 0.04, 0.02, 0.01,
0.005, and main's q3_gate.py at A = 0.04, 0.02, 0.01. Predictions: ANOMALY_A_PREDICTIONS.md (534805c).
usage: python3 anomA_smooth.py q1 | q3"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool
WT = os.environ.get("MAINWT", "/tmp/claude-0/-home-user-shape-zero/7f4f19ab-d202-57e0-98c4-ef10d02be803/scratchpad/mainwt")
sys.path.insert(0, os.path.join(WT, "04_scripts", "session")); sys.path.insert(0, os.path.join(WT, "shape_zero_tests"))
import model as M

_nl, _cub = M.Lattice._onsite_nl, M.Lattice._onsite_cubic
def nl(self, u):
    if self.well == "node":
        return (u * u).sum(axis=1, keepdims=True) * u
    return _nl(self, u)
def cub(self, u):
    if self.well == "node":
        return (((u * u).sum(axis=1)) ** 2).sum() / 4
    return _cub(self, u)
M.Lattice._onsite_nl, M.Lattice._onsite_cubic = nl, cub
HERE = os.path.dirname(os.path.abspath(__file__))
AMPS1 = (0.04, 0.02, 0.01, 0.005)
AMPS3 = (0.04, 0.02, 0.01)


def job(args):
    import amp_scaling as S
    return S.job(args)


def q1():
    jobs = [(n, gA, gB, axes, amp) for amp in AMPS1 for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10))
            for axes in ((0, 1), (0, 0))]
    with Pool(4) as p:
        res = p.map(job, jobs)
    json.dump(res, open(os.path.join(HERE, "smooth_q1.json"), "w"))


def q3():
    import q3_gate as Q
    for amp, tag in zip(AMPS3, ("smooth04", "smooth02", "smooth01")):
        sys.argv = ["q3_gate.py", "--amp", repr(amp), "--tag", tag]
        Q.main()


if __name__ == "__main__":
    {"q1": q1, "q3": q3}[sys.argv[1]]()
