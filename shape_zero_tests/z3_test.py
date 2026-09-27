#!/usr/bin/env python3
"""z3_test.py -- tests of the Z3 hypothesis (predictions: z3_predictions.txt, committed first,
43874d0). Z3a = e^{(2pi/3) JJ}; Z3b = centre of the SU(3) fixing e (identity on span{1, e},
e^{2 pi i/3} on C^3). Octonions: the D8 rung table (z1_d8_flow.oct_table), e = e_1, JJ = L_e
(the sources' J, MODEL_SPEC sec 2-3); the lattice's own JJ = rho(i I) is also used where noted.
usage: python3 z3_test.py   (~3 min)"""
import os
# recorded under node form A (per-dimer radial well); pinned so the outputs reproduce after
# the default became A' (whole-node well) on 2026-09-27.
os.environ.setdefault("SZ_J_WELL", "radial")

import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.join(HERE, "..", "04_scripts", "session")
RUNGS = os.path.join(HERE, "..", "04_scripts", "rungs")
sys.path.insert(0, SESSION)
import model as M


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m


fl = load("fl", os.path.join(RUNGS, "z1_d8_flow.py"))
v2 = load("v2", os.path.join(SESSION, "d16_spectrum_v2.py"))
E = fl.oct_table(fl.ch.oriented_lines())
I8 = np.eye(8)
Lm = lambda T, x: np.einsum("ijk,i->kj", T, x)
Rm = lambda T, x: np.einsum("ijk,j->ki", T, x)
mul = lambda x, y: np.einsum("ijk,i,j->k", E, x, y)
nrm = np.linalg.norm
rng = np.random.default_rng(33)
TH = 2 * np.pi / 3


def unit_imag(d=8):
    v = np.zeros(d); v[1:8] = rng.normal(size=7); return v / nrm(v)


def expJ(J, th):
    return np.cos(th) * np.eye(len(J)) + np.sin(th) * J          # valid since J^2 = -1


e = I8[1]
JL = Lm(E, e)                                     # the sources' J on O
P1 = np.outer(I8[0], I8[0]) + np.outer(e, e)      # span{1, e}
P2 = np.eye(8) - P1                               # C^3
Z3a = expJ(JL, TH)
Z3b = P1 + np.cos(TH) * P2 + np.sin(TH) * JL @ P2


def main():
    print("checks: JL^2 = -1:", f"{nrm(JL @ JL + I8):.1e};", "JL preserves C^3:", f"{nrm(P1 @ JL @ P2):.1e};",
          "Z3b automorphism |P(xy) - P(x)P(y)|:",
          f"{max(nrm(Z3b @ mul(x, y) - mul(Z3b @ x, Z3b @ y)) for x, y in rng.normal(size=(20, 2, 8))):.1e};",
          "Z3a automorphism:", f"{max(nrm(Z3a @ mul(x, y) - mul(Z3a @ x, Z3a @ y)) for x, y in rng.normal(size=(20, 2, 8))):.2f}")

    print("\n(1) Z1  residual L_g vs Z3")
    J16 = M.rho(1j * np.eye(8)); E16 = v2.cd(4)
    rg = np.random.default_rng(5); gv = np.zeros(16); gv[1:8] = rg.normal(size=7); gv /= nrm(gv)
    L16 = Lm(E16, gv)
    for lab, J, Lg in (("lattice n=8 (sedenion, present identification)", J16, L16),
                       ("octonion, JJ = L_e, generic g", JL, Lm(E, unit_imag()))):
        R = expJ(J, TH)
        print(f"    {lab}: ||[Z3a, L_g]|| / ||[JJ, L_g]|| = {nrm(R @ Lg - Lg @ R) / nrm(J @ Lg - Lg @ J):.4f}"
              f"  (sin 2pi/3 = {np.sin(TH):.4f}); ||[Z3a, L_g]||/||L_g|| = {nrm(R @ Lg - Lg @ R) / nrm(Lg):.3f}")
    for lab, g in (("generic g", unit_imag()), ("g = e", e)):
        Lg = Lm(E, g)
        print(f"    Z3b vs L_g, {lab}: ||[Z3b, L_g]||/||L_g|| = {nrm(Z3b @ Lg - Lg @ Z3b) / nrm(Lg):.2e}")

    print("\n    Z2  charge content of L_g: Fourier modes of e^{-th JJ} L_g e^{th JJ} (lattice n=8)")
    ths = 2 * np.pi * np.arange(24) / 24
    F = np.array([expJ(J16, -t) @ L16 @ expJ(J16, t) for t in ths])
    Fh = np.fft.fft(F, axis=0) / len(ths)
    print("      charge q:  " + "  ".join(f"{q:+d}: {nrm(Fh[q % 24]) / nrm(L16):.3f}" for q in range(-4, 5)))

    print("\n    Z3  gate-11 runs, L_g vs its anticommuting part A vs commuting part C (full Noether charge)")
    C16 = 0.5 * (L16 - J16 @ L16 @ J16); A16 = 0.5 * (L16 + J16 @ L16 @ J16)
    e0 = np.zeros(16); e0[0] = 1

    def run(kap, Cr, X):
        Et = np.zeros((16, 16, 16)); Et[0] = X.T
        lr = M.Lattice(n=8, N=512, q=1, kappa=kap, C_r=Cr, tower=(Et, e0))
        u, v = lr.packet(amp=1e-3); u[:, 8:] += 0.3 * u[:, :8]
        G = kap * J16 + Cr * X
        Q = lambda u, v: float(np.einsum("na,ab,nb->", v, J16, u) + 0.5 * np.einsum("na,ab,nb->", u, G @ J16, u))
        Q0 = Q(u, v); u, v, _ = lr.run(u, v, T=20.0)
        return Q(u, v) / Q0 - 1
    for kap in (0.0, M.KAPPA):
        for Cr in (0.05, 0.20):
            print(f"      kappa {kap:.3f} C_r {Cr:.2f}: dQ/Q  L_g {run(kap, Cr, L16):+.2e}   A {run(kap, Cr, A16):+.2e}"
                  f"   C {run(kap, Cr, C16):+.2e}", flush=True)

    print("\n(2) Z4/Z5  gauge class: symmetric 8x8 couplings commuting with ...")
    basis = []
    for i in range(8):
        for j in range(i, 8):
            S = np.zeros((8, 8)); S[i, j] = S[j, i] = 1; basis.append(S)

    def dim(ops):
        A = np.array([np.concatenate([(Z @ S - S @ Z).ravel() for Z in ops]) for S in basis])
        sv = np.linalg.svd(A, compute_uv=False)
        return len(basis) - int(np.sum(sv > 1e-9 * sv[0]))
    Jlat = M.rho(1j * np.eye(4))
    print(f"      nothing (passivity alone)          : {dim([np.zeros((8, 8))])}")
    print(f"      JJ = L_e (U(1))                    : {dim([JL])}")
    print(f"      Z3a (JJ = L_e)                     : {dim([Z3a])}")
    print(f"      lattice JJ, and its Z3a            : {dim([Jlat])}, {dim([expJ(Jlat, TH)])}")
    print(f"      JJ = L_e and the split C + C^3     : {dim([JL, P1])}")
    print(f"      Z3a and the split                  : {dim([Z3a, P1])}")
    print(f"      Z3b                                : {dim([Z3b])}")
    print(f"      Z3b restricted: JJ-breaking dims   : {dim([Z3b]) - dim([JL, P1])} beyond u(1)+u(3)")

    print("\n(3) Z6/Z7  D8 generator M = R_a + L_b")
    for lab, a, b in (("generic a, b", unit_imag(), unit_imag()), ("a = e, generic b", e, unit_imag()),
                      ("a = e, b = e", e, e)):
        Mx = Rm(E, a) + Lm(E, b)
        print(f"      {lab:18s}: ||[Z3a, M]||/||M|| = {nrm(Z3a @ Mx - Mx @ Z3a) / nrm(Mx):.2e}   "
              f"||[Z3b, M]||/||M|| = {nrm(Z3b @ Mx - Mx @ Z3b) / nrm(Mx):.2e}")

    print("\n    Z8  charge-3 terms")
    c = E[1:, 1:, 1:]
    u = rng.normal(size=7)
    print(f"      c(u, u, u) = {np.einsum('abc,a,b,c->', c, u, u, u):.1e}")
    x, y, z = rng.normal(size=(3, 8))
    cub = lambda x, y, z: np.einsum('abc,a,b,c->', c, x[1:], y[1:], z[1:])
    for lab, Rz in (("Z3b", Z3b), ("U(1) on C^3, theta = 0.3", P1 + np.cos(.3) * P2 + np.sin(.3) * JL @ P2)):
        print(f"      c(u_n, u_n+1, u_n+2) under {lab}: {cub(x, y, z):+.4f} -> {cub(Rz @ x, Rz @ y, Rz @ z):+.4f}")


if __name__ == "__main__":
    main()
