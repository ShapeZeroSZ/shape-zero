#!/usr/bin/env python3
"""circuit_error_budget.py -- error budget for the LC-gyrator ring of LAB_NOTE.md, second platform (branch
realisation-gyroscopic). Expectations committed first: circuit_error_budget_predictions.txt.

Full linear circuit, states [V_n, i_g,n, i_c,n, x_a,n, x_b,n]:
  KCL  (C_n + 2 C_in) V_n' + C_p,n (V_n' - V_{n+1}') + C_p,n-1 (V_n' - V_{n-1}')
         = -i_g,n - i_c,n + i_c,n-1 - g_s,n V_n - x_a,n + x_b,n-1
  L_g,n i_g,n' = V_n - R_g,n i_g,n ;  L_c,n i_c,n' = V_n - V_{n+1} - R_c,n i_c,n
  tau x_a,n' = -x_a,n + G_a,n V_{n+1} ;  tau x_b,n' = -x_b,n + G_b,n V_n     (tau = 0: algebraic)
Modes: eigenvalues lambda = -i w + ..., the +-k modes by largest overlap of V with e^{+-ikn}. The protocol, P1, P2
(the +-S even part at k = pi/2, bump on 1/L_g) and P3 exactly as platform_error_budget.py.
usage: python3 circuit_error_budget.py
"""
import numpy as np

C0, LG0, LC0 = 10e-9, 1e-3, 4e-3
G0 = 0.1 * np.sqrt(C0 / LG0)                    # beta-hat = 0.10
K0, CC0, B0 = 1 / (LG0 * C0), 1 / (LC0 * C0), G0 / C0
W2 = np.sqrt(K0 + 2 * CC0)                     # w(pi/2) without gyrators
S = 0.05


def eta_shape(N, kind, p=None):
    x = np.arange(N, dtype=float)
    if kind == "gauss":
        e = np.exp(-0.5 * ((x - N / 2) / p) ** 2)
    elif kind == "sech2":
        e = 1.0 / np.cosh((x - N / 2) / p) ** 2
    elif kind == "two":
        e = np.exp(-0.5 * ((x - N / 4) / p) ** 2) + np.exp(-0.5 * ((x - 3 * N / 4) / p) ** 2)
    elif kind == "ramp":
        e = x.copy()
    e = e - e.mean()
    return e / np.sqrt((e ** 2).mean())


def ideal(N):
    o = np.ones(N)
    return dict(C=C0 * o, Lg=LG0 * o, Lc=LC0 * o, Ga=G0 * o, Gb=G0 * o, gs=0 * o, Rg=0 * o, Rc=0 * o,
                Cp=0 * o, Cin=0.0, tau=0.0)


def matrix(p, bump=None, gyro=True):
    N = len(p["C"])
    invLg = 1 / p["Lg"] * (1 + (bump if bump is not None else 0))
    Ga, Gb = (p["Ga"], p["Gb"]) if gyro else (0 * p["Ga"], 0 * p["Gb"])
    tau = p["tau"]
    nst = 5 * N if tau > 0 else 3 * N
    iV, ig, ic = 0, N, 2 * N
    ia, ib = 3 * N, 4 * N
    Mm = np.diag(p["C"] + 2 * p["Cin"] + p["Cp"] + np.roll(p["Cp"], 1))
    R = np.zeros((N, nst))
    A = np.zeros((nst, nst))
    for n in range(N):
        m, l = (n + 1) % N, (n - 1) % N
        Mm[n, m] -= p["Cp"][n]; Mm[n, l] -= p["Cp"][l]
        R[n, ig + n] -= 1; R[n, ic + n] -= 1; R[n, ic + l] += 1
        R[n, iV + n] -= p["gs"][n]
        if tau > 0:
            R[n, ia + n] -= 1; R[n, ib + l] += 1
        else:
            R[n, iV + m] -= Ga[n]; R[n, iV + l] += Gb[l]
        A[ig + n, iV + n] = invLg[n]; A[ig + n, ig + n] = -p["Rg"][n] * invLg[n]
        A[ic + n, iV + n] = 1 / p["Lc"][n]; A[ic + n, iV + m] = -1 / p["Lc"][n]
        A[ic + n, ic + n] = -p["Rc"][n] / p["Lc"][n]
        if tau > 0:
            A[ia + n, ia + n] = -1 / tau; A[ia + n, iV + m] = Ga[n] / tau
            A[ib + n, ib + n] = -1 / tau; A[ib + n, iV + n] = Gb[n] / tau
    A[:N] = np.linalg.solve(Mm, R)
    return A, Mm


def modes(p, bump=None, gyro=True):
    A, _ = matrix(p, bump, gyro)
    lam, V = np.linalg.eig(A)
    return lam, V[:len(p["C"])]


def mode_w(lam, V, m, sign):
    N = V.shape[0]
    pw = np.exp(1j * sign * 2 * np.pi * m / N * np.arange(N)) / np.sqrt(N)
    sel = lam.imag < 0
    ov = np.abs(pw.conj() @ V[:, sel]) / np.linalg.norm(V[:, sel], axis=0)
    j = np.argmax(ov)
    return -lam[sel][j].imag


def dw(p, m, bump=None):
    lam, V = modes(p, bump)
    return mode_w(lam, V, m, 1) - mode_w(lam, V, m, -1)


def calibrate(p):
    N = len(p["C"])
    lam, V = modes(p, gyro=False)
    ms = np.arange(0, N // 2 + 1)
    w = np.array([0.5 * (mode_w(lam, V, m, 1) + mode_w(lam, V, m, -1)) for m in ms])
    X = np.vstack([np.ones_like(w), 2 * (1 - np.cos(2 * np.pi * ms / N))]).T
    (Kc, cc), *_ = np.linalg.lstsq(X, w ** 2, rcond=None)
    return Kc, cc, dw(p, N // 4) / 2


def C_ratio(p, eta, S=S):
    N = len(p["C"])
    Kc, cc, bc = calibrate(p)
    m = N // 4
    d0, dp, dm = dw(p, m), dw(p, m, S * eta), dw(p, m, -S * eta)
    shift = 0.5 * (dp + dm) - d0
    pred = -0.25 * bc * np.mean((Kc * S * eta) ** 2) / cc ** 2
    return shift / pred


def p1_residual(p):
    N = len(p["C"])
    ref = dw(p, N // 4)
    return max(abs(dw(p, m) / ref - np.sin(2 * np.pi * m / N)) for m in range(1, N // 2))


def max_growth(p):
    A, _ = matrix(p)
    return np.linalg.eigvals(A).real.max()


def with_tol(N, tol, rng, base=None):
    p = dict(base or ideal(N))
    for key, v in (("C", C0), ("Lg", LG0), ("Lc", LC0), ("Ga", G0), ("Gb", G0)):
        p[key] = v * (1 + tol * rng.standard_normal(N))
    p["gs"] = G0 * tol * (rng.standard_normal(N) + rng.standard_normal(N))
    return p


def realistic(N, rng, tol=0.001, Q=100, fp=5e6, Cp=10e-12, Cin=3e-12):
    p = with_tol(N, tol, rng)
    p["Rg"] = W2 * p["Lg"] / Q; p["Rc"] = W2 * p["Lc"] / Q
    p["tau"] = 1 / (2 * np.pi * fp) if fp else 0.0
    p["Cp"] = Cp * np.ones(N); p["Cin"] = Cin
    return p


def lockin_dw(p, m, bump=None, pts=601):
    """Single-node drive at node 0 (e^{-iwt}), response V_n(w), projections on e^{+-ikn}, peak located on a grid
    of +-4 design linewidths around the design-value resonance, refined by a parabola in log|P|^2."""
    N = len(p["C"])
    A, Mm = matrix(p, bump)
    f = np.zeros(A.shape[0], complex)
    f[:N] = np.linalg.solve(Mm, np.eye(N)[0])
    k = 2 * np.pi * m / N
    Qk = K0 + 2 * CC0 * (1 - np.cos(k))
    out = []
    for sign in (1, -1):
        w0 = sign * B0 * np.sin(k) + np.sqrt(B0 ** 2 * np.sin(k) ** 2 + Qk)
        lw = w0 / 100
        grid = np.linspace(w0 - 4 * lw, w0 + 4 * lw, pts)
        pw = np.exp(-1j * sign * k * np.arange(N))
        P = np.array([abs(pw @ np.linalg.solve(A + 1j * w * np.eye(A.shape[0]), -f)[:N]) ** 2 for w in grid])
        j = int(np.clip(np.argmax(P), 1, pts - 2))
        y = np.log(P[j - 1:j + 2]); h = grid[1] - grid[0]
        out.append(grid[j] + h * 0.5 * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2]))
    return out[0] - out[1]


def main():
    N = 32
    g4 = eta_shape(N, "gauss", 4)
    shapes = [("gauss", 3), ("gauss", 5), ("sech2", 4), ("two", 2.5)]
    print(f"LC-GYRATOR RING -- C = {C0 * 1e9:.0f} nF, L_g = {LG0 * 1e3:.0f} mH, L_c = {LC0 * 1e3:.0f} mH, G = {G0:.4e} S; "
          f"K = {K0:.3e} s^-2, c-hat = {CC0 / K0:.3f}, beta-hat = {B0 / np.sqrt(K0):.3f}; w(pi/2) = {W2:.4e} rad/s")
    p = ideal(N)
    print(f"X0 ideal: Delta w(pi/2) = {dw(p, 8):.5e} (2b = {2 * B0:.5e}); P1 residual {p1_residual(p):.1e}; C ratio "
          f"{C_ratio(p, g4):.5f}")
    for tol, lab in ((0.01, "X1 1%"), (0.001, "X2 0.1%")):
        rng = np.random.default_rng(2028)
        P1, R, SP = [], [], []
        for _ in range(20):
            q = with_tol(N, tol, rng)
            P1.append(p1_residual(q)); R.append(C_ratio(q, g4))
            v = np.array([C_ratio(q, eta_shape(N, s, a)) for s, a in shapes]); SP.append((v.max() - v.min()) / abs(v.mean()))
        P1, R, SP = map(np.array, (P1, R, SP))
        print(f"{lab} tolerance: P1 max {P1.max():.1e} (mean {P1.mean():.1e}); P2 C ratio mean {R.mean():.4f}, sd "
              f"{R.std():.4f}; P3 shape spread mean {SP.mean():.3f}, max {SP.max():.3f}")
    print("X3 inductor Q (series R), ideal otherwise")
    d0, r0 = dw(p, 8), C_ratio(p, g4)
    for Q in (50, 100, 300):
        q = dict(ideal(N)); q["Rg"] = W2 * LG0 / Q * np.ones(N); q["Rc"] = W2 * LC0 / Q * np.ones(N)
        print(f"  Q = {Q}: Delta w change {dw(q, 8) / d0 - 1:+.2e}; C ratio change {C_ratio(q, g4) / r0 - 1:+.2e}; "
              f"max Re lambda {max_growth(q):+.3e} s^-1")
    print("X4 VCCS bandwidth (single pole), with inductor Q")
    for fp in (5e6, 50e6):
        q = dict(ideal(N)); q["tau"] = 1 / (2 * np.pi * fp)
        q100 = dict(q); q100["Rg"] = W2 * LG0 / 100 * np.ones(N); q100["Rc"] = W2 * LC0 / 100 * np.ones(N)
        print(f"  pole {fp / 1e6:.0f} MHz: Delta w change {dw(q, 8) / d0 - 1:+.2e}; P1 residual {p1_residual(q):.1e}; C "
              f"ratio change {C_ratio(q, g4) / r0 - 1:+.2e}; max Re lambda: lossless {max_growth(q):+.3e}", end="")
        for Q in (100, 300, 500, 1000, 3000, 10000):
            qq = dict(q); qq["Rg"] = W2 * LG0 / Q * np.ones(N); qq["Rc"] = W2 * LC0 / Q * np.ones(N)
            print(f", Q={Q} {max_growth(qq):+.2e}", end="")
        print()
    print("X5 VCCS gain mismatch and Howland shunt only (20 realisations)")
    for tol in (0.01, 0.001):
        rng = np.random.default_rng(5)
        dd, gr = [], []
        for _ in range(20):
            q = dict(ideal(N))
            q["Ga"] = G0 * (1 + tol * rng.standard_normal(N)); q["Gb"] = G0 * (1 + tol * rng.standard_normal(N))
            q["gs"] = G0 * tol * (rng.standard_normal(N) + rng.standard_normal(N))
            dd.append(abs(dw(q, 8) / d0 - 1)); gr.append(max_growth(q))
        print(f"  {tol:.1%}: max |Delta w change| {max(dd):.1e}; max Re lambda (lossless inductors) {max(gr):+.2e} s^-1")
    print("X6 parasitics (C_p across L_c, C_in per VCCS input)")
    q = dict(ideal(N)); q["Cp"] = 10e-12 * np.ones(N); q["Cin"] = 3e-12
    print(f"  C_p 10 pF, C_in 3 pF: P1 residual {p1_residual(q):.1e}; C ratio change {C_ratio(q, g4) / r0 - 1:+.2e}")
    print("X7 offsets (1 mV V_os on each VCCS, 10 nA bias per input), realistic circuit: DC solution")
    rng = np.random.default_rng(7)
    q = realistic(N, rng)
    A, Mm = matrix(q)
    f = np.zeros(A.shape[0])
    Ioff = 1e-3 * G0 * (rng.choice([-1, 1], N) + rng.choice([-1, 1], N)) + 10e-9 * 2 * rng.choice([-1, 1], N)
    f[:N] = np.linalg.solve(Mm, Ioff)
    x = -np.linalg.solve(A, f)
    print(f"  max |V_dc| {np.abs(x[:N]).max():.2e} V; max |i_dc| in L_g {np.abs(x[N:2 * N]).max():.2e} A, in L_c "
          f"{np.abs(x[2 * N:3 * N]).max():.2e} A (eigenfrequencies unaffected: linear superposition)")
    print("X8 combined realistic (0.1%, Q = 100, 5 MHz pole, C_p 10 pF, C_in 3 pF, mismatch, shunts), 20 realisations")
    for NN, SS in ((32, 0.05), (32, 0.02), (64, 0.02)):
        rng = np.random.default_rng(88)
        P1, R, SP, GR = [], [], [], []
        gg = eta_shape(NN, "gauss", NN / 8)
        shp = [("gauss", NN * 3 / 32), ("gauss", NN * 5 / 32), ("sech2", NN / 8), ("two", NN * 2.5 / 32)]
        for _ in range(20):
            q = realistic(NN, rng)
            P1.append(p1_residual(q)); R.append(C_ratio(q, gg, S=SS)); GR.append(max_growth(q))
            v = np.array([C_ratio(q, eta_shape(NN, s, a), S=SS) for s, a in shp]); SP.append((v.max() - v.min()) / abs(v.mean()))
        P1, R, SP, GR = map(np.array, (P1, R, SP, GR))
        print(f"  N = {NN}, S = {SS}: P1 max {P1.max():.1e}; P2 C ratio mean {R.mean():.4f}, sd {R.std():.4f}; P3 spread "
              f"mean {SP.mean():.3f}, max {SP.max():.3f}; max Re lambda {GR.max():+.2e} s^-1 (all stable: {bool((GR < 0).all())})")
    print("X9 lock-in estimator, X8 realisation 0 (N = 32, S = 0.05)")
    rng = np.random.default_rng(88)
    q = realistic(N, rng)
    Kc, cc, bc = calibrate(q)
    e0 = dw(q, 8); ee = 0.5 * (dw(q, 8, S * g4) + dw(q, 8, -S * g4)) - e0
    l0 = lockin_dw(q, 8); ll = 0.5 * (lockin_dw(q, 8, S * g4) + lockin_dw(q, 8, -S * g4)) - l0
    pred = -0.25 * bc * np.mean((Kc * S * g4) ** 2) / cc ** 2
    print(f"  eigen: Delta w {e0:.6e}, d(Delta w) {ee:+.5e} (ratio {ee / pred:.4f}); lock-in: Delta w {l0:.6e} "
          f"({l0 / e0 - 1:+.2e}), d(Delta w) {ll:+.5e} ({ll / ee - 1:+.2e}; ratio {ll / pred:.4f})")


if __name__ == "__main__":
    main()
