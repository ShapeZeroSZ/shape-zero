#!/usr/bin/env python3
"""
em_scope_checks.py -- checks behind natural/EM_SCOPE.md (target 1, scoping only;
nothing here extends the model). Hypotheses: natural/EM_HYPOTHESES.md (7a166b8).

C1  H1: a u(1) phase on the existing VELOCITY link. In the model's literal form
        c(W_n v_{n+1} - W_{n-1} v_{n-1}) (mission 2), W = a I + b J does work unless b = 0.
        In the COVARIANT form -- the reverse hop transported with the transpose,
        c(M_n v_{n+1} - M_{n-1}^T v_{n-1}), M_n = W_n U_n -- the coupling matrix is
        antisymmetric and does no work for ANY W and theta; for symmetric W the two forms
        coincide, which is why mission 2 did not need to distinguish them.
C2  H1: the same phase on the ELASTIC link, c(U_n x_{n+1} + U_{n-1}^T x_{n-1} - 2 x_n),
        U_n = exp(J theta_n), derives from a potential (conservative) and is covariant
        under local J-rotations with theta_n -> theta_n + alpha_n - alpha_{n+1}.
C3  H4: linearised Wilson plaquette energy on a q-dimensional lattice -> the curl-curl
        operator. Its eigenvalues give the photon: at q = 3 two polarisations with
        4 sum sin^2(k/2) and one zero (longitudinal, fixed by Gauss's law); q = 2 one; q = 1 none.
C4  H4: static force of a point charge from the lattice Green's function of -Laplacian,
        as a ratio to the continuum law (1 = exact): q = 3 inverse-square; q = 2 1/r; q = 1 constant.
C5  charged lumps: node form A' has V = sqrt5 r^2/2 + r^3/3; a Q-ball needs
        min_r 2V/r^2 < K (Coleman). 2V/r^2 = K + 2r/3 > K for r > 0: none exist.
C7  (found during scoping) the existing u(n) velocity links, dressed M_n = W_n U_n in the
        covariant form, are passive for every theta and covariant under local J-rotations
        (alpha_n per site). The literal form, dressed the same way, does work.
C6  H5: the beta link's equivalent Peierls angle is arctan(beta w) -- it grows with the
        frequency w (MODEL_SPEC 1c) -- while a u(1) phase shifts every mode by the same theta.
"""

import math
import numpy as np

rng = np.random.default_rng(0)
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def rot(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def gyro_force(v, M, form):
    """Velocity-link force on every node; M[n] couples n -> n+1."""
    N = len(v)
    f = np.zeros_like(v)
    for n in range(N):
        back = M[(n - 1) % N] if form == "literal" else M[(n - 1) % N].T
        f[n] = M[n] @ v[(n + 1) % N] - back @ v[(n - 1) % N]
    return f


def c1():
    N, a, b = 7, 0.3, 0.2
    v = rng.standard_normal((N, 2))
    for form in ("literal", "covariant"):
        for bb in (0.0, b):
            M = [a * np.eye(2) + bb * J] * N
            P = float(np.sum(v * gyro_force(v, M, form)))
            print(f"  C1  {form:9s} velocity link, W = {a} I + {bb} J:  power = {P:+.3e}   "
                  f"({'passive' if abs(P) < 1e-12 else 'DOES WORK'})")


def c2():
    N, c = 7, 1.0
    x = rng.standard_normal((N, 2))
    th = rng.standard_normal(N)

    def V(x, th):
        return 0.5 * c * sum(np.sum((x[(n + 1) % N] - rot(th[n]).T @ x[n]) ** 2) for n in range(N))

    def F(x, th):
        f = np.zeros_like(x)
        for n in range(N):
            U, Um = rot(th[n]), rot(th[(n - 1) % N])
            f[n] = c * (U @ x[(n + 1) % N] + Um.T @ x[(n - 1) % N] - 2 * x[n])
        return f
    # conservative: F = -grad V
    g = np.zeros_like(x); h = 1e-6
    for n in range(N):
        for i in range(2):
            xp, xm = x.copy(), x.copy(); xp[n, i] += h; xm[n, i] -= h
            g[n, i] = (V(xp, th) - V(xm, th)) / (2 * h)
    err = np.max(np.abs(F(x, th) + g))
    # covariance: x_n -> R(al_n) x_n, th_n -> th_n + al_n - al_{n+1}
    al = rng.standard_normal(N)
    x2 = np.array([rot(al[n]) @ x[n] for n in range(N)])
    th2 = np.array([th[n] + al[n] - al[(n + 1) % N] for n in range(N)])
    Fc = np.array([rot(al[n]) @ F(x, th)[n] for n in range(N)])
    cov = np.max(np.abs(F(x2, th2) - Fc))
    print(f"  C2  elastic Peierls link: |F + grad V| = {err:.1e} (conservative);  "
          f"covariance error {cov:.1e}")


def curlcurl(k):
    """Linearised Wilson energy sum_P theta_P^2/2 in Fourier space: M = D^dag D over plaquettes."""
    q = len(k)
    d = np.exp(1j * np.array(k)) - 1.0              # forward difference symbols
    rows = []
    for m in range(q):
        for n in range(m + 1, q):
            r = np.zeros(q, complex)
            r[n] += d[m]; r[m] -= d[n]              # theta_P = D_m th_n - D_n th_m
            rows.append(r)
    if not rows:
        return np.zeros((q, q))
    D = np.array(rows)
    return (D.conj().T @ D).real if np.allclose((D.conj().T @ D).imag, 0) else D.conj().T @ D


def c3():
    for q in (1, 2, 3):
        k = rng.uniform(-math.pi, math.pi, q)
        ev = np.sort(np.linalg.eigvalsh(curlcurl(k)))
        s = sum(4 * math.sin(ki / 2) ** 2 for ki in k)
        nphys = int(np.sum(ev > 1e-10))
        ok = all(abs(e - s) < 1e-10 for e in ev if e > 1e-10)
        print(f"  C3  q = {q}: eigenvalues {np.round(ev, 6)}  vs 4 sum sin^2(k/2) = {s:.6f}  "
              f"-> {nphys} photon polarisation(s){', dispersion exact' if nphys and ok else ''}")


def c4():
    """Force of a unit point charge, -dG/dr at the midpoint r+1/2, on an L-periodic lattice.
    The periodic box needs a neutralising uniform background, whose own (exactly known)
    force -r/(qL^q)*... is added back: q = 3: r/(3L^3); q = 2: r/(2L^2); q = 1: r/L."""
    L = 128
    k = 2 * np.pi * np.fft.fftfreq(L)
    target = {3: lambda r: 1 / (4 * math.pi * r * r), 2: lambda r: 1 / (2 * math.pi * r), 1: lambda r: 0.5}
    bg = {3: lambda r: r / (3 * L ** 3), 2: lambda r: r / (2 * L ** 2), 1: lambda r: r / L}
    label = {3: "inverse-square (Coulomb)", 2: "1/r (logarithmic potential)", 1: "constant (linear potential)"}
    for q in (3, 2, 1):
        ks = np.meshgrid(*([k] * q), indexing="ij")
        lam = sum(4 * np.sin(kk / 2) ** 2 for kk in ks)
        rk = np.ones((L,) * q, complex); rk[(0,) * q] = 0.0
        lam[(0,) * q] = 1.0
        G = np.fft.ifftn(rk / lam).real
        cells = []
        for r in (4, 8, 16, 32):
            f = G[(r,) + (0,) * (q - 1)] - G[(r + 1,) + (0,) * (q - 1)]
            rm = r + 0.5
            cells.append(f"r={r:2d}: {(f + bg[q](rm)) / target[q](rm):.4f}")
        print(f"  C4  q = {q}: force / {label[q]} law = " + ", ".join(cells))


def c5():
    K = math.sqrt(5)
    r = np.linspace(1e-6, 10, 10001)
    ratio = 2 * (K * r ** 2 / 2 + r ** 3 / 3) / r ** 2
    print(f"  C5  node form A': min_r 2V/r^2 = {ratio.min():.6f} vs K = {K:.6f}  -> "
          f"{'Q-balls possible' if ratio.min() < K - 1e-9 else 'no Q-balls (Coleman condition fails)'}")


def c6():
    beta, c = 0.05, 1.0
    for K in (math.sqrt(5), 4 * math.sqrt(5), 16 * math.sqrt(5)):
        # frequency of the k = pi/2 mode at beta = 0 sets the scale
        w = math.sqrt(K + 2 * c)
        print(f"  C6  K = {K:7.3f}: carrier w = {w:.3f}; beta link's Peierls angle arctan(beta w) = "
              f"{math.atan(beta*w):.4f}  (a u(1) phase theta would be the same for every w)")


def c7():
    N = 7
    Jn = np.kron(np.eye(2), J)                          # n = 2 complex components, 4 real
    A = rng.standard_normal((4, 4)); S = A + A.T
    W = 0.5 * (S - Jn @ S @ Jn)                         # symmetric and commuting with J
    assert np.allclose(W, W.T) and np.allclose(W @ Jn, Jn @ W)
    R = lambda t: math.cos(t) * np.eye(4) + math.sin(t) * Jn
    v = rng.standard_normal((N, 4))
    th = rng.standard_normal(N)
    M = [W @ R(t) for t in th]
    for form in ("literal", "covariant"):
        P = float(np.sum(v * gyro_force(v, M, form)))
        al = rng.standard_normal(N)                     # v_n -> R(al_n) v_n,  th_n -> th_n + al_n - al_{n+1}
        v2 = np.array([R(al[n]) @ v[n] for n in range(N)])
        M2 = [W @ R(th[n] + al[n] - al[(n + 1) % N]) for n in range(N)]
        f, f2 = gyro_force(v, M, form), gyro_force(v2, M2, form)
        cov = np.max(np.abs(f2 - np.array([R(al[n]) @ f[n] for n in range(N)])))
        print(f"  C7  dressed u(2) velocity link W U(theta_n), {form:9s} form: power = {P:+.3e} "
              f"({'passive' if abs(P) < 1e-12 else 'DOES WORK'}), covariance error {cov:.1e}")


if __name__ == "__main__":
    print("EM SCOPING CHECKS (natural/EM_HYPOTHESES.md)")
    c1(); c2(); c3(); c4(); c5(); c6(); c7()
