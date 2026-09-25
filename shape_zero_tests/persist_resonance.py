#!/usr/bin/env python3
"""
persist_resonance.py -- what does "persistence at every wavelength" (no resonant
decay channel open for any wave) demand of the model's parameters?

THE MODEL (04_scripts/session/model.py, force): per node, real components u,
    u'' = -(sqrt5 u + u*u) + c lap u + kappa JJ v        (u*u ELEMENTWISE)
and, in the scalar n = 1 sector used for the pinned asymmetry and P-1,
    u'' = -(sqrt5 u + u^2) + c lap u + beta c (v[n-1] - v[n+1]).

BRANCHES (dimer psi = u0 + i u1; psi'' - i kappa psi' + Q psi = F). Free waves
e^{i(kx - w t)} with w^2 + kappa w = Q(k), Q = sqrt5 + 2c sum_a (1 - cos k_a):
    a-branch  w_a(k) = (-kappa + sqrt(kappa^2 + 4Q)) / 2     (chi, the gates' branch)
    b-branch  w_b(k) = w_a(k) + kappa   EXACTLY               (the opposite chirality)
NONLINEARITY. -u*u per real component is, per dimer,
    F = -(1-i)/4 psi^2 - (1+i)/2 |psi|^2 - (1-i)/4 conj(psi)^2 ,
all three phase combinations with nonzero coefficients: the model's own
nonlinearity is J-breaking and gives NO polarization selection rule inside a
dimer. SELECTION RULE between dimers: the square is per component and JJ is
block-diagonal, so on a free lattice different dimers never couple (at any
order). Every n reduces to one dimer.

CHANNELS (energy and crystal momentum, mod 2pi), with w_b = w_a + kappa reducing
every branch combination to a function of w_a alone:
  three-wave 1 -> 2 : G(K, k1) = w_a(K) - w_a(k1) - w_a(K - k1) = m kappa,
       m = n_y + n_z - n_x (n = 1 for a b-wave): a->aa, b->ab (m=0); b->aa (m=-1);
       a->ab, b->bb (m=1); a->bb (m=2).   b->aa is also a+a->b: second-harmonic
       and sum-frequency generation into the opposite chirality.
  four-wave, a pump's own pair (k, k -> k+p, k-p), p != 0:
       D(k, p) = 2 w_a(k) - w_a(k+p) - w_a(k-p) = m kappa, m = n_y + n_z - 2 n_x:
       aa->aa, bb->bb (m=0; needs p != 0); aa->ab (m=1); aa->bb (m=2); bb->ab (-1);
       bb->aa (-2). The (0, pi) channel of P-1 is D(pi/2, pi/2).
  four-wave 1 -> 3 : G3 = w_a(K) - w_a(k1) - w_a(k2) - w_a(K-k1-k2) = m kappa,
       m = n_y + n_z + n_w - n_x in {-1, 0, 1, 2, 3}.
Each function is continuous on a connected torus (the p != 0 region: an annulus /
punctured torus), so a channel is open iff m kappa lies in the function's range
[min, max]. 1-D ranges from dense grids; 3-D ranges by multistart local optimisation
(the 3-D numbers are numerical optima, reported as such).

THE beta SECTOR: one real branch, w(k) = beta c sin k + sqrt(beta^2 c^2 sin^2 k + Q)
(the reference script's convention, +k upper); same channel functions with a
single branch.

usage:  python3 persist_resonance.py
"""
import math
import numpy as np
from scipy.optimize import minimize

K = math.sqrt(5.0)
RNG = np.random.default_rng(7)


def wa(Q, kap):
    return 0.5 * (-kap + np.sqrt(kap * kap + 4 * Q))


def Qv(k, c):
    k = np.asarray(k, float)
    return K + 2 * c * np.sum(1 - np.cos(k), axis=-1) if k.ndim and k.shape[-1] in (1, 3) and k.ndim > 1 \
        else K + 2 * c * (1 - np.cos(k))


def kappa_star(c, q=1):
    return 2 * q * c / math.sqrt(K + 2 * q * c)


# ---------------------------------------------------------------- 1-D ranges
N1 = 1441
KS = np.linspace(-math.pi, math.pi, N1)


def ranges_1d(kap, c, beta=None):
    """(min, max) of G, H=-G, D (p != 0), G3 on the 1-D zone, for branch w_a or the beta branch."""
    if beta is None:
        w = lambda k: wa(K + 2 * c * (1 - np.cos(k)), kap)
    else:
        w = lambda k: beta * c * np.sin(k) + np.sqrt(beta ** 2 * c ** 2 * np.sin(k) ** 2 + K + 2 * c * (1 - np.cos(k)))
    A, B = np.meshgrid(KS, KS, indexing="ij")
    G = w(A) - w(B) - w(A - B)
    ps = KS[np.abs(KS) >= 0.02]
    Ak, Pp = np.meshgrid(KS, ps, indexing="ij")
    D = 2 * w(Ak) - w(Ak + Pp) - w(Ak - Pp)
    k3 = np.linspace(-math.pi, math.pi, 181)
    X, Y, Z = np.meshgrid(k3, k3, k3, indexing="ij")
    G3 = w(X) - w(Y) - w(Z) - w(X - Y - Z)
    return dict(G=(G.min(), G.max()), D=(D.min(), D.max()), G3=(G3.min(), G3.max()),
                w=(w(KS).min(), w(KS).max()))


# ---------------------------------------------------------------- 3-D ranges
def _opt_range(f, dim, starts=120):
    lo, hi = np.inf, -np.inf
    xs = list(RNG.uniform(-math.pi, math.pi, (starts, dim)))
    # symmetric points too
    for v in (0.0, math.pi / 2, math.pi):
        xs.append(np.full(dim, v))
    for x0 in xs:
        for sgn in (1, -1):
            r = minimize(lambda x: sgn * f(x), x0, method="L-BFGS-B")
            val = f(r.x)
            lo, hi = min(lo, val), max(hi, val)
    return lo, hi


def ranges_3d(kap, c):
    w = lambda k: wa(K + 2 * c * np.sum(1 - np.cos(k)), kap)
    G = lambda x: w(x[:3]) - w(x[3:]) - w(x[:3] - x[3:])

    def D(x):                                   # p kept away from 0 by construction below
        k, p = x[:3], x[3:]
        return 2 * w(k) - w(k + p) - w(k - p)
    G3 = lambda x: w(x[:3]) - w(x[3:6]) - w(x[6:]) - w(x[:3] - x[3:6] - x[6:])
    g = _opt_range(G, 6)
    d = _opt_range(lambda x: D(x) if np.linalg.norm(np.angle(np.exp(1j * x[3:]))) > 0.05 else 0.0, 6)
    g3 = _opt_range(G3, 9, starts=60)
    corner = np.full(3, math.pi)
    return dict(G=g, D=d, G3=g3, w=(w(np.zeros(3)), w(corner)))


def channel_table(r, kap):
    """Open/closed for every channel, given the ranges r."""
    inr = lambda v, rg: rg[0] - 1e-9 <= v <= rg[1] + 1e-9
    out = {}
    for m, name in ((0, "a->aa, b->ab"), (-1, "b->aa (= a+a->b)"), (1, "a->ab, b->bb"), (2, "a->bb")):
        out["3w " + name] = inr(m * kap, r["G"])
    out["4w aa->aa, bb->bb (p!=0)"] = r["D"][0] < 0 < r["D"][1]
    for m, name in ((1, "aa->ab, bb->ab"), (2, "aa->bb, bb->aa")):
        out["4w " + name] = inr(m * kap, r["D"]) or inr(-m * kap, r["D"])
    for m, name in ((-1, "b->aaa"), (0, "a->aaa, b->aab"), (1, "a->aab, b->abb"),
                    (2, "a->abb, b->bbb"), (3, "a->bbb")):
        out["1->3 " + name] = inr(m * kap, r["G3"])
    return out


def main():
    print("=" * 92)
    print("PERSISTENCE AT EVERY WAVELENGTH -- resonance conditions on the model's branches")
    print("=" * 92)
    print("  w_b = w_a + kappa exactly; nonlinearity -u*u elementwise = -(1-i)/4 psi^2 - (1+i)/2|psi|^2")
    print("  - (1-i)/4 psi*^2: no selection rule inside a dimer; dimers never couple on a free lattice.")

    # ---- the rough check the question proposes
    print("\n  ROUGH CHECK (plain dispersion w^2 = sqrt5 + 2c sum(1 - cos k); 2 w_min > w_max):")
    print(f"     1-D: c < 3 sqrt5/4 = {3 * K / 4:.4f};   3-D: c < sqrt5/4 = {K / 4:.4f}")
    for c in (1.0,):
        r = ranges_1d(0.0, c)
        print(f"     exact 1-D at c = 1, kappa = 0 (momentum included): three-wave G max = {r['G'][1]:+.4f} "
              f"-> {'OPEN' if r['G'][1] >= 0 else 'closed'}")
    ks = kappa_star(1.0)
    r = ranges_1d(ks, 1.0)
    print(f"     at kappa* = {ks:.6f}, c = 1, 1-D: w_a in [{r['w'][0]:.4f}, {r['w'][1]:.4f}], "
          f"w_b in [{r['w'][0] + ks:.4f}, {r['w'][1] + ks:.4f}]; 2 min(w_a) = {2 * r['w'][0]:.4f} < "
          f"max(w_b) = {r['w'][1] + ks:.4f}: energy alone allows b->aa")

    # ---- full channel tables at the operating point and neighbours
    for q in (1, 3):
        print(f"\n  CHANNEL TABLE, q = {q}, c = 1")
        kaps = (0.5, ks, 1.2, 1.82, 2.4, 3.0, 5.0) if q == 1 else (0.5, ks, 2.090698, 3.0, 5.0, 8.0)
        tabs = {}
        for kap in kaps:
            rr = ranges_1d(kap, 1.0) if q == 1 else ranges_3d(kap, 1.0)
            tabs[kap] = (rr, channel_table(rr, kap))
        names = list(next(iter(tabs.values()))[1].keys())
        print("     " + "channel".ljust(30) + "".join(f"k={k:<8.4f}" for k in kaps))
        for nm in names:
            print("     " + nm.ljust(30) + "".join(("OPEN      " if tabs[k][1][nm] else "closed    ") for k in kaps))
        print("     ranges: " + "; ".join(f"k={k:.3f}: G[{tabs[k][0]['G'][0]:+.3f},{tabs[k][0]['G'][1]:+.3f}] "
                                          f"D[{tabs[k][0]['D'][0]:+.3f},{tabs[k][0]['D'][1]:+.3f}] "
                                          f"G3[{tabs[k][0]['G3'][0]:+.3f},{tabs[k][0]['G3'][1]:+.3f}]"
                                          for k in kaps))

    # ---- where the b->aa (second-harmonic into b) resonance sits at kappa*
    kk = np.linspace(1e-3, math.pi, 200001)
    w = lambda k, kap: wa(K + 2 * (1 - np.cos(k)), kap)
    shg = 2 * w(kk, ks) - w(2 * kk, ks)
    i = np.argmin(shg)
    roots = kk[np.where(np.diff(np.sign(shg - ks)))[0]]
    print(f"\n  SECOND HARMONIC INTO THE b-BRANCH (b->aa at K = 2k): 2 w_a(k) - w_a(2k) = kappa")
    print(f"     at kappa*, c = 1, q = 1: min over k = {shg[i]:.6f} at k = {kk[i]:.4f}; resonant at k = "
          + ", ".join(f"{x:.5f}" for x in roots))
    print(f"     (the global min of H over (K, k1) is at k1 = K/2: the second-harmonic line)")

    # ---- bounds in (kappa, c): scan
    print("\n  BOUNDS IN (kappa, c), q = 1: kappa ranges where each requirement holds")
    print("     c      c/sqrt5  kappa*(J)   3w b->aa closed           3w a->aa closed  4w aa->ab closed  1->3 closed")
    kgrid = np.round(np.concatenate([np.linspace(0.02, 3.0, 150), np.linspace(3.05, 8, 100)]), 4)
    for c in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        kst = kappa_star(c)
        ok3b, ok3a, ok4, ok13, okall = [], [], [], [], []
        for kap in kgrid:
            rr = ranges_1d(kap, c)
            t = channel_table(rr, kap)
            a = not t["3w b->aa (= a+a->b)"]
            b = not (t["3w a->aa, b->ab"] or t["3w a->ab, b->bb"] or t["3w a->bb"])
            d = not (t["4w aa->ab, bb->ab"] or t["4w aa->bb, bb->aa"])
            e = not any(v for kk_, v in t.items() if kk_.startswith("1->3"))
            ok3b.append(a); ok3a.append(b); ok4.append(d); ok13.append(e)
            okall.append(a and b and d and e and kap >= kst)

        def spans(mask):
            s, out = None, []
            for kap, m in zip(kgrid, mask):
                if m and s is None:
                    s = kap
                if not m and s is not None:
                    out.append(f"[{s:.2f},{prev:.2f}]"); s = None
                prev = kap
            if s is not None:
                out.append(f"[{s:.2f},{kgrid[-1]:.1f}+]")
            return " ".join(out) if out else "none"
        print(f"     {c:4.2f}   {c / K:.3f}    {kst:.4f}    {spans(ok3b):24s}  {spans(ok3a):15s}  "
              f"{spans(ok4):16s}  {spans(ok13)}")
        print(f"            ALL (with kappa >= kappa*): {spans(okall)}")

    # ---- beta sector
    print("\n  THE beta SECTOR (n = 1 scalar, kappa = 0), q = 1, c = 1")
    print("     beta     w range          3w G max    4w D range (p!=0)      (0,pi) detuning 2w(-pi/2)-w(0)-w(pi)")
    for beta in (0.0, 0.02, 0.05, 0.0619, 0.08, 0.2, 0.5):
        rr = ranges_1d(0.0, 1.0, beta=beta)
        wb = lambda k: beta * np.sin(k) + np.sqrt(beta ** 2 * np.sin(k) ** 2 + K + 2 * (1 - np.cos(k)))
        det = 2 * wb(-math.pi / 2) - wb(0.0) - wb(math.pi)
        print(f"     {beta:.4f}   [{rr['w'][0]:.4f},{rr['w'][1]:.4f}]   {rr['G'][1]:+.4f}     "
              f"[{rr['D'][0]:+.4f},{rr['D'][1]:+.4f}]      {det:+.5f}")
    # beta where the (0,pi) channel of a -pi/2 pump is exactly resonant
    f = lambda b: 2 * (b * -1 + math.sqrt(b * b + K + 2)) - math.sqrt(K) - math.sqrt(K + 4)
    lo, hi = 0.0, 0.2
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    print(f"     (0, pi) channel of the -pi/2 pump exactly resonant at beta = {lo:.5f} (linear dispersion)")


if __name__ == "__main__":
    main()
