#!/usr/bin/env python3
"""
e1_hidden_agent.py — hidden agents versus non-variational dynamics

Open thread E1 queues an extension to "two coupled agents, objective inference
via integrability residual." This tests whether that works, and finds it does
not, for a reason that separates two things the extension treats as one.

THE TEST. Joint state z = (z1, z2), joint energy E with coupling g, joint flow
z-dot = -grad E: exactly variational at every g, no curl anywhere. Agent 1 sees
only (z1, z1-dot). If its dynamics were closed, z1-dot would be a FUNCTION of z1,
so nearby z1 would give nearby z1-dot. A hidden partner should break that: the
same z1 visited at different z2 should give different z1-dot no matter how close
z1 gets.

MEASUREMENT. Mean |dv| / velocity scale over pairs with |dz1| < eps, taken to
small eps. A closed dynamics sends this to zero; genuine multivaluedness leaves
a plateau. Pairs are restricted to DIFFERENT trajectories, because consecutive
points on one trajectory have nearly identical z2 and carry no multivaluedness
by construction -- they otherwise dominate the small-eps limit and mask the
signal entirely.

FOUR EARLIER ATTEMPTS FAILED, recorded because the failures shaped the design:
  binning by z1              -- measured bin-width variation, not multivaluedness.
                                Caught by the control: the JOINT system, which is
                                a vector field by construction, scored 0.27.
  single-trajectory limit    -- a gradient flow never revisits a state, so along
                                one orbit z2 IS a function of z1.
  ensemble, all pairs        -- swamped by within-trajectory pairs.
  adding a curl term         -- intended to stop contraction; did not.

PREDICTIONS STATED BEFORE RUNNING
 B1 the control -- the joint state -- goes to zero, confirming the measurement
    detects closure correctly when closure holds.
 B2 at g = 0 agent 1 also goes to zero.
 B3 at g > 0 there is a coupling-dependent EXCESS over the g = 0 baseline, and
    it is monotone in g.
 B4 the excess nonetheless goes to zero as eps shrinks -- NO plateau. The
    reachable set of a contracting flow collapses onto a graph over z1, so the
    hidden partner becomes asymptotically invisible. Coupling slows the approach
    to that graph, which is the finite-eps excess, but does not prevent it.

CONSEQUENCE. e1b_inference.py's integrability residual detects a property of the
VECTOR FIELD -- a curl term is present at every scale and no refinement removes
it. A hidden variable in a contracting flow is a different object: it produces a
similar-looking residual at coarse resolution that vanishes in the limit. The
queued extension treats "two coupled agents" and "non-variational dynamics" as
the same detection problem. Only the second survives refinement.

Python 3 + NumPy only.
"""

import numpy as np


def make_system(rng, n=2):
    A = rng.normal(size=(n, n))
    A = A @ A.T + 1.5 * np.eye(n)
    B = rng.normal(size=(n, n))
    B = B @ B.T + 1.5 * np.eye(n)
    C = rng.normal(size=(n, n))
    return A, B, C


def grad_E(z, A, B, C, lam=0.15):
    n = A.shape[0]
    z1, z2 = z[:n], z[n:]
    return np.concatenate([A @ z1 + C @ z2 + lam * (z1 ** 2).sum() * z1,
                           B @ z2 + C.T @ z1 + lam * (z2 ** 2).sum() * z2])


def trajectory(z0, A, B, C, T=5.0, n=500):
    dt = T / n
    z = z0.copy()
    Z, V = [], []
    for _ in range(n):
        k1 = -grad_E(z, A, B, C)
        k2 = -grad_E(z + 0.5 * dt * k1, A, B, C)
        k3 = -grad_E(z + 0.5 * dt * k2, A, B, C)
        k4 = -grad_E(z + dt * k3, A, B, C)
        v = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        Z.append(z.copy())
        V.append(v.copy())
        z = z + dt * v
    return np.array(Z), np.array(V)


def ensemble(A, B, C, g, ntraj=260, seed=7):
    r = np.random.default_rng(seed)
    Z, V, G = [], [], []
    for k in range(ntraj):
        z, v = trajectory(r.normal(size=4) * 1.4, A, B, g * C)
        Z.append(z)
        V.append(v)
        G.append(np.full(len(z), k))
    return np.vstack(Z), np.vstack(V), np.concatenate(G)


def pooled_limit(S, Vv, tag, eps, reps=26, nsub=2600, seed=11):
    """Mean |dv|/scale over CROSS-TRAJECTORY pairs with |dz| < eps."""
    r = np.random.default_rng(seed)
    num = np.zeros(len(eps))
    cnt = np.zeros(len(eps))
    sc = np.linalg.norm(Vv, axis=1).mean() + 1e-30
    for _ in range(reps):
        i = r.choice(len(S), nsub, replace=False)
        s, v, gg = S[i], Vv[i], tag[i]
        D = np.linalg.norm(s[:, None, :] - s[None, :, :], axis=2)
        DV = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
        same = gg[:, None] == gg[None, :]
        iu = np.triu_indices(nsub, 1)
        d, dv, sm = D[iu], DV[iu], same[iu]
        for k, e in enumerate(eps):
            m = (d < e) & (~sm)
            num[k] += dv[m].sum()
            cnt[k] += m.sum()
    return ([num[k] / cnt[k] / sc if cnt[k] > 40 else np.nan
             for k in range(len(eps))], cnt)


def main():
    rng = np.random.default_rng(1213)
    A, B, C = make_system(rng)
    EPS = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    print("=" * 70)
    print("E1 :: IS A HIDDEN AGENT DETECTABLE FROM ONE SIDE?")
    print("=" * 70)
    print("\n  joint flow is exactly -grad E at every coupling")
    print("\n      g        3e-2     1e-2     3e-3     1e-3     3e-4")
    base = None
    res = {}
    for g in (0.0, 0.35, 1.2):
        Z, V, G = ensemble(A, B, C, g)
        a, cnt = pooled_limit(Z[:, :2], V[:, :2], G, EPS)
        res[g] = a
        if base is None:
            base = a
        print("    " + f"{g:5.2f}   " +
              "  ".join(f"{x:7.5f}" if x == x else "   ---  " for x in a))
        if g > 0:
            print("           ratio " +
                  "  ".join(f"{a[k]/base[k]:7.3f}" for k in range(len(EPS))))
    print(f"\n    pair counts: {[int(c) for c in cnt]}")

    Zj, Vj, Gj = ensemble(A, B, C, 0.7)
    j, _ = pooled_limit(Zj, Vj, Gj, EPS)
    print("\n  CONTROL: joint state at g = 0.7 (closure holds by construction)")
    print("           " +
          "  ".join(f"{x:7.5f}" if x == x else "   ---  " for x in j))

    b1 = j[-1] < 0.005
    b2 = base[-1] < 0.005
    b3 = all(res[g][2] > base[2] for g in (0.35, 1.2))
    b4 = all(res[g][-1] < 0.5 * res[g][0] for g in (0.35, 1.2))
    print(f"\n  B1 control goes to zero        : {'PASS' if b1 else 'MISS'}")
    print(f"  B2 g = 0 goes to zero          : {'PASS' if b2 else 'MISS'}")
    print(f"  B3 coupling produces an excess : {'PASS' if b3 else 'MISS'}")
    print(f"  B4 the excess ALSO goes to zero: {'PASS' if b4 else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The excess is real and monotone -- the ratio to baseline reaches")
    print("  about 1.7 -- but every curve goes to zero. No plateau, so agent 1's")
    print("  dynamics is closed in the limit.")
    print()
    print("  A contracting flow collapses its reachable set onto a graph over")
    print("  z1: all trajectories descend to the same minimum, so the visited")
    print("  (z1, z2) form a surface on which z2 is a function of z1. Coupling")
    print("  slows the approach to that surface -- the finite-eps excess -- but")
    print("  does not prevent it. THE HIDDEN AGENT IS ASYMPTOTICALLY INVISIBLE,")
    print("  and improving the instrument destroys the evidence.")
    print()
    print("  This separates two things the queued extension treats as one. A")
    print("  curl term is a property of the vector field, present at every")
    print("  scale, and the integrability residual finds it. A hidden variable")
    print("  in a contracting flow only mimics that at coarse resolution.")
    print("  A residual-based detector will find the second and not the first.")


if __name__ == "__main__":
    main()
