#!/usr/bin/env python3
"""
z1_d8_plurality.py — plurality selects the flow that minimality seemed to forbid

z1_d8_minimality.py left the top rung apparently unreachable: minimality
removes the second generator, and both parameter-free laws turn out to be lower
rungs re-embedded (D2 motion for the anticommutator, D4 for the commutator).
Only the generic two-generator flow reaches octonionic content, and it carries a
parameter nothing seemed to force.

AN AUDIT FIRST. In the C1 formal proofs document the selection principle
MINIMALITY -- "no continuous parameter is introduced unless forced by a prior
principle or by the purchased plurality assumption" -- is stated in Section 1
and then never cited in any lemma, theorem or proof. Every actual elimination is
performed by something else:

    D1  first integral                 conservativity
    D2  void constant g = L^2/2        Noether, via the cyclic coordinate
    D3  no 3-dimensional case          Hurwitz / Frobenius
    D4  coupling class u(2)            passivity (Section 6, explicitly)

The word reappears later only as "role minimality," a different postulate about
Fano colourings. So minimality has never been load-bearing, and its bite at D8
is its FIRST application.

THE RESOLUTION IS IN THE PRINCIPLE AS WRITTEN. Minimality exempts what the
purchased plurality assumption forces. Plurality states that there is more than
one dynamical degree of freedom. If the parameter-free flows each have exactly
ONE, plurality excludes them and minimality's own clause then admits the second
generator. That is checkable by counting independent frequencies in the motion.

PREDICTIONS STATED BEFORE RUNNING
 Q1 both parameter-free flows have exactly one independent frequency, for every
    sampled (psi_0, a).
 Q2 the generic two-generator flow has two, for every sampled (psi_0, a, b) --
    at sufficient frequency resolution. A coarse run at T = 60 returned a single
    frequency in one trial of six; the two peaks there are separated by 0.0133,
    below that run's resolution of 1/60 = 0.0167. RECORDED because the coarse
    result was reported before being checked.
 Q3 the single-generator frequency appears again in the generic spectrum, so
    the generic flow is the one-generator motion plus an independent second one.
 Q4 therefore plurality excludes both parameter-free laws, minimality's
    exemption clause admits the second generator, and the generic flow is
    selected -- 14 terms, nonzero associator, full octonionic content.

Python 3 + NumPy only.
"""

import numpy as np
import importlib.util
import os

_here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "fl", os.path.join(_here, "z1_d8_flow.py"))
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)
ch = fl.ch


def spectrum(traj, T):
    n = len(traj)
    w = np.hanning(n)[:, None]
    F = np.abs(np.fft.rfft((traj - traj.mean(0)) * w, axis=0))
    P = (F ** 2).sum(1)
    P /= P.max()
    return np.fft.rfftfreq(n, d=T / (n - 1)), P


def peaks(fr, P, tol=0.02):
    return [fr[i] for i in range(1, len(P) - 1)
            if P[i] > P[i - 1] and P[i] > P[i + 1] and P[i] > tol]


def main():
    orient = ch.oriented_lines()
    E = fl.oct_table(orient)
    rng = np.random.default_rng(719)
    T, N = 600.0, 240000
    print("=" * 70)
    print("PLURALITY SELECTS THE D8 FLOW")
    print("=" * 70)
    print(f"\n  integration T = {T:.0f}, resolution {1/T:.5f}")
    print("  (a coarse T = 60 run merged a pair separated by 0.0133)")

    res = {"anticommutator": [], "commutator": [], "generic": []}
    firsts = {}
    for trial in range(6):
        p0 = rng.normal(size=8)
        p0 /= np.linalg.norm(p0)
        a = np.zeros(8)
        a[1:] = rng.normal(size=7)
        a[1:] /= np.linalg.norm(a[1:])
        b = np.zeros(8)
        b[1:] = rng.normal(size=7)
        b[1:] /= np.linalg.norm(b[1:])
        laws = (("anticommutator", lambda p: fl.mul(p, a, E) + fl.mul(a, p, E)),
                ("commutator", lambda p: fl.mul(p, a, E) - fl.mul(a, p, E)),
                ("generic", lambda p: fl.mul(p, a, E) + fl.mul(b, p, E)))
        for name, rhs in laws:
            fr, P = spectrum(fl.integrate(p0, rhs, T=T, n=N), T)
            pk = peaks(fr, P)
            res[name].append(len(pk))
            firsts.setdefault(name, [round(x, 5) for x in pk])

    print("\n  law               frequencies (trial 1)        counts over 6")
    for name in ("anticommutator", "commutator", "generic"):
        print(f"    {name:15s} {str(firsts[name]):26s} {res[name]}")

    ok1 = all(n == 1 for n in res["anticommutator"] + res["commutator"])
    ok2 = all(n == 2 for n in res["generic"])
    shared = set(firsts["anticommutator"]) & set(firsts["generic"])
    print(f"\n  Q1 parameter-free laws have ONE frequency : "
          f"{'PASS' if ok1 else 'MISS'}")
    print(f"  Q2 generic has TWO                        : "
          f"{'PASS' if ok2 else 'MISS'}")
    print(f"  Q3 single-generator frequency recurs      : "
          f"{'PASS' if shared else 'MISS'}   {sorted(shared)}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  Each parameter-free law is a single periodic angle -- ONE")
    print("  dynamical degree of freedom. Plurality states there is more than")
    print("  one, so it excludes both. Minimality's own wording then admits")
    print("  the second generator: it forbids a parameter 'unless forced by a")
    print("  prior principle OR BY THE PURCHASED PLURALITY ASSUMPTION'.")
    print()
    print("  The tension is resolved without adding a fourth principle. It")
    print("  required reading two existing ones jointly rather than in turn.")
    print()
    print("  Residual: plurality forces A second generator but not WHICH. The")
    print("  a-b angle survives -- now permitted by exemption rather than")
    print("  forbidden, and shown earlier to leave the outcome at 5 / 14")
    print("  regardless of its value.")


if __name__ == "__main__":
    main()
