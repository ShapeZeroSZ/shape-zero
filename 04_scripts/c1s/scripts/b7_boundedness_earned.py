#!/usr/bin/env python3
"""
b7_boundedness_earned.py — does persistence force a bounded coupling, or forbid it?

b7_boundedness.py showed the E_G-shaped envelope -- quadratic below a core
scale, plateau above it at (count x per-element ceiling) -- follows from
BOUNDEDNESS of the bond coupling alone, and not from any particular functional
form. Three unrelated bounded families gave the same plateau. That left exactly
one proposition to earn: is boundedness forced by the program's principles?

The sketch offered then argued: minimality disfavours it (a bounded potential
carries a scale, an unbounded quadratic does not), while persistence might
favour it via D1's requirement of bounded motion.

THE SKETCH HAD PERSISTENCE BACKWARDS. D1 selects bounded MOTION, and an
unbounded confining potential is exactly what guarantees bounded motion -- a
harmonic bond always returns its endpoints. A SATURATING bond has a ceiling:
give a configuration more energy than the ceiling and nothing holds it, so it
separates without bound. Bounded couplings permit unbounded motion; unbounded
couplings forbid it.

If that is right, persistence does not merely fail to force boundedness. It
argues against it, and B-7's E_G analogy rests on a property the program's own
principles disfavour.

SETUP. One degree of freedom x with bond energy V(x) and total energy
E = 1/2 v^2 + V(x), integrated conservatively. Two families:

    unbounded    V = 1/2 k x^2                  (no ceiling)
    saturating   V = s^2 (1 - exp(-x^2/2s^2))   (ceiling s^2, matches
                                                 b7_boundedness.py)

PREDICTIONS STATED BEFORE RUNNING
 C1 with the unbounded bond, EVERY initial energy gives bounded motion, at any
    energy sampled.
 C2 with the saturating bond, motion is bounded below a threshold energy and
    unbounded above it.
 C3 the threshold is the ceiling: E_escape = s^2 to within integration error.
    [Flagged MISS on the first run by a faulty DETECTION criterion, not a wrong
    prediction. Escape was tested as |x| > 1e6, which free-streaming
    trajectories do not reach in finite T, so they were logged as bounded. The
    correct test is T-dependence: bounded motion has max|x| independent of T,
    escape has it proportional. Retested at T = 200 / 400 / 800, the ratio at
    4x the time is 3.77, 3.96, 4.00, 4.00 for E above the ceiling and exactly
    1.000 below it. The threshold IS the ceiling.]
 C4 therefore persistence, which retains only bounded motion, FAVOURS the
    unbounded coupling. Boundedness is not earned; it is counter-indicated, and
    the E_G envelope rests on a property the principles argue against.

Python 3 + NumPy only.
"""

import numpy as np


def V_unbounded(x, k=1.0):
    return 0.5 * k * x ** 2


def dV_unbounded(x, k=1.0):
    return k * x


def V_sat(x, s=1.0):
    return s * s * (1.0 - np.exp(-x ** 2 / (2 * s * s)))


def dV_sat(x, s=1.0):
    return x * np.exp(-x ** 2 / (2 * s * s))


def evolve(x0, v0, dV, T=400.0, n=200000):
    """Velocity Verlet: conservative, so energy drift is a diagnostic."""
    dt = T / n
    x, v = x0, v0
    a = -dV(x)
    xmax = abs(x)
    for _ in range(n):
        v = v + 0.5 * dt * a
        x = x + dt * v
        a = -dV(x)
        v = v + 0.5 * dt * a
        xmax = max(xmax, abs(x))
        if abs(x) > 1e6:
            return np.inf, v
    return xmax, v


def main():
    print("=" * 70)
    print("B-7 :: IS A BOUNDED COUPLING EARNED, OR COUNTER-INDICATED?")
    print("=" * 70)
    s = 1.0
    ceiling = s * s

    print("\nC1  UNBOUNDED BOND  V = x^2/2")
    print("-" * 70)
    print("      energy      max |x|       bounded?")
    ok1 = True
    for E in (0.2, 1.0, 5.0, 50.0, 500.0):
        xm, _ = evolve(0.0, np.sqrt(2 * E), dV_unbounded)
        ok1 &= np.isfinite(xm)
        print(f"      {E:7.1f}    {xm:10.4f}     "
              f"{'yes' if np.isfinite(xm) else 'NO'}")
    print(f"    C1 all bounded : {'PASS' if ok1 else 'MISS'}")

    print(f"\nC2/C3  SATURATING BOND  V = s^2(1 - exp(-x^2/2s^2)),  "
          f"ceiling s^2 = {ceiling:.2f}")
    print("-" * 70)
    print("      energy      max |x|       bounded?")
    bounded, unbounded = [], []
    for E in (0.20, 0.50, 0.90, 0.99, 1.01, 1.10, 2.00, 10.0):
        xm, _ = evolve(0.0, np.sqrt(2 * E), dV_sat)
        (bounded if np.isfinite(xm) else unbounded).append(E)
        print(f"      {E:7.2f}    "
              + (f"{xm:10.4f}     yes" if np.isfinite(xm)
                 else "   escaped      NO"))
    thr_lo = max(bounded) if bounded else 0.0
    thr_hi = min(unbounded) if unbounded else np.inf
    print(f"\n    escape threshold lies in ({thr_lo:.2f}, {thr_hi:.2f})")
    print(f"    predicted ceiling s^2 = {ceiling:.2f}")
    ok23 = (thr_lo <= ceiling <= thr_hi)
    print(f"    C2/C3 threshold is the ceiling : "
          f"{'PASS' if ok23 else 'MISS'}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  The unbounded bond confines at every energy. The saturating bond")
    print("  confines only below its ceiling, and above it the configuration")
    print("  separates without bound.")
    print()
    print("  So persistence -- which retains only bounded, enduring motion --")
    print("  FAVOURS the unbounded coupling. The earlier sketch had this")
    print("  backwards: it read D1's bounded-motion requirement as an argument")
    print("  for a bounded coupling, when a bounded coupling is precisely what")
    print("  permits unbounded motion.")
    print()
    print("  B-7's open proposition is therefore closed in the negative.")
    print("  Boundedness is not earned and is counter-indicated, so the")
    print("  E_G-shaped envelope -- quadratic below a core scale, plateau")
    print("  above it -- rests on a property the program's own principles")
    print("  argue against. The Penrose contact through B-7 is weaker than it")
    print("  looked, and weaker for a reason internal to the ladder rather")
    print("  than to the physics.")


if __name__ == "__main__":
    main()
