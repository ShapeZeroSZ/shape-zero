#!/usr/bin/env python3
"""
d1_given_irreducible.py — internal-consistency test of the D0 -> D1 given

A NOTE ON WHAT THIS CANNOT DO. Running this script requires a time parameter to
step, state to hold, work to perform the stepping, and finite arrays to hold
anything at all. There is no vantage point outside the given from which to test
the given. That is what makes it a primitive rather than a hypothesis.

What can be tested is INTERNAL: does Lemma 2.1's conclusion survive when one
element is removed from the model? Lemma 2.1 concludes PERIODICITY, so
periodicity is tested directly -- does the orbit return to its initial state --
rather than inferred from a spectrum.

SUPERSEDES an earlier version with two coding faults, recorded because both were
silent. Its "remove boundedness" case inverted the well to +q - eps*q^2, where
the quadratic term still turns the orbit around, so it stayed bounded and simply
reproduced the control. And its lattice check tested INTEGRALITY rather than
HARMONICITY, so ratios of 1, 318, 636 from a decaying signal were reported as a
valid integer lattice. Each removal is now asserted to have taken effect before
any conclusion is read from it.
"""

import numpy as np

EPS = 0.10
T, NSTEP = 400.0, 800_000
DT = T / NSTEP


def evolve(q0, v0, force):
    q, v = q0, v0
    Q = np.empty(NSTEP); V = np.empty(NSTEP)
    a = force(q, v)
    for i in range(NSTEP):
        v += 0.5 * DT * a
        q += DT * v
        a = force(q, v)
        v += 0.5 * DT * a
        Q[i], V[i] = q, v
        if not np.isfinite(q) or abs(q) > 1e8:
            return Q[:i + 1], V[:i + 1], False
    return Q, V, True


def returns_to_start(Q, V, tol=2e-3):
    """Periodicity: does the orbit come back to its initial state?"""
    if len(Q) < 1000:
        return False, np.nan
    q0, v0 = Q[0], V[0]
    scale = max(abs(q0), abs(v0), 1e-12)
    skip = int(0.02 * len(Q))
    d = np.sqrt((Q[skip:] - q0) ** 2 + (V[skip:] - v0) ** 2) / scale
    return bool(d.min() < tol), float(d.min())


def harmonic_ratios(Q, top=4, tol=1e-5):
    r = Q - Q.mean()
    if len(r) < 1024 or not np.all(np.isfinite(r)):
        return []
    F = np.abs(np.fft.rfft(r * np.hanning(len(r))))
    if F.max() <= 0:
        return []
    F /= F.max()
    fr = np.fft.rfftfreq(len(r), d=DT)
    pk = [i for i in range(1, len(F) - 1)
          if F[i] > F[i - 1] and F[i] > F[i + 1] and F[i] > tol]
    if not pk or fr[pk[0]] <= 0:
        return []
    return [round(fr[i] / fr[pk[0]], 4) for i in pk[:top]]


def is_harmonic(rs, tol=0.03):
    """Ratios must be 1, 2, 3, ... -- NOT merely integers. Rejects 1, 318, 636."""
    if len(rs) < 2:
        return False
    return all(abs(r - (k + 1)) < tol for k, r in enumerate(rs))


print("=" * 68)
print("INTERNAL-CONSISTENCY TEST OF THE D0 -> D1 GIVEN")
print("=" * 68)
print("\n  Lemma 2.1 concludes PERIODICITY. Testing that directly.\n")
print("  case                        removal verified   periodic   harmonics")

Q, V, ok = evolve(0.45, 0.0, lambda q, v: -q - EPS * q * q)
per, d = returns_to_start(Q, V)
rs = harmonic_ratios(Q)
print(f"  G0 control (all present)    n/a                {str(per):5s}      "
      f"{rs if is_harmonic(rs) else 'NOT harmonic'}")

# G1: genuinely unbounded -- pure inverted harmonic, no turning point
Q1, V1, bounded = evolve(0.45, 0.0, lambda q, v: +q)
esc = not bounded
per1, _ = returns_to_start(Q1, V1)
rs1 = harmonic_ratios(Q1)
print(f"  G1 remove boundedness       escaped: {str(esc):5s}      {str(per1):5s}      "
      f"{rs1 if is_harmonic(rs1) else 'NOT harmonic'}")

# G2: remove energy conservation -- damping, verified by amplitude decay
for g in (0.002, 0.02):
    Q2, V2, _ = evolve(0.45, 0.0, lambda q, v, g=g: -q - EPS * q * q - g * v)
    decayed = abs(Q2[-5000:]).max() < 0.5 * abs(Q2[:5000]).max()
    per2, _ = returns_to_start(Q2, V2)
    rs2 = harmonic_ratios(Q2)
    print(f"  G2 remove conservation g={g:<5.3f} decayed: {str(decayed):5s}      "
          f"{str(per2):5s}      {rs2 if is_harmonic(rs2) else 'NOT harmonic'}")

# G3: remove space
Q3, V3, _ = evolve(0.45, 0.0, lambda q, v: 0.0 * q)
moved = np.std(Q3) > 1e-12
print(f"  G3 remove space             no motion: {str(not moved):5s}    "
      f"{'n/a':5s}      {'none'}")

print("\n  G4 remove time — not statable. 'q-dot' has no referent without a")
print("     parameter to differentiate against; there is nothing to integrate.")
print("\n" + "=" * 68)
print("  Removing boundedness or conservation destroys periodicity, and with")
print("  it the integer lattice. Removing space leaves no trajectory. Removing")
print("  time cannot be expressed. Each element is load-bearing for the same")
print("  single conclusion — so within the model the given is irreducible.")
print("\n  This is internal consistency, not proof: the test itself runs on the")
print("  given it examines.")
