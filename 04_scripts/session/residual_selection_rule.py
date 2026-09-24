#!/usr/bin/env python3
# ============================================================================
# STATUS: RESULT VALID AND IN THE MODEL — this file's own lattice is not
#
# The lattice and frequency extraction written here returned 0.000000 for every
# case INCLUDING the baseline, which should have been -0.100. The instrument was
# dead, not the effect.
#
# The valid result -- odd-sector residual couplings violate the synthetic-U(1)
# identity with NO threshold (r_* = 0), even-sector couplings are unconstrained
# -- was obtained by re-running against the package integrator
# (phi_gauge_nonlinear.py run_wave / mode_freq), not this code.
#
# The RESULT is a model component (MODEL_SPEC sec 4b): odd-sector residual
# couplings are forbidden outright (r_* = 0, no threshold, stable under a 40x
# even background); the even sector is unconstrained at any magnitude and is
# therefore the dynamical one.
#
# Kept for its stated protocol and predictions. Do not run it as-is; reproduce
# against phi_gauge_nonlinear.py run_wave / mode_freq as the result was.
# ============================================================================
"""
residual_selection_rule.py — does a residual coupling break the pinned asymmetry?

TASK. Take one forced identity, promote the operator to a residual-dependent
family whose value at r = 0 is the verified expression, and locate the first
surface on which the identity fails. If failure occurs for arbitrarily small
coupling, the selection rule is a THEOREM for that operator. If it occurs only
beyond some threshold, that threshold is a concrete r_*.

THE IDENTITY. Synthetic U(1) on a 1D chain with antisymmetric velocity coupling:

    omega^2 + 2 c beta sin(k) omega - W^2 = 0     =>     d_omega = -2 c beta sin(k)

exactly, for ANY W^2. The on-site nonlinearity shifts W^2 but is DIRECTION-BLIND,
so it cancels in the difference. Verified: band centre softens 0.8% over a
400-fold amplitude range while the asymmetry stays pinned to 0.5%, that residual
itself being a measured beta*A^2 correction.

THE PROMOTION. A residual-dependent term either preserves direction-blindness or
breaks it. Two families, each reducing to the verified expression at eps = 0:

  EVEN   f_even = eps * x_n^3                       direction-blind
  ODD    f_odd  = eps * (x_{n+1} - x_{n-1}) * x_n^2  direction-sensitive

Analytically, if the residual sends W^2 -> W^2 +/- delta then

    d_omega = -2 c beta s + [sqrt(X+delta) - sqrt(X-delta)] ~ -2 c beta s + delta/sqrt(X)

linear in delta with NO threshold. Tested here on the full nonlinear lattice.

PREDICTIONS STATED BEFORE RUNNING
 R1 EVEN coupling: d_omega unchanged at every eps, to numerical precision. The
    identity is insensitive to direction-blind residuals at any magnitude.
 R2 ODD coupling: d_omega shifts LINEARLY in eps with no threshold -- fitted
    exponent 1.0, and the shift already resolvable at the smallest eps tested.
 R3 therefore r_* = 0 for the odd sector: the selection rule is a theorem, not a
    stability boundary. Any non-zero direction-odd residual coupling violates the
    forced identity immediately.
 R4 and the rule is VACUOUS for the even sector -- no constraint at any size.
    The correct statement is a rule on the odd part only.

Python 3 + NumPy only.
"""

import numpy as np

N = 64
C = 1.0
K = np.pi / 2
DT = 0.01
STEPS = 24000


def run(amp, beta, direction, eps, mode):
    n = np.arange(N)
    phase = direction * K * n
    x = amp * np.cos(phase)
    v = np.zeros(N)

    def force(x, v):
        lap = np.roll(x, 1) + np.roll(x, -1) - 2 * x
        f = C * lap - (2 * x - 1.0)          # on-site phi-well linearisation
        f = f - beta * C * (np.roll(v, -1) - np.roll(v, 1))
        if eps != 0.0:
            if mode == "even":
                f = f - eps * x ** 3
            else:
                f = f - eps * (np.roll(x, -1) - np.roll(x, 1)) * x ** 2
        return f

    rec = np.zeros(STEPS)
    a = force(x, v)
    for s in range(STEPS):
        v = v + 0.5 * DT * a
        x = x + DT * v
        a = force(x, v)
        v = v + 0.5 * DT * a
        rec[s] = np.sum(x * np.cos(phase)) / N
    return rec


def freq(rec):
    r = rec - rec.mean()
    w = np.hanning(len(r))
    F = np.abs(np.fft.rfft(r * w))
    fr = np.fft.rfftfreq(len(r), d=DT) * 2 * np.pi
    i = int(np.argmax(F))
    if 0 < i < len(F) - 1:
        d = 0.5 * (F[i - 1] - F[i + 1]) / (F[i - 1] - 2 * F[i] + F[i + 1] + 1e-30)
        return fr[i] + d * (fr[1] - fr[0])
    return fr[i]


def dw(beta, eps, mode, amp=0.15):
    return freq(run(amp, beta, +1, eps, mode)) - freq(run(amp, beta, -1, eps, mode))


def main():
    B = 0.05
    base = dw(B, 0.0, "even")
    print("=" * 68)
    print("RESIDUAL SELECTION RULE FOR THE SYNTHETIC U(1) CANCELLATION")
    print("=" * 68)
    print(f"\n  verified identity at eps = 0 : d_omega = {base:+.6f}")
    print(f"  theory -2 c beta sin(k)      : {-2*C*B*np.sin(K):+.6f}")

    for mode, tag in (("even", "R1  DIRECTION-BLIND residual  (eps * x^3)"),
                      ("odd", "R2  DIRECTION-ODD residual  (eps * (x_+ - x_-) x^2)")):
        print(f"\n{tag}")
        print("      eps        d_omega        shift from eps=0")
        rows = []
        for e in (0.0, 0.001, 0.01, 0.05, 0.2):
            d = dw(B, e, mode)
            rows.append((e, d - base))
            print(f"    {e:6.3f}   {d:+.6f}     {d-base:+.3e}")
        E = np.array([e for e, _ in rows[1:]])
        S = np.array([abs(s) for _, s in rows[1:]])
        if np.all(S > 1e-9):
            p = np.polyfit(np.log(E), np.log(S), 1)[0]
            print(f"      fitted exponent in eps : {p:.3f}")
        else:
            print("      shift below numerical resolution at every eps")

    print("\n" + "=" * 68)
    print("READING")
    print("=" * 68)
    print("  If the odd shift is linear with no threshold, r_* = 0 and the")
    print("  selection rule is a THEOREM for that sector -- any non-zero")
    print("  direction-odd residual coupling violates the forced identity.")
    print("  If the even shift vanishes at all eps, the rule is VACUOUS there,")
    print("  and the correct statement constrains the odd part only.")


if __name__ == "__main__":
    main()
