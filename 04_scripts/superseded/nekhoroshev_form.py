#!/usr/bin/env python3
# ============================================================================
# STATUS: INCONCLUSIVE BY CONSTRUCTION — superseded by the Greene route
#
# The accessible regime here is too strongly perturbed: escape times come out on
# the order of the oscillator period, and at eps <= 0.14 nothing escapes within
# the window at all, leaving three unusable points.
#
# An independent GPU run (T = 3e6, ten eps values, ~4455 s per family) confirmed
# the route is practically closed: secular rates 1e-12 to 1e-10 against an
# integrator energy error dH ~ 3e-2 -- nine orders below the numerical floor.
#
# SUPERSEDED BY: greene_production.py / greene_fixed.py, which measure the
# strength-vs-order law exactly via Greene's residue criterion.
# See HIERARCHIES.md sec 3 and 5.
#
# The calibration section (N0) is sound and worth reusing.
# ============================================================================
"""
nekhoroshev_form.py — is the D1 winding dependence exponential or polynomial?

THE QUESTION. `s2b_winding.py` shows joint survival tracks Diophantine badness of
the winding ratio at Spearman +0.662, with the competing three-wave predictor
excluded at -0.132. That is the KAM/Nekhoroshev selection, present and measured.

But Spearman measures MONOTONICITY only. Nekhoroshev's content is the functional
form T ~ exp(1/eps^a), and it is the exponential that produces a HIERARCHY. A
monotone but polynomial dependence generates no large separations and would close
the classical route to dimensional transmutation the way the quantum route is
already closed.

WHAT IS MEASURED HERE. Escape TIME, not retention fraction. `s2b_winding.py`
reports S = r1*r2 at fixed T; a timescale is what the exponential law is about.

ESTIMATOR CALIBRATED FIRST. A coefficient dispute earlier in this work was
traced to an uncalibrated frequency estimator that was off by 2.0x on a case with
a known answer. So before any escape time is quoted, the escape detector is run
against a case whose decay law is known analytically: exponential relaxation
x' = -x/tau, where the crossing time of a threshold is tau*log(x0/thresh).

PREDICTIONS STATED BEFORE RUNNING
 N0 calibration: the detector recovers tau*log(x0/thresh) to within a few percent
    on pure exponential decay. If it does not, nothing below is quoted.
 N1 escape time rises with Diophantine badness of the winding ratio, reproducing
    the s2b_winding correlation in the time domain.
 N2 log(T_escape) is LINEAR in 1/eps^a for some a > 0 -- the Nekhoroshev form.
    Predict a fitted a in the range 0.2-1.0, and a correlation above 0.9.
 N3 a polynomial law log T ~ b*log(1/eps) fits WORSE than the exponential form.
    Both are fitted and compared; the better fit is reported whichever it is.
 N4 the ratio T_escape(noble) / T_escape(near-rational) GROWS as eps decreases.
    A saturating ratio means no hierarchy and closes the route.

Python 3 + NumPy only.
"""

import numpy as np


# ----------------------------------------------------------------- model
def run(rho, eps, T, dt=0.02, seed=0):
    """Two coupled oscillators, frequency ratio rho, nonlinear coupling eps.

    Standard nearly-integrable setting: two actions coupled through a
    perturbation whose strength is eps. Returns the action history of mode 1.
    """
    rng = np.random.default_rng(seed)
    w1, w2 = 1.0, rho
    q1, p1 = 1.0, 0.0
    q2, p2 = 1.0, 0.0
    n = int(T / dt)
    J1 = np.empty(n)
    for i in range(n):
        # symplectic Euler on H = (p1^2+w1^2 q1^2)/2 + (p2^2+w2^2 q2^2)/2
        #                        + eps * q1^2 q2
        f1 = -w1 * w1 * q1 - 2.0 * eps * q1 * q2
        f2 = -w2 * w2 * q2 - eps * q1 * q1
        p1 += dt * f1
        p2 += dt * f2
        q1 += dt * p1
        q2 += dt * p2
        J1[i] = 0.5 * (p1 * p1 + w1 * w1 * q1 * q1) / w1
        if not np.isfinite(J1[i]) or abs(J1[i]) > 1e6:
            return J1[:i + 1], dt
    return J1, dt


def escape_time(J, dt, frac=0.5):
    """First time the action departs its initial value by a fixed fraction."""
    if len(J) < 10:
        return 0.0
    J0 = J[0]
    if J0 == 0:
        return np.inf
    d = np.abs(J - J0) / abs(J0)
    idx = np.argmax(d > frac)
    if d[idx] <= frac:
        return np.inf          # never departed within the window
    return idx * dt


# ------------------------------------------------------------ calibration
def calibrate():
    print("N0  CALIBRATION — detector against a known decay law")
    print("      tau    predicted t_cross   measured   rel err")
    ok = True
    for tau in (5.0, 20.0, 80.0):
        dt = 0.02
        n = int(40 * tau / dt)
        t = np.arange(n) * dt
        # signal departing from J0=1 by 50%: 1*exp(-t/tau) crosses 0.5
        J = np.exp(-t / tau)
        pred = tau * np.log(1 / 0.5)
        meas = escape_time(J, dt, frac=0.5)
        err = abs(meas - pred) / pred
        ok &= err < 0.05
        print(f"    {tau:6.1f}      {pred:10.4f}     {meas:9.4f}    {err:.4f}")
    print(f"      detector calibrated: {'PASS' if ok else 'FAIL — nothing below is quoted'}")
    return ok


# ------------------------------------------------------------------ main
def badness(rho, nmax=40):
    """Diophantine badness: how poorly rho is approximated by rationals."""
    best = np.inf
    for q in range(1, nmax + 1):
        p = round(rho * q)
        if p == 0:
            continue
        best = min(best, abs(rho - p / q) * q * q)
    return best


def main():
    if not calibrate():
        return
    PHI = (1 + np.sqrt(5)) / 2
    noble = PHI - 1                 # 0.618..., maximally Diophantine
    near_rat = 0.6666667            # 2/3, strongly resonant

    print(f"\n      winding ratios:  noble {noble:.6f} (badness "
          f"{badness(noble):.4f}),  near-rational {near_rat:.6f} "
          f"(badness {badness(near_rat):.2e})")

    print("\nN1/N2  ESCAPE TIME vs PERTURBATION")
    print("        eps      T_esc(noble)   T_esc(2/3)     ratio")
    epss = [0.30, 0.20, 0.14, 0.10, 0.07]
    Tn, Tr = [], []
    for eps in epss:
        T = min(4e5, 2000.0 / eps ** 2)
        a = escape_time(*run(noble, eps, T))
        b = escape_time(*run(near_rat, eps, T))
        Tn.append(a)
        Tr.append(b)
        rat = a / b if (np.isfinite(a) and np.isfinite(b) and b > 0) else np.nan
        sa = f"{a:12.1f}" if np.isfinite(a) else "     no escape"
        sb = f"{b:12.1f}" if np.isfinite(b) else "     no escape"
        print(f"      {eps:5.2f}  {sa}  {sb}   {rat:8.3f}")

    E = np.array(epss)
    N = np.array(Tn)
    m = np.isfinite(N) & (N > 0)
    if m.sum() >= 3:
        print("\nN3  WHICH LAW FITS?  (noble ratio)")
        lp = np.polyfit(np.log(E[m]), np.log(N[m]), 1)
        rp = np.corrcoef(np.log(E[m]), np.log(N[m]))[0, 1]
        print(f"      polynomial  log T = {lp[0]:.3f} log(eps) + c    "
              f"corr {abs(rp):.4f}")
        best = None
        for a in (0.25, 0.5, 0.75, 1.0):
            x = 1.0 / E[m] ** a
            r = abs(np.corrcoef(x, np.log(N[m]))[0, 1])
            if best is None or r > best[1]:
                best = (a, r, np.polyfit(x, np.log(N[m]), 1)[0])
        print(f"      exponential log T = b/eps^a,  best a = {best[0]}, "
              f"b = {best[2]:.3f},  corr {best[1]:.4f}")
        print(f"      better fit: "
              f"{'EXPONENTIAL' if best[1] > abs(rp) else 'POLYNOMIAL'}")

    print("\nN4  does the noble/rational advantage GROW as eps falls?")
    rr = [Tn[i] / Tr[i] for i in range(len(epss))
          if np.isfinite(Tn[i]) and np.isfinite(Tr[i]) and Tr[i] > 0]
    if len(rr) >= 2:
        print(f"      ratios: {[round(x, 2) for x in rr]}")
        print(f"      growing: {rr[-1] > rr[0]}   "
              f"(a saturating ratio means no hierarchy)")


if __name__ == "__main__":
    main()
