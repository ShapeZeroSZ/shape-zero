#!/usr/bin/env python3
"""
pinned_asymmetry_headline.py — the headline, emitted not transcribed

Phase 0.2 of BUILD_CHECKLIST.md: no number appears in prose that was not emitted
by a script.

HISTORY, kept because both errors reached a lab-facing document.
  (1) An earlier version stated a bracket coefficient of 0.30 -- a factor of ten
      too large, from dividing the measured coefficient by -0.2 instead of the
      leading term -2. The tables were right; the hand-written summary was not.
  (2) The corrected 0.0305 was ALSO wrong, by a further factor of ~2.6, because
      it came from an FFT-peak frequency estimator biased 1.8-2.0x at short
      records. Timestep convergence was checked and passed -- convergence tests
      the integrator, not the instrument reading it.

  (3) The next version printed 0.044 +/- 0.015 (never quoted): phi_gauge_nonlinear.py
      seeded both directions at the beta = 0 frequency, and the coefficient
      averaged drift/A^2 over A = 0.1-0.4, where at A <= 0.2 the drift is not
      resolved. With own-root seeding (PROVENANCE §6o) that average reads
      -0.006 +/- 0.016 -- the unresolved amplitudes still dominate it.

This version imports harness.py, which locks freq_fft as uncalibrated and
provides freq_phase, calibrated to 0.3% against the exact Duffing shift. It
prints every amplitude, with a record-length check, and fits the coefficient
only from A >= 0.3, where the drift exceeds the record-length sensitivity.
"""

import os
import sys
import importlib.util
import numpy as np
import harness

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "platform", "phi_gauge_nonlinear.py")
FIT_MIN_A = 0.30      # drift resolved from here up; see the record-length column


def load_lattice():
    spec = importlib.util.spec_from_file_location("ng", PATH)
    ng = importlib.util.module_from_spec(spec)
    keep = sys.stdout
    sys.stdout = open("/dev/null", "w")
    try:
        spec.loader.exec_module(ng)
    except SystemExit:
        pass
    finally:
        sys.stdout.close()
        sys.stdout = keep
    return ng


def dw(ng, A, beta):
    """Asymmetry via the CALIBRATED estimator, not the package's FFT peak.

    Returns d_omega from the full record and from its first half: how far the
    reading moves when the record is halved bounds what the record length
    lets it resolve.
    """
    rp = ng.run_wave(A, beta, +1)
    rn = ng.run_wave(A, beta, -1)
    n = np.arange(ng.N)
    proj_p = rp @ np.exp(-1j * ng.K * n) / ng.N
    proj_n = rn @ np.exp(+1j * ng.K * n) / ng.N

    def read(sl):
        wp = harness.freq_phase(np.real(proj_p[sl] * np.conj(proj_p[0])), ng.DT)
        wn = harness.freq_phase(np.real(proj_n[sl] * np.conj(proj_n[0])), ng.DT)
        return wp - wn

    return read(slice(None)), read(slice(0, len(proj_p) // 2))


def main():
    print("=" * 66)
    print("HEADLINE — EMITTED, NOT TRANSCRIBED")
    print("=" * 66)
    print()
    if not harness.calibrate_all():
        print("\n  one or more routines failed calibration; locked ones are unusable")
    print()

    ng = load_lattice()
    c, k, B = ng.C, ng.K, 0.05
    lead = -2 * c * B * np.sin(k)
    print(f"  lattice: c = {c}, k = {k/np.pi:.3f} pi, N = {ng.N}, DT = {ng.DT}")
    print(f"  seeding: each direction at its own linear root (PROVENANCE §6o)")
    print(f"  leading term -2 c beta sin(k) = {lead:+.6f}\n")

    print("  amplitude sweep, beta = 0.05, calibrated phase estimator")
    print("  drift = (|d_omega| - |lead|) / |lead|;  T/2 check = |drift(T) - drift(T/2)|")
    print("        A     d_omega      drift      drift/A^2   T/2 check   resolved")
    rows = []
    for A in (0.10, 0.20, 0.30, 0.40):
        d, d_half = dw(ng, A, B)
        rel = (abs(d) - abs(lead)) / abs(lead)
        chk = abs(abs(d) - abs(d_half)) / abs(lead)
        ok = abs(rel) > 2 * chk
        rows.append((A, rel, chk, ok))
        print(f"      {A:5.2f}  {d:+.6f}   {rel:+.6f}   {rel/A**2:+.4f}     "
              f"{chk:.6f}    {'yes' if ok else 'NO'}")

    fit = [(A, rel) for A, rel, _, _ in rows if A >= FIT_MIN_A]
    a2 = np.array([A ** 2 for A, _ in fit])
    r = np.array([rel for _, rel in fit])
    coef = float(np.sum(a2 * r) / np.sum(a2 * a2))
    spread = float(np.ptp(r / a2))
    print(f"\n      fitted from A >= {FIT_MIN_A:.2f} only. Below that the drift is not")
    print("      resolved: halving the record moves the reading by as much as the")
    print("      drift itself (resolved = |drift| > 2 x T/2 check).")
    agree = all(ok == (A >= FIT_MIN_A) for A, _, _, ok in rows)
    print(f"      resolution flags {'agree' if agree else 'DISAGREE'} with the A >= "
          f"{FIT_MIN_A:.2f} cut{'' if agree else ' -- CHECK before quoting'}")
    print(f"\n      coefficient = {coef:+.4f}   (drift/A^2 spread over the fit: {spread:.4f})")
    print(f"      [factor-ten guard: a slip would read {10*coef:+.3f}; "
          f"|coef| below 0.5 => {'OK' if abs(coef) < 0.5 else 'CHECK'}]")

    sign = "+" if coef >= 0 else "-"
    print("\n" + "=" * 66)
    print("HEADLINE, AS THE DOCUMENT MUST QUOTE IT")
    print("=" * 66)
    print(f"\n    d_omega(k, A) = -2 c beta sin(k) * [ 1 {sign} {abs(coef):.3f} A^2 ]\n")
    print("    estimator: freq_phase, calibrated to 0.3% on the Duffing shift")
    print("    record: T = 300; the drift is resolved only for A >= 0.3 at this")
    print("    length. pinned_asymmetry_reference.py (T = 900) is the reference")
    print("    for the coefficient and its beta-independence.")


if __name__ == "__main__":
    main()
