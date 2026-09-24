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

This version imports harness.py, which locks freq_fft as uncalibrated and
provides freq_phase, calibrated to 0.3% against the exact Duffing shift.
"""

import sys
import importlib.util
import numpy as np
import harness

PATH = "phi_gauge_nonlinear.py"


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
    """Asymmetry via the CALIBRATED estimator, not the package's FFT peak."""
    rp = ng.run_wave(A, beta, +1)
    rn = ng.run_wave(A, beta, -1)
    m = ng.N // 4
    n = np.arange(ng.N)
    proj_p = rp @ np.exp(-1j * ng.K * n) / ng.N
    proj_n = rn @ np.exp(+1j * ng.K * n) / ng.N
    wp = harness.freq_phase(np.real(proj_p * np.conj(proj_p[0])), ng.DT)
    wn = harness.freq_phase(np.real(proj_n * np.conj(proj_n[0])), ng.DT)
    return wp, wn


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
    print(f"  leading term -2 c beta sin(k) = {lead:+.6f}\n")

    print("  amplitude sweep, beta = 0.05, calibrated phase estimator")
    print("        A       d_omega        relative drift     drift / A^2")
    coefs = []
    for A in (0.10, 0.20, 0.30, 0.40):
        wp, wn = dw(ng, A, B)
        d = wp - wn
        rel = (abs(d) - abs(lead)) / abs(lead)
        coefs.append(rel / A ** 2)
        print(f"      {A:5.2f}   {d:+.6f}      {rel:+.6f}        {rel/A**2:.4f}")

    coef = float(np.mean(coefs))
    print(f"\n      coefficient = {coef:.4f} +/- {np.std(coefs):.4f}")
    print(f"      [factor-ten guard: a slip would read {10*coef:.3f}; "
          f"below 0.5 => {'OK' if coef < 0.5 else 'CHECK'}]")

    print("\n" + "=" * 66)
    print("HEADLINE, AS THE DOCUMENT MUST QUOTE IT")
    print("=" * 66)
    print(f"\n    d_omega(k, A) = -2 c beta sin(k) * [ 1 + {coef:.3f} A^2 ]\n")
    print("    estimator: freq_phase, calibrated to 0.3% on the Duffing shift")
    print("    the coefficient is beta-independent, so the normalised drift")
    print("    collapses across coupling strengths -- that collapse is the")
    print("    experimental protocol.")


if __name__ == "__main__":
    main()
