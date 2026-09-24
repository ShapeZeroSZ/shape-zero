#!/usr/bin/env python3
"""
estimator_calibration.py — which frequency estimator is right?

THE DISPUTE. Two independent implementations of the pinned-asymmetry test agree
on the functional form (A^2 scaling, beta-independence, collapse) and disagree on
the coefficient by a factor of 2.6:

    package  phi_gauge_nonlinear.py : kappa = 0.0305
    independent (DOP853 + phase fit): kappa = 0.082

Ruled out already: timestep (kappa = 0.03109 at DT = 0.02 down to 0.0025, fully
converged), the force law (identical), and the seeding (both travelling, both
offset about PHI).

What differs is the ESTIMATOR:
    package     FFT peak + parabolic sub-bin interpolation, T = 300
    independent weighted complex-phase regression on one Fourier mode, T = 900

THE TEST. Neither estimator can be checked against the other. Both can be checked
against a case whose nonlinear frequency shift is known in closed form.

    Duffing oscillator:  x'' = -w0^2 x - eps x^3
    first-order shift:   w(A) = w0 * (1 + 3 eps A^2 / (8 w0^2))

That is a textbook result. Feed the same signal to both estimators and see which
recovers the known coefficient 3/(8 w0^2).

A quadratic nonlinearity is also tested, since the lattice force is quadratic:
    x'' = -w0^2 x - mu x^2   ->  w(A) = w0 * (1 - 5 mu^2 A^2 / (12 w0^4))
(the classic softening shift; sign is negative, matching the observed centre
softening in the lattice).

PREDICTIONS STATED BEFORE RUNNING
 E1 the phase-regression estimator recovers the Duffing coefficient to within a
    few percent at small A.
 E2 the FFT-peak estimator UNDERESTIMATES the shift, because a finite record with
    a drifting phase biases the peak toward the linear frequency. Predict the
    bias grows as the record shortens.
 E3 the ratio of recovered coefficients at the package's record length is of
    order the disputed factor 2.6 -- if so, the package number is an artifact and
    0.082 is the better value.
 E4 both agree in the limit of a long record, confirming the discrepancy is
    record-length bias and not a coding error.

Python 3 + NumPy only.
"""

import numpy as np


def integrate(w0, coef, power, A, T, dt):
    """x'' = -w0^2 x - coef * x^power, RK4, x(0)=A, v(0)=0."""
    n = int(T / dt)
    x, v = A, 0.0
    rec = np.empty(n)
    def acc(x):
        return -w0 * w0 * x - coef * x ** power
    for i in range(n):
        k1x, k1v = v, acc(x)
        k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x)
        k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x)
        k4x, k4v = v + dt * k3v, acc(x + dt * k3x)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        rec[i] = x
    return rec


def freq_fft(rec, dt):
    """Package-style: FFT peak with parabolic sub-bin interpolation."""
    r = rec - rec.mean()
    w = np.hanning(len(r))
    F = np.abs(np.fft.rfft(r * w))
    fr = np.fft.rfftfreq(len(r), d=dt) * 2 * np.pi
    i = int(np.argmax(F))
    if 0 < i < len(F) - 1:
        d = 0.5 * (F[i - 1] - F[i + 1]) / (F[i - 1] - 2 * F[i] + F[i + 1] + 1e-30)
        return fr[i] + d * (fr[1] - fr[0])
    return fr[i]


def freq_phase(rec, dt, discard=0.08):
    """Independent-style: analytic signal, weighted phase-slope regression."""
    n = len(rec)
    i0 = int(discard * n)
    r = rec[i0:] - rec[i0:].mean()
    t = np.arange(len(r)) * dt
    F = np.fft.rfft(r)
    F[len(F) // 2:] = 0
    an = np.fft.irfft(F, n=len(r)) * 2
    # analytic signal via Hilbert-like construction
    Ff = np.fft.fft(r)
    h = np.zeros(len(r))
    h[0] = 1
    if len(r) % 2 == 0:
        h[len(r) // 2] = 1
        h[1:len(r) // 2] = 2
    else:
        h[1:(len(r) + 1) // 2] = 2
    z = np.fft.ifft(Ff * h)
    amp = np.abs(z)
    ph = np.unwrap(np.angle(z))
    m = amp > 0.08 * amp.mean()
    tt, pp, ww = t[m], ph[m], amp[m] ** 2
    sw = ww.sum()
    swt = (ww * tt).sum()
    swt2 = (ww * tt * tt).sum()
    swp = (ww * pp).sum()
    swtp = (ww * tt * pp).sum()
    den = sw * swt2 - swt * swt
    return abs((sw * swtp - swt * swp) / den)


def main():
    w0 = 2.0581710273          # match the lattice band
    dt = 0.005
    print("=" * 68)
    print("ESTIMATOR CALIBRATION AGAINST KNOWN NONLINEAR SHIFTS")
    print("=" * 68)

    print("\nE1/E2  DUFFING  x'' = -w0^2 x - eps x^3")
    eps = 0.5
    kth = 3 * eps / (8 * w0 * w0)
    print(f"      exact first-order coefficient : {kth:.6f}")
    print("\n        T      A      k_fft      k_phase     fft/phase")
    for T in (300.0, 900.0, 2700.0):
        row = []
        for A in (0.15, 0.30):
            rec = integrate(w0, eps, 3, A, T, dt)
            kf = (freq_fft(rec, dt) - w0) / (w0 * A * A)
            kp = (freq_phase(rec, dt) - w0) / (w0 * A * A)
            row.append((A, kf, kp))
        for A, kf, kp in row:
            print(f"     {T:6.0f}  {A:5.2f}   {kf:8.5f}   {kp:8.5f}    "
                  f"{kf/kp if abs(kp)>1e-12 else float('nan'):6.3f}")

    print("\nE3  QUADRATIC  x'' = -w0^2 x - mu x^2   (the lattice's nonlinearity)")
    mu = 1.0
    kth2 = -5 * mu * mu / (12 * w0 ** 4)
    print(f"      exact first-order coefficient : {kth2:.6f}")
    print("\n        T      A      k_fft      k_phase     fft/phase")
    for T in (300.0, 900.0):
        for A in (0.15, 0.30):
            rec = integrate(w0, mu, 2, A, T, dt)
            kf = (freq_fft(rec, dt) - w0) / (w0 * A * A)
            kp = (freq_phase(rec, dt) - w0) / (w0 * A * A)
            print(f"     {T:6.0f}  {A:5.2f}   {kf:8.5f}   {kp:8.5f}    "
                  f"{kf/kp if abs(kp)>1e-12 else float('nan'):6.3f}")

    print("\n" + "=" * 68)
    print("READING")
    print("=" * 68)
    print("  whichever estimator recovers the exact coefficient is the one to")
    print("  trust for kappa. if the FFT-peak estimator underestimates by a")
    print("  factor near 2.6 at T = 300, the package's 0.0305 is an artifact")
    print("  and 0.082 is the better number.")


if __name__ == "__main__":
    main()
