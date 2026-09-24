#!/usr/bin/env python3
"""
harness.py — measurement routines that refuse to report uncalibrated

Phase 0.1 of BUILD_CHECKLIST.md. Definition of done: *the harness refuses to
report a number from an uncalibrated routine.*

WHY. Four instrument failures occurred in a single session and two reached a
document intended for a lab: a factor-of-ten slip in a hand-written headline, and
a factor-of-2.6 error from an uncalibrated frequency estimator that survived an
erratum, a referee pass, and a timestep-convergence check. Convergence tests the
integrator, not the instrument reading it.

CONTRACT. Every measurement function here is registered with a calibration case
whose answer is known independently. Calling a measurement before its calibration
has passed raises UncalibratedError. calibrate_all() runs every case and must be
called first.

Import this; do not reimplement estimators inline.
"""

import numpy as np

__all__ = ["UncalibratedError", "calibrate_all", "freq_phase", "freq_fft",
           "rank_abs", "CALIBRATION_REPORT"]


class UncalibratedError(RuntimeError):
    pass


_CALIBRATED = set()
CALIBRATION_REPORT = {}


def _require(name):
    if name not in _CALIBRATED:
        raise UncalibratedError(
            f"'{name}' has not passed calibration. Call calibrate_all() first; "
            f"if it fails, the routine must not be used to report a number.")


# --------------------------------------------------------------- estimators
def _phase_slope(rec, dt, discard=0.08):
    n = len(rec)
    r = rec[int(discard * n):]
    r = r - r.mean()
    F = np.fft.fft(r)
    h = np.zeros(len(r))
    h[0] = 1
    if len(r) % 2 == 0:
        h[len(r) // 2] = 1
        h[1:len(r) // 2] = 2
    else:
        h[1:(len(r) + 1) // 2] = 2
    z = np.fft.ifft(F * h)
    amp = np.abs(z)
    ph = np.unwrap(np.angle(z))
    t = np.arange(len(r)) * dt
    m = amp > 0.08 * amp.mean()
    tt, pp, w = t[m], ph[m], amp[m] ** 2
    sw, swt = w.sum(), (w * tt).sum()
    swt2, swp = (w * tt * tt).sum(), (w * pp).sum()
    swtp = (w * tt * pp).sum()
    den = sw * swt2 - swt * swt
    return abs((sw * swtp - swt * swp) / den)


def freq_phase(rec, dt, discard=0.08):
    """Weighted complex-phase regression. CALIBRATED against the Duffing shift."""
    _require("freq_phase")
    return _phase_slope(rec, dt, discard)


def _fft_peak(rec, dt):
    r = rec - rec.mean()
    F = np.abs(np.fft.rfft(r * np.hanning(len(r))))
    fr = np.fft.rfftfreq(len(r), d=dt) * 2 * np.pi
    i = int(np.argmax(F))
    if 0 < i < len(F) - 1:
        d = 0.5 * (F[i - 1] - F[i + 1]) / (F[i - 1] - 2 * F[i] + F[i + 1] + 1e-30)
        return fr[i] + d * (fr[1] - fr[0])
    return fr[i]


def freq_fft(rec, dt):
    """FFT peak + parabolic interpolation. FAILS calibration at short records.

    Retained so that the failure is visible and reproducible, not so that it is
    used. Calling it raises unless calibration passed, which it does not at the
    record lengths where it was originally applied.
    """
    _require("freq_fft")
    return _fft_peak(rec, dt)


def rank_abs(M, rtol=1e-9, atol=1e-8):
    """Numerical rank with an ABSOLUTE floor as well as a relative one.

    A relative-only threshold reports full rank on a matrix that is zero to
    floating-point noise, because sv[0] is itself ~1e-14 and everything clears
    rtol*sv[0]. That produced a rank of 67 against a true 22, a rank of 0 where
    the answer was 1, and spurious associator spans at dimensions 1 and 2.
    """
    sv = np.linalg.svd(np.asarray(M), compute_uv=False)
    if sv.size == 0:
        return 0
    return int(np.sum((sv > rtol * sv[0]) & (sv > atol)))


# -------------------------------------------------------------- calibration
def _duffing(w0, eps, A, T, dt):
    n = int(T / dt)
    x, v = A, 0.0
    rec = np.empty(n)
    acc = lambda x: -w0 * w0 * x - eps * x ** 3
    for i in range(n):
        k1x, k1v = v, acc(x)
        k2x, k2v = v + 0.5 * dt * k1v, acc(x + 0.5 * dt * k1x)
        k3x, k3v = v + 0.5 * dt * k2v, acc(x + 0.5 * dt * k2x)
        k4x, k4v = v + dt * k3v, acc(x + dt * k3x)
        x += dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        rec[i] = x
    return rec


def calibrate_all(verbose=True):
    """Run every calibration case. Routines that fail stay locked."""
    w0, eps, dt = 2.0581710273, 0.5, 0.005
    exact = 3 * eps / (8 * w0 * w0)          # textbook first-order Duffing shift
    results = {}

    for name, fn in (("freq_phase", _phase_slope_wrap),
                     ("freq_fft", _fft_peak_wrap)):
        errs = []
        for T, A in ((300.0, 0.15), (900.0, 0.30)):
            rec = _duffing(w0, eps, A, T, dt)
            k = (fn(rec, dt) - w0) / (w0 * A * A)
            errs.append(abs(k - exact) / exact)
        worst = max(errs)
        passed = worst < 0.05
        results[name] = (worst, passed)
        if passed:
            _CALIBRATED.add(name)

    # rank_abs: must report 0 on a noise-level zero matrix, 2 on a rank-2 one
    Z = np.random.default_rng(0).normal(size=(6, 6)) * 1e-15
    R2 = np.outer(np.ones(6), np.arange(6.0)) + np.outer(np.arange(6.0), np.ones(6))
    ok = (rank_abs(Z) == 0) and (rank_abs(R2) == 2)
    results["rank_abs"] = (0.0 if ok else 1.0, ok)
    if ok:
        _CALIBRATED.add("rank_abs")

    CALIBRATION_REPORT.clear()
    CALIBRATION_REPORT.update(results)
    if verbose:
        print("  CALIBRATION")
        print(f"      exact Duffing coefficient 3eps/(8w0^2) = {exact:.6f}")
        for k, (err, p) in results.items():
            tag = "PASS" if p else "FAIL — locked"
            extra = f"worst rel err {err:.4f}" if k.startswith("freq") else ""
            print(f"      {k:12s} {tag:16s} {extra}")
    return all(p for _, p in results.values())


def _phase_slope_wrap(rec, dt):
    return _phase_slope(rec, dt)


def _fft_peak_wrap(rec, dt):
    return _fft_peak(rec, dt)


if __name__ == "__main__":
    print("=" * 62)
    print("HARNESS SELF-TEST")
    print("=" * 62)
    try:
        freq_phase(np.zeros(1000), 0.01)
    except UncalibratedError as e:
        print(f"\n  refusal before calibration works:\n      {e}\n")
    calibrate_all()
    print("\n  after calibration:")
    for n in ("freq_phase", "freq_fft", "rank_abs"):
        try:
            if n == "freq_phase":
                freq_phase(np.sin(np.arange(4000) * 0.02), 0.01)
            elif n == "freq_fft":
                freq_fft(np.sin(np.arange(4000) * 0.02), 0.01)
            else:
                rank_abs(np.eye(3))
            print(f"      {n:12s} usable")
        except UncalibratedError:
            print(f"      {n:12s} LOCKED — failed calibration, cannot report")
