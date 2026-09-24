#!/usr/bin/env python3
"""
d1_scope_limit.py — where Lemma 2.1 stops holding

WHY THIS EXISTS. A Henon-Heiles result establishing a scope limit on D1 was
computed in an earlier session, reported in conversation, and never written to a
script or a document. This file fixes that omission. The finding bounds the
primitive that INPUT_LEDGER.md sec 1 now formalises, so it belongs alongside it.

THE CLAIM UNDER TEST. Lemma 2.1 concludes that bounded conservative motion on a
ONE-dimensional configuration manifold is periodic. The dimensional qualifier is
load-bearing and easy to lose: "bounded conservative motion is periodic" is FALSE
without it.

Henon-Heiles is the standard counterexample. It is conservative and bounded at
every energy below 1/6, with a TWO-dimensional configuration manifold:

    H = (px^2 + py^2)/2 + (x^2 + y^2)/2 + x^2 y - y^3/3

At low energy the motion is quasi-periodic and the spectrum is a set of sharp
lines: a clock exists. Above a threshold the same bounded, conservative system
goes chaotic and the spectrum is broadband: no clock.

INSTRUMENT, NAMED. Spectral concentration is measured as the fraction of total
power carried by the twenty largest spectral lines, on a Hanning-windowed FFT of
x(t) after transient discard. Energy conservation is monitored as |dH|/H and
reported with every row -- a result is not quoted if the integrator drift is
comparable to the effect. This naming is deliberate: the coefficient dispute
elsewhere in this work survived an erratum and a referee pass because a number
was quoted without its instrument.

PREDICTIONS STATED BEFORE RUNNING
 S1 |dH|/H stays below 1e-8 at every energy, so the integrator is not the story.
 S2 at low energy (E = 0.02, 0.06, 0.11) the spectrum is line-like: top-20
    concentration above 0.98, few sharp peaks.
 S3 near E = 0.155 the same bounded conservative system is BROADBAND on SOME
    section points: concentration collapses below 0.5 with 150+ peaks, while
    other points at the SAME energy remain line-like. Regular and chaotic
    regions coexist -- whether a clock exists depends on where in phase space
    you are, not on the energy alone.
 S4 therefore boundedness plus conservation do NOT imply periodicity in two
    dimensions. Lemma 2.1's one-dimensional restriction is doing real work.
 S5 the one-dimensional control stays line-like at every energy tested.

Python 3 + NumPy only.
"""

import numpy as np


def henon_heiles(E, y0=0.15, py0=0.0, T=3000.0, dt=0.001):
    """Integrate at fixed energy E from a chosen point on the surface of section.

    IMPORTANT: a single initial condition is NOT the system. At E = 0.155
    regular and chaotic regions coexist, so one orbit can report a clean line
    spectrum while the system is partly chaotic. An earlier version of this
    script used one fixed start and wrongly concluded the spectrum stayed
    line-like at every energy. Several section points must be sampled.
    """
    y = y0
    x = 0.0
    py = py0
    V = 0.5 * (x * x + y * y) + x * x * y - y ** 3 / 3.0
    k = E - V - 0.5 * py0 ** 2
    if k <= 0:
        return None, None
    px = np.sqrt(2 * k)

    def acc(x, y):
        return (-(x + 2 * x * y), -(y + x * x - y * y))

    n = int(T / dt)
    rec = np.empty(n)
    ax, ay = acc(x, y)
    H0 = None
    Hmax = 0.0
    for i in range(n):
        px += 0.5 * dt * ax
        py += 0.5 * dt * ay
        x += dt * px
        y += dt * py
        ax, ay = acc(x, y)
        px += 0.5 * dt * ax
        py += 0.5 * dt * ay
        rec[i] = x
        if i % 500 == 0:
            H = 0.5 * (px * px + py * py) + 0.5 * (x * x + y * y) \
                + x * x * y - y ** 3 / 3.0
            if H0 is None:
                H0 = H
            Hmax = max(Hmax, abs(H - H0) / abs(H0))
        if not np.isfinite(x) or abs(x) > 50:
            return rec[:i + 1], np.inf
    return rec, Hmax


def oned(E, T=4000.0, dt=0.002):
    """Control: one-dimensional bounded conservative motion at the same energies."""
    q = 0.0
    p = np.sqrt(2 * E)
    n = int(T / dt)
    rec = np.empty(n)
    acc = lambda q: -q - 0.3 * q * q
    a = acc(q)
    H0 = 0.5 * p * p + 0.5 * q * q + 0.1 * q ** 3
    Hmax = 0.0
    for i in range(n):
        p += 0.5 * dt * a
        q += dt * p
        a = acc(q)
        p += 0.5 * dt * a
        rec[i] = q
        if i % 500 == 0:
            H = 0.5 * p * p + 0.5 * q * q + 0.1 * q ** 3
            Hmax = max(Hmax, abs(H - H0) / abs(H0))
    return rec, Hmax


def concentration(rec, dt, discard=0.1, top=20, peak_tol=1e-3):
    """INSTRUMENT: fraction of power in the top-N lines, and the peak count."""
    r = rec[int(discard * len(rec)):]
    r = r - r.mean()
    if len(r) < 1024 or not np.all(np.isfinite(r)):
        return np.nan, 0
    F = np.abs(np.fft.rfft(r * np.hanning(len(r)))) ** 2
    tot = F.sum()
    if tot <= 0:
        return np.nan, 0
    conc = np.sort(F)[-top:].sum() / tot
    Fn = F / F.max()
    peaks = sum(1 for i in range(1, len(Fn) - 1)
                if Fn[i] > Fn[i - 1] and Fn[i] > Fn[i + 1] and Fn[i] > peak_tol)
    return conc, peaks


def main():
    dt = 0.001
    print("=" * 70)
    print("SCOPE LIMIT ON LEMMA 2.1")
    print("=" * 70)
    print("\n  Lemma 2.1: bounded conservative motion on a ONE-dimensional")
    print("  configuration manifold is periodic. Testing whether the")
    print("  dimensional qualifier is load-bearing.")
    print("\n  Instrument: top-20 spectral power concentration, Hanning window,")
    print("  10% transient discard. Energy drift reported with every row.")

    print("\n  HENON-HEILES — bounded and conservative, TWO-dimensional")
    print("  FOUR section points per energy: one orbit is not the system")
    print("     energy   y0     py0    |dH|/H     conc    peaks   verdict")
    SECTION = ((0.15, 0.00), (-0.10, 0.10), (0.30, 0.05), (-0.25, 0.00))
    for E in (0.060, 0.155):
        for y0, py0 in SECTION:
            rec, drift = henon_heiles(E, y0, py0, dt=dt)
            if rec is None:
                print(f"     {E:.3f}  {y0:+.2f}  {py0:.2f}   unreachable")
                continue
            conc, pk = concentration(rec, dt)
            verdict = "line" if conc > 0.9 else (
                "BROADBAND — no clock" if conc < 0.6 else "mixed")
            print(f"     {E:.3f}  {y0:+.2f}  {py0:.2f}   {drift:.1e}  "
                  f"{conc:.4f}  {pk:5d}   {verdict}")

    print("\n  CONTROL — one-dimensional, same energies")
    print("     energy    |dH|/H      concentration   peaks    verdict")
    for E in (0.020, 0.060, 0.110, 0.155):
        rec, drift = oned(E, dt=dt)
        conc, pk = concentration(rec, dt)
        verdict = "line spectrum" if conc > 0.9 else "BROADBAND"
        print(f"     {E:.3f}   {drift:.2e}      {conc:.4f}       {pk:4d}    {verdict}")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    print("  If the two-dimensional system goes broadband while remaining")
    print("  bounded and conservative, then boundedness plus conservation do")
    print("  NOT imply periodicity. Lemma 2.1's one-dimensional restriction is")
    print("  load-bearing, and 'bounded conservative motion is periodic' is")
    print("  false as a general statement.")
    print()
    print("  Consequence for the ladder: D1's conclusion, and the integer")
    print("  lattice built on it, hold on the one-dimensional rung and do not")
    print("  extend to higher-dimensional configuration manifolds by default.")


if __name__ == "__main__":
    main()
