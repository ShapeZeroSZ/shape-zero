#!/usr/bin/env python3
"""
d8_angle_readout.py -- the D8 frequency ratio against the a-b angle: closed form vs the
Fourier-peak readout (2026-09-26; INPUT_LEDGER sec 3.2).

The generic D8 flow psi' = psi a + b psi (04_scripts/rungs/z1_d8_plurality.py) is linear,
M = R_a + L_b antisymmetric; for unit imaginary a, b at angle theta its angular
frequencies are {0, 2 sin(theta/2), 2}, so the frequency ratio is exactly 1/sin(theta/2).
The recorded sweep ([1.04, 23.9] over 5-150 deg, "phi near 76 deg") has no script in the
repository; its method was almost certainly z1_d8_plurality.py's Fourier-peak readout
(Hanning window, peaks above 2% of max) at T = 600, resolution 1/600. This script
reproduces that readout -- same integrator, same T, N, spectrum() and peaks() -- at
theta = 5, 76.3 and 150 deg and compares it with the closed form and the nearest grid bin.

usage:  python3 d8_angle_readout.py
"""
import importlib.util
import os

import numpy as np

RUNGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "rungs")
spec = importlib.util.spec_from_file_location("pl", os.path.join(RUNGS, "z1_d8_plurality.py"))
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
fl = pl.fl

T, N = 600.0, 240000


def main():
    E = fl.oct_table(fl.ch.oriented_lines())
    rng = np.random.default_rng(719)
    p0 = rng.normal(size=8); p0 /= np.linalg.norm(p0)
    a = np.zeros(8); a[1:] = rng.normal(size=7); a /= np.linalg.norm(a)
    e = np.zeros(8); e[1:] = rng.normal(size=7)
    e -= (e @ a) * a; e /= np.linalg.norm(e)
    print(f"Fourier-peak readout, T = {T:.0f}, N = {N}, resolution {1 / T:.5f} cycles per unit time")
    print(f"{'theta':>7} {'exact f_low':>11} {'exact ratio':>11} {'eig ratio':>9} {'bin ratio':>9} "
          f"{'readout peaks (cycles/unit time)':>34} {'readout ratio':>13}")
    for th in (5.0, 76.3, 150.0):
        t = np.radians(th)
        b = np.cos(t) * a + np.sin(t) * e
        M = np.array([fl.mul(x, a, E) + fl.mul(b, x, E) for x in np.eye(8)]).T
        ev = np.unique(np.round(np.abs(np.linalg.eigvals(M).imag), 9))
        ev = ev[ev > 1e-9]
        f_lo = 2 * np.sin(t / 2) / (2 * np.pi)
        f_hi = 2 / (2 * np.pi)
        fr, P = pl.spectrum(fl.integrate(p0, lambda p: fl.mul(p, a, E) + fl.mul(b, p, E), T=T, n=N), T)
        df = fr[1]
        bin_ratio = (round(f_hi / df) * df) / (round(f_lo / df) * df)
        pk = pl.peaks(fr, P)
        rr = max(pk) / min(pk) if len(pk) >= 2 else float("nan")
        print(f"{th:7.1f} {f_lo:11.5f} {1 / np.sin(t / 2):11.4f} {ev.max() / ev.min():9.4f} {bin_ratio:9.4f} "
              f"{str([round(float(x), 5) for x in pk]):>34} {rr:13.4f}")


if __name__ == "__main__":
    main()
