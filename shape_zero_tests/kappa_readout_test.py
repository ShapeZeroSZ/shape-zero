#!/usr/bin/env python3
"""
kappa_readout_test.py — is kappa's side-dependence physics or a readout artifact?

SELF-CONTAINED. No imports from the ark. Paste into Colab and run.
Needs: numpy, scipy. Runtime ~10-25 min depending on the machine.

SEEDING FIX (2026-09-24). The gyro term here is beta*c*(v[n-1] - v[n+1]) -- the
reference convention (MODEL_SPEC §5), where +k takes the UPPER root. The
original seeded +k with the LOWER root and -k with the upper: the swap of
PROVENANCE §6o. Its plane-wave control accordingly gave +0.0799, the retracted
value. And for a transversely localised packet a single carrier frequency is
wrong even at the right root: each transverse component has its own W^2(q).
seed_velocity() now builds v0 in Fourier space, one branch frequency per
wavevector. `python3 kappa_readout_test.py --swapped-seed` reproduces the
original run (output: kappa_readout_test_swapped.txt; fixed run:
kappa_readout_test_output.txt). All kappa figures quoted below this note -- the side table, the
0.0797 / ~0.080 control -- were measured with the swapped seed: RETRACTED.

================================================================================
THE QUESTION
================================================================================
Measured kappa at fixed transverse width w = 2.0, periodic BC:

    side      8       12       16       24       32
    kappa   0.0175  0.0113  0.0082  0.0027  0.0018     [swapped seed: RETRACTED]

Still falling at side 32 -- a tenfold drop, not converging. Meanwhile the
PLANE-WAVE control holds at 0.0797 at every side.

HYPOTHESIS (H1, readout artifact). The readout takes a UNIFORM transverse
average before the FFT. A fixed-width packet on a growing transverse domain
occupies a shrinking fraction of the averaged sites, so the measured mode
amplitude falls with side. kappa multiplies A^2, so an artificially small
amplitude yields an artificially small kappa. The plane wave is immune because
it fills the plane at every side -- which is exactly why its control holds.

ALTERNATIVE (H2, physics). The nonlinear correction genuinely weakens for a
transversely localised packet, and the side-dependence reflects a real
approach to an isolated-beam limit.

================================================================================
WHAT THIS SCRIPT MEASURES
================================================================================
 T1  the packet's transverse FILL FRACTION at each side -- the quantity H1 says
     is doing the damage. Prediction under H1: kappa is roughly proportional to
     it; under H2 there is no such relation.

 T2  kappa under THREE readouts at each side:
       (a) uniform transverse average   -- the current method
       (b) intensity-WEIGHTED average   -- weights each transverse site by its
                                           own power, so the packet's own
                                           profile sets the weighting
       (c) single centre line           -- no transverse averaging at all
     Under H1, (b) and (c) should be side-INDEPENDENT while (a) falls.
     Under H2, all three fall together.

 T3  plane-wave control at each side and each readout. Must stay ~-0.018
     everywhere (reference: -0.0184 +/- 0.00033; was ~0.080 with the swap). If a readout breaks the plane wave, that readout is wrong and
     its localised numbers cannot be used.

 T4  the linear PINNING at each side and readout: |d|/|theory| must be
     1.000 +- 0.001. This has held in every configuration so far; if it breaks,
     something more basic is wrong and the kappa numbers are moot.

================================================================================
READ THE OUTPUT LIKE THIS
================================================================================
  * T3 fails for a readout          -> that readout is invalid, ignore its rows
  * (b) and (c) flat, (a) falls     -> H1 CONFIRMED, readout artifact.
                                       Fix: weight the readout by intensity.
                                       The kappa(w) curve must be re-measured.
  * all three fall together          -> H2, the side-dependence is physical, and
                                       the open question becomes what sets the
                                       isolated-beam limit
  * (b)/(c) flat but disagree with (a) in VALUE only, not trend
                                     -> partial artifact; report both

================================================================================
NOTES
================================================================================
 - side must be DIVISIBLE BY 4, or FFT mode side//4 is not k = pi/2. Side 10
   gave kappa = -0.497 from this alone. The script asserts this.
 - periodic transverse BC only. An open-BC plane wave is not a plane wave (it
   has edges) and fails its own control at 0.604 vs 0.080 -- do not use open BC
   for these measurements.
 - kappa is extracted as ( |d(A)| / |d_theory| - 1 ) / A^2 at A = 0.30, matching
   the reference implementation.
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------------- parameters
PHI = (1.0 + np.sqrt(5.0)) / 2.0
C = 1.0
K = np.pi / 2
BETA = 0.05
A_LIN = 0.02          # linear amplitude, for the pinning check
A_NL = 0.30           # nonlinear amplitude, for kappa
T_RUN = 300.0
RTOL = 1e-9
ATOL = 1e-12

SIDES = [8, 12, 16, 24, 32]
W_TRANSVERSE = 2.0
# --swapped-seed reproduces the original (retracted) run; see SEEDING FIX above.
SWAPPED_SEED = "--swapped-seed" in sys.argv


def build(side, width, plane_wave=False):
    """Seed a q=3 lattice. Returns (shape, x0, coord, env)."""
    assert side % 4 == 0, "side must be divisible by 4 (FFT mode side//4 = k=pi/2)"
    shape = (side, side, side)
    c = np.indices(shape).astype(float)
    coord = c[0].reshape(-1)
    if plane_wave:
        env = np.ones(side ** 3)
    else:
        e = np.ones(shape)
        for a in (1, 2):
            d = c[a] - side / 2.0
            d = (d + side / 2) % side - side / 2          # periodic distance
            e *= np.exp(-0.5 * (d / width) ** 2)
        env = e.reshape(-1)
    return shape, coord, env


def branch_omega(q0, w2, beta):
    """Positive root of  w^2 - 2*beta*c*sin(q0)*w - W^2(q) = 0.

    This lattice's gyro term is beta*c*(v[n-1] - v[n+1]) along axis 0 -- the
    REFERENCE convention (MODEL_SPEC §5) -- so a wave moving toward +axis 0
    (q0 > 0) takes the UPPER root at k = +pi/2.
    """
    b = beta * C * np.sin(q0)
    return b + np.sqrt(b * b + w2)


def seed_velocity(shape, u0, direction, beta):
    """Initial velocity that makes u0 a pure travelling packet in `direction`.

    Built in Fourier space: every wavevector component q of u0 gets its OWN
    branch frequency. A component with sigma = direction * sign(sin q0) = +1
    evolves as exp(-i w(q) t); its conjugate partner (sigma = -1) as
    exp(+i w(-q) t). So  v_hat(q) = -i * sigma * w(sigma q) * u_hat(q),  with
    W^2(q) = sqrt5 + 2c * sum_a (1 - cos q_a) including the transverse axes.
    A single carrier frequency is right only for the q_perp = 0 component; for
    a transversely localised packet it seeds the other components partly in
    the counter-propagating branch.
    """
    U = np.fft.fftn(u0.reshape(shape))
    q = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n) for n in shape], indexing="ij")
    w2 = np.sqrt(5.0) + 2 * C * sum(1 - np.cos(qa) for qa in q)
    sigma = direction * np.sign(np.round(np.sin(q[0]), 12))
    sigma[sigma == 0] = direction          # q0 = 0 or pi: no direction; u0 has none there
    V = -1j * sigma * branch_omega(sigma * q[0], w2, beta) * U
    v = np.fft.ifftn(V)
    return np.real(v).reshape(-1)


def counter_fraction(shape, u0, v0, direction, beta):
    """Norm of the counter-propagating branch in (u0, v0) over the intended one.

    Per mode, u = a exp(-i w1 t) + b exp(+i w2 t) with w1 the intended branch
    and w2 the other; returns ||b|| / ||a||. Zero for a perfect seed.
    """
    U = np.fft.fftn(u0.reshape(shape)); V = np.fft.fftn(v0.reshape(shape))
    q = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(n) for n in shape], indexing="ij")
    w2 = np.sqrt(5.0) + 2 * C * sum(1 - np.cos(qa) for qa in q)
    sigma = direction * np.sign(np.round(np.sin(q[0]), 12))
    sigma[sigma == 0] = direction
    w_1 = branch_omega(sigma * q[0], w2, beta)
    w_2 = branch_omega(-sigma * q[0], w2, beta)
    # sigma = +1 modes: u = a e^{-i w1 t} + b e^{+i w2 t}; sigma = -1 is the mirror
    b = (sigma * V + 1j * w_1 * U) / (1j * (w_1 + w_2))
    a = U - b
    return float(np.sqrt((np.abs(b) ** 2).sum() / (np.abs(a) ** 2).sum()))


def eom_factory(shape, beta):
    side = shape[0]
    N = side ** 3

    def lap(x):
        y = x.reshape(shape)
        out = np.zeros_like(y)
        for ax in range(3):
            out += np.roll(y, -1, axis=ax) + np.roll(y, 1, axis=ax) - 2 * y
        return out.reshape(-1)

    def dir0(x):
        y = x.reshape(shape)
        return (np.roll(y, 1, axis=0) - np.roll(y, -1, axis=0)).reshape(-1)

    def eom(t, z):
        x = z[:N]; v = z[N:]
        return np.concatenate([v,
                               -(x * x - x - 1.0) + C * lap(x) + beta * C * dir0(v)])
    return eom


def readout(Y, shape, mode, kind):
    """Y is (N, ntime). Returns the complex mode amplitude time series.

    kind = 'uniform'   : plain transverse mean  (the current method)
           'weighted'  : intensity-weighted transverse mean
           'centre'    : single transverse line through the middle
    """
    side = shape[0]
    Z = Y.reshape(shape + (Y.shape[1],))
    if kind == 'uniform':
        prof = Z.mean(axis=(1, 2))
    elif kind == 'weighted':
        p = (Z[..., 0] - PHI) ** 2                    # initial transverse power
        wgt = p.sum(axis=0)                           # (side, side)
        s = wgt.sum()
        if s <= 0:
            prof = Z.mean(axis=(1, 2))
        else:
            prof = np.einsum('abct,bc->at', Z, wgt) / s
    elif kind == 'centre':
        m = side // 2
        prof = Z[:, m, m, :]
    else:
        raise ValueError(kind)
    F = np.fft.fft(prof, axis=0) / side
    return F[mode]


def omega_of(mode_series, t):
    """Phase-regression frequency of a complex mode series."""
    ph = np.unwrap(np.angle(mode_series))
    w = np.abs(mode_series)
    good = w > 0.05 * w.max()
    if good.sum() < 10:
        return np.nan, np.nan
    tt = t[good]; pp = ph[good]; ww = w[good]
    Aм = np.vstack([tt, np.ones_like(tt)]).T
    Wм = np.diag(ww)
    sol, *_ = np.linalg.lstsq(Wм @ Aм, Wм @ pp, rcond=None)
    resid = float(np.sqrt(np.mean((Aм @ sol - pp) ** 2)))
    return abs(sol[0]), resid


def run_one(side, width, amp, plane_wave, kinds):
    shape, coord, env = build(side, width, plane_wave)
    N = side ** 3
    mode = side // 4
    out = {}
    freqs = {}
    for sign in (+1, -1):
        # travelling wave: x = A cos(Kx -/+ wt). At t=0 both directions share
        # x0 = A cos(Kx); the DIRECTION is carried by v0 alone.
        # (cos is even, so putting the sign inside the cosine gives the SAME
        #  wave and both runs collapse to one -- that bug produced a pinning
        #  ratio of 5.7e-05 in testing.)
        x0 = PHI + amp * env * np.cos(K * coord)
        if SWAPPED_SEED:
            # the ORIGINAL seed, kept only to reproduce the retracted numbers:
            # one carrier, +k at the LOWER root -- the §6o swap
            B = 2 * C * BETA * np.sin(K)
            w2 = np.sqrt(5.0) + 2 * C * (1 - np.cos(K))
            wd = ((-B if sign > 0 else B) + np.sqrt(B * B + 4 * w2)) / 2
            v0 = sign * amp * env * wd * np.sin(K * coord)
        else:
            v0 = seed_velocity(shape, x0 - PHI, sign, BETA)
        sol = solve_ivp(eom_factory(shape, BETA), (0, T_RUN),
                        np.concatenate([x0, v0]), method='DOP853',
                        rtol=RTOL, atol=ATOL,
                        t_eval=np.linspace(0, T_RUN, 3000))
        for kind in kinds:
            # both directions are read at the SAME +k mode; they differ by the
            # sign of v0, hence by branch frequency, not by mode index.
            m = readout(sol.y[:N], shape, mode, kind)
            w_, r_ = omega_of(m, sol.t)
            freqs.setdefault(kind, {})[sign] = (w_, r_)
    for kind in kinds:
        wp, rp = freqs[kind][+1]
        wm, rm = freqs[kind][-1]
        out[kind] = (wp - wm, max(rp, rm))
    return out


def main():
    kinds = ['uniform', 'weighted', 'centre']
    theory = -2 * C * BETA * np.sin(K)
    print("=" * 78)
    print("KAPPA READOUT TEST — is the side-dependence an artifact?")
    print("=" * 78)
    print(f"  theory linear asymmetry |d| = {abs(theory):.6f}")
    print(f"  transverse width w = {W_TRANSVERSE}, periodic BC, T = {T_RUN}")
    print("  seed: " + ("ORIGINAL, swapped roots, one carrier -- RETRACTED numbers"
                        if SWAPPED_SEED else
                        "Fourier space, own branch frequency per wavevector"))
    print()

    # ---- T1 fill fraction -------------------------------------------------
    print("T1  TRANSVERSE FILL FRACTION (what H1 says drives the damage)")
    print("     side   fill fraction   (packet power / uniform power)")
    fills = {}
    for side in SIDES:
        _, _, env = build(side, W_TRANSVERSE, False)
        e = env.reshape((side, side, side))[0]
        fills[side] = float((e ** 2).sum() / (side * side))
        shape, coord, env3 = build(side, W_TRANSVERSE, False)
        u0 = A_NL * env3 * np.cos(K * coord)
        cf = max(counter_fraction(shape, u0, seed_velocity(shape, u0, d, BETA), d, BETA)
                 for d in (+1, -1))
        if not SWAPPED_SEED:
            print(f"     {side:4d}   {fills[side]:.5f}        seed counter-branch {cf:.1e}")
        else:
            print(f"     {side:4d}   {fills[side]:.5f}")
    print()

    # ---- T2/T3/T4 ---------------------------------------------------------
    for label, pw in (("LOCALISED PACKET", False), ("PLANE-WAVE CONTROL", True)):
        print("=" * 78)
        print(f"{label}")
        print("=" * 78)
        print("     side   readout     |d|/|th| (A=0.02)   kappa (A=0.30)   resid")
        for side in SIDES:
            lin = run_one(side, W_TRANSVERSE, A_LIN, pw, kinds)
            nl = run_one(side, W_TRANSVERSE, A_NL, pw, kinds)
            for kind in kinds:
                dl, rl = lin[kind]
                dn, rn = nl[kind]
                ratio = abs(dl) / abs(theory)
                kap = (abs(dn) / abs(theory) - 1.0) / (A_NL ** 2)
                flag = ""
                if not np.isfinite(ratio) or abs(ratio - 1.0) > 1e-3:
                    flag = "  <- PINNING FAILS"
                print(f"     {side:4d}   {kind:9s}   {ratio:.6f}          "
                      f"{kap:+.5f}      {max(rl,rn):.3f}{flag}")
            print()
        print()

    print("=" * 78)
    print("HOW TO READ IT")
    print("=" * 78)
    print("  * plane-wave kappa must be ~-0.018 for a readout to be valid")
    print("    (pinned_asymmetry_reference.py: -0.0184 +/- 0.00033).")
    print("    a readout that breaks the plane wave is wrong; ignore its rows.")
    print("  * if 'weighted' and 'centre' are FLAT across side while 'uniform'")
    print("    falls  ->  H1 CONFIRMED, readout artifact. Fix the readout and")
    print("    re-measure kappa(w). The kappa(w) formula must not be published.")
    print("  * if all three fall together  ->  H2, the side-dependence is")
    print("    physical; the question becomes what sets the isolated-beam limit.")
    print("  * pinning |d|/|th| must be 1.000 +- 0.001 everywhere. If it breaks,")
    print("    something more basic is wrong and kappa is moot.")


if __name__ == "__main__":
    main()
