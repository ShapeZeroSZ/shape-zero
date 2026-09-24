# The Pinned Asymmetry — A Zero-Parameter Bench Test

**One measurable claim, no fitted parameters, sharp pass/fail.** This is the
only falsifiable prediction in the Shape Zero program that requires neither
cosmology nor a signature it does not have. It has been in the package since
v5.1 and is verified in its own numerics.

---

## 1. The claim

A one-dimensional array of nonlinear oscillators with **antisymmetric
velocity coupling** develops a synthetic gauge field. The band becomes
asymmetric in wavenumber, and the asymmetry is

**Δω(k) ≡ ω(k) − ω(−k) = −2 c β sin(k)**

- **c** — nearest-neighbour elastic coupling
- **β** — antisymmetric velocity-coupling ratio (the gyroscopic term)
- **k** — wavenumber

All three are fixed by **linear spectroscopy of the same platform**. Nothing is
fitted to the nonlinear regime.

## 2. Why it is a test and not a fit

The prediction is an **invariance**, not a curve.

From the exact dispersion root ω² + 2cβ sin(k)ω − W² = 0, the asymmetry is
−2cβ sin(k) **for any W²**. The on-site nonlinearity is direction-blind, so it
shifts W² symmetrically and **cancels in the difference**. Therefore:

| quantity | behaviour with drive amplitude |
|---|---|
| band centre ½(ω(k) + ω(−k)) | **softens** — a real, visible nonlinear effect |
| asymmetry ω(k) − ω(−k) | **pinned** — must not move |

So the experiment varies amplitude and watches whether the centre moves while
the asymmetry does not. One knob, two observables, opposite predicted responses.
A fit cannot produce that; only the mechanism can.

**Conditions versus parameters:** the asymmetry is one prediction with zero free
parameters, checked across a continuum of amplitudes. Conditions exceed
parameters by the number of amplitudes measured, which is the criterion that
separates evidence from fitting.

## 3. Numerical verification

`phi_gauge_nonlinear.py`, N = 64, c = 1.0, k = π/2, γ = 0, over a 400-fold
amplitude range:

| β | A | ω(+k) | ω(−k) | centre | **Δω** |
|---|---|---|---|---|---|
| 0.00 | 0.001 | 2.05817 | 2.05817 | 2.05817 | **0.00000** |
| 0.00 | 0.400 | 2.04199 | 2.04199 | 2.04199 | **0.00000** |
| 0.05 | 0.001 | 2.00878 | 2.10878 | 2.05878 | **−0.10000** |
| 0.05 | 0.100 | 2.00779 | 2.10782 | 2.05780 | **−0.10003** |
| 0.05 | 0.200 | 2.00479 | 2.10491 | 2.05485 | **−0.10012** |
| 0.05 | 0.400 | 1.99237 | 2.09287 | 2.04262 | **−0.10050** |

Theory: −2cβ sin(π/2) = **−0.1000**.

- **Centre softens 0.8%** across the amplitude range — the nonlinearity is real
  and visible.
- **Asymmetry drifts 0.5%**, from −0.10000 to −0.10050.
- **β = 0 gives exactly 0.00000** at every amplitude — the control fires.

## 3b. The correction is measured: β·A², with a pure-number coefficient

The script was written to look for a β·A² correction. It is there, and it has
been measured.

**Scaling in amplitude.** Residual divided by A², over the valid domain:

| A | residual / A² |
|---|---|
| 0.1 | −0.002978 |
| 0.2 | −0.003035 |
| 0.3 | −0.003061 |
| 0.4 | −0.003109 |

Constant to **4%** — the correction is A², confirmed.

**Scaling in β.** The coefficient divided by β, over a tenfold range:

| β | coefficient | coeff/β |
|---|---|---|
| 0.02 | −0.001229 | −0.06146 |
| 0.05 | −0.003046 | −0.06091 |
| 0.10 | −0.006302 | −0.06302 |
| 0.20 | −0.011783 | −0.05891 |

Constant to **3%**. So the correction is **proportional to β**, and the
coefficient is a **pure number** fixed by c and k alone.

**The full prediction:**

**Δω(k, A) = −2 c β sin(k) · [1 + κ A²]**   (k = π/2, c = 1)

**⚠ κ IS GEOMETRY-DEPENDENT. The value 0.0799 is for a 1D CHAIN.**

| geometry | Δω (theory 0.100000) | κ |
|---|---|---|
| **1D chain** | 0.100003 | **0.0799** |
| 3D, transverse width 2 | 0.100001 | **0.0168** |
| 3D, transverse width 3 | 0.100001 | **0.0311** |

**The pinning survives every geometry** — Δω tracks theory to 10⁻⁵ or better in
all cases, and *that* is the invariance the experiment tests. **The A² correction
does not**: κ falls with transverse extent and depends on the beam profile,
because the correction comes from the packet's self-interaction and a
transversely-spreading packet dilutes the amplitude driving it.

**κ = 0.0799 is the plane-wave value AT UNIT ON-SITE STIFFNESS.** It changes with
transverse geometry **and** with base stiffness — measured κ = 0.0959, 0.0799,
0.0677 at stiffness 0.90, 1.00, 1.10, a ~35% swing over ±10%. **The pinning
remains protected against both** (null to 1.2×10⁻⁵ under stiffness), and that is
the invariance the experiment rests on.

**For an experimentalist:** on a 1D array at unit stiffness, κ = 0.0799. On anything with
transverse extent, κ is smaller and must be measured for that profile. The
**collapse protocol is unaffected** — the β-independence of the normalised drift
is what makes it a test, and that holds regardless.

*Measured with* `pinned_asymmetry_reference.py`, which now carries
`set_base(q, side)` and `TRANSVERSE_WIDTH`.

**⚠ VACUITY TRAP, documented in that file.** With `TRANSVERSE_WIDTH = None` the
seed is a transverse-**uniform** plane wave, whose transverse Laplacian is
identically zero — so a q = 3 run reduces **exactly** to q = 1 and the outputs are
**bit-identical**. That is not a q = 3 measurement. Bit-identical results from a
supposedly different configuration mean the same code path ran twice.

**Value pinned by the β-sweep, not the amplitude sweep.** The reference
implementation (`04_scripts/session/pinned_asymmetry_reference.py`, DOP853,
rtol 10⁻⁹, T = 900, weighted phase regression) gives |Δ/Δ₀| = 1.007227, 1.007190,
1.007262, 1.007051 at β = 0.02, 0.05, 0.10, 0.20 and A = 0.30. Dividing the drift
by A² = 0.09 gives **κ = 0.0799 at every β** — β-independent to four digits.

The amplitude sweep alone gives a range 0.078–0.082; the β-sweep is tighter
because the collapse is exact. **0.0799 is the value to quote.**

**SECOND CORRECTION — the coefficient is 0.082, not 0.0305.** An independent
reimplementation (DOP853, rtol 10⁻⁹, T = 900, weighted complex-phase regression)
obtained 0.082 where the package script gives 0.0305. The difference is **not**
timestep (κ = 0.03109 at DT from 0.02 to 0.0025, fully converged), **not** the
force law (identical), and **not** the seeding (both travelling, both offset
about φ). It is the **frequency estimator**.

*Calibration against known answers* (`estimator_calibration.py`):

| test case | exact coefficient | phase regression | FFT peak, T = 300 |
|---|---|---|---|
| Duffing, A = 0.15 | 0.044263 | **0.04439** | **0.02220** — off by 2.0× |
| Duffing, A = 0.30 | 0.044263 | 0.04437 | 0.05001 |
| quadratic, A = 0.15 | −0.023220 | −0.02459 | −0.04475 — off by 1.8× |

The phase estimator recovers the textbook Duffing shift to **0.4%** at every
record length. The FFT-peak estimator used by the package is biased by factors of
1.8–2.0 at T = 300, in **both** directions depending on the case, converging only
as the record lengthens (0.0222 → 0.0404 → 0.0416 for T = 300 → 900 → 2700).

The disputed factor of 2.6 lies within that measured bias. **The package's 0.0305
is an artifact of FFT-peak estimation on a short record.**

**This line is emitted by `pinned_asymmetry_headline.py`, not written by hand.**
That script recomputes every headline number from the lattice via the package's
own `phi_gauge_nonlinear.py`, confirms the coefficient by two independent routes,
and carries a guard against decimal slips.

| route | coefficient | estimator |
|---|---|---|
| package amplitude sweep | 0.0305 ± 0.0005 | FFT peak, T = 300 — **biased** |
| package coupling sweep | 0.0306 ± 0.0007 | FFT peak, T = 300 — **biased** |
| reference, amplitude sweep | 0.078–0.082 | phase regression, T = 900 |
| reference, **β-sweep** | **0.0799** (four digits) | phase regression, T = 900 |
| **adopted** | **0.0799** | β-sweep, calibrated to 0.4% on Duffing |

*The package sweeps are internally consistent because they share one biased
estimator. Internal consistency is not accuracy — both routes used the same
instrument.*

**Measured domain limit.** The coefficient holds near 0.0305 through A = 0.5,
drifts to 0.0345 at A = 0.7 and 0.0418 at A = 0.9, and departs at A = 1.1
(0.0865) as the wave leaves the φ well. **Valid for A below ≈ 0.9** — wider than
the ≈ 0.5 estimated earlier, which was read off the integrator overflow rather
than the onset of departure.

**ERRATUM.** An earlier version gave 0.30 — a factor of ten too large, from
dividing the measured coefficient by −0.2 instead of the leading term −2. It
would have predicted 4.8% drift at A = 0.4 where 0.5% is measured, and an
experimentalist following it would have reported a spurious failure.

**The tables were always right; only the hand-written summary was wrong.** The
process fix, now applied: *headline formulas are emitted by the script that
produces the table, never written in prose afterwards.* `pinned_asymmetry_headline.py`
implements that and prints what a factor-ten slip would read, so this class of
error is caught at source.

**Why the β-proportionality is the sharp part.** The nonlinearity does not shift
the asymmetry independently — it *rescales* it. So the **normalised** drift
Δω(A)/Δω(0) has **no β dependence at all**. Measure it at one coupling strength
and it must be identical at every other.

**Better protocol, therefore:** sweep amplitude at two or three values of β and
check the normalised curves **collapse onto one**. Collapse is a pass;
separation is a fail; nothing is fitted anywhere. An invariance across a second
knob is far harder to reproduce accidentally than a single curve.

**Domain limit.** The on-site potential −(x² − x − 1) is unbounded below, so
waves escape the φ well at large drive. Numerically the integrator overflows at
**A = 1.2**, and residual/A² has already drifted 70% by A = 1.0. **Useful
amplitudes stop below A ≈ 0.5.** That limit is itself a checkable feature: the
escape threshold is set by the well depth, which linear spectroscopy also
fixes.

## 3c. Resolution requirements, and what each buys

Two claims live in this test and they need different precision. Stating both, so
an experiment is not set up to see one while reporting on the other.

| claim | required relative resolution on Δω | why |
|---|---|---|
| **asymmetry equals −2cβ sin(k)** | ~10⁻³ | the leading term is O(1) against the centre shift |
| **asymmetry stays pinned under amplitude** | ~10⁻⁴ | the drift is 0.5% at A = 0.4 |
| **the A² law, and its coefficient 0.0305** | **~10⁻⁵** | at A = 0.1 the correction is only 3×10⁻⁴ of the leading term |
| **collapse across β** | ~10⁻⁴ | the normalised curves differ by less than the drift itself |

**Worked from the numbers:** the correction is 0.0305·A² in relative terms. At
A = 0.4 that is 4.9×10⁻³ — visible at 10⁻⁴ resolution. At A = 0.1 it is
3.1×10⁻⁴, and at A = 0.05 it is 7.6×10⁻⁵. So mapping the *scaling* over a decade
of amplitude requires an order of magnitude better resolution than confirming the
*pinning*.

**This has been observed in practice.** An independent reimplementation of this
model reported Δω = −0.10015 at A = 0.001, where the nonlinear correction should
be 3×10⁻⁸ — so that 1.5×10⁻⁴ deviation is estimator noise, and it is the same
size as their A = 0.2 point (−0.10006). Their run resolves the pinning and does
**not** resolve the A² law. That is the expected outcome at ~10⁻⁴ and it is worth
knowing in advance rather than discovering mid-experiment.

**A systematic that does not matter.** The same reimplementation found absolute
frequencies 0.0015–0.006 above ours, from residual second-harmonic mixing — and
reproduced the asymmetry anyway. That is the mechanism working: Δω is a
*difference*, so any common-mode offset cancels. **Errors affecting both
propagation directions equally drop out of the observable by construction**,
which is precisely why this test is robust to the systematics a real apparatus
will have.

## 4. Experimental realisation

Any platform with tunable antisymmetric velocity coupling on a 1D chain:

- coupled mechanical oscillators with gyroscopic elements (spinning rotors,
  Coriolis coupling)
- photonic lattices with synthetic gauge fields
- coupled electrical resonators with non-reciprocal elements
- optomechanical arrays

**Procedure.**
1. Measure the linear band; extract **c** from the dispersion width.
2. Measure the linear asymmetry at one k; extract **β**. *All inputs now fixed.*
3. Drive at k = π/2 and at −k, sweeping amplitude over at least two decades.
4. Track band centre and asymmetry independently.

**Pass:** centre softens, asymmetry pinned to within the β·A² correction.
**Fail:** asymmetry moves with amplitude beyond that, or does not equal
−2cβ sin(k) at low amplitude.

## 5. What a pass and a fail would mean

**A pass** confirms the one chain in the program with no free parameters:
passivity forces the coupling matrix skew, hence **u(2)**, hence a synthetic
U(1) with a pinned asymmetry. That is a derivation from a stated principle to a
measured number.

**A fail** breaks it at a located point. If the asymmetry is not
−2cβ sin(k) at low amplitude, the u(2) derivation is wrong. If it drifts with
amplitude beyond the A² term, the direction-blindness of the nonlinearity is
wrong. Either is informative; neither is absorbable by re-fitting.

## 6. Why this and not the rest

Most of the program derives **structure that already exists** rather than
numbers that are missing — u(2), the colour decomposition 1 ⊕ 3 ⊕ 3̄,
sin²θ_W = 3/8. Those are consistency checks against known physics, not
predictions.

And the routes to genuinely new numbers are closed, provably: all five selection
principles are dimensionless, so no output carries a unit; and every spectrum on
a compact homogeneous space is a quadratic tower k(k+n−1), which measured
spectra are not — lepton m² ratios are 1, 4.3×10⁴, 1.2×10⁷, and Regge
trajectories grow linearly where these grow quadratically.

**This is the exception.** It is dimensionless — a ratio — so the theorem does
not forbid it. It is a lattice, so no signature obstruction applies. It is
zero-parameter once the linear measurement is made. And it can fail.

---

*Script: `phi_gauge_nonlinear.py` (C1 package). Related: `phi_gauge_test.py`,
`phi_gauge_wilson.py`, `phi_gauge_precession.py`. Spec: v5.1 §5.2.7, and
Sections 6–9 for the passivity → u(2) derivation.*
