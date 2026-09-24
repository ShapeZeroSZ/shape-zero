# Shape Zero — Experimental Predictions (v1, addendum to spec v5.3)

Status: frozen spec v5.3 is the derivation record; this addendum states what
it predicts for physical systems outside the simulations that produced it.
Every prediction below is zero-fit: all constants are computed from linear
or weakly-nonlinear measurements of the platform itself. Universality is
established (spec, S2 section): none of these predictions depends on the
golden-ratio potential — they hold for any lattice in the model class.

**Model class.** Chains (or quasi-1D lattices) of local oscillators with
(i) a quadratic on-site nonlinearity, (ii) nearest-neighbour elastic
coupling, and (iii) a directional velocity coupling c·W(v_{n+1} − v_{n−1})
— scalar W = β for P-1, matrix W for P-2/P-3. Candidate platforms:
gyroscopic mechanical metamaterials (spinner/gyro lattices), nonlinear
electrical transmission lines with gyrator elements, magnetoacoustic and
coupled-pendulum arrays. The coupling realization is the engineering step;
the predictions apply to any faithful realization.

**Scripts:** P-1 `phi_gauge_test.py`, `phi_gauge_delta.py`, `phi_gauge_decaymap.py`,
`s2_universality.py`; P-2 `phi_gauge_chiral.py`; P-3 `phi_gauge_precession.py`.

---

## P-1. The nonreciprocal decay window, located without fitting

**Calibration measurements (weak drive):**
1. Linear band ω(k) and the asymmetry slope: Δω = ω(k) − ω(−k) = −2cβ sin k.
   This law is stiffness-free — it directly yields the gauge strength cβ.
2. The self-softening coefficient at one wavenumber: Δω_self = s·A².
   Via the closed form s = −(A²-normalized) α²(4/K + 2/L₂)/(2ω), this
   yields the nonlinear coefficient α with no other unknowns.

**Predictions (strong drive), all computed from the above:**
- A forward-propagating carrier at k₀ decays into the product pair
  (q*, 2k₀ − q*) minimizing the detuning Δ = 2ω(k₀) − ω(q) − ω(2k₀ − q),
  inside a window in the gauge strength centred at
  **β_res(A) = β₀ + κ·A², with κ > 0** — the window drifts UP with drive
  amplitude, because the product modes' cross-softening in the pump field
  (DC stiffness shift plus mixing sidebands) overcompensates the pump's
  own softening. κ is computed from the calibrated (ω(k), α, c) alone.
- The reverse-propagating carrier at identical parameters is protected
  throughout the window (retention ≈ 1 while the forward wave loses tens
  of percent).

**Worked instances (simulation-verified):** φ-lattice: β_res = 0.062 +
0.064A², deepest measured retention 0.61; generic control lattice (K = 2,
α = 0.7): predicted centre 0.0754 at A = 0.4, deepest measured retention
0.528 at β = 0.074; reverse direction 1.000 / 1.002 respectively.

> **Note on what is backed.** β_res = 0.062 + 0.064A² is a fit to the
> measured retention map from `phi_gauge_decaymap.py`, not a model
> prediction: that script's own predicted window moves the other way
> (0.0619 at A = 0.10 to 0.0470 at A = 0.40). The cross-softening
> derivation of κ and the control-lattice numbers (centre 0.0754, retention
> 0.528 at β = 0.074, reverse 1.002) have no script in this repository.

> **Investigation (2026-09-24) — P-1 as stated is NOT supported. Nothing above
> is deleted; read it through this note.** Full trail: `PROVENANCE.md` §6o.
>
> 1. **The prediction script's resonance condition is incomplete.**
>    `beta_res()` in `phi_gauge_decaymap.py` holds w(0) + w(π) fixed at the
>    linear value 3.99256 (`W_SUM` is a constant) and lets only the pump soften
>    (−0.096 A²). Its curve is exactly −(pump self-shift)/(dω/dβ) = −0.0991 A²,
>    hence the downward drift. The products' own shifts in the pump field are
>    omitted. A linear (Hill/Floquet) stability analysis of the exact
>    harmonic-balance travelling wave — which reproduces the script's −0.0974 A²
>    self-shift and +0.035 βA² asymmetry term — puts the (0, π) unstable band at
>    β ∈ [0.0630, 0.0643] at A = 0.10 and [0.0653, 0.0868] at A = 0.40: the lower
>    edge stays near the linear 0.063, the upper edge moves out, midpoint
>    ≈ 0.063 + 0.08 A². The model's own window does move up; the script
>    computes something else.
> 2. **The measured map is an artifact of the plain-cosine start and the
>    fixed-time readout.** Retention at T = 400 is identical to four digits with
>    the noise switched off, reseeded, or 1000× larger (0.6117 / 0.8855 at
>    A = 0.4, β = 0.07 / 0.08): the dips are not noise-seeded instability, which
>    reaches only ~3.5 e-folds by T = 400. The bare cosine is not the nonlinear
>    wave; it launches free q = 0 and q = π oscillations of ≈ 0.027 and 0.017 —
>    exactly the product pair. Starting from the exact travelling wave, retention
>    is 0.9998 at the same cell. On a 0.0025 β grid the dip moves with readout
>    time (argmin β = 0.075 at T = 200–300, 0.0725 at T = 400, 0.0675 at
>    T ≥ 500), and one cell swings 0.81 → 0.27 → 0.78 between T = 400 and 800 —
>    MODEL_SPEC §4d.1 trap 6 (readout without a clearing criterion). All cells
>    with A ≤ 0.25 read 1.000 ± 0.005, so the fitted 0.062 + 0.064 A² rests on
>    essentially the β = 0.07 column; "deepest retention 0.61" is a T = 400
>    snapshot (min over time 0.26).
> 3. **A clean start shows broad instability, not a window.** From the exact
>    wave at A = 0.3–0.4, other pair channels (q, 2k₀ − q), q ≠ 0, grow as fast
>    as or faster than (0, π) at every β from 0.05 to 0.10 (Hill), and long runs
>    (T = 2500) lose pump energy across the whole range. The localised window
>    exists only because the plain-cosine start seeds (0, π) at O(A²).
> 4. **The reverse-direction protection holds.** The −k pump is linearly stable
>    on every channel checked (A = 0.3, 0.4; β = 0.05, 0.07, 0.08), and the
>    script's −k spot checks read 1.000.
>
> The P-1 falsifier ("the window drifts downward") is therefore not met by the
> model, but the window claim itself fails under a clean start; the directional
> protection is the part that stands. The investigation scripts (Hill analysis,
> batched clean-start runs) were scratch code and are not yet in this
> repository.

**Falsified if:** the window drifts downward with amplitude; or the reverse
direction decays inside the window; or the centre misses the zero-fit
prediction by more than the derived window width. Device reading: a passive
one-way amplitude valve whose operating point is predictable from linear
spectroscopy.

## P-2. Passivity forces u(2): non-Abelian holonomy without fine-tuning

For nodes carrying two internal dimers with an intra-node gyroscopic term
(which dynamically selects the complex structure), the zero-power condition
on a matrix-valued directional velocity coupling **requires W symmetric**,
and symmetric-plus-chirality-compatible is exactly Hermitian = the u(2)
gauge class. Predictions:
- An antisymmetric-W implementation is NOT energy-conserving and will
  exhibit parametric pumping/damping (a diagnostic signature, measured in
  our runs) — passivity itself is the selection mechanism.
- A chirality-pure wave packet crossing a coupling region with W ∝ σ_a
  precesses about axis a by θ = Σ_links Δk, with Δk computed from the
  two-branch dispersion at the packet's conserved frequency. Two regions
  with non-parallel axes traversed in opposite orders yield different
  final polarizations by a computable splitting; parallel axes yield zero.
  (Simulation instance: 59.86° measured vs 59.84° computed; control 0.00°.)

**Falsified if:** measured transport angles deviate by more than a few
percent from the dispersion computation, or the ordering splitting is
absent for non-parallel axes, or present for parallel ones.

## P-3. The amplitude-Zeeman effect (spinor self-precession)

At finite drive, the per-component quadratic nonlinearity adds a universal
on-site precession of the internal polarization about the dimer-population
axis: **Ω = C·A²·n_z**, with n_z conserved, and
C = [−1/K′ − 1/(4L₊) − 1/(4L₋)]·α²-scaled/(2ω + κ_g) computed from the
calibrated constants (κ_g the gyroscopic strength; L± the co-/counter-
rotating second-harmonic denominators). Simulation instance: C = −0.0896
predicted, −0.0899 measured, n_z conserved to three decimals.

**Falsified if:** the precession axis is not the population axis, n_z is
not conserved at leading order, or the coefficient misses the closed form
beyond experimental error.

---

**What a confirmation would mean:** the derivation chain (conservativity
selection → emergent complex structure → gauge-class transport → derived
nonlinear corrections) transfers from simulation to matter with no fitted
parameters. **What a falsification would mean:** the model class is not
faithfully realized, or the derivation chain breaks at the identified step
— each prediction names its step.
