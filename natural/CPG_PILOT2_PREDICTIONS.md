# CP-G, second pilot — the populated background (predictions)

*Committed before `cpg_pilot2.py` was written or run. The first pilot (`CPG_PILOT_PREDICTIONS.md`,
`cpg_pilot_output.txt`) linearised `main` about the **empty** vacuum and found every mode gapped
(gap 1.0864, static range ξ = 0.71). `main`'s P0 (non-isolation) says that vacuum never occurs:
every level is populated. This pilot asks the same question about the populated background.*

## The background under test, and what "driven" means here

- **`main`'s tower background, unchanged.**
  - Node form A′, κ\*, D16 (n = 8), a q = 1 ring with N = 512.
  - Components 4–7 are populated incoherently on both chirality branches, rms 0.005 per
    component — `main`'s A_U, using `main`'s own `incoherent_upper` (`shape_zero_tests/tower_populated_test.py`).
  - Components 0–3 are empty, except where a source, probe or lump is placed.
  - Seeds 0–3; T = 2000.
- **Secondary runs:**
  - A_U = 0.02, seeds 0–1, for the census only;
  - q = 3, side 16, same components, T = 300, for the range.
- **The background is closed, not driven.**
  - It is a Hamiltonian system with no reservoir and no external drive.
  - Its initial state is a random-phase Gaussian state with |a_k|² flat. That state is exactly
    stationary under the linear dynamics, and it is not thermal.
  - It relaxes only through the weak nonlinearity. The fluctuating stiffness is δK ~ 0.0025, so
    the collisional time is τ ~ (δK/(2ω + κ))⁻² ~ 10⁵–10⁶, far beyond T.
  - The one place energy is pumped is local: a lump in the D ≤ 8 part gains energy parametrically
    (`main` §6u), so it acts as a **sink** for the background. That is the only "drive" tested here.
- **Mean-field stiffness shift.** A probe in an empty component (orthogonal to the background)
  sees dK = ⟨|ψ_B|⟩ = 0.00969. A background component sees ⟨|ψ_B|⟩(1 + 1/8) = 0.0109, the extra
  1/8 being the tangent term of |ψ|ψ.

## N1 — collective-mode census

**1a. The field itself stays gapped.**
- The background components' a-branch line at k = 0 sits at **ω = 1.0864 + 0.0109/3.1446 =
  1.0899 (± 0.002)**.
- Less than 10⁻³ of the ψ spectral power lies below |ω| = 1.0.

**1b. The conserved densities are gapless, as a ballistic continuum.**
- **Densities measured:** the energy density e(x), and each background component's U(1) charge
  density ρ_j, j = 4…7 (`main`'s charge, per component).
- **Their fluctuation spectra S(q, ω) have gapless weight.** It comes from same-branch pairs
  (a†a, b†b), with frequencies ω(k + q) − ω(k).
- **It is a two-quasiparticle continuum confined to |ω| ≤ v_max q**, where v_max = 0.4859 is the
  maximum group velocity. v(k) is the same for both branches.
- **Criteria**, for modes m = 4…32 (q = 2πm/512), with the low band |ω| < 0.7:
  - the frequency below which 95% of the low-band weight lies, divided by q, is in
    **[0.85, 1.05]·v_max**;
  - the rms frequency scales as q^p with **p ∈ [0.9, 1.1]** — ballistic; diffusion would give
    p = 2;
  - **it is not a single sound mode:** less than 60% of the low-band weight lies within ±10% of
    the spectrum's peak frequency.
- **The rest of the spectrum is gapped.** Opposite-branch pairs (a†b) form a band starting at
  ω_a(0) + ω_b(0) ≈ 3.15. Less than 2% of the weight lies in 0.7 < |ω| < 2.9.

**1c. No hydrodynamic regime within reach.**
- **Proxy for the collisional rate:** the mode occupations |a_k|², |b_k|² (smoothed over 16
  modes) change by less than **2% rms** over T = 2000 at A_U = 0.005.
- At A_U = 0.02 the change is larger, roughly ∝ A_U² (so ~16×), but stays below 30%. There is
  still no diffusive peak there, and p stays in [0.9, 1.1].

**1d. Beyond τ (analytic; not tested).** The energy and charges should become diffusive: the
on-site well breaks momentum conservation, so no sound mode survives. D ~ v²τ.

**Census verdict, predicted:**
- **gapless:** the energy density and the four background U(1) charge densities (ballistic
  continua now, diffusive after τ);
- **gapped:** the field itself (1.0899) and every opposite-branch density.

## N2 — static and slow response to a localised source

**2a. Static point force, q = 1.**
- **Set-up:** F₀ = 10⁻³ on component 0 at x = 0, ramped on over t ∈ [0, 200] and averaged over
  [400, 2000]. Each seed's background-only run is subtracted (common random numbers).
- **Prediction:** ⟨δu₀(x)⟩/F₀ = G_pop(x), the Green's function of Q + dK with dK = 0.00969.
  - G_pop(0) = 0.26701 against the empty vacuum's 0.26779 — a ratio of **0.9971 (± 0.0010)**;
  - decay length **0.7219** against 0.7233. **The range gets shorter, not longer.**
- **Other components** (1, 4–7): |⟨δu⟩| < 10⁻² of δu₀(0).

**2b. The densities do not carry a static response.**
- The source enters the background only through |ψ|, at second order: δK ≈ u_s²/(2|ψ_B|) ~ 10⁻⁵.
- **Criterion:** the time-averaged background energy-density change, summed over 5 ≤ |x| ≤ 250,
  is below 1% of the source's own static field energy, and consistent with zero at 3σ across seeds.
- **Reason:** the static limit of a conserved density's response is its thermodynamic
  susceptibility, not the hydrodynamic pole. The pole gives χDq²/(−iω + Dq²) → χ as ω → 0, and χ
  is short-range here.

**2c. The range at q = 3.**
- **Set-up:** side 16, same background, F₀ = 10⁻³ at the origin in component 0, ramped over
  [0, 100] and averaged over [150, 300]; seeds 0–1, background runs subtracted.
- **Prediction:** along an axis, ⟨δu₀(r)⟩/F₀ matches G_pop (side 16) within 1% for r ≤ 3.
  - G(0)/G_vac(0) = **0.9985**;
  - decay length **0.72 sites**;
  - for r ≥ 5, |⟨δu₀⟩| < 10⁻⁴ of δu₀(0).
- **So the static range at q = 3 is Yukawa, 0.72 sites, slightly shorter than in the empty vacuum.**

**2d. Slow response: a lump as a sink (q = 1).**
- **Set-up:** a stationary lump — k₀ = 0, width 8, amplitude 0.05, a-branch per-mode launch — in
  component 0 at the ring's centre. Subtracted against its background-only run and against the
  isolated lump.
- **The lump gains energy** (`main`'s parametric heating): fractional gain f(T) in
  **[3×10⁻⁴, 5×10⁻³]** (seed mean).
- **The background loses it, and the loss streams outward ballistically.** The background
  energy change, CRN-subtracted, is:
  - negative in the far field (|x − x₀| > 30);
  - confined to |x − x₀| ≤ v_max t + 30;
  - carrying at least 50% of the background's total loss in that far field;
  - roughly flat inside the cone (1D ballistic): the mean depletion of the inner half of the cone
    over its outer half is in [0.5, 2].
- **Energy balance, as a check:** ΔE_B + ΔE_l,self + ΔE_int = 0 to within the energy drift.
- **At q = 3 (analytic; not run).** A sink of rate P depletes the background as:
  - **−P/(4π v̄ r²)** in the ballistic regime (t < τ) — a force ∝ 1/r³ on a probe through dK;
  - **−P/(4π D r)** in the diffusive regime (t ≫ τ) — **an inverse-square force**, Le Sage-like,
    lasting only as long as the sink keeps absorbing.
  - Its sign follows from `main`'s refraction result (dp away from the denser region): toward the
    sink.

## N3 — long-range correlations

- **The generic long-range correlations** of driven systems with conservation laws (the
  Garrido–Lebowitz–Spohn type) need a *sustained* drive that breaks detailed balance, with
  conserving noise. `main`'s background has neither: it is closed and Hamiltonian, with no
  reservoir. A Gaussian state that is stationary under the linear dynamics generates none at
  linear order.
- **Prediction:** the equal-time connected correlations C(r)/C(0) of e and of each ρ_j, and of
  the total background charge density:
  - are **short-range** — |C(r)/C(0)| < 0.01 for r ≥ 3;
  - are **consistent with zero at 3σ** in the mean over r ∈ [5, 255];
  - are **the same in the windows [0, 200] and [1800, 2000]** — nothing builds up.
- **No power law.**
- **The one embedded drive**, the lump sink, gives a *mean* profile (2d), not power-law
  fluctuations. A genuine non-equilibrium steady state with a sustained sink would be expected to
  have 1/r^d correlations; that is out of reach here, being analytic only.

## N4 — universality of any long-range response

The only long-range candidate predicted is the sink's depletion (2d).

**4a. Source side — not universal.** A lump's fractional energy gain is set by its branch, not by
its energy.
- **a-lump over b-lump** (amplitude 0.05, both k₀ = 0): the ratio lies **outside [0.8, 1.25]**,
  central estimate **~1.9 = ω_b(0)/ω_a(0)**. Pair creation stimulated per quantum adds ω_a + ω_b
  per pair, but the lump's energy per quantum is ω_a or ω_b.
- **A coherent background gives no sink at all** (`main` §6u, x = 0). The source strength belongs
  to the background's incoherence, not to the lump's energy.
- **a-lump at amplitude 0.02 against 0.05:** the fractional gain ratio is in [0.6, 1.6], the gain
  being stimulated.

**4b. Probe side — universal in stiffness, not in energy.**
- **`main`'s input.** Probes feel dK = |u| − |u_l|. That is universal for weak probes of either
  branch (`main`'s pairs-and-refraction predictions, 3063afd; **not yet run on `main`, so used here
  only as a prediction**). At k₀ = 0, a- and b-probes get the same dp/dt and the same acceleration,
  since dv/dp|₀ = 2/(2ω_a + κ) for both.
- **Strong probes sense less** (`main`'s 2c: dK_eff/√s ≈ 0.45–0.8 at amplitude 10⁻²). **So
  heavier lumps fall less, and the equivalence principle fails.**
- **The coupling is an additive stiffness, not a multiplicative (lapse-like) one.** So a
  lattice with a different K would give a slow probe an acceleration ∝ 1/ω².

**4c. Verdict, predicted.** Any long-range response found **does not couple to all energy
alike**, at the source or at the probe.

## Predicted outcome

**CP-G around the populated background: NOT SUPPORTED, for a different reason than around the
empty vacuum.**
- Gapless modes now exist: the conserved densities.
- But they carry **no static long-range response**, and **no power-law correlations**.
- The one long-range effect is a transient depletion around a sink. It is 1/r² (ballistic) or 1/r
  (diffusive, reached only after τ ~ 10⁵–10⁶) in density. Its source is the sink rate, not energy,
  and its effect on probes is not universal.
