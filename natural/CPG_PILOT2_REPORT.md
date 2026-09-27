# CP-G second pilot — report against the predictions of 99131c0

*Sources:*
- *predictions: `CPG_PILOT2_PREDICTIONS.md` (99131c0), committed before any script;*
- *script: `cpg_pilot2.py` (7c12643) → `cpg_pilot2_output.txt`;*
- *post-hoc checks: `cpg_pilot2_checks.py` → `cpg_pilot2_checks_output.txt` (K1–K4). They are
  labelled **post-hoc** throughout: they explain results, they do not re-score them.*

**Every miss is scored as committed**, including misses caused by a badly worded criterion.

## Validation

- **Energy:** drift at most 2.6×10⁻⁵ over all runs (RK4 at `model.DT`).
- **The static-response code:** it reproduces the vacuum Green's function to 2×10⁻⁵ at q = 1 and
  2.5×10⁻⁶ at q = 3.
- **Background mean radius:** ⟨|ψ_B|⟩ = 0.00971 (predicted 0.00969).
- **Charge — the output's "5.4×10⁻²" is a small denominator** (K4, post-hoc):
  - one component's net charge happened to be Q(0) = 8×10⁻⁶;
  - relative to that component's total quanta N_a + N_b, every component drifts by
    1.5–1.8×10⁻⁵, which is the energy-drift level;
  - **so each component's U(1) charge is conserved.**

## Scorecard

| # | prediction (99131c0) | measured | verdict |
|---|---|---|---|
| 1a | field line at k = 0: 1.0899 ± 0.002; power below \|ω\| = 1 < 10⁻³ | 1.08987 (shift +0.00345, predicted +0.00347); 3.4×10⁻⁷ | **hit** |
| 1b | ω₉₅/(v_max q) in [0.85, 1.05] for m = 4…32 | 0.988 everywhere, except the energy density at m = 4, which is **1.054** | **miss** (one point; m = 4 spans ~8 frequency bins) |
| 1b | p ∈ [0.9, 1.1] | 0.994–1.033 (A_U = 0.005); 0.985–1.030 (0.02) | **hit** |
| 1b | not a single sound mode (peak fraction < 0.6) | 0.08–0.34 | **hit** |
| 1b | weight in 0.7 < \|ω\| < 2.9 below 2% | ≤ 6×10⁻⁸ | **hit** |
| 1c | occupation change < 2% at A_U = 0.005 | **3.3%** | **miss** |
| 1c | at 0.02 "roughly ∝ A_U², so ~16×", but < 30% | 12.8% (< 30%), but only **3.9×** | < 30%: **hit**; A_U² scaling: **miss** |
| 1c | no diffusive peak at 0.02; p in range | S(q, 0)/S(q, peak) ≤ 0.15; p 0.985–1.030 | **hit** |
| 2a | G(0)/G_vac(0) = 0.9971 ± 0.0010 | 0.99709 ± 0.00001 | **hit** |
| 2a | decay length 0.7219 (vacuum 0.7233) | 0.7220 | **hit** |
| 2a | other components < 10⁻² of δu₀(0) | 7.6×10⁻⁸ | **hit** |
| 2b | far-field background change < 1% of the source energy; zero at 3σ | **23%**; signed sum +1.4×10⁻⁹ ± 3.2×10⁻¹⁰ (4.5σ) | **miss** |
| 2c | ratio to G_pop within 1% for r ≤ 3 | 1.0000–1.0001 | **hit** |
| 2c | G(0)/G_vac(0) = 0.9985 | 0.99848 | **hit** |
| 2c | decay length "0.72 sites" | **0.758** by the stated fit (ln rG, r = 1…4), which gives 0.758 for G_pop itself on side 16 | **miss** (predicted with the infinite-lattice ξ, measured with a fit that does not return it) |
| 2c | \|δu₀\| < 10⁻⁴ of δu₀(0) for r ≥ 5 | **1.35×10⁻⁴** — equal to G_pop(5)/G_pop(0) itself | **miss** (the bound contradicted the predicted Green's function) |
| 2d | lump gain f(T) in [3×10⁻⁴, 5×10⁻³] | 4.0×10⁻³ | **hit** |
| 2d | background far-field change negative | negative in 3 of 4 seeds | **miss** |
| 2d | ≥ 50% of the background's loss in the far field | in 3 of 4 seeds | **miss** |
| 2d | confined to the ballistic cone | beyond the cone ~10⁻⁹–10⁻¹⁰, inside ~10⁻⁵ | **hit** |
| 2d | flat inside the cone (inner/outer in [0.5, 2]) | −29.5, −5.9, −1.5, 0.37 (a05); no seed of any lump inside the band | **miss** |
| 2d | energy balance within drift | residual 2×10⁻⁵, equal to the drift | **hit** |
| 3 | \|C(r)/C(0)\| < 0.01 for r ≥ 3 | e, ρ₄, ρ₅, ρ₇: 0.0075–0.0088; **ρ₆: 0.0112** | **miss** (ρ₆) |
| 3 | mean over r ∈ [5, 255] zero at 3σ | **−2.0×10⁻³ ± 3×10⁻⁵** for every field | **miss** — the −1/N sum rule of per-snapshot mean subtraction (1/512 = 1.95×10⁻³); not a correlation, but missed as worded |
| 3 | early window = late window | C(1) for e: 0.141 against 0.127; the charges ±0.01 | **hit** |
| 3 | no power law | none | **hit** |
| 4a | f(a)/f(b) outside [0.8, 1.25]; central 1.9 | 1.885 | **hit** |
| 4a | f(0.02)/f(0.05) in [0.6, 1.6] | **2.23** | **miss** |

**Tally: 17 hits, 12 misses.**
- None of the misses reverses the predicted outcome.
- Four are criteria I wrote badly, where the measurement agrees with the physics the prediction
  stated: 2c's decay length, 2c's r ≥ 5 bound, 3's sum-rule offset, and 1b at m = 4.
- The rest are real misses: 1c, 2b, 2d's sign, share and flatness, 4a's amplitude, and ρ₆.

## The post-hoc checks

**K1 — the 1c overshoot is real exchange, not an offset.**
- The occupation change grows steadily: 1.2×10⁻³ at t = 10, 5.6×10⁻³ at 100, 1.4×10⁻² at 500,
  2.9×10⁻² at 2000 — roughly ∝ t^0.55.
- It is the same with the a/b split taken at the populated frequencies, so it is not a
  decomposition artefact.
- It scales ∝ A_U, not A_U². That fits A′'s nonlinearity, |ψ|ψ, which is non-analytic: first
  order in the amplitude.
- **So my collisional-time estimate (∝ A_U⁻²) was the wrong model.** The growth looks like
  random-walk exchange; if the √t trend holds, O(1) change comes at t ~ 10⁶. The spectra show no
  hydrodynamic regime within T all the same.

**K2 — 2b's far field is deterministic and scales as F₀², but it is not a static field.**
- Between F₀ = 10⁻³ and 2×10⁻³ the far-field profile has correlation +1.000 and |·| ratio 4.00.
- Between the time windows [400, 1200] and [1200, 2000] the correlation is only +0.25, and the
  magnitude grows from 0.20 to 0.38 of E_s.
- The signed sum stays about 3% of the summed |·|, and sign-alternating.
- **Reading:** the source's second-order bump in the local radius scatters background quanta. In
  each realisation the scattered waves form a deterministic speckle, random in sign, spreading
  ballistically and accumulating in time. The ensemble mean is small.
- **This is a slow dynamic response, not a static long-range one.** The committed criterion
  still fails.

**K3 — identical a- and b-lump background profiles: real, not a bug.**
- They agree to 2.6×10⁻⁶. A control lump (a-branch, k₀ = π/4) differs by 1.34 — O(1) — so the
  code distinguishes lumps.
- **Reason:** a real symmetric envelope gives |ψ_b(x,t)| = |ψ_a(x,t)| in free evolution
  (measured: 1.3×10⁻⁵ at T), and under A′ the background sees only the total radius.
- **Consequence:** the absolute energy the two lumps drain is the same (2.828×10⁻⁴ against
  2.820×10⁻⁴). Their self-energies differ by ω_b/ω_a: E_b/E_a = 1.885, against 1.894 at k = 0.
- **So 4a's number hit, but for a sharper reason than the one I gave.** The sink strength is set
  by the lump's radius profile |u|(x, t), not by its energy or its branch. The amplitude miss
  (2.23) points the same way: a lump's absorption is not proportional to its energy.

**K4 — charge.** Conserved; see Validation.

## Answers to the pilot's four questions

1. **Census.**
   - **Gapless:** the energy density and each background U(1) charge density — a ballistic
     two-quasiparticle continuum, |ω| ≤ v_max q, v_max = 0.486; not a sound mode, and not
     diffusive within reach.
   - **Gapped:** the field itself, at 1.0899, and every opposite-branch density (≥ 3.15).
2. **Static and slow response.**
   - **Static:** Yukawa, with the populated operator Q + ⟨|ψ_B|⟩, and a range *shorter* than
     in the vacuum — 0.7220 against 0.7233 at q = 1; at q = 3, G = G_pop to 10⁻⁴.
   - **Slow:**
     - second-order scattering speckle, sign-random;
     - a lump drains the background into a depletion that spreads ballistically, confined to
       the cone but not sign-definite seed by seed at q = 1.
3. **Correlations.** Short range: C(1) = 0.14 for the energy density, ≤ 0.01 beyond r = 3 apart
   from ρ₆'s 0.011 and the −1/N offset. No build-up and no power law. The background is closed,
   not driven, so none was expected.
4. **Universality.** The one long-range candidate, the sink's depletion, has a source strength
   set by the lump's radius profile: branch-blind in absolute terms and amplitude-dependent per
   unit energy. It is not set by energy.

## The shadow-force questions (analytic; not run)

**Could a shadow force from the sink's depletion be universal?** No, not under A′.
- In a Le Sage mechanism the same absorption cross-section sets both the shadow a body casts
  (its active mass) and the momentum it receives (its passive mass).
- Here the absorption follows the radius profile |u|:
  - a- and b-lumps whose energies differ by 1.89× absorb identically;
  - the fractional gain per unit energy changes by 2.2× between amplitudes 0.02 and 0.05.
- So the effective gravitational mass is a function of |u|, not of energy. That violates the
  equivalence principle at O(1), against the Eötvös-type bound of ~10⁻¹⁵.

**Can an A′ packet respond to the directional momentum flux a ballistic shadow removes?** Not
through any coupling A′ has.
- A′'s force on a packet depends only on the local total radius, through −(√5 + |u|)ψ.
- A ballistic shadow reaches the packet in two ways:
  - **(i) as a scalar deficit in density ∝ 1/r²** (the solid angle blocked). Through dK its
    gradient gives a **1/r³** refraction force, not 1/r². By `main`'s 2d the sign of the
    resulting displacement is not even universal: it follows dv/dp.
  - **(ii) through the momentum the packet absorbs.** Absorption by pair creation conserves
    crystal momentum only modulo 2π. `main`'s residual fills the whole zone flatly in k, so
    umklapp is frequent and the transfer is unreliable. It is also second order in the
    background amplitude, and weighted by the same non-universal absorption.
- **So the Le Sage momentum route needs a coupling to the residual's momentum current that A′
  lacks.** That is property R3 of target 4, and it is counted there.

## Outcome

- **The predicted outcome holds:** CP-G around the populated background is **NOT SUPPORTED**, for
  a different reason than around the empty vacuum.
  - Gapless modes exist (the conserved densities), but they carry no static long-range response
    and no power-law correlations.
  - The only long-range effect is a sink depletion whose strength follows |u| rather than
    energy, and which an A′ packet feels only as a scalar 1/r³ refraction.
- **Nothing is adopted; the count is unchanged.**
- **Carried forward to target 4** (`NATURAL_BRANCH.md`): what would be missing for gravity to
  emerge from a residual.
