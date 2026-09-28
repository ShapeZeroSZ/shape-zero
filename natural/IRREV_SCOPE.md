# Target 5 — irreversibility as the gradient half of the dynamics: scope and pilot report

*Sources:*
- *predictions: `IRREV_HYPOTHESES.md`, committed at **1c9e5bf (2026-09-28T13:56:35Z)**, before any
  script was written or run;*
- *pilots: `irrev_pilot.py` → `irrev_pilot_output.txt` (q = 1) and `irrev_q3.py` →
  `irrev_q3_output.txt` (q = 3, side 32);*
- *post-hoc check: `irrev_ratio_check.py` → `irrev_ratio_check_output.txt`, labelled post-hoc;*
- *a larger q = 3 run for a clean distance law: `irrev_q3_colab.py`, self-contained for a Colab GPU,
  not run here.*

**Every miss is scored as committed.**

## (1) The minimal GENERIC completion (I1)

**Model G:**
- **State:** `main`'s (u, v) plus one internal energy ε_i per node, with T_i = ε_i/C and
  s_i = C·ln ε_i.
- **Energy:** E = H_main + Σε_i. The Hamiltonian part is `model.py`'s force, unchanged, integrated
  by RK4.
- **Dissipative part:**
  - **absorption at matter:** an exact Ornstein–Uhlenbeck step on each tower component's *radial*
    velocity, rate γ_i = Γ·e_low,i, noise at T_i. The energy is booked exactly into ε_i;
  - **Fourier conduction** between neighbouring ε_i.
- **Principles:**

| principle | status | measured |
|---|---|---|
| total energy | **kept, exactly** (every joule removed is booked) | drift 2.0×10⁻⁵ (q = 1) and 1.9×10⁻⁶ (q = 3) of the mechanical energy — RK4 error only |
| J-compatibility (phase symmetry and each U(1) charge) | **kept**: radial kicks leave v·𝕁u_c unchanged — GENERIC's degeneracy condition, forced | charge drift ≤ 3.5×10⁻⁵ of N_a + N_b |
| A′ | **kept** (the Hamiltonian part is untouched) | — |
| P0 | kept (the tower stays populated) | — |
| **passivity / conservativity of the mechanical part** | **BROKEN, by design.** The dissipative bracket does net work on the mechanics — that is what makes a sink one-way. **It is counted as the cost of the adaptation**, carried by the "thermal variable + dissipative bracket" selection below | — |
| time-reversal symmetry | **broken, by design** | — |

## Scorecard against 1c9e5bf

| # | prediction | measured | verdict |
|---|---|---|---|
| I2 | energy ≤ 5×10⁻⁵; charge ≤ 5×10⁻⁵ | 2.0×10⁻⁵; 3.5×10⁻⁵ | **hit** |
| I2 | θ_r ≈ 5.9×10⁻⁵ ± 15% | 5.50×10⁻⁵ (−7%) | **hit** |
| I2 | initial sink rate P₀ ≈ 1.45×10⁻⁴ within ×2 | 0.85–1.18×10⁻⁴ | **hit** |
| I2 | one-way: every 100-unit window positive | 20 of 20, in all 4 seeds | **hit** |
| I2 | cumulative absorption at T in [0.03, 0.2] | 0.121–0.123 | **hit** |
| I2 | not a permanent steady sink (the rate falls as the finite ring drains) | window rates fall from ~8×10⁻⁵ to ~3.5×10⁻⁵ | **hit** |
| I2 | deterministic arrow: S never decreases | 0 decreases in 2000 samples; ΔS = +1.45×10⁴ | **hit** |
| I2 | fluctuating arrow: seed-mean S rises every window; single-run drops > −3σ | rises every window; worst −2.08σ | **hit** |
| I3a | mechanical–heat flow J ∝ δ, exponent 1.0 ± 0.1 | **1.46** | **miss** — see below |
| I3a | relaxation rate independent of δ within 10%, in [0.05, 0.4] | 0.067 (δ = 0.5), 0.054 (δ = 0.2); at δ = 0.1 and 0.05 the two seeds disagree in sign; spread 2.16 | **miss** — see below |
| I3b | pure conduction: J ∝ ΔT, exponent 1.00 ± 0.01 | J/ΔT = 1.0000 from 10⁻¹ to 10⁻⁴ | **hit** |
| I3b | slowest-mode decay rate independent of ΔT within 1%, = 1.5060×10⁻⁴ | 1.5060×10⁻⁴ at every ΔT (+0.00%) | **hit** |
| I4 | q = 1 tower deficit negative around the lump | −1.0×10⁻² to −1.3×10⁻¹ | **hit** |
| I4 | confined to the cone \|x − x₀\| ≤ v_max·t + 30 | beyond the cone ~10⁻¹⁰ | **hit** |
| I4 | q = 3 radially averaged exponent (r = 2…8, box mean subtracted) in [−2.6, −1.4] | **−1.463** | **hit** (narrowly, and see the caveat below) |
| I4 | energy-weighted sink ratio b/a = 1.885 ± 10% "by construction" | **1.506** | **miss** — post-hoc cause below |
| I4 | radius-weighted (G-R) ratio 1.00 ± 10% | 1.000 | **hit** |
| F-drag | a moving absorbing lump decelerates, \|Δv/v\| < 10⁻² | −7.6×10⁻⁴ | **hit** |
| F-range | the q = 3 law is not inverse-square | local slope −0.65 (r = 2→4) steepening to −2.3 (r = 4→8); nowhere the −1 an inverse-square force needs over a range | **hit** |

**Tally: 16 hits, 3 misses.**

**Not scored, and stated as such:**
- **F-heat** holds by construction: the sink *is* heating. Over T = 2000 the lump region's heat
  reservoir took up 0.12 — twice the lump's own mechanical energy (0.061).
- **F-univ** on the probe side rests on `main`'s refraction predictions (3063afd), which are still
  not run on `main`; it was not tested here.

### The misses, as committed

**I3a, the mechanical–heat parity test — noise-limited.**
- At δ = 0.5 and 0.2, J scales as expected (2.14×10⁻³ against 7.56×10⁻⁴, a ratio of 2.83 for a
  contrast ratio of 2.5), and the relaxation rates 0.067 and 0.054 fall in the predicted band.
- At δ ≤ 0.1 the contrast is below the thermal noise of the ~2,000 radial degrees of freedom
  (θ_r fluctuates by ~3% per sample). The two seeds disagree in sign there, and they drag the
  fitted exponent to 1.46 and the rate spread to 2.16.
- **The test could not resolve small contrasts at this size.** The committed criteria fail as
  stated, and larger ensembles would be needed to test them.
- **The conduction half (I3b), which has no mechanical noise, is exact.**

**I4, the energy-weighted sink ratio (1.51 against 1.885) — POST-HOC cause: optical thickness.**
- At Γ = 10 a tower quantum crossing the lump accumulates an integrated absorption rate
  ~Γ·e_low,peak × width / v ≈ 3, so it is absorbed almost completely. The sink is then limited by
  the tower's incoming supply, not by the lump's energy.
- **The Γ sweep** (deterministic, seeds 0–1, window [0, 50]):

  | Γ | ratio b/a | early ratio [0, 5] | a absorbed / Γ |
  |---|---|---|---|
  | 0.1 | **1.880** | 1.884 | 6.07×10⁻⁴ |
  | 0.3 | 1.870 | 1.882 | 6.03×10⁻⁴ |
  | 1 | 1.835 | 1.875 | 5.90×10⁻⁴ |
  | 3 | 1.747 | 1.856 | 5.56×10⁻⁴ |
  | 10 | **1.528** | 1.792 | 4.59×10⁻⁴ |

- **In the optically thin limit the ratio is E_b/E_a = 1.885, and absorption per unit Γ is
  constant.** As the lump thickens, the ratio falls toward the supply limit, where the sink is set
  by the lump's geometric cross-section and the incoming flux, not by its energy.
- **So a strong sink follows geometry, not energy** — Le Sage's saturation problem, in lattice
  form.

### The q = 3 caveat — box too small for a clean law

- The deficit profile, averaged over t ∈ [10, 33], is flat near the lump (its width is 3) and
  steepens outward. At larger r the time-averaged cone edge cuts it off, and beyond r ≈ 11 the
  box-mean subtraction turns it positive.
- **The fitted −1.46 lies in the committed range, but it is an average of a slope that runs from
  −0.65 to −2.3 — not a power law.** The ballistic prediction (∝ 1/r² in a steady cone) needs
  r ≫ lump width and a window in which the cone has long passed r.
- **`irrev_q3_colab.py`** (side 96, T = 120, window [60, 99], fit r = 4…24) is written for that. It
  is not run here: the side-96 runs need ~27× the side-32 compute.
- **What this pilot does show at q = 3:** the deficit falls faster than 1/r beyond the source
  (slope −2.3 at r = 4→8). **It is not inverse-square.**

## Answers

**(2) Does GENERIC supply what every earlier attempt lacked? Yes.**
- **One-way sinks:** 20 of 20 windows positive, in all seeds.
- **A monotone entropy — an arrow of time:** exact in the deterministic limit; in the seed mean with
  fluctuations.
- **All while keeping energy, charge and A′.**
- **But the sink is not permanent in a closed, finite system.** The second law ends every flow at
  equilibrium, and the rate halves as the ring drains. Steady sinks need an infinite system or a
  reservoir.

**(3) The asymmetry principle.**
- **"Flow is driven by differences and stilled by parity": HOLDS.** J ∝ ΔT exactly in conduction,
  and J scales with δ in the resolvable mechanical cases.
- **"Relaxation slows as balance is approached" (critical slowing down): does NOT hold** away from
  a critical point. The relaxation rate is independent of the contrast — exactly in conduction
  (1.5060×10⁻⁴ at every ΔT), and within the noise at δ = 0.5 and 0.2. The *flow* slows ∝ contrast;
  the relaxation *time* does not.

**(4) The attraction test, rerun.**
- **The depletion is steady only while the sink is.** It is one-way and confined to the ballistic
  cone.
- **Probes feel it through A′'s radius** as a pull toward the sink, but **not inverse-square**: the
  deficit falls faster than 1/r at q = 3 (the ballistic shadow; inverse-square would need diffusive
  tower transport over ≳ 10⁵ sites).
- **Sink strength:**
  - **follows energy only when thin, and only by construction** (γ ∝ e_low was the selection):
    1.880 at Γ = 0.1;
  - saturates toward geometry when thick (1.51 at Γ = 10);
  - under the radius-weighted variant it follows the radius exactly (1.000).
- **The old failure modes are all present:**
  - **heating** — the sink *is* heating: the absorbed energy is 2× the lump's own in T = 2000;
  - **drag** — Δv/v = −7.6×10⁻⁴;
  - **non-universality** — probe side via A′ (`main`'s 2c); source side geometric when thick;
  - **range** — not 1/r².

## (5) The count

| | parameters | selections | conditions | firm predictions |
|---|---|---|---|---|
| branch now (joint, repaired by spin-2) | 5 + a state | 7 | ~9 | 7 |
| **+ target 5, GENERIC completion (pilot; not adopted)** | **8 + 2 states** (+ C, Γ, K_th; + T₀) | **10** (+ a thermal variable per node with a dissipative bracket — **this is where passivity is broken, by design, and it is counted as the cost of the adaptation**; + absorption located at matter with rate ∝ energy density; + Fourier conduction). The radial projection and the fluctuation–dissipation form are forced (+0) | ~9 | 7, **+0** about nature — the arrow of time and the second law are known facts, and the arrow motivated the target |

**Verdict:**
- **GENERIC supplies irreversibility — one-way sinks and an arrow of time — while keeping energy,
  charge and A′, at the cost of passivity (by design) and +3 selections and +3 parameters.**
- The parity half of the asymmetry principle holds; the critical-slowing half does not, away from a
  critical point.
- **As a route to gravity it fails at accessible scales:**
  - an attraction that is not inverse-square;
  - sources that heat and drag;
  - sink strength that follows energy only in the thin limit and only by construction;
  - probes that respond non-universally.
- **Not adopted.** The gravity arc's open question stands, sharpened: irreversibility is now
  available, and it is not by itself enough.
