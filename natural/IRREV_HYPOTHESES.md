# Target 5 — irreversibility as the gradient half of the dynamics: hypotheses and predictions

*Committed before `irrev_pilot.py`, `irrev_q3.py` and `IRREV_SCOPE.md` were written or run. Scoping
and a pilot only: nothing is adopted.*

**Framework.**
- **GENERIC** (Grmela & Öttinger 1997): dx/dt = L·δE/δx + M·δS/δx.
  - L is antisymmetric (the Hamiltonian part);
  - M is symmetric positive semi-definite (the dissipative gradient part);
  - the degeneracies L·δS = 0 and M·δE = 0 keep energy conserved and entropy non-decreasing.
- **Prototype:** diffusion as gradient descent of free energy (Jordan, Kinderlehrer & Otto 1998).

## I1 — the minimal GENERIC completion of `main` (model "G")

**State per node i:** `main`'s (u_i, v_i) ∈ ℝ²ⁿ, plus **one thermal variable**, its internal
energy ε_i, with temperature T_i = ε_i/C and entropy s_i = C·ln ε_i (constant heat capacity C per
node).

**Energy:** E = H_main(u, v) + Σε_i. H_main is `model.py`'s Hamiltonian, unchanged (A′, κ\*, c).

**Dissipative part (M)** — a Langevin form with fluctuations, the stochastic GENERIC:
- **(D1) absorption by matter.**
  - Located at matter: the tower components' velocities at node i feel a friction
    γ_i = Γ·e_low,i, where e_low,i is the local energy density of the D ≤ 8 part.
  - Acts only on each tower component's **radial** velocity (along u_c):
    - the friction is γ_i·(radial velocity);
    - the noise is √(2γ_iT_i), fluctuation–dissipation at the node's own temperature;
    - the work done is booked exactly into ε_i.
  - **The radial projection is forced by J-compatibility.** GENERIC requires M to annihilate the
    gradient of every conserved quantity. Each component's U(1) charge depends on v only through
    v·𝕁u_c, and a radial kick leaves it unchanged.
- **(D2) heat conduction:** ε̇_i = K_th·Σ_j(T_j − T_i) over neighbours — Fourier's law, the JKO
  gradient flow of entropy.
- **The deterministic limit (noise off)** keeps the friction and the heat booking. Its sinks never
  stop, but entropy rises exactly monotonically.
- **Variant "G-R"** (for I4 only): γ_i = Γ_R·|u_low,i|², following the radius instead of the
  energy.

**Principles:**

| principle | kept? |
|---|---|
| total energy conservation | **kept, exactly** (every joule taken from the mechanics is booked into ε) |
| J-compatibility | **kept**: the phase symmetry, and each component's U(1) charge, conserved exactly by the radial projection |
| A′ | **kept**: the Hamiltonian part is unchanged |
| P0 | kept: the tower stays populated |
| **passivity / conservativity of the mechanical part** | **broken, by design**: the dissipative bracket does net work on the mechanics |
| time-reversal symmetry | **broken, by design** |

**Count, predicted:**
- **+3 selections:**
  - the thermal variable per node;
  - absorption located at matter with rate ∝ energy density (or ∝ radius in G-R);
  - Fourier conduction.
- The fluctuation–dissipation form and the radial projection are forced by GENERIC and by
  J-compatibility (**+0**).
- **+3 parameters:** C, Γ, K_th, and the initial temperature T₀ as a state.

## I2 — a one-way sink and an arrow of time (q = 1; D16 ring, N = 512, κ\*, tower comps 4–7 at A_U = 0.005)

**Set-up.** An a-branch lump (k₀ = 0, width 8, amplitude 0.05, E_a = 0.0613) at x₀ = 256.
C = 1000, Γ = 10, K_th = 1, T₀ = 10⁻⁵. Seeds 0–3, T = 2000.

**The tower's radial temperature**, analytic:
θ_r = ⟨(v·û)²⟩ ≈ ½·A_U²·(⟨ω_a²⟩ + ⟨ω_b²⟩)/2 ≈ **5.9×10⁻⁵ (± 15%)**.

**Predictions:**
- **Energy:** total energy E_mech + Σε conserved to ≤ 5×10⁻⁵ relative (the RK4 error only).
- **Charge:** each tower component's U(1) charge conserved to ≤ 5×10⁻⁵ of its N_a + N_b.
- **The sink:**
  - initial absorption rate P₀ = Γ·E_a·4·θ_r ≈ **1.45×10⁻⁴ (within a factor 2)**;
  - **one-way:** positive in every 100-unit window over T;
  - cumulative absorption at T between **0.03 and 0.2** (the tower holds ≈ 0.227);
  - the rate falls as the finite ring's tower drains. **Not a permanent steady sink:** in a closed,
    finite GENERIC system the second law ends every flow at equilibrium. The sink's lifetime grows
    with C and with system size.
- **The arrow:**
  - **deterministic limit:** S = ΣC·ln ε_i never decreases, at any of the ~4000 samples;
  - **fluctuating model:** the seed-mean S rises in every 100-unit window, with any single-run
    decrease < 3σ of its fluctuation.

## I3 — the asymmetry principle: flow driven by differences, stilled by parity

**Test (a) — mechanical–thermal contact.**
- Uniform absorption γ₀ = 0.05 on the tower everywhere (no lump); C = 4; K_th = 0.
- Heat starts at T₀ = θ_r(1 − δ) for δ = 0.5, 0.2, 0.1, 0.05. Seeds 0–1.
- **Net flow J(δ)** into heat over the first 5 time units ∝ δ: a fitted exponent **1.0 ± 0.1**, and
  J → 0 as δ → 0. **Flow is stilled by parity.**
- **Relaxation of the contrast θ_r(t) − T(t):** exponential, with a rate **independent of δ within
  10%**, predicted in [0.05, 0.4]. **So there is NO critical slowing down.** Near a balance point,
  linear relaxation has a fixed rate. Critical slowing down needs a critical point, where a rate
  goes to zero with a control parameter — not the approach to balance itself.

**Test (b) — pure heat conduction** (the JKO prototype; two halves of a 512-ring, contrasts
10⁻¹ … 10⁻⁴):
- J ∝ ΔT, with exponent **1.00 ± 0.01**;
- decay time of the slowest mode **independent of ΔT within 1%**;
- equal to C/(K_th·(2 − 2cos(2π/512))).

## I4 — the attraction test, rerun with irreversible sinks

**The mechanism.**
- The sink depletes the tower's mechanical energy around the lump.
- A weak probe feels the tower only through A′'s radius: dK ≈ ⟨|u_tower|⟩, and the force is
  −∇dK/(2ω_a + κ) (`main`'s refraction).
- Probes are pushed away from the denser region — toward the sink. **Attraction.**

**q = 1** (deterministic limit, CRN against the Γ = 0 run, seed 0): the tower's energy deficit is
negative near and around the lump, and confined to the cone |x − x₀| ≤ v_max·t + 30.

**q = 3** (side 32, n = 6: comps 0–3 low, lump in comp 0; comps 4–5 tower at A_U = 0.005;
deterministic, CRN, 2 seeds; the deficit averaged over t ∈ [10, 33]):
- The tower is ballistic here (its own collisional time is ~10⁶; absorption acts only at the
  lump), so the deficit is a solid-angle shadow ∝ **1/r²**: a radially averaged exponent in
  **[−2.6, −1.4]** for r = 2…8, after subtracting the box-mean level.
- **So the force on probes ∝ ∇(1/r²) ∝ 1/r³ — not inverse-square.**
- An inverse-square force (a deficit ∝ 1/r) needs diffusive transport of the tower on scales
  ≫ its mean free path (≳ 10⁵ sites here). A uniform bath coupling would not do it either: it
  Yukawa-screens the deficit, and the absorbed energy re-emerges as a heat source ∝ +1/r.

**Sink strength (q = 1, the fluctuating model, initial window [0, 50], seeds 0–1):**
- **G:** absorption rate of the b-lump over the a-lump = E_b/E_a = **1.885 ± 10%**, following
  energy — **by construction**, since γ ∝ e_low was chosen (the selection in I1). So it is a
  consistency check, not evidence.
- **G-R** (γ ∝ |u_low|², Γ_R = 10/((ω² + Q)/2) matched at the a-lump): ratio **1.00 ± 10%**,
  following the radius profile, as under A′ alone.

**Old failure modes, pre-registered as possible failures:**

| # | failure mode | prediction |
|---|---|---|
| **F-heat** | a sink is a heater: the lump's heat grows at exactly the absorption rate P. Any attraction strength ∝ P implies heating ∝ source mass at a rate tied to that strength | **present, structural.** The Le Sage calibration (`RESIDUAL_SCOPE` F5) applies once units are fixed |
| **F-drag** | a moving absorbing lump (k₀ = π/4, deterministic, CRN) loses momentum to its own asymmetric wake | its velocity change relative to the Γ = 0 run is **negative (decelerating)**, with \|Δv/v\| < 10⁻² over T |
| **F-univ** | probe side: A′'s dK coupling is not universal (`main`'s 2c: strong probes feel less; accelerations scale with 1/(2ω(2ω + κ))) | **fails universality** on the probe side. The source side follows energy only in G, by construction |
| **F-range** | the q = 3 law is 1/r³ (ballistic), not inverse-square | **present** |

## I5 — the count and the verdict (predicted)

- **Count:** +3 selections, +3 parameters (+ T₀ as a state).
- **New predictions passed about nature: 0.** The arrow of time and the second law are known
  facts; the arrow motivated this target, so it cannot count.
- **Predicted verdict:**
  - GENERIC supplies what every earlier attempt lacked — one-way sinks and a monotone entropy —
    while keeping energy, charge and A′;
  - it breaks passivity by design;
  - but the attraction it gives is **1/r³, heats its sources, drags moving ones, and is not
    universal on the probe side**;
  - **not a route to gravity at accessible scales.**
  - The parity half of the asymmetry principle holds (flow ∝ contrast). Its critical-slowing half
    does not, away from a critical point.
- **Nothing is adopted.**
