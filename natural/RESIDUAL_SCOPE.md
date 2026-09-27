# Target 4 — the residual as the missing part: scope

*Scoping only; nothing is built.*
- *The property list R1–R7 is frozen in `RESIDUAL_HYPOTHESES.md` (f20137d).*
- *§1–§3 (conflicts, derived consequences, and the outcomes that count against) were committed
  before §4 evaluated anything against the numbers.*
- *Units: SI; order-unity geometric factors are kept symbolic (C) or dropped, as stated.*

## §1 — Each property against `main`'s principles

`main`'s principles as used here:
- **passivity:** a closed, energy-conserving Hamiltonian lattice with no external drive; velocity
  couplings only in gyroscopic or transposed form;
- **J-compatibility:** invariant under the node phase rotation, so charges are conserved;
- **P0 (non-isolation):** every level populated simultaneously;
- **A′:** node form — the on-site well acts on the node's total radius |u|, and nothing else is
  sensed on site (C43; C34).

| property | passivity | J-compatibility | P0 | A′ | conflicts |
|---|---|---|---|---|---|
| **R1** isotropic flux Π at speed u, one rest frame | a flux is allowed; the lattice frame is the rest frame | yes | **supplied** (a populated background) | — | **u is bounded by the lattice's own kinematics.** Every populated mode has group velocity ≤ v_max = 0.486 (lattice units), which is no faster than the shared wave speed. So R1 with u ≫ c (needed by R6, §2) conflicts with A′'s dispersion and with the shared-speed selection (target 3). |
| **R2** sink ∝ energy, universal h | **conflict** — see below | an energy-density coupling is J-invariant: no conflict | **conflict** — see below | **conflict:** A′ absorbs by the radius profile \|u\|, not energy — a- and b-lumps with energies 1.89× apart absorb identically; amplitude changes absorption per energy by 2.2× (pilot 2, K3). R2 needs a new coupling to the local energy density, an L1-type term: an ingredient. | **three conflicts** (A′, passivity, P0) |
| **R3** coupling to the residual's momentum current | a current–current coupling j_matter·j_res can be passive in the transposed form (the EM_SCOPE C1/C7 precedent): no conflict in principle | built from U(1) currents: no conflict | — | **conflict:** A′ has no vector on-site coupling. Absorption conserves crystal momentum only modulo 2π, and `main`'s zone-filling residual makes umklapp frequent. R3 is an addition to the adopted node form (C43). | **one conflict** (A′) |
| **R4** ballistic, λ ≫ the scales where 1/r² is verified | — | — | **tension:** `main`'s own residual exchanges occupation ∝ A_U·t^0.55 (K1), so it is O(1) by t ~ 10⁶ and λ ~ v_max·10⁶ ~ 5×10⁵ sites at A_U = 0.005. λ ≥ 10¹³ m needs a lattice spacing ≥ 2×10⁷ m at that amplitude, or a far smaller amplitude — which lowers Π, which R1 and G need large. | — | **one tension**, set by the unknown lattice spacing (a selection) |
| **R5** no heating: absorbed energy leaves by a channel that neither heats nor refills the shadow | **conflict:** in a closed Hamiltonian system a populated channel re-emits what it absorbs (detailed balance), so the shadow refills | — | **conflict:** the channel must stay empty to absorb without re-emitting, but P0 populates every level | **conflict:** `main` itself measures the residual heating matter (§6u, parametric heating) | **three conflicts** |
| **R6** drag below bounds | — | — | — | via R1: needs u ≫ c (§2), beyond A′'s kinematics | **one conflict** (as R1) |
| **R7** replenishment, so G is constant | **conflict:** a sustained source of fresh residual is an external drive. An internal cycle (residual → matter → hidden channel → residual) in a closed system relaxes to equilibrium, where absorption equals emission everywhere and the shadow vanishes (Kirchhoff) | — | — | — | **one conflict** (passivity) |

**Tally: 9 conflicts and 1 tension.**
- Conflicts: A′ ×4 (R1/R6, R2, R3, R5), passivity ×3 (R2, R5, R7), P0 ×2 (R2, R5).
- Tension: R4.
- **R2 against passivity and P0.** Absorption as a net sink is not an equilibrium property. In a
  closed, fully populated Hamiltonian system, a body in equilibrium with the residual emits as
  much as it absorbs, so "sink ∝ energy" can hold only out of equilibrium. That needs R7's drive.
- **Most of these conflicts are forced by the hypothesis itself.** A shadow force needs net,
  sustained absorption, and `main`'s closed, populated, passive lattice forbids exactly that in
  the steady state.

## §2 — Consequences not on the list (derived, symbolic)

Notation:
- h: absorption area per unit mass (m²/kg; for energy E the mass is E/c²);
- I_p: the residual's momentum flux per steradian, and Π ≡ 4πI_p;
- p/E = 1/u for the residual quanta, so the energy flux per steradian is u·I_p and the energy
  density is ε ≈ Π.
- Order-unity geometric factors: C.

1. **G in terms of the residual.** A body of mass m₂ at distance r blocks a fraction h·m₂/r² of the
   flux per steradian arriving at body 1, and body 1 absorbs a fraction h·m₁ of what reaches it. So
   F = h²I_p·m₁m₂/r², and **G = h²Π/(4π)**, i.e. **Π = 4πG/h²**. The inverse square follows from
   R4. Active and passive mass are both h·m.
2. **Gravitational shielding (Majorana).** A body's attraction through intervening matter of
   column density Σ is reduced by a factor e^{−hΣ}. The hypothesis *requires* h > 0, so shielding
   is not an option but a consequence. Gravitational mass is not additive at order hΣ.
3. **Self-absorption of the residual — R2 applied to the residual's own energy.** The residual's
   energy density ε ≈ Π, i.e. a mass density Π/c², so its own mean free path is
   λ_self = c²/(hΠ) = **c²h/(4πG)**. **This fixes λ from h alone**, and R4 needs λ ≫ solar-system
   scales. The escape — exempting the residual from R2 — breaks "all energy alike".
4. **The heating rate of ordinary matter.** The absorbed power per unit mass is
   H = h·Π·u = **4πG·u/h**.
5. **Drag on a moving body.** Absorbing an isotropic flux while moving at v through the rest
   frame gives a_drag = C·h·Π·v/u = **4πG·C·v/(h·u)**, with C ≈ 4/3. The combination h·u is set by
   the orbit bound.
6. **The speed of gravity.** Changes in a shadow propagate at u, so gravitational signals and
   radiation travel at u.
7. **Ġ/G without replenishment.** Matter of cosmic mean density ρ_m depletes the residual at the
   fractional rate ρ_m·h·u. Through the drag bound (h·u is bounded below), this is bounded below
   independently of h.
8. **Light.** Photons carry energy, so under R2 they absorb the residual:
   - they are deflected like a Newtonian corpuscle — half the observed bending, PPN-equivalent
     γ = 0, as in `JOINT_PREDICTIONS.md`;
   - they gain energy at the rate H per unit mass-energy, a secular blueshift or heating of light.
9. **Clocks.** Under A′, the residual's density sets a stiffness shift dK = ⟨|ψ_B|⟩ (pilot 2, 2a).
   A shadow deficit therefore lowers local frequencies — a redshift of the right sign — but it
   follows the deficit profile, ∝ 1/r² (ballistic, R4), not the potential ∝ 1/r.
10. **The uniform part.**
    - **(a)** Under A′ it renormalises every mass gap by dK/(2ω + κ) — constant, unobservable.
    - **(b)** Its energy density ε ≈ 4πG/h². If it gravitates (R2), that is a uniform source of
      gravity, and a cosmological energy density.
    - **(c)** It defines a preferred frame: PPN α₁, α₂-type effects, suppressed by v/u.
    - **(d)** Its fluctuations heat matter parametrically (`main` §6u), separately from item 4.
11. **No tensor radiation.** Shadow dynamics is scalar/vector. There are no + and × polarisations.

## §3 — Outcomes that count against the hypothesis (committed before §4)

Each counts against target 4 if it occurs. The bounds are fixed here, before any numbers are put
in.

| # | against if | bound used |
|---|---|---|
| **F1** | the h required by any other item exceeds the shielding bound | **h ≤ 1×10⁻²² m²/kg** (Eckhardt 1990, lunar laser ranging) |
| **F2** | λ_self = c²h/(4πG) < 10¹³ m (≈ 70 AU; the inverse square is verified across the planetary system) for every h allowed by F1 | the ballistic scale needed by R4 |
| **F3** | orbit survival needs u > c(1 + 10⁻¹⁵) for every h allowed by F1 | drag e-folding time > age of the solar system: a_drag/v < 7×10⁻¹⁸ s⁻¹ (deliberately the weakest orbit bound) |
| **F4** | the speed of gravity u differs from c by more than GW170817 allows | −3×10⁻¹⁵ < (c_g − c)/c < +7×10⁻¹⁶ |
| **F5** | the heating H exceeds Earth's internal heat budget per unit mass, unless R5's channel is invoked (whose conflicts, §1, are then counted) | ≈ 47 TW / 6×10²⁴ kg ≈ 8×10⁻¹² W/kg |
| **F6** | unreplenished depletion gives \|Ġ/G\| above the bound (R7 then becomes compulsory, with its passivity conflict) | \|Ġ/G\| ≲ 10⁻¹³ yr⁻¹ (LLR; recent solutions are tighter) |
| **F7** | under `main`'s own A′ coupling, composition dependence exceeds the equivalence-principle bound | MICROSCOPE: η ≲ 10⁻¹⁵ |
| **F8** | light bending departs from the observed value | γ = 1 ± 2×10⁻⁵ (Cassini) |
| **F9** | the gravitational redshift does not scale as the potential | redshift ∝ ΔΦ confirmed to ~10⁻⁴–10⁻⁵ (GP-A; Galileo 5 and 6) |
| **F10** | a gravitating uniform part exceeds the observed cosmic energy density | ρ_crit·c² ≈ 8×10⁻¹⁰ J/m³ |
| **F11** | gravitational radiation lacks the tensor polarisations | LIGO/Virgo polarisation tests and binary-pulsar damping |

**Rule:** any one of F2, F3/F4, F7, F8, F9 or F11 occurring **falsifies target 4 as scoped**. F5, F6
and F10 falsify it unless the named escape is taken, and each escape is then counted as a further
selection with its §1 conflicts.

## §4 — Evaluation (after §3 was committed at 6badcc3)

Inputs: G = 6.674×10⁻¹¹; c = 2.998×10⁸ m/s; C = 4/3; ρ_m = 0.3 ρ_crit = 2.6×10⁻²⁷ kg/m³.
**Every quantity is taken at the most favourable allowed h** — its upper bound from F1 — unless
stated.

| # | derived value | bound | outcome |
|---|---|---|---|
| F1 | h is free below 10⁻²²; Π = 4πG/h² ≥ **8.4×10³⁴ Pa** | h ≤ 10⁻²² m²/kg | no violation on its own; it forces the rest |
| **F2** | λ_self = c²h/(4πG) ≤ **1.1×10⁴ m (11 km)**. λ ≥ 10¹³ m needs h ≥ 9.3×10⁻¹⁴ m²/kg — 9×10⁸ × the shielding bound | 10¹³ m | **occurs — falsifies.** R2 (all energy absorbs) and R4 (ballistic over the solar system) are incompatible with shielding below the LLR bound, whatever u and Π are |
| **F3** | u ≥ 4πG·C/(h × 7×10⁻¹⁸ s⁻¹) = **1.6×10³⁰ m/s = 5.3×10²¹ c** | u ≤ c(1 + 10⁻¹⁵) | **occurs** |
| **F4** | gravity's speed = u ≥ 5.3×10²¹ c | \|c_g − c\|/c ≲ 10⁻¹⁵ (GW170817) | **occurs — falsifies.** Drag and the measured speed of gravity cannot both hold, by 21 orders — Laplace's and Poincaré's objection, now closed by a direct measurement |
| F5 | H = 4πG·u/h ≥ **1.3×10⁴³ W/kg** | 8×10⁻¹² W/kg | **occurs, by 55 orders.** Survivable only through R5's channel, whose three §1 conflicts are counted; the escape adds 1 selection |
| F6 | depletion rate ρ_m·h·u ≥ ρ_m·4πG·C/(7×10⁻¹⁸ s⁻¹) = **1.3×10⁻¹¹ yr⁻¹**, independent of h | 10⁻¹³ yr⁻¹ | **occurs, by 130×.** R7 is therefore compulsory, with its passivity conflict |
| **F7** | under A′ the absorption follows \|u\|: a- vs b-lumps 1.89× per unit energy; amplitude 2.2× (pilot 2) | η ≲ 10⁻¹⁵ | **occurs — falsifies**, as committed. It says that A′ cannot supply R2: R2 needs a new energy-density coupling (§1), and F7 records that `main`'s coupling fails by 15 orders |
| **F8** | photons absorb (R2) → Newtonian deflection, γ-equivalent 0; secular blueshift at H/c² ≥ 1.5×10²⁶ s⁻¹ (bounded by F5's escape) | γ = 1 ± 2×10⁻⁵ | **occurs — falsifies** (half the bending — the defect of the scoped joint theory, not repaired) |
| **F9** | clock shift follows the deficit ∝ 1/r² (ballistic) | redshift ∝ ΔΦ ∝ 1/r to 10⁻⁴–10⁻⁵ | **occurs — falsifies** |
| F10 | ε ≈ Π ≥ 8.4×10³⁴ J/m³ = **1.1×10⁴⁴ ρ_crit c²** | ρ_crit c² | **occurs.** The escape (the residual does not gravitate) contradicts R2 — the same exemption F2 would need; counted as 1 selection |
| **F11** | shadow dynamics: no tensor polarisations | tensor GW polarisations; binary-pulsar damping | **occurs — falsifies** |

**Verdict: target 4 as scoped is FALSIFIED.**
- **Six decisive outcomes:** F2, F4, F7, F8, F9, F11.
- **Three more need escapes:** F5, F6, F10.
- **Two are internal contradictions within the frozen list** — the Pauli-style test at work, since
  the list's own unasked-for consequences contradict each other and the data:
  - **F2:** R2 plus R4 against shielding;
  - **F3/F4:** R6 against the measured speed of gravity.
- **Unlike the neutrino** (whose required properties — neutral, light, weakly interacting — were
  mutually consistent and later detected), **the properties gravity would need from a residual are
  not.** These are the classical Le Sage objections — heating (Maxwell), drag and speed (Laplace,
  Poincaré) — with the self-absorption bound and GW170817 now making them quantitative and closed.

**The count (for `NATURAL_BRANCH.md`):**
- **+7 selections** (R1–R7), **+2 escape selections** (R5-channel, residual-non-gravitating), and
  **+3 parameters** (Π, u, h; λ is fixed by h through F2).
- **1 condition consumed** (G, used as input).
- **0 new firm predictions passed**, and 6 decisive failures.
