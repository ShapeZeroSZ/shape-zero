# Gravity is subject to gravity — scoping report

*Scoping only: nothing is built or adopted.*
- *Hypotheses were committed first: `SELFCOUPLING_HYPOTHESES.md` (**c897a7e,
  2026-09-28T06:38:30Z**).*
- *Check: `selfcoupling_check.py` → `selfcoupling_check_output.txt`.*
- *The uniqueness results are cited from the literature, not rederived here.*

## Summary

**Self-coupling fixes the completion.** Demanding that the joint theory's spin-2 field couple to
its own energy fixes its nonlinear completion as **general relativity plus a cosmological
constant**, given three conditions:
- Lorentz covariance of the linear theory;
- two derivatives;
- the stress-tensor ambiguities resolved as field redefinitions.

**Consequences:**
- It closes candidate 7's open lapse: Schwarzschild, not linear and not exponential.
- It makes horizons a consequence.
- It adds no selection, and pre-empts the one that choosing a lapse would have been.
- It adds one parameter (Λ) and two firm predictions (β = 1; horizons).

**Tensions and limits:**
- It is in tension with `main`'s fixed lattice, which GR's diffeomorphism invariance cannot treat
  as exact.
- **It does not change G, and so it does nothing for the hierarchy.**

## (1) Uniqueness (SC1 — HELD, with its conditions)

**The self-coupling chain.**
- A linear spin-2 field h coupled to T_matter requires ∂_μT^μν_matter = 0. But matter exchanges
  energy-momentum with h, so the linear theory is inconsistent for self-gravitating sources.
- Sourcing h with its own stress tensor, and iterating, gives the Einstein–Hilbert action. Deser's
  first-order (Palatini) form closes the iteration in one step.
- Wald's theorem shows that any consistent self-interaction of a massless spin-2 field is
  generally covariant.
- In four dimensions, at two derivatives, the only covariant option is **Einstein–Hilbert + Λ**
  (Lovelock).

**The conditions, and what they cost here:**

| condition | status in the joint theory | cost |
|---|---|---|
| (a) Lorentz-covariant linear theory | already selected by the spin-2 repair; exact only in the continuum limit of the lattice | none new |
| (b) two derivatives | `main`'s nearest-neighbour minimality; higher-curvature terms allowed but suppressed | none new (a (+1) if minimality is not accepted as covering it) |
| (c) stress-tensor ambiguities | resolved up to field redefinitions (Deser; Wald). The critiques (Padmanabhan 2008; Butcher, Hobson & Lasenby 2009) require these assumptions, but give no alternative two-derivative completion | none new |
| **Λ** | **not fixed** | **+1 parameter** (fitted), or a selection Λ = 0 |

## (2) The nonlinear lapse — candidate 7 closed (SC2 — HELD, one addition)

Check (U = GM/rc², g₀₀ = −N² = −(1 − 2U + 2βU² + …)):

| lapse | N² to O(U²) | β | N = 0 at |
|---|---|---|---|
| linear, 1 − U (candidate 7) | 1 − 2U + U² | **1/2** | U = 1 |
| exponential, e^{−U} (candidate 7) | 1 − 2U + 2U² | 1 | **never** |
| **GR, isotropic: (1 − U/2)/(1 + U/2)** | 1 − 2U + 2U² | **1** | **U = 2** (r_iso = GM/2c², the horizon; areal r = 2GM/c²) |

- **The self-coupled lapse is GR's.**
  - It shares β = 1 with the exponential, so 1PN does not distinguish them.
  - It reaches zero, which the exponential never does.
  - Its energy is bounded below (positive-energy theorem), unlike the linear lapse.
- **Addition, post-hoc:** the linear lapse also has **β = 1/2**. With γ = 1 that gives Mercury
  (2 + 2γ − β)/3 = 7/6 of the observed perihelion advance. So candidate 7's linear option was
  already excluded at 1PN. I did not state this in the hypotheses.

## (3) Horizons — a consequence, not a choice (SC3 — HELD, conditionally)

- **Within the self-coupled theory:**
  - by Birkhoff, the exterior of any spherical mass is Schwarzschild, with a horizon at 2GM/c²
    whenever the mass lies inside it;
  - above the maximum mass of any causal equation of state (Rhoades–Ruffini), collapse cannot be
    halted;
  - given the energy conditions, a trapped surface leads on to a singularity (Penrose).
- **A′'s matter satisfies the energy conditions**: its on-site potential is positive. A′ lumps grow
  more compact with N (JOINT_PREDICTIONS candidate 4, GM/R ~ G³N²ω/b²), so lumps above A′'s
  maximum mass **must** form horizons.
- **Horizons are therefore no longer a selection.** Candidate 7's earlier "any choice made to
  produce horizons would be a selection" no longer applies.
- **Condition (SC5):**
  - this holds in the continuum;
  - on `main`'s lattice the singularity is cut off at ℓ;
  - the horizon itself is a large-scale structure, and survives when GM/c² ≫ ℓ.
- **A′'s maximum lump mass is not computed here.** That needs building: a 3-D lattice with the
  full metric.

## (4) The count (SC4 — HELD)

**Selections.**
- Self-coupling is the *all-energy* selection already counted in the joint scope (b), applied to
  gravity's own energy. **+0.**
- It pre-empts the nonlinear-lapse selection that candidate 7 would otherwise have needed. That was
  never counted, so the listed total stays **7**. **The saving is real, but it shows as
  "not added", not as a reduction.**

**Parameters: +1 (Λ).**

**New firm predictions**, none used to choose the principle — which is the all-energy principle,
chosen for the equivalence principle:
- **β = 1** — the perihelion of Mercury (42.98″/century, with γ = 1) and the absence of a Nordtvedt
  effect (η = 4β − γ − 3 = 0, lunar laser ranging). Counted **once**.
- **Horizons and their strong-field dynamics** — black holes (EHT), and binary mergers with Kerr
  ringdown (LIGO/Virgo). Counted **once**.
- Both are known facts, reproduced — as with the equal fall of the chirality branches.

**Not counted:**
- light bending and tensor waves (they motivated spin-2);
- Λ's value (fitted);
- the quadrupole-formula orbital decay of binary pulsars. It needs the self-consistent source, but
  it overlaps the tensor-wave selection, so it is left out to avoid double counting.

| | parameters | selections | conditions | firm predictions |
|---|---|---|---|---|
| joint, repaired by spin-2 (branch now) | 5 + a state | 7 | ~9 | 7 |
| **+ self-coupling, projected** | **6 + a state** (+ Λ) | **7** (the nonlinear lapse pre-empted) | **~11** | **9** (+ β = 1; + horizons) |

- **The ratio improves** from 7 predictions per 12 inputs to 9 per 13.
- **Still below 1:1** with selections counted.
- **As a projection it is not falsified**, but it rests on the continuum limit (SC5).

## (5) The tension with `main` (SC5 — HELD)

- **GR's diffeomorphism invariance makes the geometry dynamical.** `main`'s lattice is fixed — a
  background with a preferred frame. The self-coupled theory is therefore GR only in the continuum
  limit. At ℓ:
  - diffeomorphism invariance is broken, and the extra modes it would remove can reappear (the
    Boulware–Deser ghost of massive gravity is the known danger);
  - matter's couplings must become metric-dependent (the lapse on-site and the metric on bonds —
    families L1 and L2 of INSIDE_OUT_SCOPE);
  - singularities are cut off.
- The same premise, a lattice base, is what makes the photon dispersion a prediction (candidate 5).
- **Recorded as a tension, not a selection.** A lattice realisation that keeps the ghost out is not
  scoped here (Regge calculus and causal dynamical triangulations are the known routes, and both
  make the lattice itself dynamical).

## (6) The hierarchy — not addressed (SC6 — HELD)

- **Self-coupling adds only higher orders in GM/rc², plus Λ.** The weak-field coefficient G is the
  linear coupling, untouched.
- **So it does nothing for λ ~ 10⁴⁰**, the hierarchy that KK_SCOPE (λ ≤ 4) and DILUTION_SCOPE
  (dilution impossible through A′'s radius) left open.

## Hypotheses

| | statement (short) | status |
|---|---|---|
| SC1 | unique completion, GR + Λ, under (a)–(c); Λ unfixed | **HELD** (cited) |
| SC2 | the lapse is Schwarzschild; β = 1, shared with the exponential; reaches zero, unlike the exponential | **HELD** — check: β = 1, 1, and N = 0 at U = 2 for GR; never for the exponential. Addition, not stated in advance: the linear lapse has β = 1/2 |
| SC3 | horizons a consequence (Birkhoff, maximum mass, Penrose; A′ satisfies the energy conditions); continuum only | **HELD** |
| SC4 | +0 selections (one pre-empted), +1 parameter (Λ), +2 firm predictions (β = 1; horizons) | **HELD** |
| SC5 | GR diffeomorphism invariance against the fixed lattice: a tension | **HELD** |
| SC6 | G untouched; the hierarchy not addressed | **HELD** |
