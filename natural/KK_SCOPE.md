# A Kaluza–Klein reading of the joint theory — scoping report

*Scoping only: nothing is built and nothing is adopted.*
- *Hypotheses and predictions were committed first: `KK_HYPOTHESES.md` (88cf23d).*
- *Checks: `kk_scope_checks.py` → `kk_scope_checks_output.txt` (KC1–KC3).*
- *K2's ring/Higgs statement and K4's KK numbers (λ = 1 + a² = 4, Brans–Dicke ω = 0) are standard
  results, cited rather than recomputed here.*

## Summary

- **The gauge sector reads exactly as a KK structure on the node's target space.**
  - κ is the fibre–time component A₀ = −κ/2 (KC1: the model's force equals the KK-form force to
    2×10⁻¹⁵, with only that sign).
  - The Wilson links are the fibre connection.
  - The gauge-invariant coupling that JOINT_SCOPE found forced is what a reduction gives.
  - Larmor's theorem is a fibre diffeomorphism.
- **But the joint theory is not the KK reduction of a higher-dimensional gravity.**
  - The node's radius is the matter's own position on the fibre, not a modulus.
  - The circle has zero radius in A′'s vacuum.
  - The only vacuum with a finite circle, the ring, Higgses the U(1).
- **Positing the higher-dimensional metric anyway** would cut the discrete selections from 7 to
  3 (+1). But it forces every charged state to have λ ≤ 4, where nature has ~10⁴⁰–10⁴². **The KK
  reading is FALSIFIED as scoped.**

## (1) Term-by-term match (K1 — HELD)

**KC1: `model.py`'s single-node force against the Euler–Lagrange acceleration of**
L = ½|u̇|² + A₀u̇·𝕁u + ½A₀²|u|² − ½(√5 + κ²/4)|u|² − |u|³/3.
- That is, along the U(1) orbit, ½ṙ² + ½r²(θ̇ + A₀)² minus the rotating-frame potential.
- 200 random states per case:

| n | A₀ = −κ/2 | A₀ = +κ/2 |
|---|---|---|
| 1 | **1.8×10⁻¹⁵** | 5.4 (fails, as predicted) |
| 3 | **3.6×10⁻¹⁵** | 5.4 |

- **Predicted:** ≤ 10⁻¹⁰, with A₀ = −κ/2 in `model.py`'s 𝕁 convention. **Hit.**

| target-3 term | KK counterpart | status |
|---|---|---|
| gyroscopic κ𝕁u̇ | cross term r²θ̇A₀ with A₀ = −κ/2 | **exact** (KC1) |
| the κ²r²/8 rotating-frame mass shift | A₀²r²/2 from the same square | **exact** (KC1: needed for agreement) |
| Wilson links | the fibre connection g_{iθ} | match, structural |
| the forced gauge-invariant energy (JOINT_SCOPE L1) | reduction gives the stress tensor in D_μ | match — L1's forced coupling is the reduction's |
| Larmor = gauge | θ → θ + λ(x), a diffeomorphism of the total space | match |
| **Maxwell F², with free e** | from R₅, with 1/e² ∝ R²/G | **not derived** — the lattice has no fibre metric dynamics |
| **4D gravity** | from the 4D block of the higher metric | **not derived** — an ingredient, as in target 3 |

- **Under A′** the U(1) acts on all n components at once. Its orbit through a node's state is a
  circle of radius |u|, the node's total radius. So A′'s radius is the orbit radius at every n,
  and KC1 holds at n = 3 as at n = 1.
- **This is the standard fact that a charged field's U(1) gauge field is a connection on its
  circle bundle.** It is a KK structure of the *target* space. It is not a reduction of
  spacetime gravity.

## (2) The radial coordinate as the KK scalar (K2 — the identification FAILS, as predicted)

**KC2 — no third mode.** The linearised spectrum of a node about its vacuum has exactly two
distinct frequencies, **1.086435 (a) and 2.058171 (b)**, at n = 1 and n = 3 (no growth, |Re λ| ≤
3×10⁻¹⁶). There is **no separate radial (radion-like) mode**: at r = 0 radius and phase are not
separate degrees of freedom.

**Why r is not the KK scalar:**
- A KK scalar is the fibre's size at a point, a modulus shared by all matter there.
- r is the orbit radius of the matter's own state, and the charge (r²θ̇ plus the κ term) is built
  from it. It is a matter variable, not geometry.

**The resemblance is partial:**
- A′ shares one radius across all components at a site;
- `main`'s universal stiffness shift dK = |u| − |u_l| moves both branches by dK/(2ω_a + κ) alike,
  as a radion shifts every KK mass. But the forms are opposite (additive, and rising with the
  radius, against p_θ²/R², falling), and r does not set e.

**What stabilises it:**
- **the node well stabilises r — at r = 0**, the cone's apex. So in A′'s vacuum the circle has
  zero radius, and a KK coupling e ∝ 1/R would diverge;
- **the only vacuum with R ≠ 0 is the ring** (form B, `RING_FORK.md`). With a dynamical u(1) its
  phase Goldstone is eaten — the Higgs mechanism — and the **photon is massive**;
- **neither node form gives a KK vacuum with an unbroken U(1).**

## (3) What the reading forces, and the count (K3 — HELD as a projection)

**Posit a higher-dimensional metric on lattice × phase circle (1 selection).** Diffeomorphism
invariance of the total space then forces what the joint scope selected one by one:
- the Maxwell/Wilson form;
- one lapse and one shift for every sector;
- coupling to T⁰⁰ and to all energy, field energy included;
- spin-2 with a massless graviton;
- c_γ = c_g;
- and c_matter = c_γ, if matter couples minimally to the same metric.

**Still selected:**
- q = 3;
- identifying the fibre U(1) with 𝕁 (the model's own phase, not an arbitrary circle);
- (+1) how the missing radion is treated.

**e is traded for R** through e² ∝ G/R². A′ does not fix R, since its vacuum radius is 0. That is a
swap, not a saving.

| | numeric parameters | discrete selections | inputs |
|---|---|---|---|
| joint, repaired by spin-2 (current) | 5 + a state (ĉ, κ̂, e, G, shared speed) | 7 | 12 |
| **KK reading, projected** | **4 + a state** (ĉ, κ̂, G, R) | **3 (+1)** | **7 (8)** |

- **The inputs drop from 12 to 7 (8).** It is the largest reduction any target has offered.
- **But it arrives with a new strict prediction that fails** (4a below).

## (4) The known failure modes (K4)

**(a) Charge-to-mass — appears; FAILS.**
- **In KK**, charge is momentum around the circle. The pure KK states are extremal:
  λ ≡ e²q²/(4πGm²) = 1 + a² = 4 (a = √3 for one extra dimension), with the radion's attraction
  balancing the repulsion.
- **Any additional rest mass** — a 5D mass, or here the node well — **lowers λ**, so the reading
  forces **λ ≤ 4** for every charged state.
- **Nature's charged constituents have λ ~ 10⁴⁰–10⁴²** (electron ≈ 4×10⁴²). **Falsified by
  ~40 orders**, as in classical KK.
- **KC3 shows the lattice version:** every quantum has the same gauge-invariant mass and charge.
  - ω_a + κ/2 = ω_b − κ/2 = √(√5 + κ²/4) = **1.572303**;
  - |q|/m = **0.636010** = 2/(2ω_a + κ) (difference 0);
  - the lab-frame ratios (0.920 against 0.486) are gauge artefacts of the uniform A₀.
  - **One universal q/m, as for a KK tower, cannot produce nature's spread.**
  - *Rounding misses:* I predicted 1.5722 and 0.6361. The values are 1.5723 and 0.6360 at four
    digits. The identities themselves are exact.

**(b) The extra scalar — does not appear, but only because the reading is not genuine KK.**
- **In real KK a massless radion is a Brans–Dicke scalar** with ω = 0, giving γ = 1/2, against
  Cassini's 1 ± 2×10⁻⁵. So a stabilisation mechanism must be added.
- **Here** there is no separate radion (KC2). What would play it is the matter's gapped amplitude
  (≥ 1.0864, range 0.72 sites).
- **If a genuine fibre modulus were posited with the metric**, the failure returns, and the "(+1)"
  selection in (3) is its stabilisation.

**(c) Chirality — appears, as a reinterpretation.**
- Reduction on S¹ gives vector-like spectra, and the a/b branches are exactly the ±KK-momentum
  (±charge) partners: equal gauge-invariant mass, split in the lab only by the uniform potential,
  κ = 2e|A₀|.
- So **`main`'s "chirality branches" are charge-conjugate partners, not 4D chirality**, consistent
  with EM_SCOPE §4.
- That does not falsify anything in `main`: `main` never claimed Standard-Model chirality. But it
  **constrains how row 4 (the κ̂ floor κ\*) may be read.** Under a dynamical u(1), a uniform κ is a
  gauge choice, so κ\* is physical only as a statement in the lattice's own frame. EM_SCOPE flagged
  the same point.

## (5) Verdict and count (K5 — HELD)

- **What holds:**
  - the gauge sector is a KK structure on the node's target space, exact to 10⁻¹⁵;
  - it explains why JOINT_SCOPE's gauge-invariant coupling was forced.
- **What fails:**
  - the joint theory is not a reduction of higher-dimensional gravity — no modulus; the circle
    collapses in A′'s vacuum and is Higgsed in the ring's;
  - if the higher metric is posited anyway, it forces λ ≤ 4, **falsified** by ~40 orders.
- **Count.** Nothing is adopted, so **the branch's count is unchanged**, at the repaired joint row:
  - 5 parameters + a state;
  - 7 discrete selections;
  - ~9 conditions;
  - 7 firm predictions (corrected: the κ-gradient force is a verified consequence).
- **The KK row, projected:**
  - 4 + a state numeric, 3 (+1) selections;
  - ~9 conditions, plus one new condition (λ ≤ 4) that **fails**;
  - 7 firm predictions and 1 falsified;
  - **FALSIFIED as scoped.**

## Hypotheses

| | statement (short) | status |
|---|---|---|
| K1 | the couplings match a KK structure term by term; KC1 to ≤ 10⁻¹⁰ with A₀ = −κ/2; F² and 4D gravity not derived | **HELD** — 1.8×10⁻¹⁵ / 3.6×10⁻¹⁵ with A₀ = −κ/2; +κ/2 fails at 5.4 |
| K2 | r is not a KK scalar; there is no third mode at r = 0; the circle vanishes in A′'s vacuum and is Higgsed in the ring's | **HELD** — KC2: only 1.086435 and 2.058171 at n = 1, 3 |
| K3 | a posited higher metric forces six joint selections; inputs 12 → 7 (8); e swapped for R | **HELD** (projection) |
| K4 | charge-to-mass fails (λ ≤ 4 against ~10⁴⁰); no radion here; branches are charge conjugates | **HELD** — KC3 exact; rounding misses at the fourth digit (1.5722 → 1.5723, 0.6361 → 0.6360) |
| K5 | KK on target space, not a reduction of gravity; the count is unchanged; the KK row falsified as scoped | **HELD** |
