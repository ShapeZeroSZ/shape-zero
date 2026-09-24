# Hierarchies and Dimensional Transmutation

Where scale ratios could come from, given that the construction's inputs are
dimensionless. Written to a standard that permits checking: every number names
its script, and the gap between what is demonstrated and what is claimed is
stated in each section rather than at the end.

---

## 1. The problem is hierarchies, not scale

**One scale generates no ratios.** Every quantity built from a single
dimensionful input is either that input, or a pure number times it. Ratios of
such quantities are dimensionless combinations of the pure numbers — nothing new
enters. So positing one scale (§4 of `INPUT_LEDGER.md`) fixes units and produces
no separations.

**The Standard Model's hard problems are separations**, not magnitudes:
m_p/m_e ≈ 1836, the electroweak-to-Planck gap of ~10¹⁷, the neutrino mass scale
relative to everything. A theory that supplies one scale supplies none of these.

**Two routes exist.**

1. **A second independent input.** Honest, and it costs a prediction. Two inputs
   give one ratio, and that ratio is then fitted rather than derived.
2. **Dimensional transmutation.** A dimensionless coupling generates an
   exponentially separated scale from a single reference. Costs nothing.

Route 2 is the one worth wanting, because this construction has dimensionless
content in abundance and dimensionful content by posit only.

---

## 2. The quantum route is closed

Standard dimensional transmutation requires:

| ingredient | present in the ladder? |
|---|---|
| a dimensionless coupling | **yes** — κ, the a–b angle, β |
| renormalisation-group flow | **no** |
| a beta function of the right sign | requires the above |

Λ = μ exp(−1/(b·g(μ))) needs loops. The ladder is classical throughout — the
rungs are single-state dynamics (`z1_d4_rung.py` has no lattice index;
`z1_d8_attempt.py` is algebra only), and no quantisation has been performed.
Geometric quantisation was attempted and gives only integrality, which
discretises a scale rather than generating one (C1S §9).

**So the quantum route is unavailable without first quantising the ladder**,
which is a larger undertaking than anything attempted here.

---

## 3. The classical route, and why Nekhoroshev is the wrong instrument

Nekhoroshev bounds drift time below by exp(1/ε^a) — an exponentially large
timescale from a dimensionless perturbation. Correct as theory, and **practically
inaccessible**.

**Measured, on GPU.** A 2-DOF nearly-integrable system, T = 3×10⁶, dt = 0.06, ten
ε values from 0.15 to 0.02, both a noble and a near-rational winding, ~4,455
seconds per family:

- **no sustained escape at any ε** — the escape-time observable never fires
- secular drift rates 10⁻¹² to 10⁻¹⁰
- **integrator energy error dH ≈ 3×10⁻²**

The signal sits **nine orders of magnitude below the integrator's own energy
non-conservation**, and the noble secular rates are non-monotone in ε
(8.4×10⁻¹², 3.0×10⁻¹¹, 1.3×10⁻¹¹) — the signature of a numerical floor rather
than diffusion. The near/noble secular ratio also *shrinks* (84 → 9.6 → 8.5)
where the mechanism needs it to grow.

**This closes one path, not the route.** Arnold diffusion requires error control
below a signal that is itself exponentially small, over times that grow
exponentially in the quantity being measured. That is a known hard problem in
numerical dynamics, not a defect of this construction.

---

## 4. The arithmetic route: already inside the architecture

Scale separation does not require Nekhoroshev. It requires an exponential, and
one is present at D1 in the continued-fraction structure — with **no dynamics, no
long integration, and about one second of compute**.

**Convergent denominators grow geometrically.** For the noble ratio φ−1:

    denominators q_n : 1, 2, 3, 5, 8, 13, 21, 34, 55, 89
    growth ratio     : 1.615, 1.619, 1.618, 1.618        (φ = 1.618)

The denominators **are** the Fibonacci numbers, so **q_n ~ φⁿ**.

**Contrast with a near-rational** (0.6666667): denominators 1, 3, **10⁷**,
3.07×10⁹ — no regular growth, because the continued fraction terminates into a
huge partial quotient. The noble ratio gives an orderly ladder of scales; a
rational gives a cliff.

---

## 5. The strength law, measured

The arithmetic gives the *orders*. Whether the hierarchy is **dynamically real**
or merely formal depends on how resonance strength falls with order q. Measured
by **Greene's residue criterion** — exact periodic orbits, no tolerance
parameter, no statistical scan.

**Calibration.** K_c for the golden torus is known: **0.971635**. The measured
sequence descends monotonically toward it: **1.518 → 1.286 → 1.148 → 1.079 →
1.035** at q = 3, 5, 8, 13, 21 — within 6.6% of the exact value at the last
convergent. Rotation numbers track the convergents exactly (1/2, 2/3, 3/5,
5/8, 8/13) with |rot − φ⁻¹| falling 0.118 → 0.0027.

**Residues at K = 0.5**, from two independent implementations — finite-difference
Jacobian on a 48-grid, and analytic Jacobian on a 56-grid:

| q | R (finite-diff) | R (analytic) | ratio to previous |
|---|---|---|---|
| 2 | 6.25001×10⁻² | 6.25000×10⁻² | — |
| 3 | 3.52902×10⁻² | 3.52902×10⁻² | 0.56 |
| 5 | 8.56541×10⁻³ | 8.56541×10⁻³ | 0.24 |
| 8 | 1.20695×10⁻³ | 1.20697×10⁻³ | 0.14 |
| 13 | **4.06030×10⁻⁵** | **4.05749×10⁻⁵** | **0.034** |
| 21 | — | **1.55173×10⁻⁷** | **0.0038** |

Agreement to five significant figures through q = 8, three at q = 13 where the
residue is 4×10⁻⁵ and finite-difference error appears first.

**The falloff is faster than exponential in q.** A drop of **403,000×** from
q = 2 to q = 21, with the per-step ratio itself shrinking at every step — 0.56,
0.24, 0.14, 0.034, 0.0038. A pure exponential would hold that ratio constant; it
falls by two orders of magnitude across the sequence.

**So the hierarchy is dynamically real, not merely formal.** Composed with
q_n ~ φⁿ, the scale separation opens **doubly exponentially** in the convergent
index, from a dimensionless input.

---

## 6. What this gives, and what it does not

**Gives:** a mechanism generating unbounded scale separation from dimensionless
data, present at D1, verified twice by independent implementations against a
known critical value, at negligible compute cost.

**Does not give:** the *value* of any physical ratio. The mechanism produces
large numbers from small parameters; matching 1836 or 10¹⁷ requires the small
parameter to take a particular value, which returns to the input question. The
gain is structural — hierarchies become possible without a second scale.

**And the stability is bounded, not unconditional.** K_c is **finite**. Below it
the noble winding survives and the resonances that would destroy it are
exponentially suppressed; above it, even the golden torus breaks. D1 supplies a
stability criterion **with a threshold**, and any claim that structure rests on it
holds only inside that phase.

**φ is not predicted here.** φ is the maximally Diophantine number by
construction, so any noble ratio gives the same structure. What the ladder
supplies is a place where a winding ratio sits; the arithmetic and Greene's
criterion do the rest. *Attainability is not selection.*

---

## 7. Other mechanisms not yet examined

Three further dimensionless structures in the architecture could supply
separation and have not been tested:

- **rarity / inverse-probability weights** — 1/p for small p
- **thermodynamic or information-geometric penalties** — Boltzmann-type factors
  exp(−ΔF), exponential in a dimensionless argument
- **spectral properties of the strange-attractor core** — Lyapunov exponents and
  escape rates from a repeller are exponential by construction

Each is cheaper than Nekhoroshev and none has been examined. The relevant
question is no longer whether a hierarchy is *possible* — §5 settles that — but
which of these is **sharp and controllable**.

## 8. Status

| | |
|---|---|
| quantum transmutation | **closed** — no RG flow, ladder is classical |
| Nekhoroshev route | **closed practically** — signal 10⁹ below integrator error, 4,455 GPU-s, no escape |
| arithmetic route | **OPEN AND VERIFIED** — q_n ~ φⁿ, growth ratio 1.618 to four digits |
| strength-vs-order law | **MEASURED** — 403,000× falloff q = 2→21, faster than exponential, two implementations agreeing |
| instrument calibrated | **yes** — K_c → 0.971635, Greene's known value |
| hierarchy possible without a second scale | **yes** |
| any SM number derived | **no** |
| stability threshold | **finite** — K_c = 0.971635; above it even the golden torus breaks |
| untested mechanisms | rarity weights, thermodynamic penalties, attractor spectra (§7) |
