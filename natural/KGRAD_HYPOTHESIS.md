# Candidate 3 — the κ-gradient force: hypothesis and predictions

*Committed before `kgrad_test.py` was written or run. From `JOINT_PREDICTIONS.md`, candidate 3.*

**Setting — `main`'s own lattice, no dynamical links.**
- `04_scripts/session/model.py` (identical to `main`), `Lattice(n = 1, q = 1)`, with its force
  law, node form A′, time step DT = 0.02 and RK4 integrator.
- **The only change:** the scalar κ is replaced by a static site-dependent κ(x) = κ\* + κ′·(x − x₀).
- Packets at rest (k = 0), amplitude 10⁻³, pure a-branch or pure b-branch in `model.py`'s own
  convention. Branch a: ψ ∝ e^{−iω_a t} with ω_a² + κω_a = Q. Branch b: ψ ∝ e^{+iω_b t} with
  ω_b = ω_a + κ.

**Hypothesis.** A static κ gradient acts on the two chirality branches as an electric field acts
on opposite charges. It is the local-frame form of Larmor's theorem: κ(x) = 2eA₀(x) (EM_SCOPE §4).
- Eikonal: ω_{a,b}(k, x) = ω_rot(k, x) ∓ κ(x)/2, with ω_rot = √(K + κ²/4 + ck²) and v = ck/ω_rot.
- **The accelerations are**
  - **opposite part:** a_a − a_b = **c·κ′/ω_rot**;
  - **common part:** −c·κκ′/(4ω_rot²) each, from κ entering ω_rot.

**Numerical predictions** — at κ\* = 0.971737, K = √5, c = 1: ω_rot = 1.57230, c/ω_rot = 0.6360.

| κ′ | a_a | a_b | a_a − a_b |
|---|---|---|---|
| +4×10⁻⁴ | **+8.79×10⁻⁵** | **−1.665×10⁻⁴** | **+2.544×10⁻⁴** |
| −4×10⁻⁴ | −8.79×10⁻⁵ | +1.665×10⁻⁴ | −2.544×10⁻⁴ (sign reversal) |
| 0 (control) | 0 | 0 | 0 |

- The a-branch moves toward larger κ; the b-branch away.
- **Pass criterion:** the difference and the common part each within 3% of the prediction. The
  scalar sector's free fall at comparable settings was accurate to about 0.5–2%, the shortfall
  being a velocity effect.

**If it holds:** it is a derivation on `main`'s own terms — `main`'s force law, no adaptation —
and a merge candidate for `main` (charter rule 6).
