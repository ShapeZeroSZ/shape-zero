# Shape Zero — Verification Package (C1, updated through Z1)

Numerical and structural verification program for the Shape Zero
specification (spec/shape_zero_v5-3.pdf). Every quantitative claim is
backed by a self-contained script. **Environment: Python 3 + NumPy only**
(matplotlib for figures); ~1 min per script unless noted. Every prediction
was stated before measurement; the record includes the retractions,
instrument lessons, and inverted predictions, documented in the spec
changelog and the rung records.

## One command per result

| Script | Verifies |
|---|---|
| `phi_gauge_test.py` | Result 1 — U(1) synthetic gauge, Δω = −2cβ sin k (stiffness-free) |
| `phi_gauge_nonlinear.py` | Result 2 basis — nonlinear protection of the asymmetry |
| `phi_gauge_delta.py` | Result 2 corrected — closed-form PT; the documented preparation-bias retraction |
| `phi_gauge_closure.py` | Result 3 — direction-selective parametric decay |
| `phi_gauge_decaymap.py` | Result 7 — window map, β_res = 0.062 + 0.064A² (~4 min) |
| `phi_gauge_wilson.py` | Result 4 — non-commutative Wilson loops (imposed links; bridge result) |
| `phi_gauge_chiral.py` | Result 5 — dynamically generated u(2), no link matrices |
| `phi_gauge_precession.py` | Result 6 — amplitude-Zeeman precession Ω = C·A²·n_z |
| `hk_alignment_convexity.py` | Item 1 — horizon counterexample + exact local convexity |
| `s1_gradient_flow.py` | Item 1 dynamics — sharp rate; basin edge = π/2 |
| `s2_universality.py` | Universality on a generic control lattice (~5 min) |
| `s2b_winding.py` | Endurance vs Diophantine badness, Spearman +0.66 (~4 min) |
| `fano_reduction.py` | Triadic closure ⇏ Fano; the two minimal supplements |
| `s3_roles.py` | Role postulates ⇒ supplement (B) ⇒ Fano; König construction |
| `e1a_extraction.py` | Externalization — passivity vs parasitic extraction, linear response |
| `e1b_inference.py` | Objective inference; integrability residual; camouflage theorem |
| `s4_fullstack.py` | Unified model session 1 — branch split; U(1) branch-blind at 5 digits |
| `s4_session2.py` | Unified model session 2 — window dissolution; floor-as-barrier |
| `z1_d1_rung.py` | Zero ladder D1 — birth of integers; resonance tongues |
| `z1_d2_rung.py` | Zero ladder D2 — void = centrifugal (g = L²/2); phase protection exact |
| `z1_d4_rung.py` | Zero ladder D4 — structure sphere; exact-κ precession; passivity |

Top level: `shape_zero_predictions_v1.md` (three zero-fit hardware
predictions with falsification conditions) and `shape_zero_zero_ladder.md`
(the Z1 synthesis: the derivation line from zero to the Fano plane, with
the sharp questions for division-algebra researchers). Cover notes per
specialist audience in `cover_notes/`.
