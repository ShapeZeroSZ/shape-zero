# The Pinned Asymmetry — A Zero-Parameter Bench Test

**One measurable claim, no fitted parameters, sharp pass/fail.** This is the
only falsifiable prediction in the Shape Zero program that requires neither
cosmology nor a signature it does not have. It has been in the package since
v5.1 and is verified in its own numerics.

---

## 1. The claim

A one-dimensional array of nonlinear oscillators with **antisymmetric
velocity coupling** develops a synthetic gauge field. The band becomes
asymmetric in wavenumber, and the asymmetry is

**Δω(k) ≡ ω(k) − ω(−k) = −2 c β sin(k)**

- **c** — nearest-neighbour elastic coupling
- **β** — antisymmetric velocity-coupling ratio (the gyroscopic term)
- **k** — wavenumber

All three are fixed by **linear spectroscopy of the same platform**. Nothing is
fitted to the nonlinear regime.

## 2. Why it is a test and not a fit

The prediction is an **invariance**, not a curve.

From the exact dispersion root ω² + 2cβ sin(k)ω − W² = 0, the asymmetry is
−2cβ sin(k) **for any W²**. The on-site nonlinearity is direction-blind, so it
shifts W² symmetrically and **cancels in the difference**. Therefore:

| quantity | behaviour with drive amplitude |
|---|---|
| band centre ½(ω(k) + ω(−k)) | **softens** — a real, visible nonlinear effect |
| asymmetry ω(k) − ω(−k) | **pinned** — must not move |

So the experiment varies amplitude and watches whether the centre moves while
the asymmetry does not. One knob, two observables, opposite predicted responses.
A fit cannot produce that; only the mechanism can.

**Conditions versus parameters:** the asymmetry is one prediction with zero free
parameters, checked across a continuum of amplitudes. Conditions exceed
parameters by the number of amplitudes measured, which is the criterion that
separates evidence from fitting.

## 3. Numerical verification

`phi_gauge_nonlinear.py`, N = 64, c = 1.0, k = π/2, γ = 0, over a 400-fold
amplitude range, each direction seeded at its own linear root (re-run
2026-09-24, `PROVENANCE.md` §6o):

| β | A | ω(+k) | ω(−k) | centre | **Δω** |
|---|---|---|---|---|---|
| 0.00 | 0.001 | 2.05817 | 2.05817 | 2.05817 | **0.00000** |
| 0.00 | 0.400 | 2.04199 | 2.04199 | 2.04199 | **0.00000** |
| 0.05 | 0.001 | 2.00878 | 2.10878 | 2.05878 | **−0.10000** |
| 0.05 | 0.100 | 2.00781 | 2.10779 | 2.05780 | **−0.09998** |
| 0.05 | 0.200 | 2.00489 | 2.10482 | 2.05485 | **−0.09993** |
| 0.05 | 0.400 | 1.99278 | 2.09246 | 2.04262 | **−0.09968** |

Theory: −2cβ sin(π/2) = **−0.1000**.

- **Centre softens 0.8%** across the amplitude range — the nonlinearity is real
  and visible.
- **Asymmetry drifts 0.3%**, from −0.10000 to −0.09968 — its magnitude
  **shrinks** with amplitude.
- **β = 0 gives exactly 0.00000** at every amplitude — the control fires.

> **RETRACTED rows (β = 0.05), seeded with both directions at the β = 0
> frequency:** A = 0.100: 2.00779 / 2.10782 / −0.10003; A = 0.200: 2.00479 /
> 2.10491 / −0.10012; A = 0.400: 1.99237 / 2.09287 / −0.10050 — "asymmetry drifts
> 0.5%, from −0.10000 to −0.10050". The O(β) velocity mismatch left a
> counter-propagating admixture of 1.25×10⁻² (own root: 1.5×10⁻³, the finite-record
> projection floor) whose frequency pulling reversed the sign of the drift. The
> β = 0 rows, the A = 0.001 row and the centre column are unchanged.

## 3b. The correction is measured: β·A², with a pure-number coefficient

The script was written to look for a β·A² correction. It is there, and it has
been measured. Values below are from `phi_gauge_nonlinear.py`'s `run_wave` and
`mode_freq` with each direction at its own root (re-run 2026-09-24, `PROVENANCE.md`
§6o); the retracted values, from the β = 0 seed, are kept alongside.

**Scaling in amplitude.** Residual (Δω + 2cβ sin k) divided by A², β = 0.05:

| A | residual / A² | retracted (β = 0 seed) |
|---|---|---|
| 0.1 | +0.001757 | −0.002978 |
| 0.2 | +0.001793 | −0.003035 |
| 0.3 | +0.001869 | −0.003061 |
| 0.4 | +0.001996 | −0.003109 |

Constant to **14%**, rising with A (retracted: 4%). The correction is A² at
leading order; the rise is a higher-order term and appears in the reference too
(κ = −0.0176 → −0.0200 over A = 0.1–0.4, `pinned_asymmetry_reference.py`).

**Scaling in β.** The coefficient (mean residual/A² over A = 0.1–0.4) divided by
β, over a tenfold range:

| β | coefficient | coeff/β | retracted coefficient | retracted coeff/β |
|---|---|---|---|---|
| 0.02 | +0.000734 | +0.03670 | −0.001229 | −0.06146 |
| 0.05 | +0.001853 | +0.03707 | −0.003046 | −0.06091 |
| 0.10 | +0.003510 | +0.03510 | −0.006302 | −0.06302 |
| 0.20 | +0.007442 | +0.03721 | −0.011783 | −0.05891 |

Constant to **6%** (retracted: 7%). So the correction is **proportional to β**,
and the coefficient is a **pure number** fixed by c and k alone. Its sign is
opposite to the retracted table: coeff/β ≈ +0.036 agrees with second-order PT
(+0.0349, `phi_gauge_delta.py`), and κ = −(coeff/β)/(2c sin k) ≈ −0.018 agrees
with the reference β-sweep (−0.0184 ± 0.00033). `phi_gauge_closure.py`, fitting
over A = 0.05–0.20 and six β, emits δ = +0.0357·β (retracted: −0.0598·β).

**The full prediction:**

**Δω(k, A) = −2 c β sin(k) · [1 + κ A²]**   (k = π/2, c = 1)

> **⚠ κ = 0.0799 IS RETRACTED (2026-09-24). Nothing below is deleted; read it
> through this note.** `pinned_asymmetry_reference.py` seeded each direction with
> the *other* direction's root (its gyro term has the opposite sign to the platform
> scripts, so +k is the upper root there; "+" was seeded with the lower). The O(β)
> velocity mismatch biased the A² coefficient. With each direction seeded at its
> own root, the same integrator and estimator give **κ = −0.0187** (β = 0.05,
> A = 0.30) and **−0.0184 ± 0.00033** over the β-sweep — |Δ/Δ₀| = 0.998339,
> 0.998316, 0.998390, 0.998316 at β = 0.02, 0.05, 0.10, 0.20. The asymmetry
> magnitude **shrinks** with amplitude, as second-order PT predicts (−0.0175,
> `phi_gauge_delta.py`); confirmed with separate code. **Scripts:**
> `pinned_asymmetry_reference.py` (fixed), `model.py` gate 6.
>
> *κ = −0.0187 is the value **at A = 0.3**. The small-amplitude coefficient is **−0.0175** —
> derived by perturbation theory (−0.01748) and measured (−0.0176 at A = 0.10) — and
> a fourth-order term adds about 7% by A = 0.3.* [**CORRECTED 2026-09-25:** the
> true fourth-order term adds about **1.5%** at A = 0.3. −0.0187 is the
> **plain-cosine-launch** value at A = 0.3; the exact travelling wave gives
> **−0.01775** there. About 80% of the growth over −0.0175 is a launch effect —
> "κ to fourth order, and the launch", below.]
>
> **Unaffected:** the linear pinned asymmetry Δω = −2cβ sin k — the claim this
> document is built on — and the Lean missions (Prove2Me 2, 3, 4a, 4b), which are
> linear-order. The β-collapse also survives (β-independent to 3×10⁻⁴).
> **Unverified pending re-measurement:** the geometry table and κ(w, side) values,
> measured with the swapped seeding. [κ(w = 2, side) since re-measured and
> **CLOSED**, below the geometry table; the geometry table stays unverified [since
> SUPERSEDED by the width scan below; w = 3 CLOSED], and a
> localised-beam κ quoted without a box size is not a property of the beam.] **κ versus stiffness is re-measured:** the
> retracted 0.0959 / 0.0799 / 0.0677 become **−0.0214 / −0.0187 / −0.0165** at
> stiffness 0.90 / 1.00 / 1.10 (`shape_zero_tests/joint3_kappa_stiffness.py`), with
> the linear ratio at 0.99999 throughout — the pin is protected, κ is not. The earlier §3b table (−0.003046 at β = 0.05, from
> `phi_gauge_nonlinear.py`) seeded both directions at the β = 0 frequency — a
> different mismatch, since corrected: §3 and §3b now carry own-root values
> (+0.001853 at β = 0.05). Trail: `PROVENANCE.md` §6o.

**⚠ κ IS GEOMETRY-DEPENDENT. The value 0.0799 is for a 1D CHAIN.** [κ values in
this table and the next paragraph: 0.0799 RETRACTED, the rest UNVERIFIED — see
note above. The table is SUPERSEDED by the width scan ("κ across beam widths",
below): w = 3 re-measured and CLOSED.]

| geometry | Δω (theory 0.100000) | κ |
|---|---|---|
| **1D chain** | 0.100003 | **0.0799** |
| 3D, transverse width 2 | 0.100001 | **0.0168** |
| 3D, transverse width 3 | 0.100001 | **0.0311** |

**The pinning survives every geometry** — Δω tracks theory to 10⁻⁵ or better in
all cases, and *that* is the invariance the experiment tests. **The A² correction
does not**: κ falls with transverse extent and depends on the beam profile,
because the correction comes from the packet's self-interaction and a
transversely-spreading packet dilutes the amplitude driving it. [That mechanism
was tested and failed: κ is not proportional to fill fraction — see below.]

**κ(w = 2, side) re-measured (2026-09-24) — measured and open [now CLOSED, below].** `shape_zero_tests/kappa_readout_test.py` (β = 0.05, A = 0.30, q = 3, transverse
width w = 2, periodic BC, T = 300, uniform transverse readout). Its gyro term is
the reference convention, but it had seeded +k with the lower root — the §6o
swap — and one carrier frequency for every transverse component. It now builds
the initial velocity in Fourier space, each wavevector at its own branch
frequency (counter-propagating content 10⁻¹⁶; one carrier at the right root
still leaves 1.8–2.2%, the swap 4.0%). Outputs: `kappa_readout_test_output.txt`
(fixed) and `kappa_readout_test_swapped.txt` (`--swapped-seed`, the original run).

| side | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| transverse fill fraction | 0.194 | 0.087 | 0.049 | 0.022 | 0.012 |
| **κ, own-branch seed** | **−0.00447** | **−0.00289** | **−0.00204** | **−0.00115** | **−0.00047** |
| κ / κ(plane wave, −0.01869) | 0.239 | 0.155 | 0.109 | 0.062 | 0.025 |
| that ratio / fill fraction | 1.23 | 1.77 | 2.22 | 2.82 | 2.05 |
| κ, swapped seed, re-run — RETRACTED | +0.01772 | +0.01128 | +0.00801 | +0.00457 | +0.00185 |
| κ, swapped seed, as quoted in the script's docstring — RETRACTED | 0.0175 | 0.0113 | 0.0082 | **0.0027** (does not reproduce) | 0.0018 |

- **Plane-wave control: κ = −0.01869** (linear ratio 0.999993) at every side and
  readout — the **fourth independent confirmation** of the corrected value, after
  the fixed `pinned_asymmetry_reference.py`, the separate-code check, and
  `joint3_kappa_stiffness.py`. (Swapped seed: +0.07994, the retracted value.) It
  cannot test the readouts: a transverse-uniform wave is the 1D problem at every
  side.
- **Linear pinning holds** at 0.999997–1.000000 for every side and readout.
- **κ still has no side-independent limit.** It falls tenfold from side 8 to 32
  and steepens at the end (≈ side⁻³ from 24 to 32) [within error — below]. **No κ(w) is published.**
- **Mechanism unresolved** [since CLOSED — a fixed dilution with factor F = 2 − s, below]. It is **not seeding**: new/old is −0.25 at every side
  to 1%, so the fix changed κ's sign and scale but not its side-dependence. It
  is **not readout**: the uniform readout is clean (phase residual 0.003–0.011);
  the weighted and centre readouts are invalid as frequency readouts (residual
  0.5–3.5 rad) and show the same trend. It is **not proportional to fill
  fraction** — predicted before the run (transverse spreading dilutes the
  amplitude driving the shift) and **failed**: |κ|/(|κ_pw|·fill) runs 1.23 →
  2.82, not constant.
- **The original docstring's 0.0027 at side 24 does not reproduce**: the
  original script, re-run as supplied, gives +0.00457. The other four sides
  agree with it to ~2%.

Status: κ(w = 2, side) was **measured and open**; it is now **CLOSED** — below. The other transverse widths
(w = 3, the geometry table) were measured with `pinned_asymmetry_reference.py`'s
single-carrier seed and remain **unverified**. [Since SUPERSEDED by the width
scan — "κ across beam widths" below.]

**κ(w = 2, side) — CLOSED (2026-09-24).** Three GPU scans (Google Colab, Tesla
T4, PyTorch float64, fixed-step RK4 dt = 0.01; physics, Fourier-space seed and
uniform readout identical to `kappa_readout_test.py`), each with its output in
`shape_zero_tests/`: `kappa_side_gpu.py`, `kappa_boxscan_gpu.py`,
`kappa_extended_gpu.py` (`*_colab_output.txt`). All passed every validation
check; `kappa_extended_gpu.py` reproduces the table above exactly at every side
and extends it to L = 80. Re-measured here on CPU with the saved
`kappa_extended_gpu.py` (NumPy backend) at L = 20, 24, 28, 32: every κ, error bar,
[0, 150] value and energy drift identical to the Colab output.

| L | fill | κ ± err | F = κ/(κ_pw·fill) ± err | model 2 − s |
|---|---|---|---|---|
| 8 | 0.19387 | −0.00447 ± 0.00011 | 1.23 ± 0.03 | 1.35 |
| 12 | 0.08726 | −0.00289 ± 0.00014 | 1.77 ± 0.08 | 1.66 |
| 16 | 0.04909 | −0.00204 ± 0.00021 | 2.22 ± 0.23 | 1.80 |
| 20 | 0.03142 | −0.00147 ± 0.00030 | 2.50 ± 0.52 | 1.87 |
| 24 | 0.02182 | −0.00115 ± 0.00036 | 2.81 ± 0.89 | 1.91 |
| 28 | 0.01603 | −0.00075 ± 0.00034 | 2.49 ± 1.13 | 1.94 |
| 32 | 0.01227 | −0.00047 ± 0.00015 | 2.03 ± 0.63 | 1.95 |
| 36 | 0.00970 | −0.00040 ± 0.00016 | 2.22 ± 0.86 | 1.96 |
| 40 | 0.00785 | −0.00033 ± 0.00017 | 2.26 ± 1.16 | 1.97 |
| 48 | 0.00545 | −0.00025 ± 0.00019 | 2.43 ± 1.83 | 1.98 |
| 64 | 0.00307 | −0.00018 ± 0.00021 | 3.22 ± 3.64 | 1.99 |
| 80 | 0.00196 | −0.00017 ± 0.00022 | 4.59 ± 5.91 | 1.99 |

(κ over [0, 300]; the error bar is the larger of the weighted-vs-unweighted
phase-fit difference and the fit's standard error. s = share of the beam's power
in its box-wide transverse component.)

1. **The box-size dependence is static in origin.** It is present from t = 0 —
   κ(L = 16) over [0, 150] and over [0, 300] are both −0.00204 — and it is a clean
   A² coefficient (−0.00201 / −0.00202 / −0.00204 at A = 0.10 / 0.20 / 0.30, flat
   to 1%). **The prediction of a dynamic origin failed.** A slow secondary energy
   transfer into the measured wave is also present, riding on top and not the
   cause: its amplitude grows up to 11% (L = 16, T = 1200) and κ drifts ~10% over
   long runs (−0.00204 at T = 300 → −0.00184 at T = 1200), while the plane-wave
   control stays put (−0.01869 early, −0.01874 late). `kappa_side_gpu.py`.
2. **κ_box = κ_pw × fill × F**, with F rising from 1.23 ± 0.03 at L = 8 toward
   about 2. The cross-versus-self model **F = 2 − s** (the box-wide component is
   shifted by itself with weight 1 and by the beam's sideways components with
   weight 2; nothing fitted) is consistent within error — within two error bars,
   largest 1.8σ at L = 16 — at every L ≥ 12, though all eleven of those points
   lie above it. It is **9% high at L = 8** (1.35 against 1.23 ± 0.03). Deriving
   the exact cross factor for this lattice is the one refinement left. [Done:
   derived below — "The cross-modulation factor F, derived".]
   **What failed earlier still stands as failed:** dilution alone (F = 1, κ ∝
   fill — the measured F is 1.2–2.8) and dilution by transverse spreading during
   the run (the origin is static; and, as `kappa_side_gpu.py` notes, spreading
   alone cannot change the box average, since total energy is conserved). **What fits is a fixed dilution with factor
   F = 2 − s.**
3. **κ goes to zero as the box grows, keeping its sign.** Every measured value is
   negative. From L = 64 it cannot be distinguished from zero by this method
   (−0.00018 ± 0.00021 at 64, −0.00017 ± 0.00022 at 80); nothing suggests a sign
   change.
4. **No oscillation in F is resolved.** Every departure from its neighbours is
   within its error bar — e.g. +13% ± 32% at L = 24, −14% ± 31% at L = 32. The
   "oscillation" and "does not settle" readings of `kappa_boxscan_gpu.py`, which
   has no error bars, are withdrawn; so is the steepening noted above (≈ side⁻³
   from 24 to 32), which is inside the error bars.
5. **Only the plane-wave κ = −0.0187 is a real coefficient.** [−0.0187 is the
   plain-cosine-launch value at A = 0.3; the exact travelling wave gives −0.01775
   there, and −0.017480 at small amplitude — "κ to fourth order", below.] A localised beam's
   box-averaged κ has no box-independent value. **A localised-beam κ quoted
   without a box size is not a property of the beam.** The w = 3 value and the
   geometry table remain **unverified**; by this result, even re-measured they
   would describe a beam in a particular box. [Both since SUPERSEDED by the width
   scan below; the w = 3 item is CLOSED.]

*Provenance of the outputs:* the three `_colab_output.txt` files are annotated
transcripts, not raw output; the raw outputs are in the matching
`*_colab_raw.txt` files. Each opens with a note; the verdicts printed by the
first versions of `kappa_side_gpu.py` ("DYNAMIC") and `kappa_boxscan_gpu.py`
(Q1 "NO BUMP", Q2 "DOES NOT SETTLE") were wrong and were corrected afterwards,
and the scripts saved here carry the corrected reading code — re-running them
prints different verdict text over the same tables. The extended output's
validation and Q2 lines are condensed from the per-size lines the script
prints. The tables are the result.

**κ across beam widths — w = 3 CLOSED (2026-09-24).** `kappa_widthscan_gpu.py`
(Colab, Tesla T4; raw `kappa_widthscan_gpu_colab_raw.txt`, reading
`kappa_widthscan_gpu_colab_output.txt`) measures F in four groups of matched
proportions w/L, since a Gaussian beam's fill and s depend on w/L alone.
`kappa_resolution_test.py` (NumPy on CPU; `kappa_resolution_test_raw.txt`) holds
w/L = 1/4 and varies w from 1 to 6; it reproduced the Colab values for w = 2, 3
and 4 exactly. Physics, seed, readout and error bar as `kappa_extended_gpu.py`.

| w/L | w = 1 | w = 1.5 | w = 2 | w = 3 | w = 4 | w = 5 | w = 6 | 2 − s |
|---|---|---|---|---|---|---|---|---|
| 1/4 | 1.15 ± 0.01 | — | 1.23 ± 0.03 | 1.32 ± 0.04 | 1.33 ± 0.07 | 1.29 ± 0.10 | 1.34 ± 0.16 | 1.34–1.38 |
| 1/8 | — | 2.01 ± 0.12 | 2.22 ± 0.23 | 2.38 ± 0.65 | 1.61 ± 0.53 | — | — | 1.80 |
| 1/12 | — | — | 2.81 ± 0.89 | 1.99 ± 0.83 | 2.78 ± 1.66 | — | — | 1.91 |
| 1/16 | — | 2.86 ± 0.86 | 2.03 ± 0.63 | 2.60 ± 1.85 | 4.82 ± 3.09 | — | — | 1.95 |

(F = κ/(κ_pw·fill). The w = 1, 5, 6 entries are from the resolution test.)

1. **The box-size mechanism is general, not tuned to w = 2.** At matched w/L, F
   agrees across beam widths 1.5 to 4 within two combined error bars in every
   group (largest differences 1.8, 1.1, 0.7, 0.9). The test is sharp only at
   w/L = 1/4 (error bars ±0.03–0.07), partly sharp at 1/8, and **not a real test
   at 1/12 and 1/16**, where error bars reach ±3 — agreement there means only
   "not resolved".
2. **The 9% gap between F and 2 − s at w = 2, L = 8 is a narrow-beam lattice
   effect, established at w/L = 1/4:** the gap is −16.5% at w = 1 (18.6 of its
   error bars) and −8.8% at w = 2 (4.0), and within error for w = 3 to 6 (−2.2%,
   −1.4%, −3.7%, −0.1%). Holding w/L fixed holds the beam's geometry fixed, so the
   gap tracks the beam's width itself. The reading: a narrow beam carries sideways
   ripples on the scale of single lattice sites, where lattice waves differ from
   smooth space, and the model assumes smooth. It is **not** coarse sampling of
   the beam's shape — fill and s are computed on the actual grid. The prediction
   (the gap grows as the beam narrows and vanishes for wide beams) was stated
   before running and **held**. [**CORRECTED 2026-09-24:** at A = 0.30 the gap
   is **not a pure lattice effect**. It is partly the derived lattice kernel and
   partly fourth order in amplitude: from A = 0.10 to 0.30 the plane-wave κ grows
   6.5% while the narrow beam's barely changes, and at A = 0.10 the derived kernel
   accounts for the whole gap (w = 1: 1.221 ± 0.018 against 1.225; w = 2: 1.312 ±
   0.031 against 1.318). The prediction's result stands; its reading is corrected —
   see "The cross-modulation factor F, derived" below.] [**CORRECTED 2026-09-25:** the
   plane-wave growth that lowers F at A = 0.30 is **mostly a launch effect**, not
   fourth-order physics, acting through the plane-wave normalisation. With
   orbit-consistent (second-order) launches F at A = 0.30 is 1.201 (w = 1) and
   1.281 (w = 2), against 1.154 and 1.234 from plain cosines and 1.225 and 1.318
   derived at second order: the launch accounts for 66% and 56% of the gap, and the
   rest matches the plane wave's true fourth-order and velocity terms — "κ to fourth
   order, and the launch", below.]
3. **Unresolved:** whether F exceeds 2 − s at smaller w/L (10 of 14 width-scan
   points lie above it). The error bars there are too large to say. [**Since
   resolved:** it does, and the phase-coherent off-diagonal terms are why —
   derived below.]
4. **w = 3 κ, correctly seeded:** −0.00478 ± 0.00013 at L = 12, −0.00218 ± 0.00059
   at L = 24, −0.00081 ± 0.00034 at L = 36, −0.00060 ± 0.00043 at L = 48. These
   **supersede** the old w = 3 value (0.0311) and the old geometry table (1D 0.0799,
   w = 2 0.0168, w = 3 0.0311; swapped seeding, single carrier), which stay
   visible where they were quoted. By result 5 above, each is a beam in a
   particular box. **The w = 3 item is CLOSED.**

**The cross-modulation factor F, derived (2026-09-24).** `kappa_cross_pt.py`
extends the second-order perturbation theory behind the closure result
(δ = +0.0349β; the script reproduces the plane-wave κ as −0.01748 against
−0.01745) to the **cross kernel** R(q⊥) = K(q⊥)/K(0) on the lattice: the
direction-odd shift of the box-wide component caused by one sideways component
q⊥, per unit of that component's amplitude², in units of the self term. It keeps
the intermediate sum mode (2K, q⊥), the difference mode (0, −q⊥) and the DC mode,
each at its own lattice frequency. Nothing is fitted.

- **R → 2 as q⊥ → 0**, recovering the smooth model's cross-versus-self factor;
  on the lattice it falls to 1.39 at q⊥ = (π, 0) and 1.12 at (π, π).
- **The direction-odd part comes only from the sum mode.** The difference mode
  (kx = 0, frequency Ω₀ − Ω_q, the same in both directions), the DC mode and the
  frequency denominator are all even in direction; the odd part is carried by the
  sum mode at (π, q⊥), whose detuning grows with q⊥. That is why a narrow beam's
  sideways components cross-modulate less.
- **Checked directly** (`kappa_cross_kernel.py`, two-wave runs: a probe at (K, 0)
  plus one pump at (K, q⊥), 16 values of q⊥): at pump amplitude 0.10 the measured
  R matches the derived kernel within about one error bar at 15 of 16 points
  (1.389 ± 0.012 against 1.391 at (π, 0); 1.116 ± 0.011 against 1.118 at (π, π)),
  2.8σ at the smallest q⊥. At 0.30 it sits 3–4% below at large q⊥ — beyond second
  order.

Two predictions for F were recorded before comparison
(`kappa_cross_pt_output.txt`): **P3a** sums the kernel over each beam's grid
components (the diagonal terms only); **P3b** also keeps the **phase-coherent
off-diagonal terms** — three components such as (a, 0), (0, b) and (a, b) that
stay in step on the lattice, because the lattice dispersion is a sum over axes —
read out over the experiment's [0, 300] window, with linear detunings.
`kappa_cross_compare.py`, against every measured F (sharp = error bar ≤ 0.25,
match = within two error bars):

| | sharp points matched |
|---|---|
| 2 − s (smooth model) | 7 of 9 |
| P3a, diagonal kernel sum | 4 of 9 |
| **P3b, all triads** | **7 of 9** — every point except w = 1, L = 4 (1.225 against 1.15 ± 0.01) and w = 2, L = 8 (1.318 against 1.23 ± 0.03), both at A = 0.30 |

**The rise of F above 2 − s at small w/L comes from the phase-coherent
off-diagonal terms** (item 3 above): P3a stays below 2 − s there and misses the
sharp w/L = 1/8 and L = 12 points; P3b rises above it and lies within 1.1 error
bars of every non-sharp point.

**The two narrow-beam misses are fourth order in amplitude** (`kappa_cross_amplitude.py`).
Re-measured at A = 0.10 they give 1.221 ± 0.018 (w = 1, L = 4; derived 1.225)
and 1.312 ± 0.031 (w = 2, L = 8; derived 1.318). From A = 0.10 to 0.30 the
plane-wave κ grows 6.5% (−0.01755 → −0.01869) while the narrow beam's κ barely
changes (−0.004121 → −0.004147), so F, which divides by the plane-wave κ, falls.
[**CORRECTED 2026-09-25:** that plane-wave growth is mostly the plain-cosine launch,
not fourth-order physics — the exact travelling wave grows 1.5% from A → 0 to 0.30,
not 6.9% — see "κ to fourth order, and the launch", below.]

**Out-of-sample test** (`kappa_cross_oos.py`). Three never-measured beams, chosen
where P3a and P3b differ most in small boxes, measured at A = 0.10; both
predictions recorded and committed before the run
(`kappa_cross_oos_predictions.txt`, commit `3b19bf7` on branch
`claude/p1-resonance-window-disagreement-o3m1zm`). Validation passed, including
an 8 × 4 × 4 box reproducing the cubic L = 4 value (x length does not matter).

| beam | measured F | P3a | P3b |
|---|---|---|---|
| w = 0.75, 8 × 6 × 6 | 1.421 ± 0.111 | 1.536 (+1.0σ) | **1.425 (0.0σ)** |
| w = 1.25, 8 × 10 × 10 | 1.988 ± 0.123 | 1.676 (−2.6σ) | **1.994 (+0.1σ)** |
| w = 1.0, 12 × 12 × 12 | 2.127 ± 0.175 | 1.732 (−2.3σ) | **2.131 (0.0σ)** |

P3b lands on all three, including the one where it predicts **below** P3a. The
discrimination is weaker than planned: the error bars at A = 0.10 came out
0.11–0.18, not the 0.02–0.12 expected, so no configuration separates the two
predictions by the four error bars set in advance; P3a is excluded at 2.6σ and
2.3σ, and the first beam does not discriminate (1.0σ). The formal four-error-bar
criterion was **not met**. But P3b's three predictions land within **0.006** of
the measured F while the error bars are 0.11–0.18 — agreement that close would be
very unlikely by chance if those error bars reflected the true uncertainty. So at
A = 0.10 the error-bar estimator (the disagreement between weighted and unweighted
fits) is likely **conservative**.

*Caveat on order of work:* the measured F values were seen before the theory was
written. Nothing in the theory is adjustable, and neither prediction was changed
after comparison.

**Open:** the **fourth-order calculation** — the A⁴ terms that move the plane-wave
κ and the narrow-beam F at A = 0.30. P3b's detunings are also linear; the O(A²)
nonlinear shifts of the components are not in them. [**Done for the plane wave, 2026-09-25** — below;
the beam's fourth order remains open.]

**κ to fourth order, and the launch (2026-09-25).** `shape_zero_tests/kappa_pw4_pt.py`
derives the plane wave's A⁴ term by harmonic balance on the lattice (reference
convention; static shift and second harmonic to O(A⁴), third harmonic to O(A³)):

    κ(A) = −0.017480 − 0.002947·A²

It is confirmed by an exact 24-harmonic solution (the same κ₄ to every printed
digit) and by simulation launched on the exact travelling wave
(`kappa_pw4_seed.py`), which reproduces it to every printed digit with a zero
error bar. The physical value is **−0.01751 at A = 0.1**, **−0.01775 at A = 0.3**
and **−0.01797 at A = 0.4** (exact solution; the A² truncation gives −0.01795 at
0.4). The true fourth-order term adds about **1.5%** at A = 0.3.

**The measured amplitude sweep is mostly a launch effect.** The sweep −0.0176,
−0.0179, −0.0187, −0.0200 at A = 0.1–0.4 (`pinned_asymmetry_reference.py`; the same
on the side-8 cube and on a 1-D ring: −0.01755, −0.01792, −0.01869, −0.01996)
starts from a plain cosine, which omits the wave's static shift and second
harmonic and sets the velocity at the linear frequency. That adds **−0.00094 at
A = 0.3**, about **80%** of the apparent growth over −0.01748. Attribution at
A = 0.3 (`kappa_pw4_attrib.py`: the exact wave with pieces removed; identical at
T = 300 and T = 900, so a frequency shift, not a transient):

| omitted from the launch | κ increment | status |
|---|---|---|
| velocity at the linear frequency | −0.00016 | **derived** (forward/backward split of the fundamental; matches) |
| static shift | −0.00029 | **measured, not derived** |
| second harmonic | −0.00031 | **measured, not derived** |
| interaction of the two | about −0.00024 | **measured, not derived** |
| total, plain-cosine launch | −0.00094 | −0.01869 against the wave's −0.01775 |

**The static-shift and second-harmonic pieces are measured, not derived.** [**Since
derived, 2026-09-25** — "The launch pieces derived", below; the table's status column
is kept as first written.] The
leading cross-modulation formula for the free oscillations the launch leaves
behind does not capture them: for the uniform mode its direction-odd part
vanishes identically, and for the staggered (k = π) mode it gives a sixth to a
third of the measured shift (−2.5×10⁻⁶ against −1.55×10⁻⁵ and −9.9×10⁻⁶ against
−3.6×10⁻⁵ per direction), which grows roughly linearly with that mode's amplitude
while the formula grows quadratically (`kappa_pw4_attrib.py`, part 2). **Adding the second-order field
to the launch** (`kappa_seed2_test.py`) gives **−0.01787 ± 0.00006** at A = 0.3,
against the derived exact wave plus velocity term, −0.01791.

**Beam fourth-order tests** (`kappa4_predict.py`, predictions committed before
any comparison in commit `0e7f269`; `kappa4_compare.py`; `kappa4_measure.py`).
Every fourth-order term is quartic in the component amplitudes, so relative to
the second-order term it carries one more factor of the fill; the kernel is not
derived. Three hypotheses for the beam's own growth, as a fraction r of the
plane wave's: **H0** none; **S1** all quartic terms survive (incoherent local
sextic), r = fill·(6 − 3P₂ − 6s + 4s²)/F₂; **S2** only the box-wide component's
own term, r = fill·s²/F₂. At the nine sharp points (A = 0.30): H0 9/9, S1 8/9
(w = 1, L = 4 at +2.1σ), S2 9/9; the second-order F₂ alone 7/9. Two never-measured
large-fill beams at A = 0.10 and 0.30: the second-order F₂ holds there (1.0063 ±
0.014 against 1.0076; 1.0316 ± 0.014 against 1.0338), and **H0 is excluded at
large fill** — w = 3, L = 4: F(0.30)/F(0.10) at −2.2σ, F(0.30) at −3.4σ; w = 2,
L = 4: F(0.30) at −2.4σ; the box κ grows 4.4% and 3.0%. S1 and S2 both pass
(within 0.8σ), so **S1 against S2 is unresolved**; the narrow-beam evidence that
favours S2 was seen before the predictions were written. **These tests used
plain-cosine launches, so they mix physics with launch effects.** Redoing them
with orbit-consistent launches is open (MODEL_SPEC §9). [**Done 2026-09-25** — below.]

**The launch pieces derived (2026-09-25)** (`kappa_launch_pt.py`; measured by
`kappa_launch_attrib.py`, the exact wave with pieces removed, on a 1-D ring at
T = 900).

*First order — energy projection.* The energy E is conserved, so its differential
dE is invariant under the flow linearised about the wave: it vanishes on every
Floquet mode with multiplier ≠ 1 and on the phase direction. A launch error δ
therefore moves the wave's amplitude by exactly dA = dE(δ)/E′(A), and its
frequency by W′(A)·dE(δ)/E′(A). Closed forms, with W₂ and c₂ the A² coefficients
of the frequency and second harmonic and √(b² + Q) (b = cβ sin sK) the same in
both directions:

    velocity at the linear frequency:   δW = −W₂² A⁴ / √(b² + Q)
    missing second harmonic:            δW = −32 W W₂ c₂² A⁴ / √(b² + Q)
    missing static shift:               dE(δ) = 0 identically — second order

(the uniform deviation has dE = −Σₙ u″ₙ = 0 on a travelling wave). These agree with
the exact projection at small A (velocity −0.000037 against −0.000038, second
harmonic −0.000092 against −0.000093 at A = 0.15).

*Second order — a three-frequency torus.* The launch leaves free oscillations of
the uniform (k = 0) and staggered (k = π) modes. The motion is solved as a torus
u = Σ c_mjl exp i(mθ + jφ₀ + lφ_π) by harmonic balance (every term at its own
lattice wavevector and frequency; Newton with an analytic Jacobian; converged in
truncation), matched to the launch by its uniform and staggered displacements and
its energy.

| A | static shift (derived / measured) | second harmonic | fundamental only | plain cosine |
|---|---|---|---|---|
| 0.15 | −0.000029 / −0.000030 | −0.000072 / −0.000071 | −0.000112 / −0.000108 | −0.000150 / −0.000146 |
| 0.20 | −0.000079 / −0.000080 | −0.000131 / −0.000131 | −0.000254 / −0.000252 | −0.000323 / −0.000321 |
| 0.30 | −0.000278 / −0.000278 | −0.000311 / −0.000313 | −0.000794 / −0.000786 | −0.000959 / −0.000954 |
| 0.40 | −0.000583 / −0.000574 | −0.000581 / −0.000582 | −0.001698 / −0.001681 | −0.002022 / −0.002014 |

(κ increments over the exact wave.) **Derived and measured agree within about 1%
from A = 0.15 to 0.4.** Directly, the exact wave plus an explicit free uniform
mode gives per-direction shifts within 0.04% (A = 0.3, amplitude 0.02).

*The mechanism.* The wave's second harmonic (k = π, frequency 2W) is
**near-resonant with the sum of the two free-mode frequencies**, 2W ≈ Ω₀ + Ω_π —
the **same (0, π) channel as the P-1 decay window** (PROVENANCE §6o, "P-1 — what
was found": the pump's four-wave decay k₀ + k₀ → 0 + π). The detuning is **0.04
for −k against 0.24 for +k** at A = 0.3, so a free uniform mode drives a staggered
one about seven times more strongly for −k. That direction asymmetry is the
static-shift piece's odd part, and it is why the two pieces interact. A torus that
leaves the staggered mode out is 2% wrong for −k and correct for +k.

**Beam fourth-order tests, redone with orbit-consistent launches (2026-09-25)**
(`kappa4_orbit_launch.py`; predictions committed before any beam run, commit
`9e2c4c5`). The launch is the second-order forced field plus every transverse
component's velocity at its second-order nonlinear frequency. **Plane-wave check:**
it removes **96%** of the launch effect, the error bars are **25 times smaller**,
and the residual, **+0.000039 at A = 0.3**, is the missing third harmonic. The test
compares κ_box(A)/κ_box(0.10) at A = 0.30 and 0.40 against (1 + r g(A))/(1 + r g(0.10)),
with g the plane wave's derived physical growth and r from each hypothesis.

- **H0 and S1 are excluded.** H0 fails every L = 4 beam with fill ≥ 0.38 (−14σ to
  −142σ); S1 fails every L = 4 beam (+3σ to +31σ at T = 900).
- **S2's status, exactly:** by the criterion committed in advance it **passed three
  of four L = 4 beams** (w = 1, 1.5, 2) and **failed w = 3, L = 4 by +18σ** (A = 0.4,
  T = 900; +10.7σ at 0.3). It is consistent there only under a band that allows the
  launch's third-harmonic residual, and that band was **chosen after seeing the
  data — post hoc** (`kappa4_orbit_compare.py`). Under it S2 fits all eight L = 4
  points and S1 none. [**CORRECTED 2026-09-25:** with the third harmonic added to the
  launch (plane-wave residual down from +0.000039 to −0.000006 at A = 0.3;
  `kappa4_orbit3_launch.py`), **S2 fails three of the four L = 4 beams** by the
  criterion committed in advance (commit `de96bc3`): w = 1.5 (−6.1σ, −11.1σ at
  A = 0.30, 0.40; T = 900), w = 2 (−11.7σ, −21.9σ) and w = 3 (−9.9σ, −21.3σ) grow
  **more** than S2 allows. The earlier "passed three of four" came from the old
  launch's missing third harmonic, which under-read the beams' growth. The failures
  at w = 1.5 and w = 2 are **robust**; w = 3 is **marginal** — a 0.05% miss, within
  about two of the estimated remaining launch residuals (each order added to the
  launch moves the plane wave by about a sixth of the previous step); w = 1 **does
  not discriminate** (±0.0017).]
- **Post hoc** (`kappa4_orbit3_reading_output.txt`, written after the run): the
  beams' growth is a **clean A² law** — the implied r is consistent between
  A = 0.30 and 0.40 (0.351/0.350 at w = 1.5, 0.547/0.549 at w = 2, 0.750/0.758 at
  w = 3) — and it lies **0.33, 0.50 and 0.68 of the way from S2 to S1** at fills
  0.38, 0.53 and 0.74. **The fourth-order cross terms are partly present, not
  zero.** Also post hoc: each fraction is close to 0.9 × the fill (0.34, 0.48, 0.66).
- **w = 2, L = 8 was excluded:** its κ changes 2.5% with record length at A = 0.10
  (−0.004463 at T = 300, −0.004350 at T = 900), from the slow secondary energy
  transfer seen in larger boxes (`kappa_side_gpu.py`); the L = 4 boxes agree between
  T = 300 and 900.

~~Open (MODEL_SPEC §9): test S2 at w = 3, L = 4 with the third harmonic added to the launch,
and derive why the fourth-order cross terms cancel.~~ [**Superseded 2026-09-25:** the
test was done — S2 fails three of four — and the cross terms do not cancel.] Open
(MODEL_SPEC §9): derive the beam's fourth-order cross kernel, with the measured r values as
the target.

**κ = 0.0799 is the plane-wave value AT UNIT ON-SITE STIFFNESS.** It changes with
transverse geometry **and** with base stiffness — measured κ = 0.0959, 0.0799,
0.0677 at stiffness 0.90, 1.00, 1.10, a ~35% swing over ±10% [RETRACTED; corrected
−0.0214, −0.0187, −0.0165, a 26% spread — see note above]. **The pinning
remains protected against both** (null to 1.2×10⁻⁵ under stiffness), and that is
the invariance the experiment rests on.

**For an experimentalist:** on a 1D array at unit stiffness, κ = 0.0799 [RETRACTED —
use −0.0187; see note above]. On anything with
transverse extent, κ is smaller and must be measured for that profile [superseded
by the κ(w = 2, side) closure above: a localised beam's box-averaged κ depends on
the box and goes to zero as it grows — quote the plane-wave κ, or a localised κ
only with its box size]. The
**collapse protocol is unaffected** — the β-independence of the normalised drift
is what makes it a test, and that holds regardless.

*Measured with* `pinned_asymmetry_reference.py`, which now carries
`set_base(q, side)` and `TRANSVERSE_WIDTH`.

**⚠ VACUITY TRAP, documented in that file.** With `TRANSVERSE_WIDTH = None` the
seed is a transverse-**uniform** plane wave, whose transverse Laplacian is
identically zero — so a q = 3 run reduces **exactly** to q = 1 and the outputs are
**bit-identical**. That is not a q = 3 measurement. Bit-identical results from a
supposedly different configuration mean the same code path ran twice.

**Value pinned by the β-sweep, not the amplitude sweep.** The reference
implementation (`04_scripts/session/pinned_asymmetry_reference.py`, DOP853,
rtol 10⁻⁹, T = 900, weighted phase regression) gives |Δ/Δ₀| = 1.007227, 1.007190,
1.007262, 1.007051 at β = 0.02, 0.05, 0.10, 0.20 and A = 0.30. Dividing the drift
by A² = 0.09 gives **κ = 0.0799 at every β** — β-independent to four digits.

The amplitude sweep alone gives a range 0.078–0.082; the β-sweep is tighter
because the collapse is exact. **0.0799 is the value to quote.** [RETRACTED —
swapped seeding; corrected β-sweep value −0.0184 ± 0.00033, see note above.]

**SECOND CORRECTION — the coefficient is 0.082, not 0.0305.** An independent
reimplementation (DOP853, rtol 10⁻⁹, T = 900, weighted complex-phase regression)
obtained 0.082 where the package script gives 0.0305. The difference is **not**
timestep (κ = 0.03109 at DT from 0.02 to 0.0025, fully converged), **not** the
force law (identical), and **not** the seeding (both travelling, both offset
about φ). It is the **frequency estimator**.

*Calibration against known answers* (`estimator_calibration.py`):

| test case | exact coefficient | phase regression | FFT peak, T = 300 |
|---|---|---|---|
| Duffing, A = 0.15 | 0.044263 | **0.04439** | **0.02220** — off by 2.0× |
| Duffing, A = 0.30 | 0.044263 | 0.04437 | 0.05001 |
| quadratic, A = 0.15 | −0.023220 | −0.02459 | −0.04475 — off by 1.8× |

The phase estimator recovers the textbook Duffing shift to **0.4%** at every
record length. The FFT-peak estimator used by the package is biased by factors of
1.8–2.0 at T = 300, in **both** directions depending on the case, converging only
as the record lengthens (0.0222 → 0.0404 → 0.0416 for T = 300 → 900 → 2700).

The disputed factor of 2.6 lies within that measured bias. **The package's 0.0305
is an artifact of FFT-peak estimation on a short record.**

**This line is emitted by `pinned_asymmetry_headline.py`, not written by hand.**
[That script now prints −0.0205 (fitted from A ≥ 0.3, own-root seeding; its T = 300
record does not resolve the drift below that). The reference, not it, carries
the quoted coefficient. `PROVENANCE.md` §6o.]
That script recomputes every headline number from the lattice via the package's
own `phi_gauge_nonlinear.py`, confirms the coefficient by two independent routes,
and carries a guard against decimal slips.

| route | coefficient | estimator |
|---|---|---|
| package amplitude sweep | 0.0305 ± 0.0005 | FFT peak, T = 300 — **biased** |
| package coupling sweep | 0.0306 ± 0.0007 | FFT peak, T = 300 — **biased** |
| reference, amplitude sweep | 0.078–0.082 | phase regression, T = 900 |
| reference, **β-sweep** | **0.0799** (four digits) | phase regression, T = 900 |
| **adopted** | **0.0799** | β-sweep, calibrated to 0.4% on Duffing |
| **RETRACTED → corrected** | **−0.0184 ± 0.00033** (−0.0187 at β = 0.05) | same reference, each direction seeded at its own root |

*The package sweeps are internally consistent because they share one biased
estimator. Internal consistency is not accuracy — both routes used the same
instrument.*

**Measured domain limit.** The coefficient holds near 0.0305 through A = 0.5,
drifts to 0.0345 at A = 0.7 and 0.0418 at A = 0.9, and departs at A = 1.1
(0.0865) as the wave leaves the φ well. **Valid for A below ≈ 0.9** — wider than
the ≈ 0.5 estimated earlier, which was read off the integrator overflow rather
than the onset of departure.

**ERRATUM.** An earlier version gave 0.30 — a factor of ten too large, from
dividing the measured coefficient by −0.2 instead of the leading term −2. It
would have predicted 4.8% drift at A = 0.4 where 0.5% is measured, and an
experimentalist following it would have reported a spurious failure.

**The tables were always right; only the hand-written summary was wrong.** The
process fix, now applied: *headline formulas are emitted by the script that
produces the table, never written in prose afterwards.* `pinned_asymmetry_headline.py`
implements that and prints what a factor-ten slip would read, so this class of
error is caught at source.

**Why the β-proportionality is the sharp part.** The nonlinearity does not shift
the asymmetry independently — it *rescales* it. So the **normalised** drift
Δω(A)/Δω(0) has **no β dependence at all**. Measure it at one coupling strength
and it must be identical at every other.

**Better protocol, therefore:** sweep amplitude at two or three values of β and
check the normalised curves **collapse onto one**. Collapse is a pass;
separation is a fail; nothing is fitted anywhere. An invariance across a second
knob is far harder to reproduce accidentally than a single curve.

**Domain limit.** The on-site potential −(x² − x − 1) is unbounded below, so
waves escape the φ well at large drive. Numerically the integrator overflows at
**A = 1.2**, and residual/A² has already drifted 70% by A = 1.0. **Useful
amplitudes stop below A ≈ 0.5.** That limit is itself a checkable feature: the
escape threshold is set by the well depth, which linear spectroscopy also
fixes.

## 3c. Resolution requirements, and what each buys

Two claims live in this test and they need different precision. Stating both, so
an experiment is not set up to see one while reporting on the other.

| claim | required relative resolution on Δω | why |
|---|---|---|
| **asymmetry equals −2cβ sin(k)** | ~10⁻³ | the leading term is O(1) against the centre shift |
| **asymmetry stays pinned under amplitude** | ~10⁻⁴ | the drift is 0.5% at A = 0.4 |
| **the A² law, and its coefficient 0.0305** | **~10⁻⁵** | at A = 0.1 the correction is only 3×10⁻⁴ of the leading term |
| **collapse across β** | ~10⁻⁴ | the normalised curves differ by less than the drift itself |

**Worked from the numbers:** the correction is 0.0305·A² in relative terms. At
A = 0.4 that is 4.9×10⁻³ — visible at 10⁻⁴ resolution. At A = 0.1 it is
3.1×10⁻⁴, and at A = 0.05 it is 7.6×10⁻⁵. So mapping the *scaling* over a decade
of amplitude requires an order of magnitude better resolution than confirming the
*pinning*.

**This has been observed in practice.** An independent reimplementation of this
model reported Δω = −0.10015 at A = 0.001, where the nonlinear correction should
be 3×10⁻⁸ — so that 1.5×10⁻⁴ deviation is estimator noise, and it is the same
size as their A = 0.2 point (−0.10006). Their run resolves the pinning and does
**not** resolve the A² law. That is the expected outcome at ~10⁻⁴ and it is worth
knowing in advance rather than discovering mid-experiment.

**A systematic that does not matter.** The same reimplementation found absolute
frequencies 0.0015–0.006 above ours, from residual second-harmonic mixing — and
reproduced the asymmetry anyway. That is the mechanism working: Δω is a
*difference*, so any common-mode offset cancels. **Errors affecting both
propagation directions equally drop out of the observable by construction**,
which is precisely why this test is robust to the systematics a real apparatus
will have.

## 4. Experimental realisation

Any platform with tunable antisymmetric velocity coupling on a 1D chain:

- coupled mechanical oscillators with gyroscopic elements (spinning rotors,
  Coriolis coupling)
- photonic lattices with synthetic gauge fields
- coupled electrical resonators with non-reciprocal elements
- optomechanical arrays

**Procedure.**
1. Measure the linear band; extract **c** from the dispersion width.
2. Measure the linear asymmetry at one k; extract **β**. *All inputs now fixed.*
3. Drive at k = π/2 and at −k, sweeping amplitude over at least two decades.
4. Track band centre and asymmetry independently.

**Pass:** centre softens, asymmetry pinned to within the β·A² correction.
**Fail:** asymmetry moves with amplitude beyond that, or does not equal
−2cβ sin(k) at low amplitude.

## 5. What a pass and a fail would mean

**A pass** confirms the one chain in the program with no free parameters:
passivity forces the coupling matrix skew, hence **u(2)**, hence a synthetic
U(1) with a pinned asymmetry. That is a derivation from a stated principle to a
measured number.

**A fail** breaks it at a located point. If the asymmetry is not
−2cβ sin(k) at low amplitude, the u(2) derivation is wrong. If it drifts with
amplitude beyond the A² term, the direction-blindness of the nonlinearity is
wrong. Either is informative; neither is absorbable by re-fitting.

## 6. Why this and not the rest

Most of the program derives **structure that already exists** rather than
numbers that are missing — u(2), the colour decomposition 1 ⊕ 3 ⊕ 3̄,
sin²θ_W = 3/8. Those are consistency checks against known physics, not
predictions.

And the routes to genuinely new numbers are closed, provably: all five selection
principles are dimensionless, so no output carries a unit; and every spectrum on
a compact homogeneous space is a quadratic tower k(k+n−1), which measured
spectra are not — lepton m² ratios are 1, 4.3×10⁴, 1.2×10⁷, and Regge
trajectories grow linearly where these grow quadratically.

**This is the exception.** It is dimensionless — a ratio — so the theorem does
not forbid it. It is a lattice, so no signature obstruction applies. It is
zero-parameter once the linear measurement is made. And it can fail.

---

*Script: `phi_gauge_nonlinear.py` (C1 package). Related: `phi_gauge_test.py`,
`phi_gauge_wilson.py`, `phi_gauge_precession.py`. Spec: v5.1 §5.2.7, and
Sections 6–9 for the passivity → u(2) derivation.*
