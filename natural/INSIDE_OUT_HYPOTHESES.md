# CP-G, inside-out: hypotheses

*Committed with the charter's candidate principle CP-G, before the scoping in
`INSIDE_OUT_SCOPE.md`. Nothing is built. Each hypothesis is to be marked HELD, FAILED or OPEN when
a pilot or a run tests it.*

**IO1 — the teacher.** Linearised general relativity on the q = 3 lattice.
- A symmetric perturbation h_μν couples to every sector's energy-momentum: matter, the chirality
  sectors, and gauge links if present. The coupling is −½h_μν T^μν, with □h̄_μν = −16πG T_μν in
  harmonic gauge, and the shared wave speed.
- PPN γ = 1 is built in; the teacher is linear, so PPN β is **not** defined by it.
- The teacher is itself selected (spin-2, shared speed, G), so it is counted as training wheels,
  never as evidence.

**IO2 — the library, restricted to what the principles allow.** All terms are local (on-site or
nearest-neighbour), passive (conservative, or gyroscopic in the antisymmetric/transposed form) and
J-compatible (phase-invariant):
- **L1** energy-dependent on-site stiffness K(ε) — lapse-like;
- **L2** energy-dependent elastic coupling c(ε) — spatial-metric-like; it changes the local wave
  speed;
- **L3** energy- or momentum-dependent velocity links — shift-like, passive in the transposed form;
- **L4** phase-invariant nonlinear couplings between neighbours, |ψᵢ|²|ψᵢ − ψⱼ|² and the like;
- **L5** fluctuation-induced two-body couplings: integrate out the populated background of `main`'s
  own modes (P0 non-isolation), thermal or zero-point.

**IO3 — the mediator obstruction (the central hypothesis).**
- Every field of `main` is gapped: the scalar sector at √5; the J-sector branches at ω_a(k = 0) =
  1.086 and above, with healing length ξ ≈ 0.64 sites (MODEL_SPEC §1c).
- A static influence carried by local couplings of gapped fields decays as e^{−r/ξ}; a
  fluctuation-induced interaction through them decays at least as fast (range ~ξ/2).
- **So no library term L1–L5 reproduces the teacher's 1/r potential once the teacher is removed.**
  The student can reproduce free fall only in a field the teacher supplies.
- **Prediction:** the held-out two-body law at q = 3 is Yukawa-type with range ≲ ξ, not
  inverse-square.
- **The failure is avoidable only with a massless mediator:**
  - a new field — an ingredient, contradicting CP-G;
  - target 1's photon — it couples to charge, not energy; neutral lumps interact only through
    two-photon exchange, ∝ r⁻⁷ (Casimir–Polder), not 1/r²;
  - a Goldstone mode of a spontaneously broken symmetry — its couplings are derivative, so static
    sources exchange no 1/r potential.

**IO4 — free-fall training underdetermines γ.**
- Slow packets falling constrain only the lapse-like coupling (L1).
- Sparse extraction then prefers L1 alone, which is γ = 0, and fails held-out light bending.
- A γ = 1 student needs L2 in a fixed ratio to L1. Free fall cannot fix that ratio; only light
  propagation can, and putting it in training makes it a selection.

**IO5 — polarisation and Weinberg–Witten.**
- **Library L1–L5 mediates scalar (L1, L2) and vector (L3) influences, not tensor.** The held-out
  polarisation test fails.
- **Weinberg–Witten** forbids a massless spin-2 composite with a Lorentz-covariant, conserved
  stress tensor, in a Lorentz-invariant theory. The lattice is not Lorentz-invariant, so the theorem
  does not apply at the lattice scale. But the shared wave speed makes the infrared effectively
  Lorentz-invariant, where it does bite.
- An emergent graviton must therefore either:
  - (a) keep infrared Lorentz violation — constrained by the shared-speed selection and by
    observation; or
  - (b) come with an emergent linearised-diffeomorphism gauge symmetry, so its stress tensor is not
    a covariant local operator.
- **Neither is available in L1–L5.**

**IO6 — closing orbits.** With a Yukawa law (IO3), bound orbits precess and do not close (Bertrand).
The held-out closing-orbit test fails with IO3. The teacher defines closure only at Newtonian order:
the 1PN precession needs PPN β, which a linear teacher does not fix.

**IO7 — the accounting.**
- **Selections:** each training behaviour (free fall of scalar packets, of each chirality branch,
  gravitational redshift); each library family (5); the teacher's form.
- **Fitted parameters:** each non-zero learned coefficient.
- **Evidence:** only held-out passes.
- **Predicted net:** with free-fall training, 3 + 5 + ≥ 1 inputs against 0 held-out passes. **CP-G
  is predicted to fail its own test**, at the distance law first, unless IO3 fails — a long-range
  mediator emerging from `main`'s fields. That is the one outcome that would be real evidence.

**IO8 — compute and route.**
- **A CPU pilot first**, q = 1 and 2, minutes: can any L1–L5 term carry a static influence beyond
  a few ξ from a source, with no teacher? It decides IO3.
- **If IO3 fails in the pilot:** a self-contained Colab GPU script (PyTorch, float64) for teacher
  trajectories at q = 3 (48³–64³), with SINDy-style sparse regression and symbolic regression on the
  learned couplings.
- **If IO3 holds:** the GPU program is not worth running.
