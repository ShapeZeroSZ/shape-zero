# Shape Zero — Open Threads

Forward-looking research queue as of July 2026. Each item states what is
open, why it matters, and what would close it. Stated as technical
questions only; interpretive framing is deliberately absent.

---

## A. Proof-grade mathematical questions

**A-1. Proof of the role-orientation result.** Census established: of the
128 orientation assignments of the standard Fano lines, 16 yield valid
normed algebras, and all 48 proper role colorings (role-complete,
role-minimal) induce valid octonion multiplications, in 3:1
correspondence. *Open:* a proof of why proper role colorings coincide with
valid orientations. *Closes when:* a demonstration exists, or the result is
identified in the existing literature on octonion sign conventions.
*Script:* `z1_d8_census.py`.

**A-2. Status of the invariance hinge.** The coupling class was argued
SELECTED on two legs — (i) a coupling either is constructible from the
structure already present, which is equivalent to invariance under that
structure's automorphism group, or it introduces a new tensor; (ii)
transport that fails to preserve the multiplication does not preserve the
role assignment. Both legs apply the program's selection principles to
structure rather than to states within structure. *Open:* whether that
scope extension is legitimate. This is the program's single recurring
conditional; it reappears as diffeomorphism covariance in the gravity
items. *Closes when:* the extension is justified, restricted, or replaced.

**A-3. Uniqueness beyond linear algebra.** The invariant cubic form was
verified unique by nullspace computation on one algebra instance
(`z1_hinge.py`: dim 14 derivations, 1-dimensional invariant space,
overlap 1.000000). *Open:* the statement as a theorem across all valid
orientations rather than a verified instance.

**A-4. Shadow structure at seven dimensions.** The imaginary part of the
quaternions gave the odd-dimension case its content (one unpaired
direction, chirality, exact inertness — measured bit-identical). *Open:*
whether the imaginary part of the octonions plays an analogous role, and
what the corresponding inertness statement is.

---

## B. Builds queued

**B-1. S5 — arena globalization.** Two parts: (i) a unified lattice with
eight-dimensional fibers carrying the invariant cubic vertex — verify that
three-wave matrix elements are supported on the incidence lines, or
characterize the deviation; (ii) the measure-valued (HK) statement of the
articulation result: bilinear couplings dissolve, cubic content is the
first invariant. Note that S4 already exhibited (ii) for two-dimensional
fibers without being asked to. *Estimate:* 2–3 sessions.

**B-2. Metric parameter derivation, run blind.** The cone as built is flat;
its angular scale is the single remaining metric freedom, and fixing it by
the phase-boundary self-consistency condition is the parked "scale
fixing" item. *Protocol:* derive the value first, inspect afterward. A
second candidate mechanism — selecting the angle by winding survival, i.e.
sweeping the deficit against retention — is a designed experiment, not a
prediction.

**B-3. Gravity hinges.** Metric compatibility of the connection is already
banked (it is the passivity theorem in frame language). The remaining
route is Lovelock-type uniqueness: minimality plus covariant conservation.
*Hinges:* torsion (does minimality remove it?), dimension count, and the
covariance question, which is A-2 again. *Status:* a program, not a
session; no closure claimed.

**B-4. Quantum interface.** The eight-dimensional Clifford system is the
Dirac/Pauli operator algebra, and passivity forcing Hermitian generators
with unitary transport is a kinematic skeleton derived from classical
conservation. *Open and explicitly not derived:* ℏ, the Born rule,
entanglement. *Closes when:* either a derivation of measurement structure
exists, or the boundary is stated as permanent.

**B-5. E1c — coexistence terms.** Two-agent version of the winding result:
map the safe-coexistence windows for coupled agents. Pairs with E1a/E1b
(both closed) to complete the externalization extension for the v5.4 fold.

**B-6. Overdamped seam.** Gradient flows carry no angular momentum, so the
void term in that sector is not covered by the centrifugal derivation.
*Open:* derive it as an averaged-rotation remnant, or tag it permanently
as chosen.

---

## C. Prediction candidates

**C-1. Odd-fiber signature (candidate P-4).** A gyroscopic metamaterial
with three-dimensional fibers must carry exactly one polarization branch
showing no dispersion asymmetry, while the other two split. Falsifiable,
cheap, and follows from a two-line theorem.

**C-2. Deficit lensing.** Deflection independent of impact parameter
distinguishes a conical deficit from a Newtonian potential (verified in
simulation to 3×10⁻⁴ after aperture correction). Applicable wherever an
effective conical geometry can be engineered.

**C-3. Breach scaling.** Under wave loading near a channel-switched
defect, local mass transiently passes below the static bound and recovers
(timestep-converged). *Open:* breach depth as a function of local kinetic
energy — a scaling law worth measuring and stating.

---

## D. Method and instrument

**D-1. Cascade corrections.** Pairwise action invariants are exact only
for isolated triples; the seven-mode network drifted several percent.
*Open:* the correct multi-triple invariant bookkeeping.

**D-2. Readout protocol, now standing.** Quantized readouts (FFT bins,
zero-crossing counts, finite windows, finite apertures) bracket the truth;
continuous readouts (interpolated crossings, phase slopes,
subtract-then-project, analytic corrections) converge. Six occurrences
this program. Worth writing up as a short methods note — it would be
useful to others working with the same kinds of measurement.

---

## E. Number-theoretic selection (measurable, two directions)

**E-1. Fine structure near noble ratios.** The winding survival experiment
used one carrier, one amplitude, one horizon. *Open:* vary all three and
sample densely near extremal-irrational ratios to test whether the
survival plateau has predicted structure.

**E-2. The valuation contrast.** Two selection rules run in opposite
directions on the same number line: resonant coupling destroys low-order
rationals and preserves badly-approximable ratios, while auditory
consonance is defined by low-order rationals (coinciding partials,
vanishing beat rates). *Open:* whether this is a genuine contrast worth
formalizing — the same continued-fraction machinery grades both, with
opposite polarity. Tunings sit at convergents of the relevant logarithm,
which is a concrete, checkable statement rather than an analogy.

**E-3. Combination tones as an articulation readout.** The derived cubic
vertex produces combination frequencies (measured at twice the fundamental
and at DC). These are difference tones, audible and well documented since
the eighteenth century. *Open:* whether an acoustic experiment can measure
the articulation coefficient directly — the cheapest possible test of the
first-articulation result, on equipment that already exists.

---

## F. Publication and outreach path

**F-1. R1.** Draft complete. Remaining: author review pass, references,
LaTeX conversion, figure integration, repository URL, and arXiv
endorsement (typically required for first-time submitters — the outreach
contacts are plausible endorsers, so the two tracks may travel together).

**F-2. C2 outreach.** Four cover notes, sequenced. Response protocol
pre-decided: specific objection → branch; category objection → log and
move on; silence → one follow-up at roughly three weeks.

**F-3. v5.4 fold.** The specification is frozen at v5.3 by design. The
unified-model results, externalization extension, and ladder record are
queued to fold in after R1 posts.

**F-4. R2.** Sequenced after R1 and first external contact. Standing
done-criterion unchanged: every claim cites the result that licenses it,
and anything unlicensed appears in an explicit retraction list.
