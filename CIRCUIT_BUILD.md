# Circuit build — a 32-node LC-gyrator ring

*Branch `realisation-gyroscopic`, 2026-09-28. Every design choice here is ours. The predictions and failure criteria
are in `CIRCUIT_PREREGISTRATION.md` (committed before any build, SPICE run or measurement; amendment 1 corrects the
GIC values). The platform and its error budget: `LAB_NOTE.md` §6. The SPICE check of this design: §8 below and
`spice/`. The model is not a theory of nature (`LAB_NOTE.md` §5): this circuit tests whether a built lattice
realises three relations of the model's equations.*

## 1. What is built

Thirty-two identical nodes on a closed ring, joined by 32 identical bonds. With the node flux φ_n (V_n = φ̇_n) the
circuit obeys C φ̈_n = −φ_n/L_g + (φ_{n+1} + φ_{n−1} − 2φ_n)/L_c − G φ̇_{n+1} + G φ̇_{n−1}: the model's scalar
β-sector lattice with K = 1/(L_g C) = 10¹¹ s⁻², ĉ = L_g/L_c = 0.25, β̂ = G√(L_g/C) = 0.106, κ = 0.

**Node n (to ground):**
- C1 = 10.00 nF C0G capacitor, node to ground (the mass).
- L_g = 1.176 mH shielded ferrite inductor, node to ground (on-site stiffness, 85% of it).
- **Trim and bump element, in parallel with L_g (the other 15% of the on-site stiffness, the per-node trim, and
  the pinning bumps).** The pre-registration specifies an Antoniou GIC synthetic inductor (6.667 mH). **SPICE
  shows that GIC stage is not viable as specified** (§6, §8): with op-amp B wired one way it oscillates, the other
  way it latches at DC; of eight Antoniou variants only one (capacitor in position 2) is sane, and it has a negative
  series resistance (Q ≈ −20) that erodes the ring's stability margin. **Proposed replacement (ours; needs a
  pre-registration amendment before any build — not adopted):** a passive, slug-tuned shielded inductor of
  6.7 mH ± 10% (adjustable ferrite core), set per node and per bump configuration by single-node spectroscopy.
  The SPICE runs with an ideal inductor in this position (§8, variant A) are the verification of that proposal.
  1.176 mH ∥ 6.667 mH = 1.000 mH.
- The GIC as pre-registered, for the record: chain R1 = 2.000 kΩ (node→g2), R2 = 2.000 kΩ (g2→g3), R3 = 2.000 kΩ
  (g3→g4), C4 = 1.000 nF (g4→g5), R5 = 3.333 kΩ (g5→ground); op-amp A + at node, − at g3, output g2; op-amp B
  + at g3, − at g5, output g4 (amendment 2); L = C4·R1·R3·R5/R2 = 6.667 mH.

**Bond n → n+1 (between nodes a = n and b = n+1):**
- L_c = 4.000 mH shielded ferrite inductor from a to b (the spring).
- Gyrator: two voltage-controlled current sources (VCCS), one OPA4197 quad per bond:
  - VCCS B (+V_a·G into b): op-amp 1 is a unity follower of V_a; its output feeds RB1 = 2.983 kΩ into b; op-amp 2
    (+ input at b, − input at mb, output hb) with RB2 = 2.983 kΩ from hb to b, RB3 = 10.0 kΩ from mb to ground,
    RB4 = 10.0 kΩ from mb to hb (a basic Howland pump: I = V_in/RB1 when RB4/RB3 = RB2/RB1).
  - VCCS A (−V_b·G into a): op-amp 3 is a unity follower of V_b; op-amp 4 (+ input at a, − input at ma, output ha)
    with RA1 = 2.983 kΩ from a to ground, RA2 = 2.983 kΩ from ha to a, RA3 = 10.0 kΩ from the follower output to
    ma, RA4 = 10.0 kΩ from ma to ha (input on the R3 side: I = −V_in/RA1).
  - G = 1/2.983 kΩ = 3.352×10⁻⁴ S.
- Supplies ±12 V, 100 nF + 10 µF decoupling per package; a solid ground plane.

## 2. Bill of materials (32 nodes; approximate 2026 distributor prices)

| part | type | qty (incl. spares for sorting) | tolerance needed | ~unit | ~total |
|---|---|---|---|---|---|
| node capacitor 10 nF | C0G/NP0, 1% (sorted to 0.1%) | 64 | 0.1% after sorting | $0.30 | $20 |
| L_g 1.2 mH (nominal 1.176 mH) | shielded ferrite, ±5%, Q 60–150 at 60 kHz | 96 | uniform to 0.1% after sorting + trim | $1.50 | $144 |
| trim/bump inductor 6.7 mH ± 10% adjustable | slug-tuned shielded (proposed passive replacement for the GIC) | 40 | set and measured per configuration | $4.00 | $160 |
| L_c 3.9–4.0 mH | shielded ferrite, ±5%, Q 60–150 | 128 | uniform to 0.1% after sorting | $1.80 | $230 |
| OPA4197 (quad, 10 MHz) | gyrators | 34 | — | $5.50 | $187 |
| thin-film resistors 2.983 kΩ, 10.0 kΩ | 0.1%, 25 ppm/K | ~330 | 0.1% (Howland ratios and G) | $0.35 | $115 |
| drive resistor 1 MΩ, decoupling, connectors, series-R pads for Q | — | — | — | — | $50 |
| PCBs | 4-layer, 8 boards × 4 nodes, daisy-chained into a ring | 8 | — | $15 | $120 |
| ±12 V linear bench supply | — | 1 | — | — | $60 |
| **parts total** | | | | | **≈ $1100** |
| *(pre-registered GIC, not recommended: OPA2197 ×34, 1 nF C0G ×40, 2.000 kΩ 0.1% ×100, AD5272 ×34, microcontroller)* | | | | | *(≈ $350)* |

Instruments (the expensive part unless borrowed): a precision LCR meter measuring at 50–70 kHz with ≤ 0.05% basic
accuracy (for sorting); a lock-in amplifier or a USB oscilloscope/AWG with software demodulation plus a 32:1 analog
multiplexer (2 × ADG1606) and a buffer; a temperature-stable enclosure (±0.5 K during a run).

## 3. Part sorting (ours)

The predictions need **uniformity**, not absolute values: K, c and b are calibrated in situ (§5). Tolerance
targets are per-part scatter.
1. Let every part stabilise at the enclosure temperature for 12 h. Measure every node capacitor, L_g and L_c on
   the LCR meter at 60 kHz, 0.1 V drive, in one session, recording temperature.
2. Keep the 32 capacitors, 32 L_c and 32 L_g whose values lie closest to their batch medians (the best contiguous
   window of 32 in sorted order). Required: C and L_c within ±0.1% (the budget: 1% scatter wrecks P2; 0.1% gives
   ~0.2% scatter). L_g may scatter ±3%: the trim inductor sets each node's total K_n.
3. Match the four Howland resistors of each VCCS by measurement (ratio RB4/RB3 to RB2/RB1 within 0.1%).
4. After assembly, trim each node: adjust its trim inductor so that the node's single-site resonance (neighbours'
   bond inductors lifted, gyrators out) is at the common target within 0.05%. Seal the slug; record the value.

## 4. Stability window

- **Inductor Q must stay between 80 and 250** (pre-registered). SPICE with TI's OPA4197 model and the proposed
  passive trim element puts the instability threshold at Q ≈ 290 (stable at 200, unstable at 300); with the
  pre-registered GIC (version 1) it was between 300 and 450; our own model says 600–1000 (a disagreement, §6
  item 4). The gyrators' op-amp phase lag makes one propagation direction slightly active; the inductor losses must
  beat it. **Aim for Q ≈ 100 (80–150)**, well below the threshold: ferrite drum inductors typically have Q 60–150 at
  60 kHz; if a batch measures higher, add series resistance.
- Check before measuring: power up with the drive off and confirm no node exceeds 1 mV rms (no self-oscillation);
  then confirm every line in the first sweep has a finite width.

## 5. Measurement protocol (ours) — P1 first

**Drive and readout.** A sine from the generator into node 0 through 1 MΩ (a current source to ~0.1%; the
1 µS it adds at node 0 is 0.03% of that node's admittance ωC and is part of the measured circuit). Read each node's voltage in turn
through the multiplexer and buffer into the lock-in (or demodulate in software), referenced to the drive. Record
complex V_n(f) for all 32 nodes at every frequency. Settle 20 ms per frequency step, integrate ≥ 50 ms.
**Take three repeat sweeps of every configuration** (the analysis uses their scatter for the error bars).
Interleave configurations (S0, +S, −S, S0, …) so slow drift cancels.

**Order.**
1. Stability check (§4).
2. `band`: gyrators out — both VCCS of every bond disconnected by jumpers (unpowered op-amps would load the nodes
   through their protection diodes), 48–75 kHz, 25 Hz steps.
3. `p1`: gyrators powered, same sweep. → **run the analysis for the preconditions and P1 now, and record the P1
   result before any bump is applied.**
4. Measure the applied bumps: for each shape at +S and at −S, set the trim inductors and measure every node's
   single-site frequency; δK_n = K·[(f_n/f_n,0)² − 1] → `dK_<shape>_p.txt` and `dK_<shape>_m.txt` (32 values
   each). Both are needed: the realised bump never equals the design exactly (in SPICE the pre-registered GIC
   realised positive bump steps 3.1% and negative ones 1.5% larger than designed), and P2 uses ⟨δK²⟩ averaged over
   the two. Note: P3's pre-registered R values are for the exact design shapes; a realised shape that departs from
   its design by more than ~1% of the peak changes the finite-ring value and must be reported.
5. `m8_S0`, then `m8_<shape>_p` and `m8_<shape>_m` for the six shapes (Gaussian σ 4, 3, 5; sech² width 4; two
   Gaussians σ 2.5; ramp), 54–69.8 kHz, 25 Hz steps, three repeats each, interleaved with `m8_S0`.
6. Run the analysis (§7) on the files. The pre-registered failure criteria decide; nothing is tuned after
   looking.

Time: one full sweep of 32 nodes × 1081 frequencies through a multiplexer at ~70 ms per point ≈ 40 min; the
windowed sweeps ≈ 25 min. Three repeats of 15 configurations ≈ 20 h, automated.

## 6. Known issues found in SPICE (open — to be decided before any measurement)

These are reported, not resolved. Each needs a decision, and any change to the pre-registered design or criteria
needs a dated amendment committed **before** measurement.

1. **The GIC stage is not viable as pre-registered.** Version 1 (op-amp B + at g5): correct AC impedance, but a
   node driven by a 10 µA kick runs to the ±12 V rails. Version 2 (amendment 2, + at g3): the DC operating point
   latches (op-amp B at +11.75 V) or fails to converge. A search over the eight Antoniou variants
   (`spice/gic_search/`) found one sane configuration (capacitor in position 2), but with a negative series
   resistance (Q ≈ −20) that raises each node's effective Q toward ~1000, beyond the ring's stability threshold
   (item 4). **Proposal (not adopted): a passive, slug-tuned trim inductor instead** (§1); SPICE variant A (ideal
   inductor in that position) is its verification.
2. **Precondition 2 is wrong as pre-registered.** It requires ĉ = 0.250 ± 0.002; the design's own parasitics
   (10 pF across L_c, op-amp input capacitance) make the band-fit ĉ 0.2475–0.2479 in SPICE, and our own model with
   the same parasitics gives 0.2479. SPICE and our model agree; the criterion ignored the design. It would void a
   correct run.
3. **Precondition 4 (product rule ≤ 10⁻³) fails in SPICE with the real op-amp VCCS: 4.0×10⁻³.** With ideal VCCS
   SPICE gives 4.2×10⁻⁴, and our model (single-pole VCCS) gives 4.2×10⁻⁴. **Disagreement:** the real Howland VCCS
   break velocity-linearity about 10× more than our model of them. Either faster op-amps, or the criterion, must
   change.
4. **Stability threshold disagreement.** SPICE (AC poles, GIC version 1): stable at Q ≤ 300, unstable at Q ≥ 450.
   Our model (9.25 MHz VCCS pole): unstable only between Q = 600 and 1000. SPICE with an ideal on-site element
   (variant A): stable at Q ≤ 200, unstable at Q ≥ 300 (threshold ≈ 290). The pre-registered window (Q 80–250) is inside SPICE's stable range, with little margin at
   the top.
5. **P2 with the design bump fails in SPICE with GIC version 1** (R = 1.083 against the window 1.031 ± 0.015),
   because the GIC realises the bump 1.5–3.1% larger than designed; with ⟨δK²⟩ measured at ±S (as the protocol
   now specifies) it is ≈ 1.035, and with an ideal on-site element R = 1.038 — both inside the window.
6. **Full-ring transients were not run** (0.1 ms of simulated time took > 10 min with 192 op-amp models); time-domain
   stability was checked on a 4-node ring of the same cells.

## 7. Analysis script (paste into one Colab cell)

Set `DATA_DIR` to the folder of measured files; with `DATA_DIR = None` it runs a synthetic demonstration (a few
minutes). Needs only numpy. It is also committed as `circuit_analysis.py`.

```python
# circuit_analysis.py -- self-contained analysis for the 32-node LC-gyrator ring (CIRCUIT_BUILD.md).
# Paste into one Colab cell and run. Needs only numpy. (The demonstration takes a few minutes.)
#
# INPUT: frequency-response files, one per configuration, each with a header line and then rows
#   f_Hz  Re V0  Im V0  Re V1  Im V1 ... Re V31  Im V31
# (the node voltages for a 1 A-equivalent current drive into node 0; ngspice's `wrdata` with wr_singlescale
# writes exactly this). Several files for the same configuration (repeat sweeps) give error bars from their
# scatter; with one file the error bar comes from bootstrapping the fit residuals.
#   band*.txt            gyrators unpowered, 48-75 kHz
#   p1*.txt              gyrators powered, 48-75 kHz
#   m8_S0*.txt           gyrators powered, no bump, 54-70 kHz
#   m8_<shape>_p*.txt    bump +S ; m8_<shape>_m*.txt  bump -S     shapes: gauss4 gauss3 gauss5 sech24 two2.5 rampNone
#   dK_<shape>_p.txt, dK_<shape>_m.txt   the 32 measured dK_n (s^-2) of that bump at +S and at -S (single-node
#                        spectroscopy); <dK^2> is averaged over both. If absent, the design dK_n = K S eta_n is used and
#                        flagged (SPICE: the GIC realises the bump 1.5-3.1% larger than designed, so measure it).
# Leave DATA_DIR = None to run on synthetic data from a built-in model of the circuit (a demonstration only).
#
# OUTPUT: calibrations and preconditions, then P1, P2, P3 with error bars against the pre-registered
# predictions of CIRCUIT_PREREGISTRATION.md (committed before any measurement).
import glob
import os

import numpy as np

DATA_DIR = None          # e.g. "/content/data"
N, S = 32, 0.05
RNG = np.random.default_rng(0)

# ---- pre-registered predictions (CIRCUIT_PREREGISTRATION.md) -------------------------------------------------
PRED = dict(beta_hat=0.106, c_hat=0.250, K=1e11,
            P1_tol=5e-3, P2_R=1.031, P2_tol=0.015,
            P3={"gauss4": 1.035, "gauss3": 1.058, "gauss5": 1.026, "sech24": 1.058, "two2.5": 1.077, "rampNone": 0.894},
            P3_tol=0.02, P3_ramp_tol=0.03)


def eta_shape(kind, p=None):
    x = np.arange(N, dtype=float)
    if kind == "gauss":
        e = np.exp(-0.5 * ((x - N / 2) / p) ** 2)
    elif kind == "sech2":
        e = 1.0 / np.cosh((x - N / 2) / p) ** 2
    elif kind == "two":
        e = np.exp(-0.5 * ((x - N / 4) / p) ** 2) + np.exp(-0.5 * ((x - 3 * N / 4) / p) ** 2)
    else:
        e = x.copy()
    e = e - e.mean()
    return e / np.sqrt((e ** 2).mean())


SHAPES = {"gauss4": ("gauss", 4), "gauss3": ("gauss", 3), "gauss5": ("gauss", 5), "sech24": ("sech2", 4),
          "two2.5": ("two", 2.5), "rampNone": ("ramp", None)}


def design_f(m, sign, K=1e11, c=0.25e11, b=0.106 * np.sqrt(1e11)):
    """Design line frequency (Hz); sign = +1/-1 for the +k/-k branch, 0 for the unsplit band (gyrators off)."""
    k = 2 * np.pi * m / N
    Qk = K + 2 * c * (1 - np.cos(k))
    bb = b if sign else 0.0
    return (sign * bb * np.sin(k) + np.sqrt((bb * np.sin(k)) ** 2 + Qk)) / (2 * np.pi)


# ---- estimator (fixed in the pre-registration: a rational fit around each line, pole real part; implemented as a
# common-pole fit of all 32 node responses, the line identified by its residue vector) -------------------------
def vf(f, V, f0, hw=2000.0, npol=16, it=8):
    """Common-pole rational fit (vector fitting with Sanathanan-Koerner weights) of all node responses over
    f0 +- hw. Returns poles (Hz, complex), residue vectors (one row per pole), the fitted responses and the mask."""
    msk = np.abs(f - f0) <= hw
    x = (f[msk] - f0) / hw
    Y = V[msk]
    npts = len(x)
    w = np.ones(npts)
    for _ in range(it):
        Vd = np.vander(x, npol + 1, increasing=True)
        Dd = np.vander(x, npol, increasing=True)
        Qv, _ = np.linalg.qr(Vd * w[:, None])
        Mw = -(Y * w[:, None]).T[:, :, None] * Dd[None]                  # (channels, points, npol)
        Mw = Mw - np.einsum("pk,ckj->cpj", Qv, np.einsum("pk,cpj->ckj", Qv.conj(), Mw))
        rw = (Y * (x ** npol * w)[:, None]).T                             # (channels, points)
        rw = rw - np.einsum("pk,ck->cp", Qv, rw @ Qv.conj())
        bc, *_ = np.linalg.lstsq(Mw.reshape(-1, npol), -rw.reshape(-1), rcond=None)
        b = np.concatenate([bc, [1.0]])
        w = 1 / np.abs(np.polyval(b[::-1], x))
    poles = np.roots(b[::-1])
    Bm = np.hstack([1 / (x[:, None] - poles[None, :]), np.ones((npts, 1))])
    R, *_ = np.linalg.lstsq(Bm, Y, rcond=None)
    return f0 + hw * poles, R[:-1], Bm @ R, msk


def pick(poles, R, m, sign, f0, hw=2000.0):
    """The pole whose residue vector overlaps most with the plane wave of the +k (sign = +1) or -k mode. Phasor
    convention e^{+jwt}: the +k mode e^{i(kn - wt)} has shape ~ e^{-ikn}. For the band (sign 0) either works."""
    s = sign if sign else 1
    pw = np.exp(1j * s * 2 * np.pi * m * np.arange(N) / N)
    ok = (np.abs(poles.real - f0) < hw) & (np.abs(poles.imag) < hw)
    ov = np.abs(R @ pw) / (np.linalg.norm(R, axis=1) * np.sqrt(N) + 1e-300)
    ov[~ok] = -1
    return poles[int(np.argmax(ov))]


def line(f, V, m, sign, gyro=True, nboot=10):
    """Line frequency (Hz) with a noise-injection standard error. With >= 2 repeat files, lines_all also uses
    their scatter (preferred: take >= 3 repeat sweeps)."""
    f0 = design_f(m, sign if gyro else 0)
    po, R, fit, msk = vf(f, V, f0)
    fr = pick(po, R, m, sign, f0).real
    # noise-injection bootstrap: the relative noise level from fourth differences along frequency (model-free;
    # removes the smooth resonance shape), injected into the data and the line refitted. A residual or wild
    # bootstrap overestimated the error ~50x on synthetic data with known scatter: the fit residuals are not noise.
    d4 = V[4:] - 4 * V[3:-1] + 6 * V[2:-2] - 4 * V[1:-3] + V[:-4]
    sig = np.median(np.abs(d4) / np.abs(V[2:-2])) / (np.sqrt(70) * np.sqrt(2 * np.log(2)))
    boots = []
    for _ in range(nboot):
        Vb = V * (1 + sig * (RNG.standard_normal(V.shape) + 1j * RNG.standard_normal(V.shape)))
        pb, Rb, _, _ = vf(f, Vb, f0)
        boots.append(pick(pb, Rb, m, sign, f0).real)
    return fr, np.std(boots)


# ---- data ---------------------------------------------------------------------------------------------------
def load(pattern):
    files = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    out = []
    for fn in files:
        d = np.loadtxt(fn, skiprows=1)
        out.append((d[:, 0], d[:, 1::2] + 1j * d[:, 2::2]))
    return out


def synth(config, noise=1e-3):
    """Synthetic response of the design circuit (ideal gyrators with a 9.25 MHz pole, Q = 100, C_p = 10 pF,
    0.1% component scatter, a random 0.1% relative noise on every voltage). A demonstration only."""
    C, Lg, Lc, G = 10e-9, 1e-3, 4e-3, 0.106 * np.sqrt(10e-9 / 1e-3)
    w2 = np.sqrt(1.5e11)
    rng = np.random.default_rng(42)
    Cn, Lgn, Lcn = (v * (1 + 1e-3 * rng.standard_normal(N)) for v in (C, Lg, Lc))
    gyro = not config.startswith("band")
    f = np.arange(48e3, 75e3 + 1, 25.0) if config in ("band", "p1") else np.arange(54e3, 69.8e3 + 1, 25.0)
    invL = 1 / Lgn
    if config.startswith("m8_") and config != "m8_S0":
        name, sg = config[3:-2], (1 if config.endswith("_p") else -1)
        invL = invL * (1 + sg * S * eta_shape(*SHAPES[name]))
    V = np.zeros((len(f), N), complex)
    for i, fi in enumerate(f):
        w = 2 * np.pi * fi
        Gw = G / (1 + 1j * w / (2 * np.pi * 9.25e6))
        Y = np.diag(1j * w * Cn + 1 / (w2 * Lgn / 100 + 1j * w / invL))
        for n in range(N):
            m = (n + 1) % N
            yc = 1 / (w2 * Lcn[n] / 100 + 1j * w * Lcn[n]) + 1j * w * 10e-12
            Y[n, n] += yc; Y[m, m] += yc; Y[n, m] -= yc; Y[m, n] -= yc
            if gyro:
                Y[n, m] += Gw; Y[m, n] -= Gw
        I = np.zeros(N, complex); I[0] = 1
        V[i] = np.linalg.solve(Y, I)
    V *= 1 + noise * (rng.standard_normal(V.shape) + 1j * rng.standard_normal(V.shape))
    return [(f, V)]


def get(config):
    if DATA_DIR is None:
        return synth(config)
    d = load(config + "*.txt")
    if not d:
        raise FileNotFoundError(f"no files for {config} in {DATA_DIR}")
    return d


def lines_all(data, ms, sign, gyro=True):
    """Mean over repeat files; error = scatter of repeats (if >= 2 files) combined with the bootstrap error."""
    out = {}
    for m in ms:
        vals = [line(f, V, m, sign, gyro) for f, V in data]
        fr = np.mean([v[0] for v in vals])
        err = np.sqrt(np.mean([v[1] ** 2 for v in vals]) / len(vals)
                      + (np.var([v[0] for v in vals], ddof=1) / len(vals) if len(vals) > 1 else 0.0))
        out[m] = (fr, err)
    return out


# ---- analysis -----------------------------------------------------------------------------------------------
def main():
    print("CIRCUIT ANALYSIS --", "SYNTHETIC DEMONSTRATION DATA" if DATA_DIR is None else f"data from {DATA_DIR}")
    ms = range(0, 17)
    band = get("band")
    bp, bm = lines_all(band, ms, 1, gyro=False), lines_all(band, ms, -1, gyro=False)
    wb = np.array([np.pi * (bp[m][0] + bm[m][0]) for m in ms])
    zero = max(abs(bp[m][0] - bm[m][0]) for m in range(1, 16))
    X = np.vstack([np.ones(17), 2 * (1 - np.cos(2 * np.pi * np.array(ms) / N))]).T
    (K, c), *_ = np.linalg.lstsq(X, wb ** 2, rcond=None)
    fitres = np.abs(np.sqrt(X @ [K, c]) / wb - 1).max()
    p1 = get("p1")
    Lp, Lm = lines_all(p1, range(1, 16), 1), lines_all(p1, range(1, 16), -1)
    dw = {m: 2 * np.pi * (Lp[m][0] - Lm[m][0]) for m in range(1, 16)}
    dwe = {m: 2 * np.pi * np.hypot(Lp[m][1], Lm[m][1]) for m in range(1, 16)}
    b = dw[8] / 2
    prod = max(abs(4 * np.pi ** 2 * Lp[m][0] * Lm[m][0] / (K + 2 * c * (1 - np.cos(2 * np.pi * m / N))) - 1) for m in range(1, 16))
    print("\nCALIBRATIONS AND PRECONDITIONS")
    print(f"  1 zero check: max |f+ - f-| gyrators off = {zero / (dw[8] / 2 / np.pi):.1e} of Delta f(pi/2)   "
          f"[{'ok' if zero / (dw[8] / 2 / np.pi) < 1e-3 else 'FAIL - run void'}]")
    print(f"  2 band: K = {K:.5e} s^-2, c-hat = {c / K:.4f}, max fit residual {fitres:.1e}   "
          f"[{'ok' if abs(c / K - 0.25) <= 0.002 and fitres <= 2e-3 else 'FAIL - run void'}]")
    print(f"  3 beta-hat = {b / np.sqrt(K):.4f}   [{'ok' if abs(b / np.sqrt(K) - 0.106) <= 0.001 else 'FAIL - run void'}]")
    print(f"  4 product rule: max |w+ w- / Q(k) - 1| = {prod:.1e}   [{'ok' if prod <= 1e-3 else 'FAIL - run void'}]")
    print("\nP1 -- Delta w(m)/Delta w(8) - sin k   (failure: |residual| > 5e-3 + 3 sigma)")
    fails = 0
    for m in range(1, 16):
        r = dw[m] / dw[8] - np.sin(2 * np.pi * m / N)
        sig = abs(dw[m] / dw[8]) * np.hypot(dwe[m] / abs(dw[m]), dwe[8] / dw[8])
        bad = abs(r) > PRED["P1_tol"] + 3 * sig
        fails += bad
        print(f"  m = {m:2d}: Delta f = {dw[m] / (2 * np.pi):9.2f} Hz; residual {r:+.2e} +- {sig:.1e}  {'FAIL' if bad else 'pass'}")
    print(f"  P1: {'FAILS' if fails else 'passes'} ({fails} wavenumber(s) outside)")
    print("\nP2, P3 -- R = d(Delta w) / (-(1/4) b <dK^2> / c^2) at m = 8, +-S even part")
    base = get("m8_S0")
    b0p, b0m = lines_all(base, [8], 1)[8], lines_all(base, [8], -1)[8]
    d0, e0 = 2 * np.pi * (b0p[0] - b0m[0]), 2 * np.pi * np.hypot(b0p[1], b0m[1])
    for name in ["gauss4", "gauss3", "gauss5", "sech24", "two2.5", "rampNone"]:
        vals = []
        for sg in ("p", "m"):
            dd = get(f"m8_{name}_{sg}")
            lp, lm = lines_all(dd, [8], 1)[8], lines_all(dd, [8], -1)[8]
            vals.append((2 * np.pi * (lp[0] - lm[0]), 2 * np.pi * np.hypot(lp[1], lm[1])))
        shift = 0.5 * (vals[0][0] + vals[1][0]) - d0
        se = np.sqrt(0.25 * vals[0][1] ** 2 + 0.25 * vals[1][1] ** 2 + e0 ** 2)
        fp = None if DATA_DIR is None else os.path.join(DATA_DIR, f"dK_{name}_p.txt")
        fm = None if DATA_DIR is None else os.path.join(DATA_DIR, f"dK_{name}_m.txt")
        if fp and os.path.exists(fp) and os.path.exists(fm):
            # the realised bump differs between +S and -S (the GIC is slightly nonlinear): average <dK^2> over both
            dK2, flag = 0.5 * (np.mean(np.loadtxt(fp) ** 2) + np.mean(np.loadtxt(fm) ** 2)), ""
        else:
            dK2, flag = np.mean((K * S * eta_shape(*SHAPES[name])) ** 2), "  (design dK used -- measure it)"
        pred = -0.25 * b * dK2 / c ** 2
        R, sR = shift / pred, se / abs(pred)
        if name == "gauss4":
            bad = abs(R - PRED["P2_R"]) > PRED["P2_tol"] + 3 * sR
            print(f"  P2 (gauss4): d(Delta f) = {shift / (2 * np.pi):+.2f} Hz; R = {R:.4f} +- {sR:.4f}; predicted "
                  f"{PRED['P2_R']} (window +-{PRED['P2_tol']} + 3 sigma)  {'FAILS' if bad else 'passes'}{flag}")
        else:
            tol = PRED["P3_ramp_tol"] if name == "rampNone" else PRED["P3_tol"]
            bad = abs(R - PRED["P3"][name]) > tol + 3 * sR
            print(f"  P3 ({name}): R = {R:.4f} +- {sR:.4f}; predicted {PRED['P3'][name]} (window +-{tol} + 3 sigma)  "
                  f"{'FAILS' if bad else 'passes'}{flag}")


main()
```

## 8. SPICE netlist (ngspice)

TI's OPAx197 macro-model is not redistributed here: download https://www.ti.com/lit/zip/SBOMA34 and place
`OPAx197.LIB` next to the netlist. Put `set ngbehavior=psa` in `.spiceinit` (PSpice compatibility). The netlist
below is the nominal 32-node ring with the proposed passive trim element and an AC sweep (`gen_ring.netlist(gic=True)`
gives the pre-registered GIC version); `spice/gen_ring.py` writes it and its variants (bumps, Q,
gyrators off, transients); `spice/spice_verify.py` runs the verification. **This checks the design, not the
model: ngspice solves the circuit's equations, and the circuit was designed to realise the model's equations.**

```spice
* 32-node LC-gyrator ring (shape_zero realisation-gyroscopic branch)
.include OPAx197.LIB
VCC vcc 0 12
VEE vee 0 -12
* ---- cells (design choices ours) ----
.subckt NODE a vcc vee params: R5=3333.3
C1 a 0 10n
LG a xg 0.001176
RG xg 0 4.55463
* trim/bump element: passive slug-tuned inductor (proposed replacement for the GIC; CIRCUIT_BUILD.md sec 6 item 1)
LGIC a 0 {R5*2e-6}
.ends NODE
.subckt BOND a b vcc vee params: Rset=2983.28
LC a yc 0.004
RC yc b 15.4919
CP a b 1e-11
* VCCS B: +V(a)/Rset into b
XB1 a bufa vcc vee bufa OPAx197
RB1 bufa b {Rset}
RB2 hb b {Rset}
RB3 mb 0 10k
RB4 mb hb 10k
XB2 b mb vcc vee hb OPAx197
* VCCS A: -V(b)/Rset into a
XA1 b bufb vcc vee bufb OPAx197
RA1 0 a {Rset}
RA2 ha a {Rset}
RA3 bufb ma 10k
RA4 ma ha 10k
XA2 a ma vcc vee ha OPAx197
.ends BOND

XN0 n0 vcc vee NODE params: R5=3340.91
XN1 n1 vcc vee NODE params: R5=3340.91
XN2 n2 vcc vee NODE params: R5=3340.91
XN3 n3 vcc vee NODE params: R5=3340.91
XN4 n4 vcc vee NODE params: R5=3340.91
XN5 n5 vcc vee NODE params: R5=3340.91
XN6 n6 vcc vee NODE params: R5=3340.91
XN7 n7 vcc vee NODE params: R5=3340.91
XN8 n8 vcc vee NODE params: R5=3340.91
XN9 n9 vcc vee NODE params: R5=3340.91
XN10 n10 vcc vee NODE params: R5=3340.91
XN11 n11 vcc vee NODE params: R5=3340.91
XN12 n12 vcc vee NODE params: R5=3340.91
XN13 n13 vcc vee NODE params: R5=3340.91
XN14 n14 vcc vee NODE params: R5=3340.91
XN15 n15 vcc vee NODE params: R5=3340.91
XN16 n16 vcc vee NODE params: R5=3340.91
XN17 n17 vcc vee NODE params: R5=3340.91
XN18 n18 vcc vee NODE params: R5=3340.91
XN19 n19 vcc vee NODE params: R5=3340.91
XN20 n20 vcc vee NODE params: R5=3340.91
XN21 n21 vcc vee NODE params: R5=3340.91
XN22 n22 vcc vee NODE params: R5=3340.91
XN23 n23 vcc vee NODE params: R5=3340.91
XN24 n24 vcc vee NODE params: R5=3340.91
XN25 n25 vcc vee NODE params: R5=3340.91
XN26 n26 vcc vee NODE params: R5=3340.91
XN27 n27 vcc vee NODE params: R5=3340.91
XN28 n28 vcc vee NODE params: R5=3340.91
XN29 n29 vcc vee NODE params: R5=3340.91
XN30 n30 vcc vee NODE params: R5=3340.91
XN31 n31 vcc vee NODE params: R5=3340.91
XB0 n0 n1 vcc vee BOND params: Rset=2983.28
XB1 n1 n2 vcc vee BOND params: Rset=2983.28
XB2 n2 n3 vcc vee BOND params: Rset=2983.28
XB3 n3 n4 vcc vee BOND params: Rset=2983.28
XB4 n4 n5 vcc vee BOND params: Rset=2983.28
XB5 n5 n6 vcc vee BOND params: Rset=2983.28
XB6 n6 n7 vcc vee BOND params: Rset=2983.28
XB7 n7 n8 vcc vee BOND params: Rset=2983.28
XB8 n8 n9 vcc vee BOND params: Rset=2983.28
XB9 n9 n10 vcc vee BOND params: Rset=2983.28
XB10 n10 n11 vcc vee BOND params: Rset=2983.28
XB11 n11 n12 vcc vee BOND params: Rset=2983.28
XB12 n12 n13 vcc vee BOND params: Rset=2983.28
XB13 n13 n14 vcc vee BOND params: Rset=2983.28
XB14 n14 n15 vcc vee BOND params: Rset=2983.28
XB15 n15 n16 vcc vee BOND params: Rset=2983.28
XB16 n16 n17 vcc vee BOND params: Rset=2983.28
XB17 n17 n18 vcc vee BOND params: Rset=2983.28
XB18 n18 n19 vcc vee BOND params: Rset=2983.28
XB19 n19 n20 vcc vee BOND params: Rset=2983.28
XB20 n20 n21 vcc vee BOND params: Rset=2983.28
XB21 n21 n22 vcc vee BOND params: Rset=2983.28
XB22 n22 n23 vcc vee BOND params: Rset=2983.28
XB23 n23 n24 vcc vee BOND params: Rset=2983.28
XB24 n24 n25 vcc vee BOND params: Rset=2983.28
XB25 n25 n26 vcc vee BOND params: Rset=2983.28
XB26 n26 n27 vcc vee BOND params: Rset=2983.28
XB27 n27 n28 vcc vee BOND params: Rset=2983.28
XB28 n28 n29 vcc vee BOND params: Rset=2983.28
XB29 n29 n30 vcc vee BOND params: Rset=2983.28
XB30 n30 n31 vcc vee BOND params: Rset=2983.28
XB31 n31 n0 vcc vee BOND params: Rset=2983.28
IDRV 0 n0 dc 0 ac 1
.control
set wr_singlescale
set wr_vecnames
ac lin 1081 48000 75000
wrdata ring32_ac.txt v(n0) v(n1) v(n2) v(n3) v(n4) v(n5) v(n6) v(n7) v(n8) v(n9) v(n10) v(n11) v(n12) v(n13) v(n14) v(n15) v(n16) v(n17) v(n18) v(n19) v(n20) v(n21) v(n22) v(n23) v(n24) v(n25) v(n26) v(n27) v(n28) v(n29) v(n30) v(n31)
.endc
.end
```

### 8.1 SPICE results against our own simulation

ngspice 42 with TI's OPAx197 model (OPA4197/OPA2197); our model: `shape_zero_tests/circuit_error_budget.py` with the
same parasitics (10 pF across L_c, 6.5 pF op-amp input capacitance, inductor Q = 100) and a 9.25 MHz single-pole
VCCS (the SPICE unit test of one VCCS: |G| within 3×10⁻⁵ of design at 60 kHz, phase lag 6.5 mrad, output
conductance 1.4×10⁻⁷ S ∥ 6.5 pF). Estimator: a common-pole rational fit of all 32 node responses, each line
identified by its residue vector (validated on the exact model to 10⁻⁴ Hz and on the P2 ratio to 10⁻⁶). Outputs:
`spice/results/` (the verification and diagnostic outputs, the variant-A stability run, and transcribed unit-test
and transient results); raw data are regenerated by the scripts.
**The first-pass SPICE AC results (with GIC version 1) describe a circuit that is unstable in the time domain**;
they are kept for the record and for the parts of the circuit the GIC does not touch.

| quantity | pre-registered / own model | SPICE, GIC v1 | SPICE, ideal on-site element (variant A) | SPICE, all ideal (C) |
|---|---|---|---|---|
| zero check (gyrators out) | < 10⁻³ | 0 | — | — |
| band-fit ĉ | 0.250 ± 0.002 / own 0.2479 | 0.2475 | 0.2479 | 0.2479 |
| β̂ | 0.106 ± 0.001 / own 0.1057 | 0.1053 | 0.1055 | 0.1058 |
| product rule | ≤ 10⁻³ / own 4.2×10⁻⁴ | 4.0×10⁻³ | 4.0×10⁻³ | 4.2×10⁻⁴ |
| P1 max residual | ≤ 5×10⁻³ / own 1.0×10⁻³ | 1.6×10⁻³ | 1.6×10⁻³ | 1.0×10⁻³ |
| P2 R (design δK) | 1.031 ± 0.015 / own 1.0308 | **1.083** | 1.038 | 1.0305 |
| P3 R: Gaussian σ 3, σ 5, sech², two, ramp | 1.058, 1.026, 1.058, 1.077, 0.894 / own 1.054, 1.022, 1.054, 1.072, 0.918 | 1.108, 1.073, 1.108, 1.126, 0.980 | not run | not run |
| stability (AC poles, 32 nodes) | own: unstable between Q 600 and 1000 | stable ≤ 300, unstable ≥ 450 | stable at Q ≤ 200, unstable at Q ≥ 300 (threshold ≈ 290) | stable at 450 |
| stability (transient, 4 nodes) | — | **runs to the rails** at Q 100 and 1000 | decays at Q 100 (1.66×10³ s⁻¹) and 1000 (97 s⁻¹) | — |

Resonances: SPICE lines lie between +14 Hz (low m) and −95 Hz (m = 14) of our model's — up to 1.3×10⁻³, largest at
the top of the band. P1 is insensitive to it (both branches shift together).
