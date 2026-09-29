#!/usr/bin/env python3
"""
ampA_mechanisms.py -- anomaly A, the open first-order mechanism (2026-09-29): candidate mechanisms
under A' beyond the per-direction Peierls angle, each derived with NO new parameter, and their
predicted contributions to every certification slope. Committed before any comparison with the
recorded slopes (see ampA_MECHANISMS.md for the derivations, patterns and disclosures).

Exact first-order statement (derived): linearising A''s force -(sqrt5 + |u|) u about the linear
solution u0 gives, at first order in amplitude, the LINEAR lattice with the SCALAR potential
V(x, t) = |u0(x, t)| -- common to every internal component. So every first-order mechanism is the
interplay of a scalar potential, moving with the packet, with the direction-dependent propagation in
the link segments. Under the smooth diagnostic the potential is |u0|^2 ~ A^2: every candidate below
is linear in V and vanishes there by construction.

Candidates computed here (one-branch reduction, packet on the a-branch; design choices ours):
  ANGLE  per-direction Peierls angle (the recorded law), now spectrum-averaged at q = 1 too: per mode
         k and ramp site, the in-segment phase solved at Omega = w(k) + dw with Q fixed,
         dw = A_eff(t_site) / (2 w0 + kappa).
  BORN   the scalar potential acting OUTSIDE the in-segment frequency law: first-order Born
         integral of -i V psi / (2 w0 + kappa) over 0 < t < T (readout), with each segment applied as
         an instantaneous per-mode transfer T_s(k) = sum_j P_j exp(i [Phi_j(k) - Phi_0(k)]) at its
         centre-crossing time (Phi_j: WKB phase over the ramp, Phi_0 the free phase over the same
         sites). Contains: (K4) the shared radius acting on eigen-channels displaced by their group
         delays (between and after the segments), and (K1') the chirp that the potential imprints on
         the packet before a segment, changing the spectral average of the rotation.
  K3b    non-adiabatic ramp: exact plane-wave transfer-matrix phase of each eigen-direction at
         (Omega, V) against the WKB sum -- the ratio of d(relative phase)/dV, applied to ANGLE.
  K1     envelope averaging: the recorded weight A_eff = <F^3>/<F^2> is the first-order one (ray
         theory, derived); the PEAK-amplitude variant is reported as the alternative.
  K2     dwell time: contained in ANGLE (phase = splitting x dwell, both at Omega); the variant in
         which the dwell does NOT change (envelope at the linear group velocity) is
         ANGLE x [1 - 2 w/(2 w + kappa)].
usage: python3 ampA_mechanisms.py  -> ampA_mechanisms_predictions.txt / .json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M
import q3_gate as Q

KAP = float(M.KAPPA)
W0 = 0.5 * (-KAP + np.sqrt(KAP ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(M.K0)))))
D0 = 2 * W0 + KAP
V0 = 2 * M.C * np.sin(M.K0) / D0


# ------------------------------------------------------------------ geometry
def setup(q):
    if q == 1:
        shape, n0, width, segs = (1200,), 20, 8.0, (60, 80)
    else:
        shape, n0, width, segs = (260, 8, 8), Q.X0, Q.WIDTH, Q.SEGS
    ks = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m) for m in shape], indexing="ij")
    Qt = 2 * M.C * sum(1 - np.cos(k) for k in ks[1:]) if q == 3 else np.zeros(shape)
    om = 0.5 * (-KAP + np.sqrt(KAP ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(ks[0])) + Qt)))
    c = np.indices(shape).astype(float)
    r2 = (c[0] - n0) ** 2
    for a in range(1, len(shape)):
        d = c[a] - shape[a] / 2.0
        r2 = r2 + d ** 2
    env = np.exp(-0.5 * r2 / width ** 2)
    phi = env * np.exp(1j * M.K0 * (c[0] - n0))
    return dict(shape=shape, n0=n0, segs=segs, kx=ks[0], Qt=Qt, om=om, phi_hat=np.fft.fftn(phi))


def seg_phase(kx, om, Qt, geig, Omega=None):
    """sum over the ramp of the in-segment wavenumber (WKB), per mode; Q fixed at the linear value,
    Peierls at Omega (default: the linear om)."""
    Om = om if Omega is None else Omega
    rhs = om * om + KAP * om - M.SQ5 - Qt
    tot = np.zeros_like(kx)
    for wgt in M.RAMP:
        g = geig * wgt
        f = lambda k: 2 * M.C * (1 - np.cos(k)) - 2 * M.C * g * Om * np.sin(k) - rhs
        lo = np.full_like(kx, 1e-6); hi = np.full_like(kx, np.pi - 1e-6)
        flo = f(lo)
        for _ in range(60):
            mid = 0.5 * (lo + hi); fm = f(mid)
            left = flo * fm <= 0
            hi = np.where(left, mid, hi); lo = np.where(left, lo, mid); flo = np.where(left, flo, fm)
        tot = tot + 0.5 * (lo + hi)
    return tot


# ------------------------------------------------------------------ the model
class Model:
    def __init__(self, q, n, spec, T, amp):
        """spec: [(axis, g), (axis, g)] for the two segments, in order."""
        self.q, self.n, self.T, self.amp = q, n, T, amp
        self.s = setup(q)
        s = self.s
        self.G = M.generators(n)
        self.mask = (s["kx"] > 0) & (s["kx"] < np.pi)       # a right-moving packet
        self.ts = [(st + len(M.RAMP) / 2.0 - s["n0"]) / V0 for st in s["segs"]]
        self.trans, self.eig = [], []
        for (ax, g), st in zip(spec, s["segs"]):
            e, U = np.linalg.eigh(self.G[ax])
            ph0 = seg_phase(s["kx"], s["om"], s["Qt"], 0.0)
            phs = [np.where(self.mask, seg_phase(s["kx"], s["om"], s["Qt"], g * e[j]) - ph0, 0.0)
                   for j in range(n)]
            self.trans.append((e, U, phs, g, st))

    def apply(self, psi_hat, seg, phases=None):
        e, U, phs, g, st = self.trans[seg]
        phs = phs if phases is None else phases
        c = np.einsum("ij,...j->...i", U.conj().T, psi_hat)
        c = c * np.stack([np.exp(1j * p) for p in phs], axis=-1)
        return np.einsum("ij,...j->...i", U, c)

    def lin_hat(self, t, psi0_hat, t_from=0.0, segs_after=None):
        """propagate a k-space state from t_from to t, applying segments crossed in between."""
        s = self.s
        out = psi0_hat
        cur = t_from
        for i, ts in enumerate(self.ts):
            if t_from < ts <= t:
                out = out * np.exp(-1j * s["om"] * (ts - cur))[..., None]
                out = self.apply(out, i)
                cur = ts
        return out * np.exp(-1j * s["om"] * (t - cur))[..., None]

    def initial(self):
        s = self.s
        p = np.zeros(s["shape"] + (self.n,), complex)
        p[..., 0] = self.amp * s["phi_hat"] * self.mask
        return p

    def coords(self, psi_hat):
        psi = np.fft.ifftn(psi_hat, axes=tuple(range(self.q if self.q == 1 else 3)))
        psi = psi.reshape(-1, self.n)
        r = psi.T @ psi.conj()
        tr = np.real(np.trace(r))
        return np.array([np.real(np.trace(S @ r)) / tr for S in self.G])

    def run_lin(self):
        return self.lin_hat(self.T, self.initial())

    def run_born(self, dt=1.0, window=(0.0, np.inf)):
        """first-order state change from the scalar potential, instantaneous transfers; the potential
        acts only for window[0] <= t < window[1]."""
        ax = tuple(range(len(self.s["shape"])))
        p0 = self.initial()
        acc = np.zeros_like(p0)
        for t in np.arange(0.5 * dt, self.T, dt):
            if not (window[0] <= t < window[1]):
                continue
            ph = self.lin_hat(t, p0)
            x = np.fft.ifftn(ph, axes=ax)
            V = np.sqrt((np.abs(x) ** 2).sum(-1, keepdims=True))
            src = np.fft.fftn(-1j * dt * V * x / D0, axes=ax)
            acc = acc + self.lin_hat(self.T, src, t_from=t)
        return acc

    def run_angle(self, Aeff_site):
        """ANGLE: per-mode in-segment phase at Omega = om + dw(site)."""
        s = self.s
        p = self.initial()
        cur = 0.0
        for i, ts in enumerate(self.ts):
            p = p * np.exp(-1j * s["om"] * (ts - cur))[..., None]
            e, U, phs, g, st = self.trans[i]
            ph0 = seg_phase(s["kx"], s["om"], s["Qt"], 0.0)
            new = []
            for j in range(self.n):
                tot = np.zeros_like(s["kx"])
                for isite, wgt in enumerate(M.RAMP):
                    dw = Aeff_site[i][isite] / D0
                    tot = tot + seg_phase_one(s["kx"], s["om"], s["Qt"], g * wgt * e[j], s["om"] + dw)
                new.append(np.where(self.mask, tot - ph0, 0.0))
            p = self.apply(p, i, new)
            cur = ts
        return p * np.exp(-1j * s["om"] * (self.T - cur))[..., None]


def seg_phase_one(kx, om, Qt, g, Om):
    rhs = om * om + KAP * om - M.SQ5 - Qt
    f = lambda k: 2 * M.C * (1 - np.cos(k)) - 2 * M.C * g * Om * np.sin(k) - rhs
    lo = np.full_like(kx, 1e-6); hi = np.full_like(kx, np.pi - 1e-6)
    flo = f(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi); fm = f(mid)
        left = flo * fm <= 0
        hi = np.where(left, mid, hi); lo = np.where(left, lo, mid); flo = np.where(left, flo, fm)
    return 0.5 * (lo + hi)


def aeff_sites(model):
    """A_eff (and the peak) of the linear free envelope at each ramp site's crossing time."""
    s = model.s
    ax = tuple(range(len(s["shape"])))
    ph = model.amp * s["phi_hat"] * model.mask
    out, peak = [], []
    for st in s["segs"]:
        a, pk = [], []
        for i in range(len(M.RAMP)):
            t = (st + i - s["n0"]) / V0
            F = np.abs(np.fft.ifftn(ph * np.exp(-1j * s["om"] * t), axes=ax))
            a.append((F ** 3).sum() / (F ** 2).sum()); pk.append(F.max())
        out.append(a); peak.append(pk)
    return out, peak


# ------------------------------------------------------------------ K3b: exact ramp transmission
def transfer_phase(h, g, Om, V):
    """arg of the transmission of a right-moving plane wave through the ramp, eigenvalue h of the
    segment generator, frequency Om, uniform on-site potential V (outside k0 = pi/2 fixed by V)."""
    Wn = np.zeros(40)
    Wn[10:10 + len(M.RAMP)] = g * h * np.array(M.RAMP)
    rhs = M.SQ5 + V + 2 * M.C - Om ** 2 - KAP * Om
    # outside: C (e^{ik} + e^{-ik}) = rhs  -> cos k = rhs / 2C
    k = np.arccos(rhs / (2 * M.C))
    # iterate backwards from a pure transmitted wave at the right end
    N = len(Wn)
    psi = np.zeros(N + 2, complex)
    psi[N + 1] = np.exp(1j * k * (N + 1)); psi[N] = np.exp(1j * k * N)
    for m in range(N, 0, -1):
        Wm, Wm1 = Wn[m] if m < N else 0.0, Wn[m - 1]
        # C(1 - i Om W_m) psi_{m+1} + C(1 + i Om W_{m-1}) psi_{m-1} = rhs psi_m
        psi[m - 1] = (rhs * psi[m] - M.C * (1 - 1j * Om * Wm) * psi[m + 1]) / (M.C * (1 + 1j * Om * Wm1))
    # decompose psi at sites 0, 1 into incident A e^{ikn} + reflected B e^{-ikn}
    Mx = np.array([[1, 1], [np.exp(1j * k), np.exp(-1j * k)]])
    A, B = np.linalg.solve(Mx, psi[:2])
    return np.angle(1 / A)


def k3b_ratio(g, V=1e-4):
    Om0, Om1 = W0, W0 + V / D0
    rel = lambda Om, VV: transfer_phase(1, g, Om, VV) - transfer_phase(-1, g, Om, VV)
    exact = (rel(Om1, V) - rel(Om0, 0.0)) / V
    wkb = sum(2 * g * w / (1 + (g * w * W0) ** 2) for w in M.RAMP) / D0
    return exact / wkb, exact, wkb


# ------------------------------------------------------------------ slopes
def cases(q):
    if q == 1:
        rows = json.load(open(os.path.join(HERE, "nodewell_1d.json")))
        Tq = {(r["n"], tuple(r["axes"])): r["T"] for r in rows if r["amp"] == 1e-3}
        return [(n, gA, gB, Tq[(n, (0, 1))], Tq[(n, (0, 0))]) for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10))]
    d = json.load(open(os.path.join(HERE, "q3_gate_runs_260x8_nodeA1.json")))
    t = {(r["n"], r["job"]): r["t"] for r in d["runs"]}
    return [(n, Q.G[n][0], Q.G[n][1], t[(n, "AB")], t[(n, "fAB")]) for n in (2, 3)]


AMP_LIN = 1e-8      # model amplitude: exact first order (quadratic terms ~1e-3 of the rotation effects)
SCALE = 1e-3 / AMP_LIN


def pangle(a, b):
    """angle between Bloch vectors in degrees, precise for tiny angles (model.angle uses arccos)."""
    a = np.asarray(a) / np.linalg.norm(a); b = np.asarray(b) / np.linalg.norm(b)
    return np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(a - b) / 2)))


def main():
    amp = AMP_LIN
    res, lines = {}, []
    for q in (1, 3):
        for n, gA, gB, Tsplit, Tfloor in cases(q):
            fA, fB = (gA, gB) if q == 1 else Q.GFLOOR[n]
            specs = {"AB": [(0, gA), (1, gB)], "BA": [(1, gB), (0, gA)],
                     "fAB": [(0, fA), (0, fB)], "fBA": [(0, fB), (0, fA)]}
            co = {}
            for job, sp in specs.items():
                T = Tsplit if job in ("AB", "BA") else Tfloor
                mdl = Model(q, n, sp, T, amp)
                A, P = aeff_sites(mdl)
                lin = mdl.run_lin()
                ang = mdl.run_angle(A)
                angpk = mdl.run_angle(P)
                born = mdl.run_born()
                co[job] = {"lin": mdl.coords(lin), "angle": mdl.coords(ang), "peak": mdl.coords(angpk),
                           "born": mdl.coords(lin + born), "total": mdl.coords(ang + born)}
                print(q, n, job, "done", flush=True)
            ang_ = lambda a, b: SCALE * pangle(a, b)
            key = f"q={q} u({n})"
            res[key] = {}
            for var in ("angle", "peak", "born", "total"):
                d = {}
                for o in ("AB", "BA"):
                    d[f"per-order {o}"] = ang_(co[o][var], co[o]["lin"])
                d["split"] = (ang_(co["AB"][var], co["BA"][var]) - ang_(co["AB"]["lin"], co["BA"]["lin"]))
                d["floor"] = ang_(co["fAB"][var], co["fBA"][var]) - ang_(co["fAB"]["lin"], co["fBA"]["lin"])
                # the first-order Bloch-vector change, for the vector comparison
                d["dco"] = {o: (SCALE * (np.asarray(co[o][var]) - np.asarray(co[o]["lin"]))).tolist() for o in ("AB", "BA")}
                res[key][var] = d
            lines.append(f"\n  {key}  (slopes in deg per 1e-3; split = split(model) - split(linear))")
            for var in ("angle", "peak", "born", "total"):
                d = res[key][var]
                lines.append(f"    {var:<6} split {d['split']:+.5f}   per-order AB {d['per-order AB']:.5f}   "
                             f"BA {d['per-order BA']:.5f}   floor {d['floor']:+.6f}")
    k3 = {g: k3b_ratio(g) for g in (0.08, 0.10, 0.12, 0.15)}
    lines.append("\n  K3b  exact ramp transmission / WKB, d(relative phase)/dV: " +
                 ", ".join(f"g={g}: {v[0]:.4f}" for g, v in k3.items()))
    fac = 1 - 2 * W0 / D0
    lines.append(f"  K2   no-dwell-change variant: ANGLE x {fac:.4f}")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "ampA_mechanisms_predictions.txt"), "w").write(
        "PREDICTIONS (committed before comparison with the recorded slopes) -- anomaly A mechanisms\n" + out + "\n")
    # Born by time window (pre: before segment 1's centre crossing; mid; post: after segment 2's)
    # and the readout-time dependence (T x 1, 1.5, 2) -- q = 1, the split pair
    lines.append("\n  BORN by window and readout time (q = 1, split pair; deg per 1e-3):")
    res["born_windows"], res["T_scan"] = {}, {}
    for n, gA, gB, Tsplit, _ in cases(1):
        sp = {"AB": [(0, gA), (1, gB)], "BA": [(1, gB), (0, gA)]}
        wres, tres = {}, {}
        for fac_T in (1.0, 1.5, 2.0):
            cc = {}
            for job, spc in sp.items():
                mdl = Model(1, n, spc, Tsplit * fac_T, amp)
                A, _ = aeff_sites(mdl)
                lin, ang = mdl.run_lin(), mdl.run_angle(A)
                t1, t2 = mdl.ts
                cc[job] = {"lin": mdl.coords(lin), "total": mdl.coords(ang + mdl.run_born())}
                if fac_T == 1.0:
                    for nm, win in (("pre", (0, t1)), ("mid", (t1, t2)), ("post", (t2, np.inf))):
                        cc[job][nm] = mdl.coords(lin + mdl.run_born(window=win))
            ang_ = lambda a, b: SCALE * pangle(a, b)
            vs = ("total", "pre", "mid", "post") if fac_T == 1.0 else ("total",)
            for v in vs:
                d = {f"per-order {o}": ang_(cc[o][v], cc[o]["lin"]) for o in ("AB", "BA")}
                d["split"] = ang_(cc["AB"][v], cc["BA"][v]) - ang_(cc["AB"]["lin"], cc["BA"]["lin"])
                (wres if v != "total" else tres)[v if v != "total" else fac_T] = d
                lines.append(f"    u({n}) {('T x %.1f total' % fac_T) if v == 'total' else 'born ' + v:<16} split {d['split']:+.5f}"
                             f"   per-order AB {d['per-order AB']:.5f}   BA {d['per-order BA']:.5f}")
        res["born_windows"][n], res["T_scan"][n] = wres, {str(k): v for k, v in tres.items()}
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "ampA_mechanisms_predictions.txt"), "w").write(
        "PREDICTIONS (committed before comparison with the recorded slopes) -- anomaly A mechanisms\n" + out + "\n")
    res["K3b"] = {str(g): v for g, v in k3.items()}
    res["K2_factor"] = fac
    json.dump(res, open(os.path.join(HERE, "ampA_mechanisms_predictions.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
