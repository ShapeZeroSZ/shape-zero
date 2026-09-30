#!/usr/bin/env python3
"""
ampA_regions.py -- anomaly A, 2026-09-30: where the outside factor lives and what the in-segment term is.
Predictions: ampA_REGIONS_PREDICTIONS.md (committed before any run). Design choices ours.

  predict  -- model predictions (ray bookkeeping + K4), written to ampA_regions_predictions.json/.txt
  run      -- the lattice runs, written to ampA_regions_runs.json
  evaluate -- comparison, written to ampA_regions_compare_output.txt

Runs (A' nonlinearity masked, each against a linear run of the same geometry and readout time):
  (1) certification geometry, q = 1 and q = 3: OUT split into BEFORE (x < segment-1 start),
      BETWEEN (segment-1 end <= x < segment-2 start), AFTER (x >= segment-2 end). The IN, OUT, ALL
      and linear runs are reused from ampA_masked_runs.json (same code, same geometry).
  (2) q = 3 with a TRANSVERSELY UNIFORM packet (the x-profile of the certification packet, uniform in
      y and z; per-mode launch): linear, IN, ALL.
  (3) width scan (ours): q = 1, ONE segment (u(2), axis 0, g = 0.12, start 300), packet widths 4, 8,
      16, 32 at n0 = 150 on N = 2000, read at T = 700: linear and ALL.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "04_scripts", "session"))
import model as M
import q3_gate as Q
import ampA_mechanisms as X
import ampA_masked as MK

L = len(M.RAMP)
AMP = 1e-3
WIDTHS = (4.0, 8.0, 16.0, 32.0)
WS = dict(N=2000, n0=150, start=300, T=700.0, g=0.12)


# ------------------------------------------------------------------ model side
def make_setup(q, shape, n0, width, segs, transverse_uniform=False):
    ks = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(m) for m in shape], indexing="ij")
    Qt = 2 * M.C * sum(1 - np.cos(k) for k in ks[1:]) if q == 3 else np.zeros(shape)
    om = 0.5 * (-X.KAP + np.sqrt(X.KAP ** 2 + 4 * (M.SQ5 + 2 * M.C * (1 - np.cos(ks[0])) + Qt)))
    c = np.indices(shape).astype(float)
    r2 = (c[0] - n0) ** 2
    if not transverse_uniform:
        for a in range(1, len(shape)):
            r2 = r2 + (c[a] - shape[a] / 2.0) ** 2
    phi = np.exp(-0.5 * r2 / width ** 2) * np.exp(1j * M.K0 * (c[0] - n0))
    return dict(shape=shape, n0=n0, segs=segs, kx=ks[0], Qt=Qt, om=om, phi_hat=np.fft.fftn(phi))


class RModel(X.Model):
    def __init__(self, q, n, spec, T, amp, s):
        self.q, self.n, self.T, self.amp, self.s = q, n, T, amp, s
        self.G = M.generators(n)
        self.mask = (s["kx"] > 0) & (s["kx"] < np.pi)
        self.ts = [(st + L / 2.0 - s["n0"]) / X.V0 for st in s["segs"]]
        self.trans = []
        for (ax, g), st in zip(spec, s["segs"]):
            e, U = np.linalg.eigh(self.G[ax])
            ph0 = X.seg_phase(s["kx"], s["om"], s["Qt"], 0.0)
            phs = [np.where(self.mask, X.seg_phase(s["kx"], s["om"], s["Qt"], g * e[j]) - ph0, 0.0) for j in range(n)]
            self.trans.append((e, U, phs, g, st))


def V_of_t(mdl, t):
    ax = tuple(range(len(mdl.s["shape"])))
    F = np.abs(np.fft.ifftn(mdl.amp * mdl.s["phi_hat"] * mdl.mask * np.exp(-1j * mdl.s["om"] * t), axes=ax))
    return (F ** 3).sum() / (F ** 2).sum()


def region_V(mdl):
    """Ray bookkeeping: effective potential per region for each segment's rotation (see the .md)."""
    s = mdl.s
    t = lambda x: (x - s["n0"]) / X.V0
    if len(s["segs"]) == 1:
        t1i = t(s["segs"][0])
        return {"ALL": [mdl_V(mdl, t1i)]}
    s1, s2 = s["segs"]
    t1i, t1o, t2i = t(s1), t(s1 + L), t(s2)
    V1i, V1o, V2i = mdl_V(mdl, t1i), mdl_V(mdl, t1o), mdl_V(mdl, t2i)
    return {"BEFORE": [V1i, V1i], "IN": [0.0, V1o - V1i], "BETWEEN": [0.0, V2i - V1o], "AFTER": [0.0, 0.0],
            "ALL": [V1i, V2i], "OUT": [V1i, V2i - V1o + V1i]}


def mdl_V(mdl, t):
    return V_of_t(mdl, t)


def model_vectors(mdl):
    """dco per region (ray bookkeeping), plus the K4 (Born) windows and the per-site angle law."""
    lin = mdl.coords(mdl.run_lin())
    out = {}
    for R, Vs in region_V(mdl).items():
        A = [[v] * L for v in Vs]
        out[R] = X.SCALE * (mdl.coords(mdl.run_angle(A)) - lin)
    if len(mdl.s["segs"]) == 2:
        t1, t2 = mdl.ts
        out["K4_pre"] = X.SCALE * (mdl.coords(lin_plus(mdl, mdl.run_born(window=(0, t1)))) - lin)
        out["K4_mid"] = X.SCALE * (mdl.coords(lin_plus(mdl, mdl.run_born(window=(t1, t2)))) - lin)
    A, _ = X.aeff_sites(mdl)
    out["ANGLE_per_site"] = X.SCALE * (mdl.coords(mdl.run_angle(A)) - lin)
    return {k: v.tolist() for k, v in out.items()}


def lin_plus(mdl, born):
    return mdl.run_lin() + born


def geometries():
    rows = json.load(open(os.path.join(HERE, "nodewell_1d.json")))
    T1 = {r["n"]: r["T"] for r in rows if r["amp"] == 1e-3 and r["axes"] == [0, 1]}
    d = json.load(open(os.path.join(HERE, "q3_gate_runs_260x8_nodeA1.json")))
    t3 = {r["n"]: r["t"] for r in d["runs"] if r["job"] == "AB"}
    G = []
    for n in (2, 3):
        gq1 = {2: (0.12, 0.08), 3: (0.15, 0.10)}[n]
        G.append(("q1", 1, n, gq1, T1[n], make_setup(1, (1200,), 20, 8.0, (60, 80))))
        G.append(("q3", 3, n, Q.G[n], t3[n], make_setup(3, (260, 8, 8), Q.X0, Q.WIDTH, Q.SEGS)))
        G.append(("q3u", 3, n, Q.G[n], t3[n], make_setup(3, (260, 8, 8), Q.X0, Q.WIDTH, Q.SEGS, True)))
    return G


def predict():
    res, lines = {}, ["PREDICTIONS (ray bookkeeping + K4; committed before any run) -- dco vectors in *.json; "
                      "rotation-angle magnitudes below, deg per 1e-3"]
    for tag, q, n, (gA, gB), T, s in geometries():
        for o, sp in (("AB", [(0, gA), (1, gB)]), ("BA", [(1, gB), (0, gA)])):
            mdl = RModel(q, n, sp, T, X.AMP_LIN, s)
            v = model_vectors(mdl)
            res[f"{tag} u({n}) {o}"] = v
            lines.append(f"  {tag:<3} u({n}) {o}: " + "  ".join(
                f"{k} {np.degrees(np.linalg.norm(np.array(val))):.5f}" for k, val in v.items()))
            print(lines[-1], flush=True)
    for w in WIDTHS:
        s = make_setup(1, (WS["N"],), WS["n0"], w, (WS["start"],))
        mdl = RModel(1, 2, [(0, WS["g"])], WS["T"], X.AMP_LIN, s)
        lin = mdl.coords(mdl.run_lin())
        A, _ = X.aeff_sites(mdl)
        dv = X.SCALE * (mdl.coords(mdl.run_angle(A)) - lin)
        res[f"width {w}"] = {"ANGLE_per_site": dv.tolist(), "Aeff_mid": A[0][6]}
        lines.append(f"  width {w:>4}: single-segment rotation change {np.degrees(np.linalg.norm(dv)):.5f} deg per 1e-3 "
                     f"(A_eff at the segment centre {A[0][6]:.3e} x 1e-3/1e-8)")
        print(lines[-1], flush=True)
    json.dump(res, open(os.path.join(HERE, "ampA_regions_predictions.json"), "w"), indent=1)
    open(os.path.join(HERE, "ampA_regions_predictions.txt"), "w").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ lattice side
def mask_regions(kind, shape, segs):
    x = np.indices(shape)[0].reshape(-1)
    s1, s2 = segs
    return {"BEFORE": (x < s1), "BETWEEN": (x >= s1 + L) & (x < s2), "AFTER": (x >= s2 + L)}[kind].astype(float)


def packet3_uniform(lat, amp):
    c = np.indices(lat.shape).astype(float)
    env = np.exp(-0.5 * (c[0] - Q.X0) ** 2 / Q.WIDTH ** 2).reshape(-1)
    ph = (M.K0 * (c[0] - Q.X0)).reshape(-1)
    u = np.zeros((lat.N, lat.D)); v = np.zeros((lat.N, lat.D))
    u[:, 0], u[:, 1] = amp * env * np.cos(ph), amp * env * np.sin(ph)
    psi = (u[:, 0] + 1j * u[:, 1]).reshape(lat.shape)
    dpsi = np.fft.ifftn(-1j * lat.branch_omega() * np.fft.fftn(psi)).reshape(-1)
    v[:, 0], v[:, 1] = dpsi.real, dpsi.imag
    return u, v


class MaskedW(M.Lattice):
    def __init__(self, n, N, mask):
        super().__init__(n=n, N=N, well="node")
        self.mask = mask[:, None]

    _onsite_nl = MK.Masked1._onsite_nl
    _onsite_cubic = MK.Masked1._onsite_cubic


def job(a):
    tag, q, n, order, kind, T = a
    if tag == "q1":
        segs = (60, 80)
        gA, gB = {2: (0.12, 0.08), 3: (0.15, 0.10)}[n]
        spec = {"AB": [(60, 0, gA), (80, 1, gB)], "BA": [(60, 1, gB), (80, 0, gA)]}[order]
        lat = MK.Masked1(n, mask_regions(kind, (1200,), segs))
        W, Wm = M.make_links(lat, spec)
        u, v = lat.packet(width=8.0, n0=20, amp=AMP, per_mode=True)
        u, v, drift = lat.run(u, v, T, W, Wm)
    elif tag in ("q3", "q3u"):
        mask = (mask_regions(kind, (260, 8, 8), Q.SEGS) if kind in ("BEFORE", "BETWEEN", "AFTER")
                else MK.mask_for(kind, (260, 8, 8), Q.SEGS))
        lat = MK.Masked3(n, float(M.KAPPA), mask)
        W, Wm = M.make_links(lat, Q.spec_for(n, order))
        u, v = lat.packet3(AMP) if tag == "q3" else packet3_uniform(lat, AMP)
        E0 = lat.energy(u, v)
        t = 0.0
        while t < T - 1e-9:
            u, v, _ = lat.run(u, v, Q.CHECK, W, Wm); t += Q.CHECK
        drift = abs(lat.energy(u, v) - E0) / abs(E0)
    else:  # width scan
        w = float(tag.split("_")[1])
        mask = np.zeros(WS["N"]) if kind == "lin" else np.ones(WS["N"])
        lat = MaskedW(2, WS["N"], mask)
        W, Wm = M.make_links(lat, [(WS["start"], 0, WS["g"])])
        u, v = lat.packet(width=w, n0=WS["n0"], amp=AMP, per_mode=True)
        u, v, drift = lat.run(u, v, WS["T"], W, Wm)
    co, _ = lat.readout(u, v)
    return dict(tag=tag, q=q, n=n, order=order, kind=kind, T=T, co=co.tolist(), drift=float(drift))


def jobs():
    J = []
    for tag, q, n, g, T, s in geometries():
        for o in ("AB", "BA"):
            if tag in ("q1", "q3"):
                J += [(tag, q, n, o, k, T) for k in ("BEFORE", "BETWEEN", "AFTER")]
            else:
                J += [(tag, q, n, o, k, T) for k in ("lin", "IN", "ALL")]
    for w in WIDTHS:
        J += [(f"w_{w}", 1, 2, "AB", k, WS["T"]) for k in ("lin", "ALL")]
    # longest first
    J.sort(key=lambda a: (not a[0].startswith("w_"), a[0] != "q1"))
    return J


def run():
    """incremental: each finished job is appended to ampA_regions_runs.jsonl; finished jobs are skipped."""
    part = os.path.join(HERE, "ampA_regions_runs.jsonl")
    done = set()
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            done.add((r["tag"], r["n"], r["order"], r["kind"]))
    todo = [a for a in jobs() if (a[0], a[2], a[3], a[4]) not in done]
    with Pool(4) as p, open(part, "a") as f:
        for r in p.imap_unordered(job, todo, chunksize=1):
            f.write(json.dumps(r) + "\n"); f.flush()
    res = [json.loads(line) for line in open(part)]
    json.dump(res, open(os.path.join(HERE, "ampA_regions_runs.json"), "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] in ("predict", "run"):
    {"predict": predict, "run": run}[sys.argv[1]]()


# ------------------------------------------------------------------ evaluation (written after the predictions commit 000deb4)
def evaluate():
    P = json.load(open(os.path.join(HERE, "ampA_regions_predictions.json")))
    R = json.load(open(os.path.join(HERE, "ampA_regions_runs.json")))
    Mk = json.load(open(os.path.join(HERE, "ampA_masked_runs.json")))
    co = {}
    for r in R:
        co[(r["tag"], r["n"], r["order"], r["kind"])] = np.array(r["co"])
    for r in Mk:
        if r["tag"] == "cert":
            co[({1: "q1", 3: "q3"}[r["q"]], r["n"], r["order"], r["kind"])] = np.array(r["co"])
    nrm = np.linalg.norm
    proj = lambda v, m: float(v @ m / (m @ m))
    deg = lambda v: float(np.degrees(nrm(v)))
    L_ = ["EVALUATION against ampA_REGIONS_PREDICTIONS.md (000deb4); magnitudes in deg per 1e-3"]
    L_.append("\n(1) OUT split -- factor = projection onto the model vector (BEFORE: ray + K4 pre; BETWEEN: ray + K4 mid)")
    for tag in ("q1", "q3"):
        for n in (2, 3):
            for o in ("AB", "BA"):
                lin = co[(tag, n, o, "lin")]
                d = {k: co[(tag, n, o, k)] - lin for k in ("BEFORE", "BETWEEN", "AFTER", "IN", "OUT", "ALL")}
                m = {k: np.array(v) for k, v in P[f"{tag} u({n}) {o}"].items()}
                mB, mW = m["BEFORE"] + m["K4_pre"], m["BETWEEN"] + m["K4_mid"]
                add = nrm(d["BEFORE"] + d["BETWEEN"] + d["AFTER"] - d["OUT"]) / nrm(d["OUT"])
                L_.append(f"  {tag} u({n}) {o}: |BEFORE| {deg(d['BEFORE']):.5f} (model {deg(mB):.5f}, factor {proj(d['BEFORE'], mB):.3f}); "
                          f"|BETWEEN| {deg(d['BETWEEN']):.5f} (model {deg(mW):.5f}, factor {proj(d['BETWEEN'], mW):+.3f}, "
                          f"proj on BEFORE-model {proj(d['BETWEEN'], mB):+.3f}); "
                          f"|AFTER|/|ALL| {nrm(d['AFTER']) / nrm(d['ALL']):.4f}; additivity {add:.4f}; "
                          f"|IN| {deg(d['IN']):.5f} (ray model {deg(m['IN']):.5f})")
    L_.append("\n(2) q = 3, transversely uniform packet")
    for n in (2, 3):
        for o in ("AB", "BA"):
            lin = co[("q3u", n, o, "lin")]
            dI, dA = co[("q3u", n, o, "IN")] - lin, co[("q3u", n, o, "ALL")] - lin
            m = {k: np.array(v) for k, v in P[f"q3u u({n}) {o}"].items()}
            mA = m["ALL"] + m["K4_pre"] + m["K4_mid"]
            L_.append(f"  u({n}) {o}: |IN|/|ALL| {nrm(dI) / nrm(dA):.3f}; |ALL| {deg(dA):.5f}; ALL factor on (ray + K4) {proj(dA, mA):.3f}; "
                      f"IN projection on (ray + K4) {proj(dI, mA):+.3f}")
    L_.append("\n(3b) width scan, single segment, u(2) g = 0.12")
    for w in WIDTHS:
        dv = co[(f"w_{w}", 2, "AB", "ALL")] - co[(f"w_{w}", 2, "AB", "lin")]
        mv = np.array(P[f"width {w}"]["ANGLE_per_site"])
        L_.append(f"  width {w:>4}: measured {deg(dv):.5f}, predicted {deg(mv):.5f}, factor {proj(dv, mv):.3f}, "
                  f"cos {dv @ mv / (nrm(dv) * nrm(mv)):+.4f}")
    L_.append("\n  drifts: max " + f"{max(r['drift'] for r in R):.1e}")
    out = "\n".join(L_)
    print(out)
    open(os.path.join(HERE, "ampA_regions_compare_output.txt"), "w").write(out + "\n")


if __name__ == "__main__" and sys.argv[1] == "evaluate":
    evaluate()
