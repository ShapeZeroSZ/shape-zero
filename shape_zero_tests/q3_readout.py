#!/usr/bin/env python3
"""
q3_readout.py — gate-7 / gate-8 quantities at q = 3 under three readouts.

Geometry (MODEL_SPEC 4d requirements): full transverse slab, segments 20 apart
(>= len(RAMP) = 12, no overlap), axis-aware model.py force and make_links.
Lattice 320 x 12 x 12, long along the propagation axis so nothing wraps back into
a segment window before clearing. Packet: isotropic 3D Gaussian, width 3, centred
at x0 = 30 on the transverse centre, k0 = pi/2 along axis 0, colour 0.

One evolution per configuration, snapshotted at
  fixed   : T = 180 (ordering_test's default fixed time)
  centroid: centroid cumulative displacement past the LAST segment end + 2
            (run_until_exit's certificate, MODEL_SPEC 4d)
  clear   : every segment window [s-10, s+12+10) holds < 1e-6 of the weight
usage: q3_readout.py <n> <job>   job in AB BA fAB fBA single
"""
import os, sys, json
import numpy as np
# model.py pinned to the archive version this script was run with (948b09e8);
# MODEL_DIR overrides it.
sys.path.insert(0, os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_versions", "948b09e8")))
import model as M

L0, S, WIDTH, X0 = 320, 12, 3.0, 30
SEGS = (50, 70)
THR, CHECK, TMAX = 1e-6, 1.0, 2500.0
G = {2: (0.12, 0.08), 3: (0.15, 0.15)}          # MODEL_SPEC 4d gate-7 strengths
GFLOOR = {2: (0.12, 0.08), 3: (0.15, 0.10)}     # Abelian floor needs UNEQUAL strengths


class L3(M.Lattice):
    def __init__(self, n):
        super().__init__(n=n, N=S ** 3, q=3, shape=S)
        self.shape = (L0, S, S); self.N = L0 * S * S

    def packet3(self, amp=1e-3):
        c = np.indices(self.shape).astype(float)
        r2 = np.zeros(self.shape)
        for a, ctr in zip(range(3), (X0, S / 2.0, S / 2.0)):
            d = c[a] - ctr
            d = (d + self.shape[a] / 2) % self.shape[a] - self.shape[a] / 2
            r2 += d ** 2
        env = np.exp(-0.5 * r2 / WIDTH ** 2).reshape(-1)
        ph = (M.K0 * (c[0] - X0)).reshape(-1)
        u = np.zeros((self.N, self.D)); v = np.zeros((self.N, self.D))
        u[:, 0], u[:, 1] = amp * env * np.cos(ph), amp * env * np.sin(ph)
        v[:, 0], v[:, 1] = self.omega * u[:, 1], -self.omega * u[:, 0]
        return u, v


def spec_for(n, job):
    gA, gB = G[n]
    fA, fB = GFLOOR[n]
    s0, s1 = SEGS
    return {"AB": [(s0, 0, gA), (s1, 1, gB)], "BA": [(s0, 1, gB), (s1, 0, gA)],
            "fAB": [(s0, 0, fA), (s1, 0, fB)], "fBA": [(s0, 0, fB), (s1, 0, fA)],
            "single": [(s0, 0, 0.12)]}[job]


def axis0_profile(lat, u, v):
    w = (u * u + (v * v) / lat.omega ** 2).sum(axis=1).reshape(lat.shape)
    return w.sum(axis=(1, 2))


def snapshot(lat, u, v, t, E0, spec):
    co, pur = lat.readout(u, v)
    prof = axis0_profile(lat, u, v)
    x = np.arange(L0)
    wins = [float(prof[(x >= s - 10) & (x < s + len(M.RAMP) + 10)].sum() / prof.sum())
            for s, _, _ in spec]
    return dict(t=t, co=co.tolist(), pur=pur, windows=wins, Qt=M.transverse_Q(lat, u, v),
                drift=abs(lat.energy(u, v) - E0) / abs(E0))


def main():
    n, job = int(sys.argv[1]), sys.argv[2]
    spec = spec_for(n, job)
    lat = L3(n)
    W, Wm = M.make_links(lat, spec)
    u, v = lat.packet3()
    E0 = lat.energy(u, v)
    last_end = max(s for s, _, _ in spec) + len(M.RAMP)
    ang = 2 * np.pi * np.arange(L0) / L0

    def centroid(uu, vv):
        p = axis0_profile(lat, uu, vv)
        return (np.angle((p * np.exp(1j * ang)).sum()) % (2 * np.pi)) * L0 / (2 * np.pi)

    start = prev = centroid(u, v); cum = 0.0; t = 0.0
    out = {"n": n, "job": job, "spec": spec}
    while t < TMAX:
        u, v, _ = lat.run(u, v, CHECK, W, Wm); t += CHECK
        cur = centroid(u, v); step = cur - prev
        step += L0 if step < -L0 / 2 else (-L0 if step > L0 / 2 else 0)
        cum += step; prev = cur
        if "fixed" not in out and t >= 180.0 - 1e-9:
            out["fixed"] = snapshot(lat, u, v, t, E0, spec)
        if "centroid" not in out and start + cum > last_end + 2.0:
            out["centroid"] = snapshot(lat, u, v, t, E0, spec)
        if "centroid" in out and "fixed" in out:
            s = snapshot(lat, u, v, t, E0, spec)
            if max(s["windows"]) < THR:
                out["clear"] = s
                break
        if int(t) % 50 == 0:
            print(json.dumps({"progress": t, "windows": snapshot(lat, u, v, t, E0, spec)["windows"],
                              "cum": cum}), file=sys.stderr, flush=True)
    out["cum_disp"] = cum
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
