#!/usr/bin/env python3
"""
gate7_readout.py — does model.py's gate-7 ordering test read out too early?

Mirrors model.ordering_test (q = 1 branch) with the lattice size N and the
readout time T as parameters, and measures the packet weight left in each
segment window [start - 10, start + len(RAMP) + 10) at readout.

  old    : model.py as shipped   — N = 200,  T = 180 (calls model.ordering_test itself)
  old1200: same fixed T = 180 on N = 1200 (separates geometry from timing)
  clear  : N = 1200, T = first time every window holds < 1e-6 of the weight
           (in BOTH orderings; the later of the two, +10%), same T for AB and BA
"""
import os, sys, json
import numpy as np

# model.py pinned to the archive version this script was run with (c49da46f);
# MODEL_DIR overrides it.
sys.path.insert(0, os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_versions", "c49da46f")))
import model as M

THR = 1e-6
SEG = (60, 80)
N0, WIDTH = 20, 8.0


def weights(lat, u, v):
    return (u * u + (v * v) / lat.omega ** 2).sum(axis=1)


def window_fracs(lat, u, v):
    w = weights(lat, u, v)
    idx = np.arange(lat.N)
    out = []
    for s in SEG:
        m = (idx >= s - 10) & (idx < s + len(M.RAMP) + 10)
        out.append(float(w[m].sum() / w.sum()))
    return out


def specs(gA, gB, axes):
    a0, a1 = axes
    return (("AB", [(SEG[0], a0, gA), (SEG[1], a1, gB)], (1, 0)),
            ("BA", [(SEG[0], a1, gB), (SEG[1], a0, gA)], (0, 1)))


def clear_time(lat, spec):
    W, Wm = M.make_links(lat, spec)
    u, v = lat.packet(width=WIDTH, n0=N0)
    t, seen = 0.0, False
    while t < 3000:
        u, v, _ = lat.run(u, v, 5.0, W, Wm); t += 5.0
        fr = window_fracs(lat, u, v)
        if max(fr) > 1e-3:
            seen = True
        if seen and max(fr) < THR:
            return t
    raise RuntimeError("never cleared")


def ordering(n, gA, gB, N, T, axes=(0, 1)):
    lat = M.Lattice(n=n, N=N)
    psi0 = np.zeros(n, dtype=complex); psi0[0] = 1.0
    if T == "clear":
        T = 1.1 * max(clear_time(lat, sp) for _, sp, _ in specs(gA, gB, axes))
    out = {}
    for name, spec, order in specs(gA, gB, axes):
        W, Wm = M.make_links(lat, spec)
        u, v = lat.packet(width=WIDTH, n0=N0)
        u, v, drift = lat.run(u, v, T, W, Wm)
        co, pur = lat.readout(u, v)
        gs = (gA, gB) if order == (1, 0) else (gB, gA)
        ax = (axes[0], axes[1]) if order == (1, 0) else (axes[1], axes[0])
        Upred = M.U_segment(lat, ax[1], gs[1]) @ M.U_segment(lat, ax[0], gs[0])
        out[name] = (co, M.coords_of_state(lat, Upred @ psi0), pur, drift,
                     window_fracs(lat, u, v))
    return out, T


def summary(n, gA, gB, N, T):
    r, Tused = ordering(n, gA, gB, N, T)
    f, Tf = ordering(n, gA, gB, N, T, axes=(0, 0))
    mAB, pAB, _, dr, wAB = r["AB"]; mBA, pBA, _, _, wBA = r["BA"]
    return dict(n=n, N=N, T=Tused, T_floor=Tf,
                split=M.angle(mAB, mBA), floor=M.angle(f["AB"][0], f["BA"][0]),
                simprod_AB=M.angle(mAB, pAB), simprod_BA=M.angle(mBA, pBA),
                pred_split=M.angle(pAB, pBA), drift=dr,
                win_AB=wAB, win_BA=wBA, win_floor=f["AB"][4] + f["BA"][4])


if __name__ == "__main__":
    for n, gA, gB in ((2, 0.12, 0.08), (3, 0.15, 0.10)):
        # reproduction of model.py's own numbers, via its own functions
        r = M.ordering_test(n, gA, gB)
        ms = M.angle(r["AB"][0], r["BA"][0]); fl = M.abelian_floor(n, gA, gB)
        print(json.dumps(dict(tag="model.py", n=n, split=ms, floor=fl,
              simprod_AB=M.angle(r["AB"][0], r["AB"][1]),
              simprod_BA=M.angle(r["BA"][0], r["BA"][1]))), flush=True)
        for tag, N, T in (("old", 200, 180.0), ("old1200", 1200, 180.0),
                          ("clear", 1200, "clear")):
            s = summary(n, gA, gB, N, T); s["tag"] = tag
            print(json.dumps(s), flush=True)
