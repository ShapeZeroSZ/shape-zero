#!/usr/bin/env python3
"""
gate3_purity_modes.py -- model.py gate 3's chirality purity under the two launches
and the two readouts, at kappa = 0.5 and kappa* (gate 3's setup: n = 2, q = 1,
N = 200, width 8, T = 60).

  carrier readout  (Lattice.readout):       chi/bar split with the carrier omega
  per-mode readout (Lattice.readout_modes): each Fourier mode split with its own
                                            branch frequencies -- constant in free
                                            evolution
Purity is 1 - |bar| / |chi| (amplitude ratio), as in gate 3.

usage:  python3 gate3_purity_modes.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "04_scripts", "session"))
import model as M


def main():
    print("gate 3 setup: n = 2, q = 1, N = 200, width 8; purity 1 - |bar|/|chi|")
    print("   kappa      launch     t     carrier readout   per-mode readout")
    for kap in (0.5, float(M.KAPPA)):
        for pm in (False, True):
            lat = M.Lattice(n=2, kappa=kap)
            u, v = lat.packet(per_mode=pm)
            for T in (0.0, 60.0):
                uu, vv = (u, v) if T == 0 else lat.run(u, v, T)[:2]
                print(f"   {kap:.6f}   {'per-mode' if pm else 'carrier':8s}  {T:4.0f}     "
                      f"{lat.readout(uu, vv)[1]:.5f}           {lat.readout_modes(uu, vv)[1]:.5f}")


if __name__ == "__main__":
    main()
