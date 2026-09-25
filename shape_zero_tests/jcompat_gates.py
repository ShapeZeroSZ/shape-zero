#!/usr/bin/env python3
"""
jcompat_gates.py -- run 04_scripts/session/model.py's gates at a chosen
gyroscopic ratio kappa (the intra-node F = kappa JJ v), unchanged otherwise.

model.py binds KAPPA = 0.5 into Lattice's default arguments when it is loaded,
so patching the module afterwards would not reach them. This driver loads the
source, replaces the single line "KAPPA = 0.5" with the requested value, checks
that exactly one line changed, and executes it as __main__ from model.py's own
directory (so its relative imports and harness resolve as usual).

usage:  python3 jcompat_gates.py <kappa>
"""

import os
import runpy
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "04_scripts", "session", "model.py")


def main():
    kappa = float(sys.argv[1])
    src = open(MODEL).read()
    old = "KAPPA = 0.5\n"
    if src.count(old) != 1:
        raise SystemExit("model.py no longer has exactly one 'KAPPA = 0.5' line")
    new = src.replace(old, f"KAPPA = {kappa!r}\n")
    sdir = os.path.dirname(os.path.abspath(MODEL))
    with tempfile.NamedTemporaryFile("w", suffix="_model.py", dir=sdir, delete=False) as f:
        f.write(new)
        path = f.name
    try:
        print(f"=== model.py gates with KAPPA = {kappa} (one line changed) ===", flush=True)
        sys.path.insert(0, sdir)
        os.chdir(sdir)
        runpy.run_path(path, run_name="__main__")
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
