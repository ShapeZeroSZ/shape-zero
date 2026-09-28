"""Post-hoc design search, written after the GIC latch was found: Antoniou GIC topologies on one node
(10 nF || 1.176 mH Q100 || GIC), capacitor at position 2 or 4, all op-amp input assignments. Checks: DC op-point
without convergence aids (no op-amp output near a rail), transient kick decays, AC inductance ~6.67 mH."""
import subprocess, numpy as np, os, itertools
H = os.path.dirname(os.path.abspath(__file__))
res = []
for cpos in (2, 4):
    Z = {1: "R 2k", 2: "C 1n" if cpos == 2 else "R 2k", 3: "R 2k", 4: "C 1n" if cpos == 4 else "R 2k", 5: "R 3333.3"}
    for sa, sb in itertools.product((0, 1), (0, 1)):
        xa = "a g3" if sa == 0 else "g3 a"          # op-amp A: inputs node1(a) and node3
        xb = "g3 g5" if sb == 0 else "g5 g3"        # op-amp B: inputs node3 and node5
        chain = ["a", "g2", "g3", "g4", "g5", "0"]
        el = "\n".join(f"{Z[i][0]}{i} {chain[i-1]} {chain[i]} {Z[i].split()[1]}" for i in range(1, 6))
        base = f""".include ../OPAx197.LIB
VCC vcc 0 12
VEE vee 0 -12
C1 a 0 10n
LG a xg 1.176m
RG xg 0 4.555
{el}
XA {xa} vcc vee g2 OPAx197
XB {xb} vcc vee g4 OPAx197
"""
        tag = f"C{cpos}_A{sa}_B{sb}"
        op = base + ".control\nop\nprint v(a) v(g2) v(g4)\n.endc\n.end\n"
        open(f"{H}/{tag}_op.cir", "w").write("* op\n" + op)
        r = subprocess.run(["ngspice", "-b", f"{tag}_op.cir"], cwd=H, capture_output=True, text=True, timeout=300).stdout
        vals = {l.split("=")[0].strip(): float(l.split("=")[1].split(",")[0]) for l in r.splitlines() if l.strip().startswith("v(") and "=" in l}
        opok = bool(vals) and max(abs(vals.get("v(g2)", 99)), abs(vals.get("v(g4)", 99))) < 5
        tr = base + f"IK 0 a pwl(0 0 5u 0 10u 10u 15u 0)\n.options method=gear reltol=1e-3\n.control\nset wr_singlescale\ntran 50n 300u 0 50n\nwrdata {tag}_tr.txt v(a)\n.endc\n.end\n"
        open(f"{H}/{tag}_tr.cir", "w").write("* tr\n" + tr)
        subprocess.run(["ngspice", "-b", f"{tag}_tr.cir"], cwd=H, capture_output=True, timeout=600)
        try:
            d = np.loadtxt(f"{H}/{tag}_tr.txt"); t, v = d[:, 0], d[:, 1]
            v = v - np.median(v[t > 250e-6])
            e1 = np.abs(v[(t > 20e-6) & (t < 60e-6)]).max(); e2 = np.abs(v[t > 250e-6]).max(); trok = e2 < e1 and e1 < 0.1
        except Exception:
            e1 = e2 = float("nan"); trok = False
        ac = base + "IIN 0 a dc 0 ac 1\n.control\nac lin 3 50k 70k\nprint frequency v(a)\n.endc\n.end\n"
        acb = ac.replace("C1 a 0 10n\nLG a xg 1.176m\nRG xg 0 4.555\n", "")      # GIC alone
        open(f"{H}/{tag}_ac.cir", "w").write("* ac\n" + acb)
        r = subprocess.run(["ngspice", "-b", f"{tag}_ac.cir"], cwd=H, capture_output=True, text=True, timeout=300).stdout
        Ls = []
        for l in r.splitlines():
            p = l.split()
            if len(p) >= 4 and p[0].isdigit():
                f = float(p[1]); z = complex(float(p[2].rstrip(",")), float(p[3]))
                Ls.append((z.imag / (2 * np.pi * f), z.imag / z.real if z.real else np.inf))
        print(f"{tag}: op {'ok' if opok else 'BAD'} {vals}; transient {'decays' if trok else 'NOT'} ({e1:.2e} -> {e2:.2e}); "
              f"L(mH),Q at 50/60/70 kHz {[(round(a*1e3,3), round(b,1)) for a,b in Ls]}", flush=True)
