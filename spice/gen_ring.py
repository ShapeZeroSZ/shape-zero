#!/usr/bin/env python3
"""gen_ring.py -- writes the ngspice netlist of the 32-node LC-gyrator ring (design: CIRCUIT_PREREGISTRATION.md,
amended GIC values) with TI's OPAx197 macro-model (download separately: https://www.ti.com/lit/zip/SBOMA34, file
OPAx197.LIB; not redistributed here). Design choices are ours.

Cells:
  NODE a        C = 10 nF to ground; L_g,phys = 1.176 mH with series R (Q = 100 at w(pi/2)); Antoniou GIC
                (R1 = R2 = R3 = 2 kOhm, C4 = 1 nF, R5 programmable; L = C4 R1 R3 R5 / R2 = 2e-6 R5) in parallel.
                Op-amp A: + node, - g3, out g2; op-amp B: + g3, - g5, out g4. [CORRECTED 2026-09-28: op-amp B's inputs
                were + g5, - g3, which gives the right AC impedance but an unstable node -- found in the SPICE
                transient (spice/out_gic_v1 keeps the results of that version).]
  BOND a b      L_c = 4 mH with series R (Q = 100) and C_p across; two VCCS, each an OPAx197 follower + basic Howland
                (R1 = R2 = Rset, R3 = R4 = 10 kOhm): B injects +V_a/Rset into b, A injects -V_b/Rset into a.
usage (library): netlist(R5=[...32], Q=100, Cp=10e-12, analysis="ac"|"tran", gic=True|False, vccs="opamp"|"ideal",
                 nodes=32) -> str
STATUS (2026-09-28): gic=True is the pre-registered GIC with amendment 2 -- it latches at DC (results/). gic=False
puts a plain inductor in its place: the proposed passive trim element (CIRCUIT_BUILD.md sec 6 item 1).
"""
import numpy as np

N = 32
C = 10e-9
LGP = 1.176e-3
LC = 4e-3
RSET = 2983.28
W2 = np.sqrt(1e11 + 2 * 0.25e11)          # w(pi/2), gyrators off: sets the series R for a given Q
LGIC0 = 1 / (1 / 1e-3 - 1 / LGP)          # 6.667 mH so that L_g,phys || L_GIC = 1.000 mH
GIC_K = 1e-9 * 2e3 * 2e3 / 2e3            # L = GIC_K * R5


def r5_for(L):
    return L / GIC_K


def cells(Q=100.0, Cp=10e-12, gic=True, vccs="opamp"):
    rg = W2 * LGP / Q
    rc = W2 * LC / Q
    gic_block = ("RG1 a g2 2k\nRG2 g2 g3 2k\nRG3 g3 g4 2k\nCG4 g4 g5 1n\nRG5 g5 0 {R5}\n"
                 "XGA a g3 vcc vee g2 OPAx197\nXGB g3 g5 vcc vee g4 OPAx197") if gic else \
        "* diagnostic variant: ideal inductor in place of the GIC\nLGIC a 0 {R5*2e-6}"
    vccs_block = ("* VCCS B: +V(a)/Rset into b\nXB1 a bufa vcc vee bufa OPAx197\nRB1 bufa b {Rset}\nRB2 hb b {Rset}\n"
                  "RB3 mb 0 10k\nRB4 mb hb 10k\nXB2 b mb vcc vee hb OPAx197\n"
                  "* VCCS A: -V(b)/Rset into a\nXA1 b bufb vcc vee bufb OPAx197\nRA1 0 a {Rset}\nRA2 ha a {Rset}\n"
                  "RA3 bufb ma 10k\nRA4 ma ha 10k\nXA2 a ma vcc vee ha OPAx197") if vccs == "opamp" else \
        "* diagnostic variant: ideal VCCS\nGB 0 b a 0 {1/Rset}\nGA a 0 b 0 {1/Rset}"
    return f"""* ---- cells (design choices ours) ----
.subckt NODE a vcc vee params: R5=3333.3
C1 a 0 10n
LG a xg {LGP:.6g}
RG xg 0 {rg:.6g}
{gic_block}
.ends NODE
.subckt BOND a b vcc vee params: Rset={RSET}
LC a yc {LC:.6g}
RC yc b {rc:.6g}
CP a b {Cp:.6g}
{vccs_block}
.ends BOND
"""


def netlist(R5=None, Q=100.0, Cp=10e-12, gyro=True, analysis="ac", f=(48e3, 75e3, 25.0), tran=(2e-3, 2e-7),
            out="ac_out.txt", Rset=None, nodes=None, gic=True, vccs="opamp"):
    N = nodes or globals()["N"]
    R5 = np.full(N, r5_for(LGIC0)) if R5 is None else np.asarray(R5)
    Rs = np.full(N, RSET) if Rset is None else np.asarray(Rset)
    s = ["* 32-node LC-gyrator ring (shape_zero realisation-gyroscopic branch)", ".include OPAx197.LIB",
         "VCC vcc 0 12", "VEE vee 0 -12", cells(Q, Cp, gic, vccs)]
    for n in range(N):
        s.append(f"XN{n} n{n} vcc vee NODE params: R5={R5[n]:.6g}")
    for n in range(N):
        if gyro:
            s.append(f"XB{n} n{n} n{(n + 1) % N} vcc vee BOND params: Rset={Rs[n]:.6g}")
        else:  # gyrators unpowered: bond inductor and C_p only
            rc = W2 * LC / Q
            s += [f"LC{n} n{n} yc{n} {LC:.6g}", f"RC{n} yc{n} n{(n + 1) % N} {rc:.6g}", f"CP{n} n{n} n{(n + 1) % N} {Cp:.6g}"]
    vs = " ".join(f"v(n{n})" for n in range(N))
    if analysis == "ac":
        f0, f1, df = f
        npts = int(round((f1 - f0) / df)) + 1
        s += ["IDRV 0 n0 dc 0 ac 1", ".control", "set wr_singlescale", "set wr_vecnames",
              f"ac lin {npts} {f0:.6g} {f1:.6g}", f"wrdata {out} {vs}", ".endc", ".end"]
    else:
        T, h = tran
        s += ["IDRV 0 n0 dc 0 pwl(0 0 5u 0 10u 10u 15u 0)", ".options reltol=1e-3 rshunt=1e9", ".control",
              "set wr_singlescale", "set wr_vecnames",
              f"tran {h:.6g} {T:.6g} 0 {h:.6g}", f"wrdata {out} {vs}", ".endc", ".end"]
    return "\n".join(s) + "\n"


if __name__ == "__main__":
    open("ring32_nominal_ac.cir", "w").write(netlist())
    print("wrote ring32_nominal_ac.cir")
