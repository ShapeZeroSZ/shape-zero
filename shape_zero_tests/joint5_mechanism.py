#!/usr/bin/env python3
"""joint5_mechanism.py -- what carries C_q(L)? For each box: the kernel's largest terms (the
coupled mode p, its denominator d_p(w_k) against the probe's scale, and its share of C), the
fraction of C carried by the 10 largest terms, and the kernel's IR shape at q = 3: the summand
1/(d_{k+p} d'_k) along the longitudinal axis p_x and along a transverse axis p_perp (IR-flat would
mean the +p/-p pair sum tends to a constant in every direction). usage: python3 joint5_mechanism.py"""
import numpy as np
import joint5_kernel as K

C_, B_, S0 = 1.0, 0.05, np.sqrt(5.0)


def top_terms(q, L):
    r = K.kernel(q, L, L // 4, 1 / 8, detail=True)
    tot = r["C_pred"]
    lines = []
    for sign in (1, -1):
        for val, d, idx in r["top"][sign][:3]:
            p = tuple(int(i if i <= L // 2 else i - L) for i in idx)
            lines.append(f"{'+k' if sign > 0 else '-k'} p={p} d_p={d:+.2e} term={sign * val:+.4f}")
    return tot, lines


def ir_shape():
    k0 = np.pi / 2
    w = B_ * np.sin(k0) + np.sqrt(B_ ** 2 * np.sin(k0) ** 2 + S0 + 2 * (1 - np.cos(k0)))
    dpr = 2 * B_ * np.sin(k0) - 2 * w
    d = lambda px, pt: S0 + 2 * (1 - np.cos(k0 + px)) + 2 * (1 - np.cos(pt)) + 2 * B_ * w * np.sin(k0 + px) - w * w
    print("  q = 3 kernel near the probe, pair sum [1/d(+p) + 1/d(-p)]/d'_k (IR-flat -> the same constant in every direction):")
    for h in (0.2, 0.1, 0.05, 0.025):
        lon = (1 / d(h, 0) + 1 / d(-h, 0)) / dpr
        tra = (1 / d(0, h) + 1 / d(0, -h)) / dpr
        print(f"    |p| = {h:5.3f}: longitudinal {lon:+.3e}   transverse {tra:+.3e}")


if __name__ == "__main__":
    for q in (2, 3):
        for L in (8, 12, 16, 20, 24, 28, 32):
            tot, lines = top_terms(q, L)
            print(f"q={q} L={L}: C_pred {tot:+.5f}; largest terms: " + " | ".join(lines))
    ir_shape()
