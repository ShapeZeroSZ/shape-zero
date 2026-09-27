#!/usr/bin/env python3
"""
joint5_kernel.py -- the per-mode second-order kernel of Joint #5, summed over each box's actual
k-grid: the derived prediction of C_q(L) (compare kappa_box's box-grid kernel).

For u'' = -K(x) u + c lap u + beta c (u'[x0-1] - u'[x0+1]), K = s (1 + S eta), a plane wave
e^{i p.x - i w t} has the dispersion function
    d_p(w) = w0^2(p) + 2 beta c w sin p0 - w^2,   w0^2(p) = s + 2c sum_a (1 - cos p_a),
and the probe k = (2 pi m / L, 0, ..) solves d_k(w_k) = 0 (+k upper root, -k lower root). The
on-site perturbation V = s S eta(x) couples k to p with <p|V|k> = s S eta_hat(p - k),
eta_hat(q) = (1/n) sum_x eta(x) e^{-i q.x}; <k|V|k> = 0 (mean-zero eta). To second order
    delta w(k) = sum_{p != k} s^2 S^2 |eta_hat(p - k)|^2 / ( d_p(w_k) d'_k(w_k) ),
    d'_k(w) = 2 beta c sin k0 - 2 w,
and C = [delta w(+k) - delta w(-k)] / S^2. The kernel per mode is the summand; the grid sum is
over the box's own p. Also reported: the IR part (|p - k| <= 2 pi/8, the sec 5b.6a split), the
five largest contributions with their denominators, and the admixture ratio
max_p s S |eta_hat(p-k)| / |d_p(w_k)| at S = 0.02 (<< 1: second order is valid).
usage: python3 joint5_kernel.py q L [m] [frac]
"""
import sys

import numpy as np

C, BETA, S0 = 1.0, 0.05, np.sqrt(5.0)


def eta_blob(q, L, sigma):
    X = np.indices((L,) * q).astype(float)
    r2 = sum((X[a] - L / 2.0) ** 2 for a in range(q))
    e = np.exp(-0.5 * r2 / sigma ** 2)
    e -= e.mean()
    return e / np.sqrt((e ** 2).mean())


def kernel(q, L, m=None, frac=1 / 8, sigma=None, S=0.02, detail=False):
    m = m if m is not None else L // 4
    sigma = sigma if sigma is not None else frac * L
    eta = eta_blob(q, L, sigma)
    eh = np.fft.fftn(eta) / eta.size                      # eta_hat(q) on the grid
    ks = np.meshgrid(*[2 * np.pi * np.fft.fftfreq(L) * L / L for _ in range(q)], indexing="ij")
    ks = [2 * np.pi * np.fft.fftfreq(L)[tuple(slice(None) if b == a else None for b in range(q))] * np.ones((L,) * q)
          for a in range(q)]
    k0 = 2 * np.pi * m / L
    w0sq_all = S0 + 2 * C * sum(1 - np.cos(k) for k in ks)
    out = {}
    for sign in (+1, -1):
        kk = sign * k0
        w = sign * BETA * C * np.sin(k0) + np.sqrt(BETA ** 2 * C ** 2 * np.sin(k0) ** 2 + S0 + 2 * C * (1 - np.cos(k0)))
        dprime = 2 * BETA * C * np.sin(kk) - 2 * w
        d = w0sq_all + 2 * BETA * C * w * np.sin(ks[0]) - w * w
        # p - k index: shift eta_hat by the probe's grid index along axis 0
        sh = np.roll(eh, sign * m, axis=0)                   # sh[p] = eta_hat(p - k)
        amp2 = np.abs(sh) ** 2
        probe = tuple([(sign * m) % L] + [0] * (q - 1))
        d[probe] = np.inf
        contrib = S0 ** 2 * amp2 / (d * dprime)
        dq = np.sqrt(sum(((kk_ - (k0 * sign if a == 0 else 0) + np.pi) % (2 * np.pi) - np.pi) ** 2
                         for a, kk_ in enumerate(ks)))
        out[sign] = dict(total=contrib.sum(), ir=contrib[dq <= 2 * np.pi / 8 + 1e-12].sum(),
                         admix=(S0 * S * np.sqrt(amp2) / np.abs(d)).max(),
                         top=sorted([(abs(contrib[i]), contrib[i], d[i], i) for i in np.ndindex(contrib.shape)], reverse=True)[:5])
    Cp = out[1]["total"] - out[-1]["total"]
    Cir = out[1]["ir"] - out[-1]["ir"]
    res = dict(q=q, L=L, m=m, sigma=sigma, C_pred=Cp, C_IR=Cir, admix=max(out[1]["admix"], out[-1]["admix"]))
    if detail:
        res["top"] = {s_: [(t[1], t[2], t[3]) for t in out[s_]["top"]] for s_ in (1, -1)}
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    r = kernel(int(a[0]), int(a[1]), int(a[2]) if len(a) > 2 else None, float(a[3]) if len(a) > 3 else 1 / 8, detail=True)
    print(r)
