#!/usr/bin/env python3
"""
octonionic_term_covariant.py — the covariant version

octonionic_term_variation.py showed B.box phi is not a total derivative, but
tested the FLAT, UNCONSTRAINED version: phi : T^2 -> R^7 with a coordinate
Laplacian and a CONSTANT c. That is only valid at a single point of the target.

WHY. The target is the orbit of octonion structures. A point of it IS a
structure-constant array, and the canonical identification of its tangent space
with Im(O) is built from the structure constants AT THAT POINT (coset_audit.py:
v |-> L_v, (L_v)_ab = c_vab). So the tensor contracting the field and the field
itself are the same object. Treating c as constant is a one-point approximation.

THE FIX. Work with the Maurer-Cartan form instead of coordinates. Lift the map
to the group: g : T^2 -> Spin(7), A = g^{-1} dg, split A = A_h + A_m along
spin(7) = g2 + m. Then

    v_mu = (A_m,mu) . 1          the m-part read as an element of Im(O)
    D_mu v_nu = d_mu v_nu + A_h,mu . v_nu        (covariant, g2 acts on Im O)
    tau = D_1 v_1 + D_2 v_2                       the tension field
    L = c_{abc} eps^{mu nu} v_mu^a v_nu^b tau^c = 2 c(v_1, v_2, tau)

Here c IS constant, legitimately -- everything has been pulled back to the
identity coset by the group element, which is exactly what the Maurer-Cartan
form is for. The position dependence now lives in A_h, and appears through the
covariant derivative rather than through a varying c.

g2 and m are built in the 8-dimensional spinor representation, matching
z1_hinge.py: g2 = {X in spin(7) : X.1 = 0}, and the canonical map m -> Im(O)
is X |-> X.1, whose image is the tangent space to the orbit of 1, i.e. S^7.

PREDICTIONS STATED BEFORE RUNNING
 P1 INSTRUMENT: dim spin(7) = 21, dim g2 = 14, dim m = 7.
 P2 INSTRUMENT: A_mu = g^{-1} d_mu g lies in spin(7) -- residual off the
    21-dimensional span <= 1e-10.
 P3 INSTRUMENT: Maurer-Cartan holds, d_1 A_2 - d_2 A_1 + [A_1, A_2] = 0, to
    spectral accuracy. This is the check that the lift is a genuine map into
    the group rather than an arbitrary field.
 P4 INSTRUMENT: v_mu is purely imaginary -- the real component of A_m . 1
    vanishes to 1e-12, confirming the canonical map lands where it should.
 P5 THE RESULT: the covariant integral over the torus is NONZERO, well above
    the residuals above. Predicted on the grounds that the flat computation
    already gave a nonzero answer and the covariant corrections are additive
    rather than cancelling -- but this is the prediction that could fail, and
    a failure would mean the flat result was an artefact of holding c fixed.
 P6 the variation is nonzero, so the term enters the equations of motion of
    the constrained model and not only of the unconstrained one.
 P7 ADDED AFTER THE FIRST RUN, AND MISSED AS FIRST STATED: predicted S ~
    AMP^3 (pure cubic). Measured spread 16.2%, systematic not noisy. The
    prediction was naive: tau = d_mu v_mu + A_h,mu . v_mu has a derivative
    piece at O(amp) and a CONNECTION piece at O(amp^2), so the correct model
    is S = A*amp^3 + B*amp^4. Refitted: max relative deviation 2.1e-03, with
    B/A = -3.66. The quartic coefficient IS the covariant correction, so the
    miss is confirmation that the connection term contributes rather than a
    numerical fault. Test below now checks the cubic+quartic model.

RECORDED MISS. The first run used AMP = 0.8 and P2/P3 both FAILED -- residual
off spin(7) 4.6e-03, Maurer-Cartan 0.5% relative. Diagnosis: exp of a
band-limited field is NOT band-limited, so at large amplitude the group-valued
field carries power past Nyquist and the spectral derivatives alias. Confirmed
by an amplitude sweep in which both residuals fall about three orders per
factor-three reduction, the signature of spectral truncation. Separately
verified that the 21-dimensional span is bracket-closed to 2.8e-17 and that
expm is orthogonal to 2.2e-15, so neither the algebra nor the exponential was
at fault. AMP is now 0.03; the sweep is retained as P7.

Python 3 + NumPy only.
"""

import numpy as np
from itertools import permutations

N = 48
LBOX = 2.0 * np.pi
KMAX = 4
AMP = 0.03
TOL = 1e-9

KX = np.fft.fftfreq(N, d=LBOX / N) * 2.0 * np.pi
KXG, KYG = np.meshgrid(KX, KX, indexing='ij')
CELL = (LBOX / N) ** 2


def d1(f):
    return np.real(np.fft.ifft2(1j * KXG[..., None, None] * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))


def d2(f):
    return np.real(np.fft.ifft2(1j * KYG[..., None, None] * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))


def d1v(f):
    return np.real(np.fft.ifft2(1j * KXG[..., None] * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))


def d2v(f):
    return np.real(np.fft.ifft2(1j * KYG[..., None] * np.fft.fft2(f, axes=(0, 1)), axes=(0, 1)))


def oriented_lines():
    LINES = [tuple(sorted(((i + s - 1) % 7) + 1 for s in (0, 1, 3)))
             for i in range(1, 8)]
    used = {p: set() for p in range(1, 8)}
    assign = {}

    def bt(li):
        if li == len(LINES):
            return True
        Ln = LINES[li]
        for perm in permutations(range(3)):
            if all(perm[j] not in used[Ln[j]] for j in range(3)):
                for j in range(3):
                    used[Ln[j]].add(perm[j])
                assign[Ln] = perm
                if bt(li + 1):
                    return True
                for j in range(3):
                    used[Ln[j]].discard(perm[j])
                del assign[Ln]
        return False

    bt(0)
    return [tuple(Ln[assign[Ln].index(r)] for r in (0, 1, 2)) for Ln in LINES]


def octonion_table(oriented):
    E = np.zeros((8, 8, 8))
    E[0, :, :] = np.eye(8)
    E[:, 0, :] = np.eye(8)
    for i in range(1, 8):
        E[i, i, 0] = -1
    for (a, b, c) in oriented:
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            E[x, y, z] = 1
            E[y, x, z] = -1
    return E


def imaginary_c(oriented):
    c = np.zeros((7, 7, 7))
    for (a, b, d) in oriented:
        A, B, D = a - 1, b - 1, d - 1
        for (x, y, z), sg in [((A, B, D), 1), ((B, D, A), 1), ((D, A, B), 1),
                              ((B, A, D), -1), ((A, D, B), -1), ((D, B, A), -1)]:
            c[x, y, z] = sg
    return c


def expm_stack(A, terms=40):
    """Matrix exponential on a stack of (...,8,8), scaling and squaring."""
    nrm = np.max(np.linalg.norm(A, axis=(-2, -1)))
    n = max(0, int(np.ceil(np.log2(max(nrm, 1e-12)))) + 2)
    B = A / (2.0 ** n)
    R = np.broadcast_to(np.eye(8), A.shape).copy()
    T = np.broadcast_to(np.eye(8), A.shape).copy()
    for k in range(1, terms):
        T = T @ B / k
        R = R + T
    for _ in range(n):
        R = R @ R
    return R


def main():
    oriented = oriented_lines()
    E = octonion_table(oriented)
    c = imaginary_c(oriented)
    I8 = np.eye(8)
    print("=" * 70)
    print("OCTONIONIC TERM :: COVARIANT VERSION")
    print("=" * 70)

    # ---- P1 the algebra and its splitting ---------------------------
    Ls = [np.column_stack([np.einsum('ijk,i,j->k', E, I8[i], I8[j])
                           for j in range(8)]) for i in range(1, 8)]
    flat, basis = [], []
    for i in range(7):
        for j in range(i + 1, 7):
            v = (Ls[i] @ Ls[j] - Ls[j] @ Ls[i]).ravel()
            for b0 in flat:
                v = v - (v @ b0) * b0
            if np.linalg.norm(v) > 1e-8:
                flat.append(v / np.linalg.norm(v))
                basis.append((v / np.linalg.norm(v)).reshape(8, 8))
    Q21 = np.array(flat).T
    act = np.array([B @ I8[0] for B in basis])          # 21 x 8
    U, s, Vt = np.linalg.svd(act.T)
    rk = int(np.sum(s > 1e-8))
    ker = Vt[rk:]                                        # g2 coefficients
    img = Vt[:rk]                                        # m coefficients
    g2 = [sum(k[i] * basis[i] for i in range(21)) for k in ker]
    mm = [sum(k[i] * basis[i] for i in range(21)) for k in img]
    print(f"\nP1  dim spin(7) = {len(basis)}, dim g2 = {len(g2)}, "
          f"dim m = {len(mm)}"
          f"   [21/14/7 -> {'PASS' if (len(basis), len(g2), len(mm)) == (21, 14, 7) else 'MISS'}]")
    Qg = np.linalg.qr(np.array([X.ravel() for X in g2]).T)[0]
    Qm = np.linalg.qr(np.array([X.ravel() for X in mm]).T)[0]

    # ---- build a random map into the group --------------------------
    rng = np.random.default_rng(17)
    mask = (np.abs(KXG) <= KMAX) & (np.abs(KYG) <= KMAX)
    Y = np.zeros((N, N, 8, 8))
    for B in basis:
        sp = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * mask
        f = np.real(np.fft.ifft2(sp))
        Y += (AMP * f / np.std(f))[..., None, None] * B
    g = expm_stack(Y)
    gT = np.swapaxes(g, -1, -2)
    A1 = gT @ d1(g)
    A2 = gT @ d2(g)

    # ---- P2 A lies in spin(7) ---------------------------------------
    def off_span(A):
        v = A.reshape(N * N, 64)
        return np.max(np.abs(v - (v @ Q21) @ Q21.T))
    o1, o2 = off_span(A1), off_span(A2)
    print(f"\nP2  residual of A off spin(7) : {max(o1, o2):.3e}"
          f"   [predicted ~0 -> {'PASS' if max(o1, o2) < 1e-10 else 'MISS'}]")

    # ---- P3 Maurer-Cartan -------------------------------------------
    mc = d1(A2) - d2(A1) + (A1 @ A2 - A2 @ A1)
    scale = np.max(np.abs(A1)) ** 2
    print(f"\nP3  Maurer-Cartan residual : {np.max(np.abs(mc)):.3e}"
          f"   (scale {scale:.3e})"
          f"   [{'PASS' if np.max(np.abs(mc)) < 1e-8 * max(scale, 1) else 'MISS'}]")

    # ---- split and read off v ---------------------------------------
    def split(A):
        v = A.reshape(N * N, 64)
        Ah = ((v @ Qg) @ Qg.T).reshape(N, N, 8, 8)
        Am = ((v @ Qm) @ Qm.T).reshape(N, N, 8, 8)
        return Ah, Am

    Ah1, Am1 = split(A1)
    Ah2, Am2 = split(A2)
    w1 = np.einsum('xyij,j->xyi', Am1, I8[0])
    w2 = np.einsum('xyij,j->xyi', Am2, I8[0])
    realpart = max(np.max(np.abs(w1[..., 0])), np.max(np.abs(w2[..., 0])))
    print(f"\nP4  real component of A_m . 1 : {realpart:.3e}"
          f"   [predicted ~0 -> {'PASS' if realpart < 1e-12 else 'MISS'}]")
    v1, v2 = w1[..., 1:], w2[..., 1:]

    # ---- covariant derivative and tension field ---------------------
    def covD(Ah, vfield, mu):
        base = d1v(vfield) if mu == 1 else d2v(vfield)
        full = np.zeros(vfield.shape[:-1] + (8,))
        full[..., 1:] = vfield
        act_ = np.einsum('xyij,xyj->xyi', Ah, full)
        return base + act_[..., 1:]

    tau = covD(Ah1, v1, 1) + covD(Ah2, v2, 2)
    L = 2.0 * np.einsum('abc,xya,xyb,xyc->xy', c, v1, v2, tau)
    S = np.sum(L) * CELL
    print(f"\nP5  covariant action S = {S:+.8e}")
    print(f"    |S| vs Maurer-Cartan residual scale : "
          f"{abs(S)/max(np.max(np.abs(mc)),1e-30):.2e}"
          f"   [predicted nonzero -> {'PASS' if abs(S) > 1e-6 else 'MISS'}]")

    # ---- P6 variation -----------------------------------------------
    def action_of(Yf):
        gg = expm_stack(Yf)
        ggT = np.swapaxes(gg, -1, -2)
        a1, a2 = ggT @ d1(gg), ggT @ d2(gg)
        h1, m1 = split(a1)
        h2, m2 = split(a2)
        u1 = np.einsum('xyij,j->xyi', m1, I8[0])[..., 1:]
        u2 = np.einsum('xyij,j->xyi', m2, I8[0])[..., 1:]
        t = covD(h1, u1, 1) + covD(h2, u2, 2)
        return np.sum(2.0 * np.einsum('abc,xya,xyb,xyc->xy', c, u1, u2, t)) * CELL

    print("\nP6  DIRECTIONAL VARIATION")
    print("-" * 70)
    okP6 = True
    for t_ in range(3):
        Xi = np.zeros((N, N, 8, 8))
        for B in basis:
            sp = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * mask
            f = np.real(np.fft.ifft2(sp))
            Xi += (f / np.std(f))[..., None, None] * B
        h = 1e-5
        dS = (action_of(Y + h * Xi) - action_of(Y - h * Xi)) / (2 * h)
        okP6 &= abs(dS) > 1e-6
        print(f"    direction {t_}:  dS/dt = {dS:+.8e}")
    print(f"    {'PASS' if okP6 else 'MISS'}")

    print("\n P7  AMPLITUDE SCALING  (term is cubic -> S ~ AMP^3)")
    print("-" * 70)
    print("      amp        S            S / amp^3")
    ratios = []
    for amp in (0.01, 0.02, 0.03, 0.05):
        r2 = np.random.default_rng(17)
        Ya = np.zeros((N, N, 8, 8))
        for B in basis:
            sp = (r2.normal(size=(N, N)) + 1j * r2.normal(size=(N, N))) * mask
            f = np.real(np.fft.ifft2(sp))
            Ya += (amp * f / np.std(f))[..., None, None] * B
        Sa = action_of(Ya)
        ratios.append(Sa / amp ** 3)
        print(f"    {amp:6.3f}   {Sa:+.6e}   {Sa/amp**3:+.6e}")
    amps = np.array([0.01, 0.02, 0.03, 0.05])
    rr = np.array(ratios)
    Afit = np.vstack([np.ones_like(amps), amps]).T
    coef = np.linalg.lstsq(Afit, rr, rcond=None)[0]
    dev = np.max(np.abs(rr - Afit @ coef) / np.abs(rr))
    spread = (max(ratios) - min(ratios)) / abs(np.mean(ratios))
    print(f"    pure-cubic spread      : {100*spread:.2f}%   (predicted <10% -> MISS)")
    print(f"    cubic+quartic fit      : A = {coef[0]:+.2f}, B = {coef[1]:+.2f}")
    print(f"    max rel dev from fit   : {dev:.2e}"
          f"   [{'PASS' if dev < 1e-2 else 'MISS'}]")
    print(f"    B/A = {coef[1]/coef[0]:+.3f}  -- the connection piece of tau")

    print("\n" + "=" * 70)
    print("READING")
    print("=" * 70)
    if abs(S) > 1e-6 and okP6:
        print("  The flat result SURVIVES the covariant treatment. The term is")
        print("  not a total derivative on the constrained target either, and")
        print("  it enters the equations of motion of the actual sigma model.")
    else:
        print("  The covariant integral does not reproduce the flat result --")
        print("  the flat answer was an artefact of holding c fixed.")


if __name__ == "__main__":
    main()
