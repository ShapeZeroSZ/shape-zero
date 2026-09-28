#!/usr/bin/env python3
"""selfcoupling_check.py -- SC2 (SELFCOUPLING_HYPOTHESES.md, c897a7e): the 1PN content of candidate 7's lapses
against GR's. With U = GM/(r c^2), PPN g00 = -(1 - 2U + 2 beta U^2 + ...). Where does each lapse reach zero?"""
import sympy as s
U = s.symbols("U", positive=True)
cases = {
    "linear lapse 1 - U (candidate 7)": 1 - U,
    "exponential lapse e^-U (candidate 7)": s.exp(-U),
    "GR, isotropic (1 - U/2)/(1 + U/2)": (1 - U / 2) / (1 + U / 2),
}
for name, N in cases.items():
    g00 = s.series(N ** 2, U, 0, 3).removeO()
    beta = s.simplify(s.Poly(g00, U).coeff_monomial(U ** 2) / 2)
    zero = s.solve(s.Eq(N, 0), U)
    print(f"{name:40s} N^2 = {s.expand(g00)};  beta = {beta};  N = 0 at U = {zero if zero else 'never'}")
