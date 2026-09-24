# For specialists in unbalanced optimal transport (Hellinger–Kantorovich)

**Ask (one sharp question):** is HK²(ν, ·) 2-convex along *generalized*
geodesics restricted to the interaction-connected regime (base distances
below the π/2 horizon)?

Context (spec §5.1): we withdrew our own global convexity claim after
finding the counterexample — HK² between Diracs acquires a concave kink at
the transport–reaction horizon (`hk_alignment_convexity.py`, jump matched
to 5 digits). The surviving local theorem: total mass is exactly quadratic
along HK geodesics (F″ = 2L²), validated dynamically with a sharp rate
(`s1_gradient_flow.py`). New since v5.3: in the inertial regime the mass
floor is a barrier, not a wall — transient dt-converged breaches near
horizon defects (`s4_session2.py`) — and at the D2 rung the void term is
derived as the centrifugal term, g = L²/2 (`z1_d2_rung.py`).
