"""
lc_soliton: GPU-accelerated LC director dynamics

Typical notebook usage:
    from lc_soliton import *

If you prefer explicit imports:
    from lc_soliton import (
        LCVarDirichletState, init_lc_state, build_theta_bias_IC,
        LC_dn_from_theta, make_kspace, intens, advance_theta_timestep,
        theta_newton_step, lc_warn_bias_and_stiffness,
        # internals:
        make_dst1_ortho_wrappers, dst1o, idst1o,
    )
"""

__version__ = "0.1.0"

# ---- Core imports ----
try:
    from .lc_core import (
        LCVarDirichletState,
        init_lc_state,
        build_theta_bias_IC,
        LC_dn_from_theta,
        make_kspace,
        intens,
        advance_theta_timestep,
        theta_newton_step,
        lc_warn_bias_and_stiffness,   # 👈 add this line
    )
except Exception:
    pass

# ---- Internal helpers you want accessible ----
try:
    from ._internals import (
        make_dst1_ortho_wrappers,
        dst1o,
        idst1o,
    )
except Exception:
    pass

# ---- Public symbols ----
__all__ = [
    "LCVarDirichletState",
    "init_lc_state",
    "build_theta_bias_IC",
    "LC_dn_from_theta",
    "make_kspace",
    "intens",
    "advance_theta_timestep",
    "theta_newton_step",
    "lc_warn_bias_and_stiffness",   # 👈 include here too
    "make_dst1_ortho_wrappers",
    "dst1o",
    "idst1o",
]
