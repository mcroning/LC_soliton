"""
lc_soliton: GPU-accelerated LC director dynamics

One-line notebook usage:
    from lc_soliton import *

If you prefer explicit imports:
    from lc_soliton import (
        LCVarDirichletState, init_lc_state, build_theta_bias_IC,
        LC_dn_from_theta, make_kspace, intens,
        advance_theta_timestep, theta_newton_step,
        # internals surfaced for notebooks:
        make_dst1_ortho_wrappers, dst1o, idst1o,
    )
"""

__version__ = "0.1.0"

# ---- Core, stable public surface ----
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
    )
except Exception as _e:  # keep import robust in partial installs
    # You can log or print a gentle message if you want
    pass

# ---- Select internals promoted for notebook convenience ----
# We keep the module private (_internals) but re-export specific helpers.
try:
    from ._internals import (
        make_dst1_ortho_wrappers,
        dst1o,
        idst1o,
    )
except Exception:
    pass

# ---- Build __all__ for "from lc_soliton import *" ----
__all__ = [
    # core
    "LCVarDirichletState",
    "init_lc_state",
    "build_theta_bias_IC",
    "LC_dn_from_theta",
    "make_kspace",
    "intens",
    "advance_theta_timestep",
    "theta_newton_step",
    # internals surfaced for notebooks
    "make_dst1_ortho_wrappers",
    "dst1o",
    "idst1o",
]

