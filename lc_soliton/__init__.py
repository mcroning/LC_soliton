# lc_soliton/__init__.py
__version__ = "0.1.2"

from .lc_core import (
    LCVarDirichletState,
    init_lc_state,
    build_theta_bias_IC,
    LC_dn_from_theta,
    make_kspace,
    advance_theta_timestep,
    theta_newton_step,
    lc_warn_bias_and_stiffness,
)

from ._internals import (
    ufft, uifft, dst1o, idst1o, make_dst1_ortho_wrappers
)

__all__ = [
    "LCVarDirichletState", "init_lc_state", "build_theta_bias_IC",
    "LC_dn_from_theta", "make_kspace",
    "advance_theta_timestep", "theta_newton_step",
    "lc_warn_bias_and_stiffness",
    "ufft", "uifft", "dst1o", "idst1o", "make_dst1_ortho_wrappers",
]
