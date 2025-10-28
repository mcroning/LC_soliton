# lc_soliton/__init__.py
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("lc_soliton")
except PackageNotFoundError:
    __version__ = "0+local"


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

from .steady_api import solve_theta_steady_slice
from .steady_demo import demo_steady_path, quick_view_slice

__all__ = [
    "solve_theta_steady_slice",
    "demo_steady_path",
    "quick_view_slice",
    "LCVarDirichletState", "init_lc_state", "build_theta_bias_IC",
    "LC_dn_from_theta", "make_kspace",
    "advance_theta_timestep", "theta_newton_step",
    "lc_warn_bias_and_stiffness",
    "ufft", "uifft", "dst1o", "idst1o", "make_dst1_ortho_wrappers",
]
