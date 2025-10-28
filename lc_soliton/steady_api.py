"""
User-facing steady-state solver primitives.
- solve_theta_steady_slice: public per-slice steady solver (inexact-Newton + line search)
- (internal) theta_newton_step is expected to live in your package internals
"""

def _get_xp(*arrays):
    import numpy as _np
    try:
        import cupy as _cp  # noqa: F401
        for a in arrays:
            if a is not None and hasattr(a, "__cuda_array_interface__"):
                return __import__("cupy")
    except Exception:
        pass
    return _np

def _norm_inf(xp, arr):
    return float(xp.max(xp.abs(arr))) if arr is not None else 0.0


def solve_theta_steady_slice(
    theta_init,
    amp,
    state,
    *,
    b,
    bi,
    nl_tol=1e-6,
    step_tol=1e-7,
    max_newton=20,
    pcg_itmax=200,
    linesearch=True,
    residual_fn=None,
    newton_step_fn=None,
):
    """
    Drive theta to (local) steady state for THIS slice given optical field 'amp'.
    CPU/GPU agnostic: backend inferred from arrays (NumPy/CuPy).

    Parameters
    ----------
    theta_init : np.ndarray or cupy.ndarray
    amp        : np.ndarray or cupy.ndarray (complex)
    state      : object with material params and helpers; may expose lc_residual_norm(...)
    b, bi      : floats (dimensionless LC params)
    nl_tol     : nonlinear residual tolerance (relative)
    step_tol   : max-norm step tolerance (rad)
    max_newton : max Newton iterations per slice
    pcg_itmax  : max PCG iterations inside Newton step
    linesearch : enable line search in Newton step
    residual_fn: optional callable (theta, amp, b, bi) -> float rel residual
    newton_step_fn: optional callable performing a single Newton step

    Returns
    -------
    theta_ss, info(dict)
    """
    xp = _get_xp(theta_init, amp)

    # Try to use a residual on state if one isn't supplied
    if residual_fn is None:
        residual_fn = getattr(state, "lc_residual_norm", None)

    def _resid_rel(theta):
        if residual_fn is None:
            return None
        try:
            return float(residual_fn(theta, amp, b=b, bi=bi))
        except Exception:
            return None

    # Import the internal stepper if not injected
    if newton_step_fn is None:
        # Adjust this import to your actual internal path
        from .solver_internals import theta_newton_step as _theta_newton_step
        newton_step_fn = _theta_newton_step

    theta = theta_init
    r0 = _resid_rel(theta)
    r_prev = r0

    for it in range(1, max_newton + 1):
        # Inexact-Newton forcing: tie PCG tolerance to current progress
        if r_prev is None or it == 1:
            pcg_tol = 1e-3
        else:
            pcg_tol = min(5e-1, 0.9 * r_prev)  # simple, robust

        theta_next = newton_step_fn(
            theta, amp, state, b=b, bi=bi,
            pcg_itmax=pcg_itmax, pcg_tol=pcg_tol, linesearch=linesearch
        )

        step_inf = _norm_inf(xp, theta_next - theta)
        theta = theta_next

        r = _resid_rel(theta)

        # Convergence: prefer residual if available; else step-only
        resid_ok = (r is not None) and ((r0 is None and step_inf <= step_tol) or (r <= nl_tol))
        if resid_ok and step_inf <= step_tol:
            return theta, {"converged": True, "it": it, "rel_res": r, "step_inf": step_inf}
        if (r is None) and (step_inf <= step_tol):
            return theta, {"converged": True, "it": it, "rel_res": None, "step_inf": step_inf}

        r_prev = r

    return theta, {"converged": False, "it": max_newton, "rel_res": r_prev, "step_inf": step_inf}
