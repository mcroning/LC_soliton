# lc_soliton/_internals.py
from __future__ import annotations
import cupy as cp
import numpy as np

# ---------- unitary FFT helpers ----------
def ufft(a, axis):   return cp.fft.fft(a, axis=axis, norm="ortho")
def uifft(A, axis):  return cp.fft.ifft(A, axis=axis, norm="ortho")

# ---------- Orthonormal DST-I / IDST-I ----------
def dst1o(u, axis=0):
    n = u.shape[axis]
    m = 2*(n+1)

    pad = list(u.shape); pad[axis] = m
    v = cp.zeros(pad, dtype=cp.complex64)

    sl  = [slice(None)]*u.ndim
    sa  = sl.copy(); sa[axis] = slice(1, n+1)
    sb  = sl.copy(); sb[axis] = slice(n+2, 2*n+2)
    s_u = sl.copy(); s_u[axis] = slice(0, n)
    s_r = sl.copy(); s_r[axis] = slice(None, None, -1)

    uC = u.astype(cp.complex64, copy=False)
    v[tuple(sa)] =  uC[tuple(s_u)]
    v[tuple(sb)] = -uC[tuple(s_r)][tuple(s_u)]

    V = ufft(v, axis=axis)
    out = (-2.0 * V.imag)[tuple(sa)]
    return out

def idst1o(U, axis=0):
    return dst1o(U, axis=axis)

# --- one-time calibration (do at init after Nx, Ny are known) ---
def make_dst1_ortho_wrappers(dst1o_f, idst1o_f, n, Ny):
    G = cp.random.randn(n, Ny).astype(cp.float32)
    H = cp.random.randn(n, Ny).astype(cp.float32)
    c1 = float(cp.linalg.norm(idst1o_f(dst1o_f(G, axis=0), axis=0)) / cp.linalg.norm(G))
    c2 = float(cp.linalg.norm(idst1o_f(dst1o_f(H, axis=0), axis=0)) / cp.linalg.norm(H))
    c = 0.5*(c1 + c2)
    alpha = 1.0 / (c ** 0.5)

    def dst1_unit(X, axis=0):  return alpha * dst1o_f(X, axis=axis)
    def idst1_unit(Y, axis=0): return alpha * idst1o_f(Y, axis=axis)
    return dst1_unit, idst1_unit, c, alpha

def intens(amp, coh=True):
    """
    Compute total intensity from one or more beam amplitudes.
    Parameters
    ----------
    amp : array or list of arrays
        Complex field amplitudes, shape (..., Nx, Ny)
    coh : bool
        If True, beams are coherent (sum fields first).
        If False, beams are incoherent (sum intensities).

    Returns
    -------
    Ixy : ndarray (real)
        Total intensity map.
    """
    xp = cp.get_array_module(amp)
    A = amp
    if isinstance(amp, (list, tuple)):
        A = xp.stack(amp, axis=0)
    elif A.ndim == 2:
        A = A[None, ...]

    if coh:
        # coherent sum: |Σ a_i|²
        Ixy = xp.abs(xp.sum(A, axis=0))**2
    else:
        # incoherent sum: Σ |a_i|²
        Ixy = xp.sum(xp.abs(A)**2, axis=0)
    return Ixy.astype(xp.float32, copy=False)

def theta_newton_step(theta_in, amp, state, *, b, bi,
                      pcg_itmax=80, pcg_tol=1e-8, linesearch=True, Ixy=None, coh=True):
    theta = theta_in
    Nx, Ny = theta.shape
    # rebuild D if geometry changed
    _ensure_state_init(state, Nx, Ny, state.d_use, state.fy)

    # intensity
    if Ixy is None:
        # Use the package-wide intensity routine to respect coherence
        # Returns float32; promote to float64 for solver math
        Ixy = intens(amp, coh).astype(cp.float64, copy=False)
    else:
        Ixy = cp.asarray(Ixy, dtype=cp.float64)

    Kxy  = (b + bi * Ixy).astype(cp.float64, copy=False)
    Kint = Kxy[1:-1, :]
    theta_int = theta[1:-1, :].astype(cp.float64, copy=False)

    LpT  = _Lp_theta_real(theta_int, state).astype(cp.float64, copy=False)
    Fint = (LpT - Kint * cp.sin(2.0 * theta_int))
    Vint = (2.0 * Kint * cp.cos(2.0 * theta_int))

    tau  = cp.maximum(-cp.min(Vint) + 1e-6, 0.0)
    Veff = Vint + tau

    F_spec = state.dst1_unit_c( ufft(Fint, axis=1), axis=0 )
    U      = _pcg_var(F_spec, Veff, state, itmax=pcg_itmax, tol=pcg_tol)
    Uy     = state.idst1_unit_c(U, axis=0)
    delta  = uifft(Uy, axis=1).real

    def res_norm(th):
        th_int = th[1:-1, :]
        return float(cp.linalg.norm(
            _Lp_theta_real(th_int, state) - Kxy[1:-1, :] * cp.sin(2.0 * th_int)
        ))

    res0, alpha, accepted = res_norm(theta), 1.0, False
    for _ in range(6) if linesearch else range(1):
        theta_trial = theta.copy()
        theta_trial[1:-1, :] = theta_int - alpha * delta
        if res_norm(theta_trial) <= res0 * (1 - 1e-3 * alpha):
            theta = theta_trial; accepted = True; break
        alpha *= 0.5
    if not accepted:
        theta[1:-1, :] = theta_int - delta
    theta[0, :], theta[-1, :] = 0.0, 0.0
    return theta