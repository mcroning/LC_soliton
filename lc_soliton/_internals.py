# lc_soliton/_internals.py
from __future__ import annotations
import cupy as cp

# ---------- Orthonormal DST-I via unitary FFT ----------
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

    V = cp.fft.fft(v, axis=axis, norm="ortho")
    return (-2.0 * V.imag)[tuple(sa)]

def idst1o(U, axis=0):
    return dst1o(U, axis=axis)

def make_dst1_ortho_wrappers(dst1o_fn, idst1o_fn, n, Ny):
    # Hook for specialization/compilation; pass-through for now.
    return dst1o_fn, idst1o_fn, None, None

# ---------- Operator diagonals, buffers, guards ----------
def _ensure_state_init(state, Nx, Ny, d_use):
    need = (not state.ready) or (state.Nx != Nx) or (state.Ny != Ny) or (state.d_use != d_use)
    if not need: return

    n = Nx - 2
    mx = cp.arange(1, n+1, dtype=cp.float32)[:, None]
    dx = d_use / (n + 1)
    lamx = (4.0/(dx*dx)) * cp.sin(0.5*cp.pi*mx/(n+1))**2  # (n,1)

    # periodic y-frequencies (unitary FFT-compatible); scale sits in d_use
    fy  = cp.fft.fftfreq(Ny, d=(1.0*1.0/Ny))
    ky2 = (2.0*cp.pi*fy[None, :])**2                       # (1,Ny)

    D = (d_use**2/4.0) * (lamx + ky2)                      # (n,Ny)

    shape = (n, Ny)
    state.buf_U    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_p    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_Ap   = cp.zeros(shape, dtype=cp.complex64)
    state.buf_r    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_z    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_real = cp.zeros(shape, dtype=cp.float32)

    state.lamx = lamx; state.ky2 = ky2; state.D = D.astype(cp.float32, copy=False)
    state.U_prev = None
    state.Nx, state.Ny, state.d_use = Nx, Ny, d_use
    state.ready = True

def _ensure_state_matches(theta, state):
    Nx, Ny = theta.shape
    if (state.D is None) or (state.D.shape != (Nx-2, Ny)):
        n   = Nx - 2
        dx  = state.d_use / (n + 1)
        mx  = cp.arange(1, n+1, dtype=cp.float32)[:, None]
        lamx = (4.0/(dx*dx)) * cp.sin(0.5*cp.pi*mx/(n+1))**2
        fy  = cp.fft.fftfreq(Ny, d=(1.0*1.0/Ny))
        ky2 = (2.0*cp.pi*fy[None, :])**2
        state.lamx = lamx; state.ky2 = ky2
        state.D    = (state.d_use**2/4.0)*(lamx + ky2).astype(cp.float32)

# ---------- Spectral operator and PCG ----------
def _apply_A_var(U, V_int, state):
    Uy        = state.idst1_unit_c(U, axis=0)
    u_int_c   = cp.fft.ifft(Uy, axis=1, norm="ortho")      # (n,Ny)
    Au_spec   = state.D * U \
              + state.dst1_unit_c( cp.fft.fft(V_int * u_int_c.real, axis=1, norm="ortho"), axis=0 )
    return Au_spec

def _pcg_var(F, V_int, state, itmax=100, tol=1e-8, U0=None):
    U = cp.zeros_like(F) if U0 is None else U0.copy()
    r = F - _apply_A_var(U, V_int, state)
    z = r / (state.D + cp.maximum(V_int, 0).astype(state.D.dtype))
    p = z.copy()
    rz0 = cp.vdot(r, z)

    for _ in range(itmax):
        Ap = _apply_A_var(p, V_int, state)
        alpha = rz0 / (cp.vdot(p, Ap) + 1e-30)
        U    += alpha * p
        r    -= alpha * Ap
        if float(cp.linalg.norm(r) / (cp.linalg.norm(F) + 1e-30)) < tol:
            break
        z    = r / (state.D + cp.maximum(V_int, 0).astype(state.D.dtype))
        rz1  = cp.vdot(r, z)
        beta = rz1 / (rz0 + 1e-30)
        p    = z + beta * p
        rz0  = rz1
    return U

def _Lp_theta_real(theta_int, state):
    Th  = state.dst1_unit_c( cp.fft.fft(theta_int, axis=1, norm="ortho"), axis=0 )
    LpT = cp.fft.ifft( state.idst1_unit_c(state.D * Th, axis=0), axis=1, norm="ortho").real
    return LpT
