# lc_soliton/_internals.py
from __future__ import annotations
import cupy as cp

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
