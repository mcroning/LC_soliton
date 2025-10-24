# lc_soliton/lc_core.py
from __future__ import annotations
import cupy as cp
from cupyx.scipy import special as spspec
from ._internals import (
    _ensure_state_init, _ensure_state_matches,
    _apply_A_var, _pcg_var, _Lp_theta_real,
    dst1o, idst1o, make_dst1_ortho_wrappers,
)

__all__ = [
    "LCVarDirichletState", "init_lc_state",
    "build_theta_bias_IC", "LC_dn_from_theta",
    "theta_newton_step", "advance_theta_timestep",
    "make_kspace", "ufft", "uifft",
    "intens", "intens_beams", "warn_if_bias_too_strong",
]

# ---------- unitary FFT helpers (periodic on selected axis) ----------
def ufft(a, axis):   return cp.fft.fft(a, axis=axis, norm="ortho")
def uifft(A, axis):  return cp.fft.ifft(A, axis=axis, norm="ortho")

def make_kspace(Nx: int, Ny: int, dx: float, dy: float, *, dtype=cp.float32):
    """
    FFT-aligned k-space: returns kx_col (Nx,1), ky (Ny,), kxy2 (Nx,Ny) in rad/length.
    """
    fx = cp.fft.fftfreq(Nx, d=dx);  fy = cp.fft.fftfreq(Ny, d=dy)
    kx = (2*cp.pi*fx).astype(dtype); ky = (2*cp.pi*fy).astype(dtype)
    kx2d, ky2d = cp.meshgrid(kx, ky, indexing="ij")
    return kx[:, None], ky, (kx2d**2 + ky2d**2).astype(dtype)

# ---------- state container for Dirichlet-in-x / periodic-in-y ----------
class LCVarDirichletState:
    """Caches DST/FFT diagonals and work buffers for θ-operator with Dirichlet in x."""
    def __init__(self):
        self.ready = False
        self.Nx = self.Ny = self.d_use = None
        self.lamx = self.ky2 = self.D = None
        # complex-safe orthonormal DST-I wrappers (installed at init)
        self.dst1_unit = self.idst1_unit = None
        self.dst1_unit_c = self.idst1_unit_c = None
        # PCG buffers
        self.buf_U = self.buf_p = self.buf_Ap = None
        self.buf_r = self.buf_z = self.buf_real = None
        # warm-start
        self.U_prev = None
        # optional preconditioner shift
        self.mu = None

def init_lc_state(state: LCVarDirichletState, *, Nx: int, Ny: int, d_use: float):
    """
    Build/refresh Dirichlet-in-x operator diagonal and complex-safe DST-I wrappers.
    """
    _ensure_state_init(state, Nx, Ny, d_use)
    if (state.dst1_unit_c is None) or (state.idst1_unit_c is None):
        n = Nx - 2
        dst1_u, idst1_u, _, _ = make_dst1_ortho_wrappers(dst1o, idst1o, n, Ny)
        state.dst1_unit = dst1_u
        state.idst1_unit = idst1_u
        state.dst1_unit_c  = lambda A, axis=0: dst1_u(A.real, axis=axis) + 1j*dst1_u(A.imag, axis=axis)
        state.idst1_unit_c = lambda B, axis=0: idst1_u(B.real, axis=axis) + 1j*idst1_u(B.imag, axis=axis)
    return state

# ---------- exact 1D bias initializer (enforces b_c = π²/8) ----------
def build_theta_bias_IC(state: LCVarDirichletState, b: float, *, eps_clip=1e-12, return_1d=False):
    Nx = int(state.Nx or 256); Ny = int(state.Ny or 256)
    b_c = cp.pi**2 / 8.0
    if b <= float(b_c):
        th1d = cp.zeros(Nx, dtype=cp.float64)
        return th1d if return_1d else cp.tile(th1d[:, None], (1, Ny))

    u = cp.linspace(-1.0, 1.0, Nx, dtype=cp.float64)
    two_b = 2.0 * float(b)

    def Ksq_minus_2b(mf: float):
        K = spspec.ellipk(mf)
        return float(cp.asnumpy(K*K - two_b))

    lo, hi = 1e-12, 1 - 1e-12
    for _ in range(80):
        mid = 0.5*(lo+hi)
        (lo if Ksq_minus_2b(lo)*Ksq_minus_2b(mid) > 0 else hi) == hi
        if Ksq_minus_2b(lo)*Ksq_minus_2b(mid) <= 0: hi = mid
        else:                                        lo = mid
        if abs(hi-lo) < 1e-14: break
    m = 0.5*(lo+hi)
    theta0 = cp.arcsin(cp.sqrt(m))

    arg = cp.sqrt(2.0*b) * u
    sn, cn, dn, _ = spspec.ellipj(arg, float(m))
    cd = cn/dn
    s = cp.clip(cp.sin(theta0) * cd, -1.0 + eps_clip, 1.0 - eps_clip)
    th1d = cp.arcsin(s)
    return th1d if return_1d else cp.tile(th1d[:, None], (1, Ny))

def LC_dn_from_theta(theta, ne, no, refin):
    th = theta.astype(cp.float64, copy=False)
    ct, st = cp.cos(th), cp.sin(th)
    n_eff  = (ne*no) / cp.sqrt((ne*ct)**2 + (no*st)**2)
    return (n_eff - refin).astype(cp.float32, copy=False)

# ---------- intensity helpers ----------
def intens(amp, coh: bool, out=None):
    """|a+b|^2 if coh else |a|^2+|b|^2 for the first two beams."""
    a, b = amp[0], amp[1]
    if out is None: out = cp.empty(a.shape, a.real.dtype)
    if coh:
        s = a + b; out[...] = s.real*s.real + s.imag*s.imag
    else:
        out[...] = (a.real*a.real + a.imag*a.imag +
                    b.real*b.real + b.imag*b.imag)
    return out

def intens_beams(amp, coh: bool):
    return cp.array((cp.abs(amp[0])**2, cp.abs(amp[1])**2))

# ---------- θ solvers ----------
def theta_newton_step(theta_in, amp, state: LCVarDirichletState, *, b, bi,
                      pcg_itmax=80, pcg_tol=1e-8, linesearch=True, Ixy=None):
    theta = theta_in
    Nx, Ny = theta.shape
    _ensure_state_matches(theta, state)

    # intensity
    if Ixy is None:
        A = amp if amp.ndim == 3 else amp[None, ...]
        Ixy = cp.real(cp.sum(A * cp.conj(A), axis=0)).astype(cp.float64)
    else:
        Ixy = cp.asarray(Ixy, dtype=cp.float64)

    Kxy  = (cp.float64(b) + cp.float64(bi) * Ixy)
    Kint = Kxy[1:-1, :]
    theta_int = theta[1:-1, :].astype(cp.float64, copy=False)

    # Lp θ on interior
    LpT  = _Lp_theta_real(theta_int, state).astype(cp.float64, copy=False)

    # residual and Jacobian-diagonal
    Fint = (LpT - Kint * cp.sin(2.0 * theta_int))
    Vint = (2.0 * Kint * cp.cos(2.0 * theta_int))

    # LM/SPD guard
    tau  = cp.maximum(-cp.min(Vint) + 1e-6, 0.0)
    Veff = Vint + tau

    # solve (L_p + Veff) δ = F
    F_spec = state.dst1_unit_c( ufft(Fint, axis=1), axis=0 )
    U      = _pcg_var(F_spec, Veff, state, itmax=pcg_itmax, tol=pcg_tol)
    Uy     = state.idst1_unit_c(U, axis=0)
    delta  = uifft(Uy, axis=1).real

    # backtracking on ||F||
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

def advance_theta_timestep(theta, amp, state: LCVarDirichletState, *,
                           dt, b, bi, Ixy=None,
                           k_correct_every=None, k_index=None,
                           newton_iters=0, newton_tol=1e-6,
                           mobility=1.0):
    """
    IMEX θ-step for ∂t θ = M(Δθ + K sin 2θ). Set mobility=M [1/s], dt [s].
    Dirichlet in x (DST-I), periodic in y (FFT).
    """
    xp = cp
    if Ixy is None:
        A = xp.asarray(amp);  A = A if A.ndim == 3 else A[None, ...]
        Ixy = xp.real(xp.sum(A * xp.conj(A), axis=0))
    Ixy = Ixy.astype(xp.float64, copy=False)

    T = xp.array(theta, dtype=xp.float64, copy=False)
    Tint = T[1:-1, :]
    Kxy  = (b + bi * Ixy).astype(xp.float64, copy=False)
    Kint = Kxy[1:-1, :]

    Fexp = (Kint * xp.sin(2.0 * Tint)).astype(xp.float64, copy=False)

    Fexp_spec = state.dst1_unit_c( ufft(Fexp, axis=1), axis=0 )
    Tn_spec   = state.dst1_unit_c( ufft(Tint, axis=1), axis=0 )

    Lam = state.D.astype(xp.float64, copy=False)  # eigenvalues of (+Δ)
    alpha = float(mobility) * float(dt)

    # IMEX: (I + αΛ) T^{n+1} = T^n + α Fexp
    Tnp1_spec = (Tn_spec + alpha * Fexp_spec) / (1.0 + alpha * Lam)

    Uy         = state.idst1_unit_c(Tnp1_spec, axis=0)
    Tint_next  = uifft(Uy, axis=1).real

    Tnext = T.copy()
    Tnext[1:-1, :] = Tint_next
    Tnext[0, :], Tnext[-1, :] = 0.0, 0.0

    if (newton_iters and k_correct_every and k_index is not None and
        ((k_index + 1) % k_correct_every == 0)):
        for _ in range(int(newton_iters)):
            Tnext = theta_newton_step(Tnext, amp, state, b=b, bi=bi,
                                      pcg_itmax=80, pcg_tol=newton_tol,
                                      linesearch=True, Ixy=Ixy)
    return Tnext

def lc_warn_bias_and_stiffness(theta_field, state, *, b, bi, Ixy=None,
                            warn_theta_max_rad=1.20,   # ~69°
                            warn_eta_soft=2e-3,        # |2K cos2θ| / median(D)
                            warn_eta_hard=1e-2,        # stronger flag
                            label="pre-Newton"):
  """
  Emit a warning when steady-state Newton is likely to be slow/unstable.
  Triggers if:
    - max(theta) exceeds warn_theta_max_rad, or
    - eta_soft = ||2K cos(2θ)||_inf / median(D) exceeds thresholds.

  Returns a dict with the metrics so you can log if desired.
  """
  import cupy as cp

  # ensure float64 CuPy array
  th = cp.asarray(theta_field)
  if th.dtype != cp.float64:
      th = th.astype(cp.float64)
  Nx, Ny = th.shape

  # conservative default: zero optical drive
  if Ixy is None:
      Ixy = cp.zeros((Nx, Ny), dtype=cp.float64)
  else:
      Ixy = cp.asarray(Ixy)
      if Ixy.dtype != cp.float64:
          Ixy = Ixy.astype(cp.float64)

  # interior quantities
  th_int  = th[1:-1, :]
  Kxy     = (b + bi * Ixy).astype(cp.float64)
  Vint    = 2.0 * Kxy[1:-1, :] * cp.cos(2.0 * th_int)
  Dmed    = float(cp.median(state.D.astype(cp.float64)))
  vmax    = float(cp.max(Vint))
  vmin    = float(cp.min(Vint))
  eta_soft = max(abs(vmax), abs(vmin)) / (Dmed + 1e-30)
  indef    = max(0.0, -vmin) / (Dmed + 1e-30)
  thmax    = float(cp.max(cp.abs(th_int)))

  should_warn = (thmax > warn_theta_max_rad) or (eta_soft > warn_eta_soft) or (indef > warn_eta_soft)

  if should_warn:
      badge = "⚠️"
      sev   = "HARD" if (eta_soft >= warn_eta_hard or thmax > (warn_theta_max_rad+0.15)) else "SOFT"
      print(
  f"""{badge} Steady-state may be stiff/unstable ({sev}) [{label}]
    max|θ| = {thmax:.3f} rad  (limit ~ {warn_theta_max_rad:.2f})
    strength metric η = ||2K cos(2θ)||∞ / median(D) = {eta_soft:.3e}
    indefiniteness    = max(-2K cos(2θ)) / median(D) = {indef:.3e}
    range 2K cos(2θ): [{vmin:.3e}, {vmax:.3e}],   median(D) = {Dmed:.3e}
    Remedies: lower V_app (b) or power, reduce dz, run time-dependent IMEX with corrector OFF,
              or enable Levenberg–Marquardt damping in theta_newton_step.
  """)
      return {"theta_max": thmax, "eta": eta_soft, "indef": indef, "Dmed": Dmed, "vmin": vmin, "vmax": vmax}
