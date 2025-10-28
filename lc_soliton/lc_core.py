# lc_soliton/lc_core.py
from __future__ import annotations
import cupy as cp
from cupyx.scipy import special as spspec

from ._internals import (
    ufft, uifft, dst1o, idst1o, make_dst1_ortho_wrappers, intens
)

# ---------------- State ----------------
class LCVarDirichletState:
    def __init__(self):
        self.ready = False
        # geometry / spectra
        self.Nx = None; self.Ny = None; self.d_use = None
        self.fy = None
        self.D = None; self.lamx = None; self.ky2 = None
        # precond shift etc.
        self.mu = None
        # work buffers
        self.buf_U = self.buf_p = self.buf_Ap = None
        self.buf_r = self.buf_z = None
        self.buf_real = None
        # unitary DST wrappers
        self.dst1_unit = None
        self.idst1_unit = None
        self.dst1_unit_c = None
        self.idst1_unit_c = None

# ---------------- K-space ----------------
def make_kspace(Nx, Ny, dx, dy, *, dtype=cp.float32):
    fx = cp.fft.fftfreq(Nx, d=dx)
    fy = cp.fft.fftfreq(Ny, d=dy)
    kx = (2*cp.pi*fx).astype(dtype )
    ky = (2*cp.pi*fy).astype(dtype )
    kx_col = kx[:, None]
    kx2d, ky2d = cp.meshgrid(kx, ky, indexing="ij")
    kxy2 = (kx2d**2 + ky2d**2).astype(dtype )
    return kx_col, ky, kxy2

# ---------------- Optics ----------------
def LC_dn_from_theta(theta, ne, no, refin):
    th64 = theta.astype(cp.float64 )
    ct, st = cp.cos(th64), cp.sin(th64)
    n_eff  = (ne*no) / cp.sqrt((ne*ct)**2 + (no*st)**2)
    return (n_eff - refin).astype(cp.float32 )

# ---------------- Bias profile (exact) ----------------
def build_theta_bias_IC(state, b, *, eps_clip=1e-12, return_1d=False):
    xp = cp
    Nx = int(getattr(state, "Nx", 256))
    Ny = int(getattr(state, "Ny", 256))

    b_c = xp.pi**2 / 8.0
    if b <= float(b_c):
        theta_1d = xp.zeros(Nx, dtype=xp.float64)
        return theta_1d if return_1d else xp.tile(theta_1d[:, None], (1, Ny))

    u = xp.linspace(-1.0, 1.0, Nx, dtype=xp.float64)
    two_b = 2.0 * float(b)

    def Ksq_minus_2b(m):
        Km = spspec.ellipk(m)
        return float(cp.asnumpy(Km*Km - two_b))

    m_lo, m_hi = 1e-12, 1.0 - 1e-12
    for _ in range(80):
        m_mid = 0.5 * (m_lo + m_hi)
        if Ksq_minus_2b(m_lo) * Ksq_minus_2b(m_mid) <= 0.0:
            m_hi = m_mid
        else:
            m_lo = m_mid
        if abs(m_hi - m_lo) < 1e-14:
            break
    m = 0.5 * (m_lo + m_hi)
    theta0 = xp.arcsin(xp.sqrt(m))

    arg = xp.sqrt(2.0 * b) * u
    sn, cn, dn, _ = spspec.ellipj(arg, float(m))
    cd = cn / dn
    s = xp.sin(theta0) * cd
    s = xp.clip(s, -1.0 + eps_clip, 1.0 - eps_clip)
    theta_1d = xp.arcsin(s)
    return theta_1d if return_1d else xp.tile(theta_1d[:, None], (1, Ny))

# ---------------- Init / Geometry ----------------
def init_lc_state(state: LCVarDirichletState, Nx, Ny, xaper, fy):
    """Initialize spectral diagonals/buffers and orthonormal DST-I wrappers."""
    _ensure_state_init(state, Nx, Ny, xaper, fy)
    if state.dst1_unit_c is None:
        n = Nx - 2
        dst1_u, idst1_u, _, _ = make_dst1_ortho_wrappers(dst1o, idst1o, n, Ny)
        state.dst1_unit    = dst1_u
        state.idst1_unit   = idst1_u
        state.dst1_unit_c  = lambda A, axis=0: dst1_u(A.real, axis=axis) + 1j*dst1_u(A.imag, axis=axis)
        state.idst1_unit_c = lambda B, axis=0: idst1_u(B.real, axis=axis) + 1j*idst1_u(B.imag, axis=axis)
    return state

def _ensure_state_init(state, Nx, Ny, d_use, fy):
    need = (not state.ready) or (state.Nx != Nx) or (state.Ny != Ny) or (state.d_use != d_use)
    if not need:
        return
    n   = Nx - 2
    mx  = cp.arange(1, n+1, dtype=cp.float32)[:, None]
    dx  = d_use / (n + 1)
    lamx = (4.0/(dx*dx)) * cp.sin(0.5*cp.pi*mx/(n+1))**2
    assert fy.shape == (Ny,), f"fy must be length Ny, got {fy.shape}"
    ky2  = (fy[None,:])**2

    D = (d_use**2/4.0) * (lamx + ky2)
    shape = (n, Ny)
    state.buf_U    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_p    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_Ap   = cp.zeros(shape, dtype=cp.complex64)
    state.buf_r    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_z    = cp.zeros(shape, dtype=cp.complex64)
    state.buf_real = cp.zeros(shape, dtype=cp.float32)

    state.fy = fy.astype(cp.float32 )
    state.lamx = lamx
    state.ky2  = ky2
    state.D    = D.astype(cp.float32 )
    state.Nx, state.Ny, state.d_use = Nx, Ny, d_use
    state.ready = True

# ---------------- Warnings (your function) ----------------
def lc_warn_bias_and_stiffness(theta_field, state, *, b, bi, Ixy=None,
                               warn_theta_max_rad=1.20, warn_eta_soft=2e-3,
                               warn_eta_hard=1e-2, label="pre-Newton"):
    th = cp.asarray(theta_field, dtype=cp.float64)
    Nx, Ny = th.shape
    if Ixy is None:
        Ixy = cp.zeros((Nx, Ny), dtype=cp.float64)
    else:
        Ixy = cp.asarray(Ixy, dtype=cp.float64)

    th_int  = th[1:-1, :]
    Kxy     = (b + bi * Ixy).astype(cp.float64)
    Vint    = 2.0 * Kxy[1:-1, :] * cp.cos(2.0 * th_int)
    Dmed    = float(cp.median(state.D.astype(cp.float64)))
    vmax    = float(cp.max(Vint)); vmin = float(cp.min(Vint))
    eta_soft = max(abs(vmax), abs(vmin)) / (Dmed + 1e-30)
    indef    = max(0.0, -vmin) / (Dmed + 1e-30)
    thmax    = float(cp.max(cp.abs(th_int)))

    should_warn = (thmax > warn_theta_max_rad) or (eta_soft > warn_eta_soft) or (indef > warn_eta_soft)
    if should_warn:
        sev = "HARD" if (eta_soft >= warn_eta_hard or thmax > (warn_theta_max_rad+0.15)) else "SOFT"
        print(f"⚠️ {label}: max|θ|={thmax:.3f}, η={eta_soft:.3e}, indef={indef:.3e}, D̃={Dmed:.3e} [{sev}]")
    return {"theta_max": thmax, "eta": eta_soft, "indef": indef, "Dmed": Dmed, "vmin": vmin, "vmax": vmax}

# ---------------- Linear operator helpers ----------------
def _Lp_theta_real(theta_int, state):
    Th  = state.dst1_unit_c( ufft(theta_int, axis=1), axis=0 )
    LpT = uifft( state.idst1_unit_c(state.D * Th, axis=0), axis=1 ).real
    return LpT

def _apply_A_var(U, V_int, state):
    Uy        = state.idst1_unit_c(U, axis=0)
    u_int_c   = uifft(Uy, axis=1)
    Au_spec   = state.D * U \
              + state.dst1_unit_c( ufft(V_int * u_int_c.real, axis=1), axis=0 )
    return Au_spec

def _pcg_var(F, V_int, state, itmax=100, tol=1e-8, U0=None):
    U   = cp.zeros_like(F) if U0 is None else U0.copy()
    r   = F - _apply_A_var(U, V_int, state)
    z   = r / (state.D + cp.maximum(V_int, 0).astype(state.D.dtype))
    p   = z.copy()
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

# ---------------- Newton / Time step ----------------
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
        Ixy = intens(amp, coh).astype(cp.float64 )
    else:
        Ixy = cp.asarray(Ixy, dtype=cp.float64)

    Kxy  = (b + bi * Ixy).astype(cp.float64 )
    Kint = Kxy[1:-1, :]
    theta_int = theta[1:-1, :].astype(cp.float64 )

    LpT  = _Lp_theta_real(theta_int, state).astype(cp.float64 )
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

def advance_theta_timestep(theta, amp, state, *,
                           dt, b, bi, Ixy=None, coh=True,
                           k_correct_every=None, k_index=None,
                           newton_iters=0, newton_tol=1e-6,
                           mobility=1.0):
    xp = cp
    # --- intensity (GPU) respecting coherence flag ---
    if Ixy is None:
        # Use the same central intensity function used elsewhere
        Ixy = intens(amp, coh)
    Ixy = xp.asarray(Ixy, dtype=xp.float64 )

    T = xp.array(theta, dtype=xp.float64 )
    Tint = T[1:-1, :]
    Kxy  = (b + bi * Ixy).astype(xp.float64 )
    Kint = Kxy[1:-1, :]

    Fexp = (Kint * xp.sin(2.0 * Tint)).astype(xp.float64 )

    dst1_unit_c  = state.dst1_unit_c
    idst1_unit_c = state.idst1_unit_c

    Fexp_spec = dst1_unit_c( ufft(Fexp, axis=1), axis=0 )
    Tn_spec   = dst1_unit_c( ufft(Tint, axis=1), axis=0 )

    Lam = state.D.astype(xp.float64 )
    alpha = float(mobility) * float(dt)

    Tnp1_spec = (Tn_spec + alpha * Fexp_spec) / (1.0 + alpha * Lam)

    Uy         = idst1_unit_c(Tnp1_spec, axis=0)
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
