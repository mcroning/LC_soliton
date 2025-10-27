#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, cupy as cp
from lc_soliton.lc_core import (
    LCVarDirichletState, init_lc_state,
    build_theta_bias_IC, advance_theta_timestep, theta_newton_step,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=128)
    ap.add_argument("--xaper", type=float, default=10.0, help="domain size in x (same units as d)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=1e-3, help="seconds")
    ap.add_argument("--b", type=float, default=1.0)
    ap.add_argument("--bi", type=float, default=0.3)
    ap.add_argument("--intensity", type=float, default=1.0, help="uniform I(x,y)")
    ap.add_argument("--mobility", type=float, default=1.0, help="(K_Frank/gamma1)*(4/d^2) [1/s]")
    ap.add_argument("--newton_every", type=int, default=0, help="0=off; polish every k steps")
    ap.add_argument("--save", default="theta_out.npz")
    ap.add_argument("--coh", default= True")
    args = ap.parse_args()

    Nx, Ny = args.Nx, args.Ny
    state = LCVarDirichletState()
    init_lc_state(state, Nx=Nx, Ny=Ny, d_use=args.xaper)

    # initial bias θ(x) from exact elliptic solution (tiles along y)
    theta = build_theta_bias_IC(state, b=args.b)

    # two-beam “amp” container; here we just set a uniform amplitude -> I = const
    amp = cp.zeros((2, Nx, Ny), dtype=cp.complex64)
    amp[0, ...] = cp.sqrt(args.intensity).astype(cp.complex64)

    for k in range(args.steps):
        theta = advance_theta_timestep(
            theta, amp, state,
            dt=args.dt, b=args.b, bi=args.bi, mobility=args.mobility, coh=args.coh,
            k_correct_every=args.newton_every or None,
            k_index=k, newton_iters=1 if args.newton_every else 0
        )

    cp.savez(args.save, theta=cp.asnumpy(theta), meta=dict(
        Nx=Nx, Ny=Ny, xaper=args.xaper, dt=args.dt, steps=args.steps, b=args.b, bi=args.bi,
        intensity=args.intensity, mobility=args.mobility,
    ))
    print(f"Saved {args.save}")

if __name__ == "__main__":
    main()
