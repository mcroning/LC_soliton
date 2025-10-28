import argparse, json, sys, os
from .steady_demo import demo_steady_path, quick_view_slice

def main():
    p = argparse.ArgumentParser(prog="lc-soliton", description="LC steady-state demo and viewers")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo-steady", help="Run a demo steady z-march and write compressed outputs")
    d.add_argument("--outdir", default="steady_demo")
    d.add_argument("--picard", type=int, default=1)
    d.add_argument("--chunks", default="128,128,8")
    d.add_argument("--theta-q", default="int16,-1.57079632679,1.57079632679")  # dtype,lo,hi
    d.add_argument("--int-q", default="uint16,0.0,auto")                      # dtype,lo,hi(auto)
    # In practice you’ll likely load amp0/theta_bias/kernels/state from your app
    d.add_argument("--config", required=True, help="Path to a Python file that builds amp0, theta_bias, kernels, state, params")

    v = sub.add_parser("view", help="Quickly view a slice from a zarr")
    v.add_argument("zarr_path")
    v.add_argument("k", type=int)
    v.add_argument("--clim", default=None)
    v.add_argument("--cmap", default="viridis")

    args = p.parse_args()

    if args.cmd == "view":
        clim = None
        if args.clim:
            lo, hi = map(float, args.clim.split(","))
            clim = (lo, hi)
        quick_view_slice(args.zarr_path, args.k, clim=clim, cmap=args.cmap)
        return

    if args.cmd == "demo-steady":
        # Load a config module that provides the ready-to-use objects
        cfg_path = os.path.abspath(args.config)
        namespace = {}
        with open(cfg_path, "r") as f:
            code = compile(f.read(), cfg_path, "exec")
            exec(code, namespace, namespace)

        amp0        = namespace["amp0"]
        theta_bias  = namespace["theta_bias"]
        h_full      = namespace["h_full"]
        h_half      = namespace["h_half"]
        dz          = namespace["dz"]
        niter       = namespace["niter"]
        windowxy    = namespace["windowxy"]
        state       = namespace["state"]
        b           = namespace["b"]
        bi          = namespace["bi"]

        cx, cy, cz = map(int, args.chunks.split(","))
        tq_dtype, tq_lo, tq_hi = args.theta_q.split(",")
        iq_dtype, iq_lo, iq_hi = args.int_q.split(",")
        tq = (tq_dtype, float(tq_lo), float(tq_hi))
        iq = (iq_dtype, float(iq_lo), None if iq_hi.lower()=="auto" else float(iq_hi))

        out = demo_steady_path(
            amp0, theta_bias,
            h_full=h_full, h_half=h_half, dz=dz, niter=niter,
            windowxy=windowxy, state=state, b=b, bi=bi,
            outdir=args.outdir, picard_passes=args.picard,
            q_theta=tq, q_int=iq, chunks=(cx, cy, cz)
        )
        print(json.dumps(out, indent=2))
