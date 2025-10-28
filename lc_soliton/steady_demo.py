"""
Demo march for new users:
- Runs a full z-range, using solve_theta_steady_slice per slice.
- Writes compressed, viewable Zarr datasets for theta and intensity.
- Optionally emits thumbnails and a simple MP4 flythrough.

This is a convenience wrapper; the *primary* user path is calling solve_theta_steady_slice
in their own z-loop and extracting custom data.
"""

def _xp_of(*arrs):
    import numpy as _np
    try:
        import cupy as _cp
        if any(getattr(a, "__cuda_array_interface__", None) is not None for a in arrs if a is not None):
            return __import__("cupy")
    except Exception:
        pass
    return _np

def _to_host(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return x.get()
    except Exception:
        pass
    return x

def _make_zarr(path, shape, dtype, chunks, clevel=5, zstd=True, bitshuffle=True):
    import zarr, numcodecs
    compressor = numcodecs.Blosc(
        cname="zstd" if zstd else "lz4",
        clevel=clevel,
        shuffle=numcodecs.Blosc.BITSHUFFLE if bitshuffle else numcodecs.Blosc.SHUFFLE,
    )
    root = zarr.open_group(zarr.DirectoryStore(path), mode="w")
    arr = root.create_dataset("data", shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
    return root, arr

def _quantize(arr_host, dtype, lo, hi):
    import numpy as np
    if dtype == "uint16":
        if hi is None:
            hi = float(arr_host.max()) if arr_host.size else 1.0
        scale = 65535.0 / max(hi - lo, 1e-30)
        q = np.clip((arr_host - lo) * scale, 0, 65535).astype(np.uint16 )
        return q, dict(dtype=dtype, lo=float(lo), hi=float(hi))
    elif dtype == "int16":
        # Map [lo,hi] to int16 range symmetrically
        scale = 32767.0 / max(hi - lo, 1e-30)
        q = ((arr_host - lo) * scale - 32768.0).clip(-32768, 32767).astype(np.int16 )
        return q, dict(dtype=dtype, lo=float(lo), hi=float(hi))
    else:
        return arr_host.astype(dtype ), dict(dtype=dtype, lo=float(lo), hi=float(hi or 0.0))

from .steady_api import solve_theta_steady_slice

def demo_steady_path(
    amp0, theta_bias, *, h_full, h_half, dz, niter, windowxy, state, b, bi,
    outdir="steady_demo", picard_passes=1,
    q_theta=("int16", -3.14159265/2, 3.14159265/2),  # (dtype, lo, hi)
    q_int=("uint16", 0.0, None),                      # None => auto per-slice max
    chunks=(128,128,8), clevel=5, zstd=True,
    make_movies=True, make_thumbnails=True,
    slice_kwargs=None
):
    import os, json
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    xsamp, ysamp = theta_bias.shape
    xp = _xp_of(amp0, theta_bias, windowxy, h_full, h_half)

    # Create compressed, chunked stores
    theta_root, theta_ds = _make_zarr(os.path.join(outdir, "theta_path.zarr"),
                                      (xsamp, ysamp, niter), q_theta[0], chunks, clevel, zstd)
    I_root, I_ds = _make_zarr(os.path.join(outdir, "Ixy_path.zarr"),
                              (xsamp, ysamp, niter), q_int[0], chunks, clevel, zstd)
    theta_root.attrs["quant"] = dict(name="theta", dtype=q_theta[0], lo=q_theta[1], hi=q_theta[2])
    I_root.attrs["quant"] = dict(name="intensity", dtype=q_int[0], lo=q_int[1], hi=None)

    meta = dict(xsamp=int(xsamp), ysamp=int(ysamp), niter=int(niter), dz=float(dz),
                b=float(b), bi=float(bi), picard_passes=int(picard_passes))
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    j32 = 1j
    amp_in = amp0.copy()
    theta_ss = theta_bias.copy()
    slice_kwargs = dict(slice_kwargs or {})

    for _pass in range(max(1, picard_passes)):
        amp = amp_in.copy()
        for k in range(niter):
            # CN-style leg
            if k == 0 or k == niter - 1:
                h_curr, z_step = h_half, dz / 2.0
            else:
                h_curr, z_step = h_full, dz
            amp = xp.fft.ifft2(xp.fft.fft2(amp) * h_curr) * windowxy

            # Per-slice steady solve
            theta_ss, info = solve_theta_steady_slice(theta_ss, amp, state, b=b, bi=bi, **slice_kwargs)

            # Intensity (host), quantize + write
            Ik = _to_host((amp * xp.conj(amp)).real)
            Iq, _ = _quantize(Ik, *q_int)
            I_ds[..., k] = Iq

            # Theta (host), quantize + write
            Th = _to_host(theta_ss)
            Tq, _ = _quantize(Th, *q_theta)
            theta_ds[..., k] = Tq

            # Nonlinear phase
            dn_k = state.LC_dn_from_theta(theta_ss) if hasattr(state, "LC_dn_from_theta") \
                   else LC_dn_from_theta(theta_ss, state.ne, state.no, state.refin)
            amp *= xp.exp(j32 * state.kout * dn_k * z_step)

            # Thumbnails
            if make_thumbnails and (k % max(1, niter // 16) == 0):
                import matplotlib.pyplot as plt
                tdir = os.path.join(outdir, "snapshots"); os.makedirs(tdir, exist_ok=True)
                plt.imsave(os.path.join(tdir, f"I_{k:04d}.png"), Ik, cmap="viridis")

        amp_in = amp  # for next Picard pass

    # Optional: small MP4 from thumbnails
    if make_movies:
        import os, glob, imageio.v2 as iio
        pngs = sorted(glob.glob(os.path.join(outdir, "snapshots", "I_*.png")))
        if pngs:
            movdir = os.path.join(outdir, "movies"); os.makedirs(movdir, exist_ok=True)
            with iio.get_writer(os.path.join(movdir, "intensity_flythrough.mp4"), fps=12) as w:
                for p in pngs:
                    w.append_data(iio.imread(p))

    return {
        "theta_zarr": os.path.join(outdir, "theta_path.zarr"),
        "Ixy_zarr": os.path.join(outdir, "Ixy_path.zarr"),
        "summary": os.path.join(outdir, "summary.json"),
        "outdir": outdir,
    }


def quick_view_slice(zarr_path, k, clim=None, cmap="viridis", title=None):
    """Tiny viewer: de-quantize and show slice k."""
    import zarr, numpy as np, matplotlib.pyplot as plt, os
    g = zarr.open_group(zarr_path, mode="r")
    A = g["data"][..., k]  # quantized
    q = g.attrs.get("quant", {})
    dtype, lo, hi = q.get("dtype"), q.get("lo", 0.0), q.get("hi", 1.0)
    A = A.astype(np.float32 )
    if dtype == "uint16":
        scale = 1.0 if hi is None else (hi - lo)
        A = lo + A * (scale / 65535.0)
    elif dtype == "int16":
        A = lo + (A + 32768.0) * ((hi - lo) / 32767.0)
    plt.imshow(A, cmap=cmap, clim=clim)
    plt.title(title or f"{os.path.basename(zarr_path)} : k={k}")
    plt.colorbar(); plt.show()
