#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

# Use a non-interactive backend if no display (cluster-safe)
if not os.environ.get("DISPLAY"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="file produced by run_theta2d.py")
ap.add_argument("--cmap", default="twilight", help="matplotlib colormap")
ap.add_argument("--save", default=None, help="path to save PNG (if omitted, show window)")
args = ap.parse_args()

data = np.load(args.npz, allow_pickle=True)
theta = data["theta"]

plt.imshow(theta, origin="lower", cmap=args.cmap)
plt.colorbar(label="θ (rad)")
plt.title(args.npz)
plt.tight_layout()

if args.save:
    plt.savefig(args.save, dpi=200)
    print(f"Saved {args.save}")
else:
    plt.show()
