# Quick viewer for theta_out.npz
import argparse, numpy as np, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("npz", help="file produced by run_theta2d.py")
ap.add_argument("--cmap", default="twilight")
args = ap.parse_args()

data = np.load(args.npz, allow_pickle=True)
theta = data["theta"]
plt.imshow(theta, origin="lower", cmap=args.cmap)
plt.colorbar(label="θ (rad)")
plt.title(args.npz)
plt.tight_layout()
plt.show()
