# 🧠 LC Soliton Simulator

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![GPU](https://img.shields.io/badge/CUDA-enabled-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-informational)]()

GPU-accelerated solver for **liquid-crystal (LC)** director dynamics and optical field coupling.  
Implements both time-dependent and steady-state nematic director equations with electric (RF/DC) and optical driving.  
Designed for use with **CuPy** (CUDA) or **NumPy** backends.

---

## ✨ Features

- **GPU acceleration** with CuPy (automatic CPU fallback)  
- **Transient and steady-state** LC solvers in 2D  
- **Newton–Anderson** steady solver with stability checks  
- **Command-line examples and visualization utilities**  
- **Reproducible runs** (YAML configs, random seeds)  
- **PDF documentation** in the `docs/` folder

---

## 🧭 User-Facing Functions

Two main functions form the core of LC Soliton Simulator’s user API.

| Function | Purpose | Notes |
|-----------|----------|-------|
| `advance_theta_timestep(state, dt, ...)` | Advances the LC director field in **time** using the transient PDE | For dynamic evolution problems |
| `solve_theta_steady_slice(theta, amp, state, b, bi, ...)` | Finds the **steady-state** director field at one z-slice given the current optical field | For static or equilibrium solutions |

### 1️⃣ `solve_theta_steady_slice`

```python
from lc_soliton import solve_theta_steady_slice

theta_ss, info = solve_theta_steady_slice(theta_prev, amp_k, state, b=b, bi=bi)
```

| Argument | Description |
|-----------|-------------|
| `theta_prev` | Initial director field (CuPy or NumPy array) |
| `amp_k` | Complex optical amplitude at this slice |
| `state` | Simulation container (grid, material, helpers) |
| `b`, `bi` | Dimensionless LC parameters |
| `nl_tol`, `step_tol` | Nonlinear and step-size tolerances (`1e-6`, `1e-7`) |
| `max_newton`, `pcg_itmax` | Iteration limits for Newton and PCG solves |
| `linesearch` | Enable line search (default `True`) |

**Returns**  
- `theta_ss` – steady-state director field for this slice  
- `info` – dict with `converged`, `it`, `rel_res`, `step_inf`  

Automatically detects backend (CuPy → GPU | NumPy → CPU).

---

### 2️⃣ `advance_theta_timestep`

Integrates the LC equation forward in time for dynamic simulations.

```python
from lc_soliton import advance_theta_timestep

theta_next = advance_theta_timestep(theta, state, dt=1e-3, b=b, bi=bi, intensity=Ixy)
```

| Argument | Description |
|-----------|-------------|
| `theta` | Current director field |
| `state` | Simulation container |
| `dt` | Physical timestep |
| `b`, `bi` | LC parameters |
| `intensity` | Optical intensity pattern |

Use this for time-dependent relaxation or driven transients.

---

## ▶️ Quickstart Examples

### ⏱ Transient Simulation (Time-step)

```bash
python examples/run_theta2d.py --Nx 128 --Ny 128 --xaper 10.0   --steps 500 --dt 1e-3 --b 1.0 --bi 0.3 --intensity 1.0   --mobility 4.0 --save theta_out.npz
```

Plot results:

```bash
python examples/plot_field.py theta_out.npz
```

---

### 🧩 Steady-State (Single Slice)

```python
from lc_soliton import solve_theta_steady_slice

theta_ss, info = solve_theta_steady_slice(theta_bias, amp0, state, b=b, bi=bi)
print(info)
```

Typical use: inside a z-loop for beam propagation, replacing time evolution with Newton convergence.

---

### 📊 Quick Visualization

```python
from lc_soliton import quick_view_slice

quick_view_slice("steady_demo/Ixy_path.zarr", k=200, clim=(0,0.3), cmap="viridis")
quick_view_slice("steady_demo/theta_path.zarr", k=200, cmap="twilight")
```

---

### 🎬 Full Steady-State Demo (`demo_steady_path`)

A convenience wrapper for **new users** or quick visualization.  
Runs a full z‑range propagation, saving compressed Zarr datasets and optional movies.

```python
from lc_soliton import demo_steady_path

out = demo_steady_path(
    amp0, theta_bias,
    h_full=h_full, h_half=h_half, dz=dz, niter=niter,
    windowxy=windowxy, state=state, b=b, bi=bi,
    outdir="steady_demo", picard_passes=1,
    q_theta=("int16", -np.pi/2, np.pi/2),
    q_int=("uint16", 0.0, None)
)
```

Outputs:

- `steady_demo/theta_path.zarr` – compressed θ(x,y,z)  
- `steady_demo/Ixy_path.zarr` – compressed intensity |A|²(x,y,z)  
- `steady_demo/snapshots/` – PNG thumbnails  
- `steady_demo/movies/` – MP4 flythrough  
- `steady_demo/summary.json` – metadata

Viewable via `quick_view_slice()` or `lc-soliton view` CLI.

---

## 🧠 Governing Equations

The LC director tilt θ(x, y, t) obeys

$$
\frac{\gamma_1}{K}\frac{\partial\theta}{\partial t}
= \nabla_{xy}^2\theta
 + \frac{\epsilon_0\Delta\epsilon_{RF}E^2}{2K}\sin(2\theta)
 + \frac{\epsilon_0n_a^2|E_{op}|^2}{4K}\sin(2\theta)
$$

where *K* is the Frank constant, γ₁ the rotational viscosity, and Eₒₚ the optical field envelope.  
Steady state → set ∂ₜθ = 0.

Dimensionless form:

$$
\frac{\partial\theta}{\partial t'} = \nabla_{xy}^2\theta + b\sin(2\theta) + b_iI(x,y)\sin(2\theta)
$$

with

$$
b = \frac{\epsilon_0\Delta\epsilon_{RF}V^2}{8K},
\qquad
b_i = \frac{\epsilon_0n_a^2d^2}{16K}\langle|E_{op}|^2\rangle
$$

---

## 🧩 Mobility and Timescale

Define a mobility parameter to express physical time (s):

$$
\text{mobility} = \frac{K}{\gamma_1}\frac{4}{d^2} \quad [\mathrm{s}^{-1}]
$$

Then `--dt` corresponds to seconds, and the natural timescale is τ₀ = γ₁ / K.

Example (typical nematic):

$$
K=10\,\mathrm{pN},\quad
\gamma_1=0.1\,\mathrm{Pa·s},\quad
d=10\,\mu\mathrm{m}
\Rightarrow\text{mobility}\approx4.0
$$

---

## 🧪 Boundary Conditions

Choose using `--bc` :

- **Dirichlet:** θ|∂Ω = 0  
- **Neumann:** ∂ₙθ|∂Ω = 0

---

## 🗂 Project Structure

```
lc_soliton/
├── solvers/
├── utils/
├── examples/
├── tests/
├── docs/
└── README.md
```

---

## 🧬 Cluster / Slurm Example

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:20:00
#SBATCH -J lc-theta
#SBATCH -o lc-theta.%j.out

source ~/.bashrc
conda activate lc_soliton

python examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0   --steps 1000 --dt 5e-4 --b 1.1 --bi 0.4 --intensity 1.0   --mobility 4.0 --save theta_cluster.npz
```

Submit:

```bash
sbatch examples/slurm_run_theta.sh
```

---

## 🧰 Build & Maintenance Targets

make install          # Editable install with dev tools (pytest, ruff)
make test             # Run test suite
make lint             # Static analysis (Ruff)
make cuda-info        # GPU/CuPy diagnostics

#### --- Docs management ---
make docs-deps        # install ReportLab for PDF generation
make docs             # build all docs/*.pdf files
make docs-open        # open docs/ directory
make docs-clean       # remove generated PDFs

#### --- Cleaning utilities ---
make clean            # remove temporary build artifacts and caches (safe)
make clean-dry        # preview what 'clean' would remove
make clean-cupy-cache # remove CuPy kernel caches
make clean-slurm      # remove Slurm outputs
make superclean       # clean + remove extra build leftovers (safe)
make distclean FORCE=1 # SUPER-aggressive cleanup (data/results)

---

## 📄 Documentation

- [LC_PDE_Derivation.pdf](docs/LC_PDE_Derivation.pdf) — Governing equations  
- [Usage_Guide.pdf](docs/Usage_Guide.pdf) — CLI and runtime options  
- [Development_Guide.pdf](docs/Development_Guide.pdf) — Architecture and contributing guide  

---

## 📜 License

MIT License © 2025 Mark Cronin‑Golomb

---

## 🧩 Citation

```
Cronin‑Golomb, M. (2025).
LC Soliton Simulator: GPU‑accelerated LC solver.
GitHub repository.
```

_Developed and maintained at the Cronin‑Golomb Lab (Tufts University)._

---

### 📘 Solver Reference

For detailed solver and acronym definitions, see:  
📄 [Solver_Reference.pdf](./Solver_Reference.pdf)

Summarizes:
- Function APIs for `advance_theta_timestep()` and `theta_newton_step()`  
- Numerical methods (PCG, SPD, LM shift, IMEX)  
- LC director discretization and GPU implementation details

---
