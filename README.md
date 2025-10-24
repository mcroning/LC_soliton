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

## ⚙️ Installation

```bash
git clone https://github.com/mcroning/lc_soliton.git
cd lc_soliton
conda env create -f environment.yml
conda activate lc_soliton
make install
```

To verify GPU visibility:
```bash
make cuda-info
```

---

## ▶️ Quickstart

Run a 2‑D transient LC simulation:

```bash
python examples/run_theta2d.py --Nx 128 --Ny 128 --xaper 10.0   --steps 500 --dt 1e-3 --b 1.0 --bi 0.3 --intensity 1.0   --mobility 4.0 --save theta_out.npz
```

Plot results:

```bash
python examples/plot_field.py theta_out.npz
```

---

## 🧠 Governing Equations

The LC director tilt $\theta(x, y, t)$ evolves according to:

$$
\frac{\gamma_1}{K} \frac{\partial \theta}{\partial t}
= \nabla_{xy}^2 \theta
 \frac{\epsilon_0 \Delta \epsilon_{\mathrm{RF}} E^2}{2K} \sin(2\theta)
 \frac{\epsilon_0 n_a^2 |E_{\mathrm{op}}|^2}{4K} \sin(2\theta)
$$


where $K$ is the Frank elastic constant, $\gamma_1$ the rotational viscosity, and $E_{\mathrm{op}}$ the optical field envelope.  
For steady state, set $\partial_t \theta = 0$.

Dimensionless form:

$$
\frac{\partial \theta}{\partial t'} = \nabla_{xy}^2 \theta + b \sin(2\theta) + b_i I(x,y) \sin(2\theta)
$$



with

$$
b = \frac{\epsilon_0 \Delta \epsilon_{\mathrm{RF}} V^2}{8K},
\qquad
b_i = \frac{\epsilon_0 n_a^2 d^2}{16K} \langle |E_{\mathrm{op}}|^2 \rangle
$$

---

## 🧩 Mobility and Timescale

Define a mobility parameter to express physical time (seconds):


$\text{mobility} = \frac{K}{\gamma_1}\frac{4}{d^2} \quad [\mathrm{s}^{-1}]$


Then `--dt` corresponds to seconds, and the natural timescale is $\tau_0 = \gamma_1 / K$.

Example (typical nematic):

$$
K = 10\,\mathrm{pN}, \quad
\gamma_1 = 0.1\,\mathrm{Pa\cdot s}, \quad
d = 10\,\mu\mathrm{m}
\Rightarrow \text{mobility} \approx 4.0
$$

---

## 🧪 Boundary Conditions

Choose using `--bc`:

- **Dirichlet:** $\theta|_{\partial \Omega} = 0$  
- **Neumann:** $\frac{\partial \theta}{\partial n}\big|_{\partial \Omega} = 0$

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

Submit with:

```bash
sbatch examples/slurm_run_theta.sh
```

---

### 🧰 Build & Maintenance Targets


make install          # Editable install with dev tools (pytest, ruff)
make test             # Run test suite
make lint             # Static analysis (Ruff)
make cuda-info        # GPU/CuPy diagnostics

# --- Docs management ---
make docs-deps        # install ReportLab for PDF generation
make docs             # build all docs/*.pdf files
make docs-open        # open docs/ directory
make docs-clean       # remove generated PDFs

# --- Cleaning utilities ---
make clean            # remove temporary build artifacts and caches (safe)
make clean-dry        # preview what 'clean' would remove
make clean-cupy-cache # remove CuPy kernel caches
make clean-slurm      # remove Slurm outputs
make superclean       # clean + remove extra build leftovers (safe)
make distclean FORCE=1 # SUPER-aggressive cleanup (data/results)

---

## 📄 Documentation

Detailed derivations and usage instructions are provided as downloadable PDFs:

- [LC_PDE_Derivation.pdf](docs/LC_PDE_Derivation.pdf) — Governing equation derivation  
- [Usage_Guide.pdf](docs/Usage_Guide.pdf) — Command-line and runtime options  
- [Development_Guide.pdf](docs/Development_Guide.pdf) — Architecture and contributing guide  

---

## 📜 License

MIT License © 2025 Mark Cronin-Golomb

---

## 🧩 Citation

```
Cronin-Golomb, M. (2025).
LC Soliton Simulator: GPU-accelerated LC solver.
GitHub repository.
```

---

_Developed and maintained at the Cronin-Golomb Lab (Tufts University)._
