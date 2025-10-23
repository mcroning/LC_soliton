# 🧠 LC Soliton Simulator

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![GPU](https://img.shields.io/badge/CUDA-enabled-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-informational)]()

GPU-accelerated solver for liquid-crystal (LC) director dynamics and optical field coupling.  
Implements time-dependent and steady-state forms of the nematic director PDE with optical and electric driving, optimized for CUDA (CuPy).

---

## ⚙️ Installation

### Clone and install the environment

```bash
git clone https://github.com/mcroning/lc_soliton.git
cd lc_soliton
```

#### 🧩 GPU (CUDA systems, e.g., RTX/A100)
```bash
conda env create -f environment.yml
conda activate lc_soliton
make install
```

#### 🍎 CPU (Mac or non-CUDA systems)
```bash
conda env create -f environment-cpu.yml
conda activate lc_soliton_cpu
make install
```

Optional GPU check:
```bash
make cuda-info
```

---

## ▶️ Quickstart: run a 2-D θ simulation

```bash
python examples/run_theta2d.py --Nx 128 --Ny 128 --xaper 10.0 \
  --steps 500 --dt 1e-3 --b 1.0 --bi 0.3 --intensity 1.0 \
  --mobility 1.0 --save theta_out.npz
```

Visualize the result:
```bash
python examples/plot_field.py theta_out.npz
```

This displays a color map of the director field θ(x, y).

---

## 🧪 Using physical time (recommended)

Set the *mobility* parameter as:

\[
\text{mobility} = \frac{K_{\text{Frank}}}{\gamma_1}\frac{4}{d^2} \; [\text{s}^{-1}]
\]

with  
\(K_{\text{Frank}}\) = elastic constant (N),  
\( \gamma_1 \) = rotational viscosity (Pa·s),  
\( d \) = cell thickness (m).

Example (typical nematic):  
\(K = 10\,\text{pN}, \gamma_1 = 0.1\,\text{Pa·s}, d = 10\,\mu\text{m}\)  
→ `--mobility 4.0`

Then `--dt` is in seconds, and results can be reported in \( t/\tau_0 \)  
with \( \tau_0 = \gamma_1 / K \).

---

## 🧬 Cluster / Slurm run

Example job script (`examples/slurm_run_theta.sh`):

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -J lc-theta
#SBATCH -o lc-theta.%j.out

source ~/.bashrc
conda activate lc_soliton
# module load cuda/12.2   # if your cluster requires it

python examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0 \
  --steps 1000 --dt 5e-4 --b 1.1 --bi 0.4 --intensity 1.0 \
  --mobility 4.0 --save theta_clust_
