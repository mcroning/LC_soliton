#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -t 00:10:00
#SBATCH -J lc-theta
#SBATCH -o lc-theta.%j.out

source ~/.bashrc
conda activate lc_soliton   # your env name
# module load cuda/12.2     # if your site requires it

python examples/run_theta2d.py --Nx 256 --Ny 256 --xaper 10.0 \
  --steps 1000 --dt 5e-4 --b 1.1 --bi 0.4 --intensity 1.0 --mobility 4.0 \
  --save theta_cluster.npz --coh False
