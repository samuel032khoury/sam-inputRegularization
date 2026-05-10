#!/bin/bash
# =============================================================
# SAM CIFAR-10 WideResNet-28-10 Slurm job script for USC CARC
# -------------------------------------------------------------
# Usage:
#   1) Fill in the --account field below.
#   2) Upload this file (with the code) to CARC: ~/sam/run_sam.slurm
#   3) Submit:   sbatch ~/sam/run_sam.slurm
#   4) Status:   squeue -u $USER
#   5) Tail log: tail -f ~/sam/logs/sam_<jobid>.out
# =============================================================

#SBATCH --job-name=sam_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                   # any GPU type (P100/V100/A100/A40)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00                # 48h max; ~17h on P100 for 200 epochs
#SBATCH --account=REPLACE_ME           # REQUIRED. Check: sacctmgr show assoc user=$USER
#SBATCH --output=logs/sam_%j.out
#SBATCH --error=logs/sam_%j.err

set -uo pipefail      # No -e: we want to do manual sanity checks first

# === Environment load (order MUST be:
#       legacy -> gcc -> cuda -> cudnn -> conda) ===
# cuda/11.8.0 and cudnn/8.4 only resolve under the
# legacy/CentOS7 + gcc/11.3.0 parents; the wrong order silently fails.
module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load cudnn/8.4.0.27-11.6
module load conda

# In Slurm non-interactive shells, `conda activate` often "fake-activates":
# the command succeeds without error but `python` still points at base.
# Fix: source the hook, hard-override PATH, then hash -r.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam-env || true
export PATH="$HOME/.conda/envs/sam-env/bin:$PATH"
hash -r               # clear bash's cached python path

# Tell JAX/XLA where to find the CUDA tooling (ptxas, libdevice).
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
# Quiet down TF warnings (train.py disables TF on GPU; TF is just a dep).
export TF_CPP_MIN_LOG_LEVEL=2

# Avoid re-downloading CIFAR-10 on every run.
export TFDS_DATA_DIR=/scratch1/$USER/tfds
mkdir -p "$TFDS_DATA_DIR"

# === Output directory (checkpoints + tensorboard event files) ===
export OUTDIR=/scratch1/$USER/sam_experiments/sam_rho0.05
mkdir -p "$OUTDIR"

# === Log this job's configuration ===
echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "ptxas     : $(which ptxas)"
echo "Python    : $(which python)"
echo "OUTDIR    : $OUTDIR"
echo "Start     : $(date)"
echo "=========================================="

# === Fail-fast pre-launch check (if it fails, abort instead of wasting GPU time) ===
python -c "import sys, jax, absl, flax; print('python:', sys.executable); print('jax:', jax.__version__, jax.devices()); print('flax:', flax.__version__)" \
  || { echo "ENV BROKEN — imports failed, aborting"; exit 1; }
echo "=========================================="

# === Training ===
cd "$HOME"
python -m sam.sam_jax.train \
  --dataset cifar10 \
  --model_name WideResnet28x10 \
  --output_dir "$OUTDIR" \
  --image_level_augmentations basic \
  --batch_level_augmentations cutout \
  --num_epochs 200 \
  --batch_size 128 \
  --learning_rate 0.1 \
  --weight_decay 5e-4 \
  --sam_rho 0.05 \
  --evaluate_every 5

echo "=========================================="
echo "End       : $(date)"
echo "=========================================="
