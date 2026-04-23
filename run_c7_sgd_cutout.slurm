#!/bin/bash
# =============================================================
# Condition 7: SGD + Cutout
# Backbone: WideResNet-28-10
# Augmentation: basic (random crop + horizontal flip) + Cutout
# NOTE: 这个和旧的 run_baseline.slurm 是等价的 config, 改名对齐 8-condition 设计.
#       之前 run_baseline.slurm 的输出在 /scratch1/$USER/sam_experiments/no_sam,
#       可以 mv 成 /scratch1/$USER/sam_experiments/c7_sgd_cutout 以对齐命名 (可选).
# =============================================================

#SBATCH --job-name=c7_sgd_cutout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=vsharan_1861
#SBATCH --output=logs/c7_sgd_cutout_%j.out
#SBATCH --error=logs/c7_sgd_cutout_%j.err

set -uo pipefail

module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load cudnn/8.4.0.27-11.6
module load conda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam-env || true
export PATH="$HOME/.conda/envs/sam-env/bin:$PATH"
hash -r

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_CPP_MIN_LOG_LEVEL=2

export TFDS_DATA_DIR=/scratch1/$USER/tfds
mkdir -p "$TFDS_DATA_DIR"

export OUTDIR=/scratch1/$USER/sam_experiments/c7_sgd_cutout
mkdir -p "$OUTDIR"

echo "=========================================="
echo "Condition 7: SGD + Cutout"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "OUTDIR    : $OUTDIR"
echo "Start     : $(date)"
echo "=========================================="

python -c "import sys, jax, absl, flax; print('python:', sys.executable); print('jax:', jax.__version__, jax.devices()); print('flax:', flax.__version__)" \
  || { echo "ENV BROKEN — imports failed, aborting"; exit 1; }
echo "=========================================="

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
  --sam_rho 0.0 \
  --evaluate_every 5

echo "=========================================="
echo "End       : $(date)"
echo "=========================================="
