#!/bin/bash
# =============================================================
# Baseline (no SAM) CIFAR-10 WideResNet-28-10 Slurm job
# 和 run_sam.slurm 一样, 只把 --sam_rho 改成 0.0, OUTDIR 改名
# =============================================================

#SBATCH --job-name=sam_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                   # 不限定型号, 谁空用谁
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00                # P100 上基线约 10-11h (SAM 的一半)
#SBATCH --account=REPLACE_ME           # 🚨 必填!
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

set -uo pipefail    # 不用 -e, 手动验证再 fail

# legacy -> gcc -> cuda -> cudnn -> conda (顺序不能乱)
module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load cudnn/8.4.0.27-11.6
module load conda

# 硬兜底 PATH: 绕过 Slurm 里 conda activate 的"假激活"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam-env || true
export PATH="$HOME/.conda/envs/sam-env/bin:$PATH"
hash -r

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export TF_CPP_MIN_LOG_LEVEL=2

export TFDS_DATA_DIR=/scratch1/$USER/tfds
mkdir -p "$TFDS_DATA_DIR"

export OUTDIR=/scratch1/$USER/sam_experiments/no_sam
mkdir -p "$OUTDIR"

echo "Job ID: $SLURM_JOB_ID on $(hostname), start $(date)"
echo "GPU   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python: $(which python)"

# Fail-fast: import 检查不过就别浪费 GPU
python -c "import sys, jax, absl, flax; print('python:', sys.executable); print('jax:', jax.__version__, jax.devices()); print('flax:', flax.__version__)" \
  || { echo "ENV BROKEN — imports failed, aborting"; exit 1; }

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

echo "End $(date)"
