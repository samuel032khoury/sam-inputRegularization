#!/bin/bash
# =============================================================
# Condition 9: SGD + Mixup+Cutout  (NOVEL — augmentation stacking)
# Backbone: WideResNet-28-10
# Augmentation: basic (random crop + horizontal flip) + mixcut
#   mixcut = cutout(mixup(x))  i.e. Mixup pixel blend -> then punch a cutout hole
#   label = 软 label (Mixup 的线性混合, Cutout 不动 label)
# Optimizer: SGD, rho = 0.0
#
# 对标: 和 C3 (Mixup), C7 (Cutout) 比, 测"两种 aug 叠加"是否有边际收益.
#       C9 vs C3 → 叠加 Cutout 在 Mixup 之上
#       C9 vs C7 → 叠加 Mixup 在 Cutout 之上
#       C10 (SAM + mixcut) 会是 C9 的 SAM 版本.
#
# 注意: 用的是 google-research/sam 原生的 `mixcut` flag (= cutout(mixup(x))),
#       不是 Yun 2019 CutMix. 这是"Mixup + Cutout" 组合, 符合 slide 里的 Condition 名字.
# =============================================================

#SBATCH --job-name=c9_sgd_mixcut
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=vsharan_1861
#SBATCH --output=logs/c9_sgd_mixcut_%j.out
#SBATCH --error=logs/c9_sgd_mixcut_%j.err

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

export OUTDIR=/scratch1/$USER/sam_experiments/c9_sgd_mixcut
mkdir -p "$OUTDIR"

echo "=========================================="
echo "Condition 9: SGD + mixcut = cutout(mixup(x))"
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
  --batch_level_augmentations mixcut \
  --num_epochs 200 \
  --batch_size 128 \
  --learning_rate 0.1 \
  --weight_decay 5e-4 \
  --sam_rho 0.0 \
  --evaluate_every 5

echo "=========================================="
echo "End       : $(date)"
echo "=========================================="
