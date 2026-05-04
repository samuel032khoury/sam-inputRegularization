#!/bin/bash
# =============================================================
# Condition 10: SAM + mixcut  (= cutout(mixup(x)))
# Backbone: WideResNet-28-10
# Augmentation: basic (random crop + horizontal flip) + mixcut
# Optimizer: SAM, rho = 0.05
#
# 是 C9 (SGD + mixcut, 96.97%) 的 SAM 版本.
# 对标:
#   - C10 vs C9   → SAM 在 mixcut 之上的额外贡献 (期待 ~+0.42, 和其他 aug 一致)
#   - C10 vs C5   → 在 mixup 之上多叠一层 cutout 是否还有边际收益
#   - C10 vs C6   → 和 SAM+CutMix 97.68% 的 overall winner 比, 看 mixcut 是不是更弱
#   - C10 vs C8   → 和 SAM+Cutout 97.43% 比, 看 mixup-blend 是否反向贡献
#
# 注意: mixcut = cutout(mixup(x)), 不是 Yun 2019 CutMix. 用的是 google-research/sam
#       原生的 mixcut flag, label 是 Mixup 的 soft label (Cutout 不动 label).
# =============================================================

#SBATCH --job-name=c10_sam_mixcut
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --account=vsharan_1861
#SBATCH --output=logs/c10_sam_mixcut_%j.out
#SBATCH --error=logs/c10_sam_mixcut_%j.err

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

export OUTDIR=/scratch1/$USER/sam_experiments/c10_sam_mixcut
mkdir -p "$OUTDIR"

echo "=========================================="
echo "Condition 10: SAM + mixcut = cutout(mixup(x))"
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
  --sam_rho 0.05 \
  --evaluate_every 5

echo "=========================================="
echo "End       : $(date)"
echo "=========================================="
