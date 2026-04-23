#!/bin/bash
# =============================================================
# SAM CIFAR-10 WideResNet-28-10 Slurm job script for USC CARC
# -------------------------------------------------------------
# 用法:
#   1) 填好下面的 --account 字段
#   2) 把这个文件和代码一起上传到 CARC: ~/sam/run_sam.slurm
#   3) 提交: sbatch ~/sam/run_sam.slurm
#   4) 查状态: squeue -u $USER
#   5) 查日志: tail -f ~/sam/logs/sam_<jobid>.out
# =============================================================

#SBATCH --job-name=sam_cifar10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1                   # 不限定型号, 谁空用谁 (P100/V100/A100/A40 都行)
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00                # 最长 48h; P100 上大约 17h 可完整跑 200 epoch
#SBATCH --account=REPLACE_ME           # 🚨 必填! 查: sacctmgr show assoc user=$USER
#SBATCH --output=logs/sam_%j.out
#SBATCH --error=logs/sam_%j.err

set -uo pipefail      # 注意: 没用 -e, 方便我们在 activate 前手动验证

# === 环境加载 (顺序必须: legacy -> gcc -> cuda -> cudnn -> conda) ===
# cuda/11.8.0 和 cudnn/8.4 在 legacy/CentOS7 + gcc/11.3.0 parent 下, 顺序错了 load 不上
module purge
module load legacy/CentOS7
module load gcc/11.3.0
module load cuda/11.8.0
module load cudnn/8.4.0.27-11.6
module load conda

# Slurm non-interactive shell 里 conda activate 经常"假激活": 命令不报错,
# 但 python 依然指向 conda base. 解决办法 = source hook + 硬 override PATH + hash -r
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam-env || true
export PATH="$HOME/.conda/envs/sam-env/bin:$PATH"
hash -r               # 清 bash 对 python 位置的缓存

# JAX/XLA 找 CUDA 工具 (ptxas, libdevice)
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
# 压低 TF 冗余 warning (train.py 禁用 TF GPU, TF 只是个依赖)
export TF_CPP_MIN_LOG_LEVEL=2

# 防止 CIFAR-10 每次重下载
export TFDS_DATA_DIR=/scratch1/$USER/tfds
mkdir -p "$TFDS_DATA_DIR"

# === 输出目录 (checkpoint + tensorboard 日志) ===
export OUTDIR=/scratch1/$USER/sam_experiments/sam_rho0.05
mkdir -p "$OUTDIR"

# === 记录本次 job 的配置 ===
echo "=========================================="
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "ptxas     : $(which ptxas)"
echo "Python    : $(which python)"
echo "OUTDIR    : $OUTDIR"
echo "Start     : $(date)"
echo "=========================================="

# === 启动前 fail-fast 检查 (没通过就退出, 别浪费 GPU 排队) ===
python -c "import sys, jax, absl, flax; print('python:', sys.executable); print('jax:', jax.__version__, jax.devices()); print('flax:', flax.__version__)" \
  || { echo "ENV BROKEN — imports failed, aborting"; exit 1; }
echo "=========================================="

# === 训练 ===
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
