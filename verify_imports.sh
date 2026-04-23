#!/bin/bash
# =============================================================
# Layer 2 verification: static import check
# Run this on CARC head node or inside a salloc session.
# Expected: 3x "OK" lines. Any FAIL means we abort before submitting jobs.
# =============================================================

set -e

echo "=== [1/3] Loading environment ==="
module purge
module load legacy/CentOS7 gcc/11.3.0 cuda/11.8.0 cudnn/8.4.0.27-11.6 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam-env
export PATH="$HOME/.conda/envs/sam-env/bin:$PATH"
hash -r

# Head node 的 RLIMIT_NPROC 很严, OpenBLAS/OMP 默认想开几十个线程会触顶
# 把线程数压到 1, 只是做 import check, 不需要并行
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2

cd "$HOME"

echo ""
echo "=== [2/3] augmentation module has cutmix/mixup/cutout ==="
python -c "
from sam.sam_jax.datasets import augmentation
for fn_name in ['cutmix', 'mixup', 'cutout', 'weak_image_augmentation']:
    assert hasattr(augmentation, fn_name), f'FAIL: {fn_name} missing'
    print(f'  - augmentation.{fn_name}: OK')
print('OK: all augmentation functions exist')
"

echo ""
echo "=== [3/3] dataset_source accepts cutmix ==="
# 直接读源文件, 避开 inspect.getsource() 对继承 __init__ 的 wrapper_descriptor 问题
python -c "
from pathlib import Path
src_path = Path.home() / 'sam' / 'sam_jax' / 'datasets' / 'dataset_source.py'
assert src_path.exists(), f'FAIL: {src_path} not found'
src = src_path.read_text()

# Check 1: assert relaxed to include mixup/mixcut/cutmix
assert \"'mixup'\" in src and \"'mixcut'\" in src and \"'cutmix'\" in src, \
    'FAIL: assert not updated for all 3 augmentations'
print('  - assert includes mixup/mixcut/cutmix: OK')

# Check 2: cutmix elif branch exists
assert 'augmentation.cutmix' in src, \
    'FAIL: cutmix branch missing (no augmentation.cutmix reference)'
print('  - elif cutmix branch exists: OK')

# Check 3: existing branches still there (compatibility)
for keyword in ['augmentation.cutout', 'augmentation.mixup']:
    assert keyword in src, f'FAIL: {keyword} branch broken'
print('  - existing cutout/mixup branches intact: OK')

print('OK: dataset_source accepts all augmentation types')
"

echo ""
echo "=== All checks PASSED ==="
echo "Safe to proceed to visualization check (visualize_augmentations.py)"
