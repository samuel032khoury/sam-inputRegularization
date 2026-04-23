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
python -c "
import inspect
from sam.sam_jax.datasets import dataset_source
src = inspect.getsource(dataset_source.TFDSDatasetSource.__init__)
assert \"'cutmix'\" in src, 'FAIL: cutmix branch missing in dataset_source.py'
# Check the assert has been relaxed
assert 'mixup' in src and 'mixcut' in src and 'cutmix' in src, \
    'FAIL: assert not updated'
print('OK: dataset_source accepts all augmentation types')
"

echo ""
echo "=== All checks PASSED ==="
echo "Safe to proceed to visualization check (visualize_augmentations.py)"
