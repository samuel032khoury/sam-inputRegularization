"""Visualize mixup / cutout / cutmix on real CIFAR-10 batches.

Layer 3 verification: by eye, confirm each augmentation behaves as expected.

Usage (on CARC, after loading env):
    python visualize_augmentations.py
    # Output: aug_check.png in current directory
    # scp to local:
    #   scp $USER@discovery.usc.edu:~/aug_check.png ~/Desktop/

What to look for (correct behavior):
    - Row 1 (Original):  normal CIFAR-10 images
    - Row 2 (Mixup):     SEMI-TRANSPARENT blend of two images (double-exposure look)
    - Row 3 (Cutout):    black rectangle somewhere in the image
    - Row 4 (CutMix):    a rectangle containing a DIFFERENT image pasted in
                         (NOT semi-transparent, NOT black -- a hard cut)

If CutMix row looks semi-transparent or black, the implementation is wrong.
If any row is identical to Original, the augmentation did nothing.
"""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CARC
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sam.sam_jax.datasets import augmentation


NUM_SAMPLES = 6  # images per row
NUM_CLASSES = 10


def one_hot(labels, num_classes=NUM_CLASSES):
  return tf.one_hot(tf.cast(labels, tf.int32), num_classes)


def load_batch(n=NUM_SAMPLES):
  """Load a small CIFAR-10 batch for visualization."""
  ds = tfds.load('cifar10', split='train', shuffle_files=False)
  ds = ds.batch(n).take(1)
  for ex in ds:
    images = tf.cast(ex['image'], tf.float32) / 255.0
    labels = one_hot(ex['label'])
    return {'image': images, 'label': labels}


def imshow_row(ax_row, images, title):
  ax_row[0].set_ylabel(title, fontsize=12, rotation=0, ha='right', va='center')
  for ax, img in zip(ax_row, images):
    img_np = np.clip(img.numpy(), 0, 1)
    ax.imshow(img_np)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
  tf.config.experimental.set_visible_devices([], 'GPU')  # CPU is enough
  batch = load_batch()
  images_orig = batch['image']

  # Apply each augmentation
  print('Applying mixup...')
  mixup_out = augmentation.mixup(batch, alpha=1.0)
  print('Applying cutout...')
  cutout_out = augmentation.cutout(batch)
  print('Applying cutmix...')
  cutmix_out = augmentation.cutmix(batch, alpha=1.0)

  # Print label sanity check
  print('')
  print('=== Label sanity check (should all be close to 1.0) ===')
  print(f"Original labels (one row):  sum = {tf.reduce_sum(batch['label'][0]).numpy():.3f}")
  print(f"Mixup labels   (one row):  sum = {tf.reduce_sum(mixup_out['label'][0]).numpy():.3f}")
  print(f"CutMix labels  (one row):  sum = {tf.reduce_sum(cutmix_out['label'][0]).numpy():.3f}")
  print('')
  print(f"Mixup  first label mix:  {mixup_out['label'][0].numpy().round(3)}")
  print(f"CutMix first label mix:  {cutmix_out['label'][0].numpy().round(3)}")
  print('  (should see 2 non-zero entries -- the two mixed classes)')
  print('')

  # Plot
  fig, axes = plt.subplots(4, NUM_SAMPLES, figsize=(NUM_SAMPLES * 1.5, 6.5))
  imshow_row(axes[0], images_orig, 'Original')
  imshow_row(axes[1], mixup_out['image'], 'Mixup')
  imshow_row(axes[2], cutout_out['image'], 'Cutout')
  imshow_row(axes[3], cutmix_out['image'], 'CutMix')
  plt.suptitle('Augmentation sanity check\n'
               '(Mixup=blend, Cutout=black box, CutMix=pasted patch)',
               fontsize=11)
  plt.tight_layout(rect=[0.05, 0, 1, 0.95])
  out_path = 'aug_check.png'
  plt.savefig(out_path, dpi=120, bbox_inches='tight')
  print(f'Saved: {out_path}')
  print('')
  print('Next step:')
  print(f'  scp $USER@discovery.usc.edu:$PWD/{out_path} ~/Desktop/')
  print('Then open ~/Desktop/aug_check.png and verify each row visually.')


if __name__ == '__main__':
  main()
