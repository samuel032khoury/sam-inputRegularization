#!/usr/bin/env python3
"""Run a single trained checkpoint on the CIFAR-10 test set and dump
per-class accuracy + 10x10 confusion matrix + wrong-index list to JSON.

This version mirrors the model/optimizer/checkpoint API used by
google-research/sam (old flax.nn.Module + flax.optim) so that
checkpoints written by sam_jax.train can be restored without any
patching of the original repo.

Auto-walks: pass the top-level condition dir
(`/scratch1/$USER/sam_experiments/c1_sgd_baseline`); the script
discovers the leaf training dir
(`<top>/lr_*/wd_*/rho_*/seed_*/`) and the `checkpoints/` subdir
inside it.

Usage:
    python -m sam.scripts.eval_confusion \
        --ckpt_dir   /scratch1/$USER/sam_experiments/c1_sgd_baseline \
        --model_name WideResnet28x10 \
        --batch_size 200 \
        --out_json   /scratch1/$USER/sam_experiments/c1_sgd_baseline/confusion.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

# IMPORTANT: absl FLAGS must be marked-as-parsed BEFORE we import any
# repo module that references FLAGS at apply-time (wide_resnet.py reads
# FLAGS.use_additional_skip_connections inside Module.apply, and several
# other repo modules will eventually touch FLAGS too).  We do this
# unconditionally with a no-arg pseudo-parse — every flag keeps its
# default, which matches how train.py was invoked for our 10 conditions.
from absl import flags as _absl_flags
_absl_flags.FLAGS([sys.argv[0]])  # parse only argv[0] => all defaults

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import optim
from flax.training import checkpoints

# only the small handful of repo imports we actually need
from sam.sam_jax.datasets import dataset_source as ds_module
from sam.sam_jax.models import load_model


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def find_checkpoint_dir(top: Path) -> Path:
    """top = .../c1_sgd_baseline. We need the deepest 'checkpoints/' below it.

    Layout written by sam_jax.train:
      <top>/lr_<lr>/wd_<wd>/rho_<rho>/seed_<seed>/checkpoints/checkpoint_<epoch>
    """
    candidates = sorted(glob(str(top / "lr_*/wd_*/rho_*/seed_*/checkpoints")))
    if not candidates:
        # fall back: maybe user already gave the leaf
        if (top / "checkpoints").is_dir():
            return top / "checkpoints"
        # or the leaf without the 'checkpoints/' suffix
        ckpts = sorted(top.glob("checkpoint_*"))
        if ckpts:
            return top
        raise FileNotFoundError(
            f"No checkpoint dir found under {top} "
            f"(looked for lr_*/wd_*/rho_*/seed_*/checkpoints).")
    if len(candidates) > 1:
        print(f"[warn] multiple checkpoint dirs found, picking last:")
        for c in candidates:
            print(f"      {c}")
    return Path(candidates[-1])


def build_model(model_name: str, batch_size: int, image_size: int = 32,
                num_classes: int = 10):
    """Use the repo's own factory so the architecture matches training.

    NOTE: do NOT pass `prng_key=...`. The repo's `get_model` does
    `if not prng_key:` (load_model.py:115), which raises
    "truth value of array is ambiguous" when handed a jax.Array.
    Letting it default to None makes the repo build PRNGKey(0) itself.
    """
    return load_model.get_model(
        model_name=model_name,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
    )


def restore(model, state, ckpt_dir: Path):
    """Restore the trained Flax model + batch-norm state from disk.

    sam_jax.train wraps params in optim.Momentum; we mimic that exactly so
    the dict shape matches what was saved.
    """
    optimizer_def = optim.Momentum(
        learning_rate=0.1, beta=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = optimizer_def.create(model)
    train_state = dict(optimizer=optimizer, model_state=state, epoch=0)
    restored = checkpoints.restore_checkpoint(str(ckpt_dir), train_state)
    if restored is None or restored.get("optimizer") is None:
        raise FileNotFoundError(f"No checkpoint loaded from {ckpt_dir}")
    print(f"[eval_confusion] restored from epoch={restored.get('epoch')}")
    return restored["optimizer"].target, restored["model_state"]


@jax.jit
def predict(model, state, x):
    with flax.nn.stateful(state, mutable=False):
        logits = model(x, train=False)
    return logits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", required=True,
                   help="top-level condition dir, e.g. .../c1_sgd_baseline")
    p.add_argument("--model_name", default="WideResnet28x10")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    top = Path(args.ckpt_dir).expanduser().resolve()
    print(f"[eval_confusion] top dir = {top}")
    ckpt_dir = find_checkpoint_dir(top)
    print(f"[eval_confusion] ckpt dir = {ckpt_dir}")

    # 1) build skeleton model + state
    print("[eval_confusion] building model skeleton...")
    model, state = build_model(args.model_name, args.batch_size)

    # 2) restore trained weights from disk
    model, state = restore(model, state, ckpt_dir)

    # 3) build the CIFAR-10 test pipeline (none/none for clean inputs)
    print("[eval_confusion] loading CIFAR-10 test set...")
    src = ds_module.Cifar10(
        batch_size=args.batch_size,
        image_level_augmentations="none",
        batch_level_augmentations="none",
    )
    test_ds = src.get_test()

    # 4) inference loop
    print("[eval_confusion] running inference...")
    preds_all, labels_all = [], []
    for batch in test_ds:
        x = jnp.asarray(batch["image"])
        y_onehot = jnp.asarray(batch["label"])
        logits = predict(model, state, x)
        preds_all.append(np.asarray(jnp.argmax(logits, axis=-1)))
        labels_all.append(np.asarray(jnp.argmax(y_onehot, axis=-1)))
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)

    n = len(labels)
    correct = (preds == labels)
    acc = float(correct.mean())
    err = 1.0 - acc
    print(f"[eval_confusion] N={n}  acc={acc:.4f}  err={err:.4f}")

    # 5) confusion matrix + per-class breakdown
    cm = np.zeros((10, 10), dtype=np.int64)
    for t, p_ in zip(labels, preds):
        cm[t, p_] += 1
    per_class = {}
    for i, name in enumerate(CIFAR10_CLASSES):
        n_i = int(cm[i].sum())
        c_i = int(cm[i, i])
        per_class[name] = {
            "n": n_i,
            "correct": c_i,
            "errors": n_i - c_i,
            "acc": c_i / max(n_i, 1),
        }

    # 6) wrong-index list (for cross-condition diff)
    wrong_idx = np.where(~correct)[0].tolist()

    out = {
        "top_dir": str(top),
        "ckpt_dir": str(ckpt_dir),
        "n": int(n),
        "correct": int(correct.sum()),
        "errors": int((~correct).sum()),
        "accuracy": acc,
        "error_rate": err,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": CIFAR10_CLASSES,
        "wrong_indices": wrong_idx,
    }
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval_confusion] wrote {out_path}")


if __name__ == "__main__":
    main()
