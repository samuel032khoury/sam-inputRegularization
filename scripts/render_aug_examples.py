#!/usr/bin/env python3
"""Render a 1x6 figure showing the four augmentations applied to one
CIFAR-10 sample, for the Methods section of the report.

Layout:
    [orig A] [orig B] [Cutout] [Mixup] [CutMix] [mixcut]

Pure numpy + matplotlib + tensorflow_datasets (no jax / flax dependency).
Run on your local Mac with the .venv activated:

    python scripts/render_aug_examples.py --out report/aug_examples.pdf

If you don't have tensorflow_datasets locally, use --use_offline to load
two .npy CIFAR-10 samples from --offline_dir (you can pre-dump them
with the snippet at the bottom of this file).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------- pure-numpy reimplementations of the four augs -------------
def cutout(img: np.ndarray, size: int = 16,
           rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    H, W, _ = img.shape
    cy = rng.integers(0, H)
    cx = rng.integers(0, W)
    y0, y1 = max(0, cy - size // 2), min(H, cy + size // 2)
    x0, x1 = max(0, cx - size // 2), min(W, cx + size // 2)
    out = img.copy()
    out[y0:y1, x0:x1, :] = 0.0
    return out


def mixup(a: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    return (lam * a + (1 - lam) * b).astype(a.dtype)


def cutmix(a: np.ndarray, b: np.ndarray, lam: float,
           rng: np.random.Generator | None = None) -> tuple[np.ndarray, float]:
    """Real Yun 2019 CutMix: pastes a random rect from b onto a, with the
    label weight = actual pasted area (which may differ from sampled lam if
    the box is clipped at the boundary)."""
    rng = rng or np.random.default_rng(0)
    H, W, _ = a.shape
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = rng.integers(0, H)
    cx = rng.integers(0, W)
    y0 = max(0, cy - cut_h // 2)
    y1 = min(H, cy + cut_h // 2)
    x0 = max(0, cx - cut_w // 2)
    x1 = min(W, cx + cut_w // 2)
    out = a.copy()
    out[y0:y1, x0:x1, :] = b[y0:y1, x0:x1, :]
    actual_lam = 1 - ((y1 - y0) * (x1 - x0)) / (H * W)
    return out, actual_lam


def mixcut(a: np.ndarray, b: np.ndarray, lam: float,
           rng: np.random.Generator | None = None) -> np.ndarray:
    """sam_jax/dataset_source.py's mixcut: cutout(mixup(a, b))."""
    return cutout(mixup(a, b, lam), rng=rng)


# ----------- CIFAR-10 sample loader ---------------------------------------
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def load_two_samples(use_offline: bool, offline_dir: Path,
                     seed: int = 7) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return (img_a, img_b, label_a_name, label_b_name) — both float in [0,1]."""
    rng = np.random.default_rng(seed)
    if use_offline:
        a = np.load(offline_dir / "sample_a.npy")
        b = np.load(offline_dir / "sample_b.npy")
        # filename hints the class (sample_a_cat.npy etc) is up to user
        return a / 255.0, b / 255.0, "A", "B"

    import tensorflow_datasets as tfds
    ds = tfds.load("cifar10", split="train", as_supervised=False).take(20)
    rows = list(ds)
    # pick two distinct classes that visually differ — say cat vs ship
    pick = {"cat": None, "ship": None}
    for r in rows:
        lbl = int(r["label"])
        name = CIFAR10_NAMES[lbl]
        if name in pick and pick[name] is None:
            pick[name] = r["image"].numpy().astype(np.float32) / 255.0
        if all(v is not None for v in pick.values()):
            break
    if any(v is None for v in pick.values()):
        # fallback: just take the first two
        pick = {}
        for i, r in enumerate(rows[:2]):
            pick[CIFAR10_NAMES[int(r["label"])]] = r["image"].numpy() / 255.0
    (na, a), (nb, b) = list(pick.items())[:2]
    return a, b, na, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output PDF/PNG path")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--use_offline", action="store_true",
                    help="load images from --offline_dir/sample_{a,b}.npy")
    ap.add_argument("--offline_dir", type=str,
                    default="scripts/aug_offline_samples")
    ap.add_argument("--lam", type=float, default=None,
                    help="override Mixup/CutMix λ (default 0.5 for vis)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    a, b, na, nb = load_two_samples(args.use_offline,
                                    Path(args.offline_dir).expanduser(),
                                    seed=args.seed)
    print(f"Sample A: {na}   Sample B: {nb}")

    # NOTE: in real training, λ ~ Beta(1,1) which spans [0,1] uniformly.
    # For VISUALIZATION we hard-code λ ≈ 0.5 so the blended panels actually
    # *look* blended (Beta-sampled λ near 0 or 1 makes Mixup look like a
    # plain copy of one input).  The CLI flag --lam lets you override.
    lam = args.lam if args.lam is not None else 0.5

    img_cutout = cutout(a, size=16, rng=np.random.default_rng(args.seed + 1))
    img_mixup = mixup(a, b, lam)
    img_cutmix, cm_actual = cutmix(a, b, lam,
                                    rng=np.random.default_rng(args.seed + 2))
    img_mixcut = mixcut(a, b, lam,
                        rng=np.random.default_rng(args.seed + 3))

    panels = [
        ("(a) original A", a, na),
        ("(b) original B", b, nb),
        (f"(c) Cutout", img_cutout, na),
        (f"(d) Mixup λ={lam:.2f}", img_mixup, f"{na}/{nb}"),
        (f"(e) CutMix λ={cm_actual:.2f}", img_cutmix, f"{na}/{nb}"),
        (f"(f) mixcut λ={lam:.2f}", img_mixcut, f"{na}/{nb}"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(9.0, 1.85))
    for ax, (title, img, lbl) in zip(axes, panels):
        # interpolation='nearest' so 32x32 CIFAR-10 pixels stay sharp
        # instead of being bilinearly smeared into a blur.
        ax.imshow(np.clip(img, 0, 1), interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(lbl, fontsize=7, color="#4A5575")
        ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.10,
                        wspace=0.06)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format=out.suffix.lstrip("."), bbox_inches="tight")
    print(f"Wrote {out}")
    if out.suffix == ".pdf":
        png = out.with_suffix(".png")
        fig.savefig(png, format="png", dpi=180, bbox_inches="tight")
        print(f"Wrote {png}")


if __name__ == "__main__":
    main()


# ----------- one-time: dump 2 CIFAR-10 samples to .npy (offline use) -----
# python -c "
# import tensorflow_datasets as tfds, numpy as np, pathlib
# ds = list(tfds.load('cifar10', split='train', as_supervised=False).take(20))
# names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
# pick = {'cat': None, 'ship': None}
# for r in ds:
#     n = names[int(r['label'])]
#     if n in pick and pick[n] is None:
#         pick[n] = r['image'].numpy()
# pathlib.Path('scripts/aug_offline_samples').mkdir(parents=True, exist_ok=True)
# np.save('scripts/aug_offline_samples/sample_a.npy', pick['cat'])
# np.save('scripts/aug_offline_samples/sample_b.npy', pick['ship'])
# "
