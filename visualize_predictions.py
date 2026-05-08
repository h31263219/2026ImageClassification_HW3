"""Render predicted instance masks alongside ground truth on val images.

Usage
-----
    python visualize_predictions.py --data_dir ./hw3-data-release \
        --checkpoint ./output/best_ap50_model.pth \
        --out ./output/qualitative.png
"""

from __future__ import annotations

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CellInstanceDataset, list_train_samples, split_train_val
from model import build_model

CLASS_NAMES = ["bg", "class1", "class2", "class3", "class4"]
PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],     # class1 red
        [60, 180, 75],     # class2 green
        [0, 130, 200],     # class3 blue
        [245, 130, 48],    # class4 orange
    ],
    dtype=np.uint8,
)


def _overlay(image: np.ndarray, masks: np.ndarray, labels: np.ndarray,
             alpha: float = 0.45) -> np.ndarray:
    """Blend instance masks into an HxWx3 uint8 image."""
    out = image.astype(np.float32).copy()
    for m, lbl in zip(masks, labels):
        color = PALETTE[int(lbl)].astype(np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./hw3-data-release")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument("--num_images", type=int, default=4)
    parser.add_argument("--out", default="./output/qualitative.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_train_samples(args.data_dir)
    _, val_ids = split_train_val(samples, args.val_ratio, args.seed)
    val_set = CellInstanceDataset(args.data_dir, val_ids, augment=False)

    model = build_model(num_classes=5, pretrained_backbone=False,
                        box_score_thresh=args.score_thresh)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    n = min(args.num_images, len(val_set))
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.0 * n))
    if n == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r in range(n):
            image, target = val_set[r]
            img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_masks = target["masks"].numpy().astype(bool)
            gt_labels = target["labels"].numpy()

            out = model([image.to(device)])[0]
            scores = out["scores"].detach().cpu().numpy()
            keep = scores >= args.score_thresh
            pred_masks = (out["masks"].detach().cpu().numpy()[:, 0]
                          >= args.mask_thresh)
            pred_masks = pred_masks[keep]
            pred_labels = out["labels"].detach().cpu().numpy()[keep]

            axes[r, 0].imshow(img_np)
            axes[r, 0].set_title(f"Image  ({val_set.sample_ids[r][:8]}…)")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(_overlay(img_np, gt_masks, gt_labels))
            axes[r, 1].set_title(f"Ground truth ({len(gt_labels)} inst)")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(_overlay(img_np, pred_masks, pred_labels))
            axes[r, 2].set_title(
                f"Prediction ({int(keep.sum())} inst, "
                f"score≥{args.score_thresh:.2f})"
            )
            axes[r, 2].axis("off")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=PALETTE[c] / 255.0,
                      label=CLASS_NAMES[c])
        for c in range(1, 5)
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.005))
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
