"""Visualisation helpers: training curves and per-class confusion matrix.

Usage
-----
Plot training curves (train loss + val AP50 over epochs):

    python visualize.py curves --history ./output/history.json \
                                --out ./output/training_curves.png

Compute and plot a class confusion matrix on the validation set. The
matrix is built by matching every predicted instance to the GT instance
with the highest mask-IoU (>= ``--iou_thresh``) and recording the pair
(true class, predicted class). Predictions with no GT match go to a
``background`` row, and missed GT instances go to a ``missed`` column.

    python visualize.py confmat --data_dir ./hw3-data-release \
                                 --checkpoint ./output/best_ap50_model.pth \
                                 --out ./output/confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


CLASSES = ["class1", "class2", "class3", "class4"]


def cmd_curves(args: argparse.Namespace) -> None:
    with open(args.history, "r", encoding="utf-8") as f:
        history = json.load(f)
    epochs = [r["epoch"] for r in history]
    loss = [r.get("loss", None) for r in history]
    val_ap50 = [r.get("val_AP50", None) for r in history]
    val_ap = [r.get("val_AP", None) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, loss, label="train loss", color="tab:blue")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].grid(True, alpha=0.3)

    eval_ep = [e for e, v in zip(epochs, val_ap50) if v is not None]
    eval_ap50 = [v for v in val_ap50 if v is not None]
    eval_ap = [v for v in val_ap if v is not None]
    axes[1].plot(eval_ep, eval_ap50, marker="o", color="tab:orange",
                 label="val AP50")
    if any(v is not None for v in eval_ap):
        axes[1].plot(eval_ep, eval_ap, marker="s", color="tab:green",
                     label="val AP")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved training curves to {args.out}")


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def cmd_confmat(args: argparse.Namespace) -> None:
    import torch
    from dataset import CellInstanceDataset, list_train_samples, split_train_val
    from model import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_train_samples(args.data_dir)
    _, val_ids = split_train_val(samples, args.val_ratio, args.seed)
    val_set = CellInstanceDataset(args.data_dir, val_ids, augment=False)

    model = build_model(num_classes=5, pretrained_backbone=False,
                        box_score_thresh=args.score_thresh)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    n = len(CLASSES)
    # Rows = true class (4 classes + 1 "missed" row for FN-like accounting?)
    # Actually we use rows = true (1..4 + bg-FP), cols = pred (1..4 + missed-FN).
    # Build a (n+1) x (n+1) matrix where the extra row/col is "background".
    cm = np.zeros((n + 1, n + 1), dtype=np.int64)
    bg = n  # index of background

    with torch.no_grad():
        for idx in range(len(val_set)):
            image, target = val_set[idx]
            out = model([image.to(device)])[0]

            gt_masks = target["masks"].numpy().astype(bool)
            gt_labels = target["labels"].numpy()  # 1..4

            pred_masks = (out["masks"].detach().cpu().numpy()[:, 0]
                          >= args.mask_thresh) if out["masks"].numel() else np.zeros((0,))
            pred_scores = out["scores"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy()

            keep = pred_scores >= args.score_thresh
            pred_masks = pred_masks[keep] if pred_masks.size else pred_masks
            pred_labels = pred_labels[keep]

            matched_gt = set()
            for j in range(len(pred_labels)):
                best_iou, best_g = 0.0, -1
                for g in range(len(gt_labels)):
                    if g in matched_gt:
                        continue
                    iou = _mask_iou(pred_masks[j], gt_masks[g])
                    if iou > best_iou:
                        best_iou, best_g = iou, g
                if best_iou >= args.iou_thresh and best_g >= 0:
                    cm[gt_labels[best_g] - 1, pred_labels[j] - 1] += 1
                    matched_gt.add(best_g)
                else:
                    cm[bg, pred_labels[j] - 1] += 1  # false positive

            for g in range(len(gt_labels)):
                if g not in matched_gt:
                    cm[gt_labels[g] - 1, bg] += 1  # missed (false negative)

    labels = CLASSES + ["bg / miss"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n + 1)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n + 1)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Confusion matrix (IoU >= {args.iou_thresh})")
    for i in range(n + 1):
        for j in range(n + 1):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"Saved confusion matrix to {args.out}")
    np.save(args.out.replace(".png", ".npy"), cm)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("curves", help="Plot training curves")
    p1.add_argument("--history", default="./output/history.json")
    p1.add_argument("--out", default="./output/training_curves.png")
    p1.set_defaults(func=cmd_curves)

    p2 = sub.add_parser("confmat", help="Plot confusion matrix on val set")
    p2.add_argument("--data_dir", default="./hw3-data-release")
    p2.add_argument("--checkpoint", required=True)
    p2.add_argument("--val_ratio", type=float, default=0.1)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--score_thresh", type=float, default=0.5)
    p2.add_argument("--mask_thresh", type=float, default=0.5)
    p2.add_argument("--iou_thresh", type=float, default=0.5)
    p2.add_argument("--out", default="./output/confusion_matrix.png")
    p2.set_defaults(func=cmd_confmat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
