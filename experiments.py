"""Additional experiments grounded in the trained model + dataset.

We do not have time/budget for several full-retrain ablations, so this
script runs three architectural / data-driven analyses that are still
empirical (hypothesis → measurement → implication):

(E1) Anchor-scale coverage. Compute the distribution of GT instance
     sizes (sqrt(area) of the box). Quantify how many of them are
     smaller than torchvision's default smallest anchor (32 px) — this
     directly motivates the custom (8, 16, 32, ...) anchor stack.

(E2) Score-threshold sweep at inference. Run the trained model once on
     the validation set with a permissive score_thresh, then re-evaluate
     COCO segm AP50 at increasing thresholds (0.05 → 0.7). This shows
     the precision–recall trade-off and locates the optimal operating
     point.

(E3) Class-imbalance vs. per-class accuracy. Cross-tabulate the number
     of training instances per class with the confusion-matrix
     diagonal/recall on validation. Quantifies how much the long-tail
     class4 hurts performance.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import (
    CLASS_NAMES,
    CellInstanceDataset,
    list_train_samples,
    split_train_val,
)
from model import build_model
from utils import evaluate_ap50, make_gt_coco, predictions_to_coco


# ---------------------------------------------------------------------------
# E1: anchor-scale coverage
# ---------------------------------------------------------------------------
def experiment_anchor_scales(args: argparse.Namespace) -> None:
    samples = list_train_samples(args.data_dir)
    train_ids, _ = split_train_val(samples, args.val_ratio, args.seed)
    ds = CellInstanceDataset(args.data_dir, train_ids, augment=False)

    sizes_by_class: Dict[str, List[float]] = {c: [] for c in CLASS_NAMES}
    for i in range(len(ds)):
        _, t = ds[i]
        boxes = t["boxes"].numpy()
        labels = t["labels"].numpy()
        for b, lbl in zip(boxes, labels):
            w, h = b[2] - b[0], b[3] - b[1]
            sizes_by_class[CLASS_NAMES[int(lbl) - 1]].append(float(np.sqrt(w * h)))

    all_sizes = np.concatenate([np.asarray(v) for v in sizes_by_class.values()])
    n = len(all_sizes)
    cuts = [8, 16, 32, 64, 128]
    pct = {c: 100.0 * (all_sizes <= c).mean() for c in cuts}
    print("Instance size distribution (sqrt(box area), pixels)")
    print(f"  total instances:        {n}")
    print(f"  median sqrt(area):      {np.median(all_sizes):.1f}")
    print(f"  10th / 90th percentile: "
          f"{np.percentile(all_sizes, 10):.1f} / {np.percentile(all_sizes, 90):.1f}")
    for c in cuts:
        print(f"  fraction <= {c:>3} px:    {pct[c]:5.1f}%")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 200, 60)
    bottom = np.zeros(len(bins) - 1)
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange"]
    for c, color in zip(CLASS_NAMES, colors):
        h, _ = np.histogram(sizes_by_class[c], bins=bins)
        ax.bar(bins[:-1], h, width=np.diff(bins), bottom=bottom,
               label=f"{c} (n={len(sizes_by_class[c])})",
               color=color, edgecolor="white", linewidth=0.4)
        bottom += h
    for c in cuts:
        ax.axvline(c, color="grey", linestyle="--", alpha=0.6)
        ax.text(c, ax.get_ylim()[1] * 0.92, f"{c}px",
                ha="center", va="top", fontsize=8, color="grey")
    ax.set_xlabel("sqrt(box area)  [pixels]")
    ax.set_ylabel("# instances")
    ax.set_title("E1: GT instance-size distribution motivates 8/16-px anchors")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(args.output_dir, "exp_anchor_sizes.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    return {"pct_le": pct, "median": float(np.median(all_sizes)), "total": int(n)}


# ---------------------------------------------------------------------------
# E2: score-threshold sweep on val set
# ---------------------------------------------------------------------------
@torch.no_grad()
def experiment_score_threshold(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = list_train_samples(args.data_dir)
    _, val_ids = split_train_val(samples, args.val_ratio, args.seed)
    val_set = CellInstanceDataset(args.data_dir, val_ids, augment=False)

    # Use a very permissive score so we keep all candidates; we filter
    # later in Python.
    model = build_model(num_classes=5, pretrained_backbone=False,
                        box_score_thresh=0.01, box_detections_per_img=600)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    coco_gt, sample_to_image_id = make_gt_coco(val_set, range(len(val_set)))
    raw_results: List[dict] = []
    for idx in range(len(val_set)):
        image, _ = val_set[idx]
        image_id = sample_to_image_id[int(idx)]
        out = model([image.to(device)])[0]
        raw_results.extend(predictions_to_coco(out, image_id, score_thresh=0.0))
    print(f"Total raw predictions across val set: {len(raw_results)}")

    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    rows = []
    for t in thresholds:
        kept = [r for r in raw_results if r["score"] >= t]
        m = evaluate_ap50(coco_gt, kept) if kept else {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        rows.append({"thresh": t, "n": len(kept),
                     "AP": m["AP"], "AP50": m["AP50"], "AP75": m["AP75"]})
        print(f"  score_thresh={t:.2f}  n={len(kept):5d}  "
              f"AP={m['AP']:.4f}  AP50={m['AP50']:.4f}  AP75={m['AP75']:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ts = [r["thresh"] for r in rows]
    ax.plot(ts, [r["AP50"] for r in rows], marker="o", label="AP50")
    ax.plot(ts, [r["AP"] for r in rows], marker="s", label="AP")
    ax.plot(ts, [r["AP75"] for r in rows], marker="^", label="AP75")
    ax.set_xlabel("Inference score threshold")
    ax.set_ylabel("Validation segm AP")
    ax.set_title("E2: AP vs. inference score threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(args.output_dir, "exp_score_threshold.png")
    plt.savefig(out, dpi=150)
    out_json = os.path.join(args.output_dir, "exp_score_threshold.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")
    print(f"Saved {out_json}")


# ---------------------------------------------------------------------------
# E3: class-imbalance vs. per-class recall
# ---------------------------------------------------------------------------
def experiment_class_imbalance(args: argparse.Namespace) -> None:
    samples = list_train_samples(args.data_dir)
    train_ids, _ = split_train_val(samples, args.val_ratio, args.seed)
    train_set = CellInstanceDataset(args.data_dir, train_ids, augment=False)

    # Count instances + images per class in the training split.
    inst_per_class = {c: 0 for c in CLASS_NAMES}
    img_per_class = {c: 0 for c in CLASS_NAMES}
    for i in range(len(train_set)):
        _, t = train_set[i]
        labels = t["labels"].numpy()
        seen = set()
        for lbl in labels:
            cname = CLASS_NAMES[int(lbl) - 1]
            inst_per_class[cname] += 1
            seen.add(cname)
        for cname in seen:
            img_per_class[cname] += 1

    # Read confusion matrix produced by visualize.py confmat.
    cm_path = os.path.join(args.output_dir, "confusion_matrix.npy")
    cm = np.load(cm_path)
    # rows = GT, cols = pred (4 classes + bg/miss). recall = diag / row-sum.
    recalls = {}
    for k, c in enumerate(CLASS_NAMES):
        row = cm[k]
        recalls[c] = float(row[k]) / float(row.sum() + 1e-9)

    print(f"{'class':<8}  {'#train_inst':>11}  {'#train_imgs':>11}  {'recall@IoU0.5':>14}")
    for c in CLASS_NAMES:
        print(f"{c:<8}  {inst_per_class[c]:>11}  {img_per_class[c]:>11}  "
              f"{recalls[c]:>14.3f}")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    ax1.bar(x - 0.18, [inst_per_class[c] for c in CLASS_NAMES], width=0.36,
            label="# train instances", color="tab:blue")
    ax1.set_ylabel("# training instances", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x); ax1.set_xticklabels(CLASS_NAMES)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, [recalls[c] for c in CLASS_NAMES], width=0.36,
            label="val recall@IoU0.5", color="tab:orange")
    ax2.set_ylabel("Validation recall (IoU≥0.5)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1)
    plt.title("E3: training-set frequency vs. per-class validation recall")
    plt.tight_layout()
    out = os.path.join(args.output_dir, "exp_class_imbalance.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./hw3-data-release")
    parser.add_argument("--checkpoint", default="./output/best_ap50_model.pth")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exps", nargs="+",
                        default=["anchor", "score", "imbalance"],
                        choices=["anchor", "score", "imbalance"])
    args = parser.parse_args()

    if "anchor" in args.exps:
        print("\n=== Experiment 1: anchor coverage ===")
        experiment_anchor_scales(args)
    if "score" in args.exps:
        print("\n=== Experiment 2: score-threshold sweep ===")
        experiment_score_threshold(args)
    if "imbalance" in args.exps:
        print("\n=== Experiment 3: class imbalance ===")
        experiment_class_imbalance(args)


if __name__ == "__main__":
    main()
