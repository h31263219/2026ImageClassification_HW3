"""Print per-module trainable-parameter counts and save a bar chart.

This script is referenced in the report to prove the model stays well
under the 200M trainable-parameter cap mandated by the homework spec.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt

from model import build_model, count_trainable_parameters


def _group_params(model) -> dict:
    """Aggregate trainable parameters by top-level module."""
    groups = {
        "backbone.body (ResNet-50)": 0,
        "backbone.fpn": 0,
        "rpn (anchor gen + RPN head)": 0,
        "roi_heads.box_head": 0,
        "roi_heads.box_predictor": 0,
        "roi_heads.mask_head": 0,
        "roi_heads.mask_predictor": 0,
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        if name.startswith("backbone.body"):
            groups["backbone.body (ResNet-50)"] += n
        elif name.startswith("backbone.fpn"):
            groups["backbone.fpn"] += n
        elif name.startswith("rpn"):
            groups["rpn (anchor gen + RPN head)"] += n
        elif name.startswith("roi_heads.box_head"):
            groups["roi_heads.box_head"] += n
        elif name.startswith("roi_heads.box_predictor"):
            groups["roi_heads.box_predictor"] += n
        elif name.startswith("roi_heads.mask_head"):
            groups["roi_heads.mask_head"] += n
        elif name.startswith("roi_heads.mask_predictor"):
            groups["roi_heads.mask_predictor"] += n
        else:
            groups.setdefault("other", 0)
            groups["other"] += n
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./output/model_size.png")
    args = parser.parse_args()

    model = build_model(num_classes=5, pretrained_backbone=False)
    total = count_trainable_parameters(model)
    print(f"Total trainable parameters: {total:,} ({total/1e6:.2f} M)")
    print(f"Cap allowed by spec       : 200,000,000 (200.00 M)")
    print(f"Headroom                  : {(200_000_000 - total)/1e6:.2f} M\n")

    groups = _group_params(model)
    for k, v in groups.items():
        print(f"  {k:35s} {v:>12,}  ({v/1e6:5.2f} M)")

    labels = list(groups.keys())
    values = [groups[k] / 1e6 for k in labels]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, values, color="tab:blue")
    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_title(
        f"Trainable parameters per module — total {total/1e6:.2f} M  "
        f"(< 200 M cap)"
    )
    ax.axvline(200, color="red", linestyle="--", label="200 M cap")
    ax.legend(loc="lower right")
    for bar, v in zip(bars, values):
        ax.text(v + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}M", va="center", fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
