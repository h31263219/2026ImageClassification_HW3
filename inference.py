"""Inference script: produces ``test-results.json`` and a submission zip."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from dataset import CellTestDataset, test_collate_fn
from model import build_model
from utils import predictions_to_coco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for HW3")
    parser.add_argument("--data_dir", type=str, default="./hw3-data-release")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101"])
    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--zip_name", type=str, default="submission.zip")
    return parser.parse_args()


@torch.no_grad()
def run_inference(args: argparse.Namespace) -> List[Dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CellTestDataset(args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
    )

    model = build_model(
        num_classes=5,
        backbone_name=args.backbone,
        pretrained_backbone=False,
        box_score_thresh=args.score_threshold,
    )
    state = torch.load(args.checkpoint, map_location=device)
    state_dict = state["model"] if "model" in state else state
    model.load_state_dict(state_dict)
    model.to(device).eval()

    results: List[Dict] = []
    for i, (images, metas) in enumerate(loader):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, meta in zip(outputs, metas):
            results.extend(
                predictions_to_coco(
                    out,
                    image_id=meta["image_id"],
                    score_thresh=args.score_threshold,
                    mask_thresh=args.mask_threshold,
                )
            )
        if (i + 1) % 10 == 0 or (i + 1) == len(loader):
            print(f"  processed {i+1}/{len(loader)} (total instances: {len(results)})")
    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = run_inference(args)
    json_path = os.path.join(args.output_dir, "test-results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} instances -> {json_path}")

    zip_path = os.path.join(args.output_dir, args.zip_name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="test-results.json")
    print(f"Submission zip -> {zip_path}")


if __name__ == "__main__":
    main()
