"""Training entry point for HW3 instance segmentation."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from dataset import (
    CellInstanceDataset,
    collate_fn,
    list_train_samples,
    split_train_val,
)
from model import build_model, count_trainable_parameters
from utils import (
    AverageMeter,
    evaluate_ap50,
    make_gt_coco,
    predictions_to_coco,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN for HW3")
    parser.add_argument("--data_dir", type=str, default="./hw3-data-release")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="resnet50",
                        choices=["resnet50", "resnet101"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--default_anchors", action="store_true",
                        help="Use the default torchvision anchors "
                             "{32,64,128,256,512} instead of the custom "
                             "{8,16,32,64,128} stack — for the §4.1 ablation.")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed-precision training")
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Two parameter groups so we can use a smaller LR on the backbone."""
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(p)
        else:
            head_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )


def warmup_lr_factor(step: int, warmup_iters: int) -> float:
    if step >= warmup_iters:
        return 1.0
    return max(1e-3, step / max(1, warmup_iters))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step_ref: List[int],
) -> Dict[str, float]:
    model.train()
    meters = {k: AverageMeter() for k in
              ("loss", "loss_classifier", "loss_box_reg",
               "loss_mask", "loss_objectness", "loss_rpn_box_reg")}

    base_lrs = [g["lr"] for g in optimizer.param_groups]
    n_batches = len(loader)
    start = time.time()

    for i, (images, targets) in enumerate(loader):
        # Skip rare empty annotations (Mask R-CNN crashes on them).
        if any(t["boxes"].shape[0] == 0 for t in targets):
            continue

        global_step = global_step_ref[0]
        factor = warmup_lr_factor(global_step, args.warmup_iters)
        for g, base in zip(optimizer.param_groups, base_lrs):
            g["lr"] = base * factor

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in t.items() if k != "sample_id"} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=not args.no_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if not torch.isfinite(loss):
            print(f"  [skip] non-finite loss at step {global_step}")
            global_step_ref[0] += 1
            continue

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        meters["loss"].update(float(loss.item()))
        for k, v in loss_dict.items():
            if k in meters:
                meters[k].update(float(v.item()))

        if i % 20 == 0:
            elapsed = time.time() - start
            print(
                f"Epoch {epoch} [{i:4d}/{n_batches}] "
                f"loss={float(meters['loss']):.4f} "
                f"cls={float(meters['loss_classifier']):.4f} "
                f"box={float(meters['loss_box_reg']):.4f} "
                f"mask={float(meters['loss_mask']):.4f} "
                f"rpn_obj={float(meters['loss_objectness']):.4f} "
                f"rpn_box={float(meters['loss_rpn_box_reg']):.4f} "
                f"lr={optimizer.param_groups[1]['lr']:.2e} "
                f"({elapsed:.1f}s)"
            )
        global_step_ref[0] += 1

    return {k: float(m) for k, m in meters.items()}


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_dataset: CellInstanceDataset,
    val_indices: List[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    coco_gt, sample_to_image_id = make_gt_coco(val_dataset, range(len(val_dataset)))
    results: List[Dict] = []
    for idx in range(len(val_dataset)):
        image, target = val_dataset[idx]
        image_id = sample_to_image_id[int(idx)]
        out = model([image.to(device)])[0]
        results.extend(predictions_to_coco(out, image_id, score_thresh=0.05))
    metrics = evaluate_ap50(coco_gt, results)
    return metrics


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    samples = list_train_samples(args.data_dir)
    train_ids, val_ids = split_train_val(samples, args.val_ratio, args.seed)
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")

    train_set = CellInstanceDataset(args.data_dir, train_ids, augment=True)
    val_set = CellInstanceDataset(args.data_dir, val_ids, augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    model = build_model(
        num_classes=5,
        backbone_name=args.backbone,
        pretrained_backbone=not args.no_pretrained,
        default_anchors=args.default_anchors,
    )
    print(f"Anchors: {'default torchvision' if args.default_anchors else 'custom (8/16/32/64/128)'}")
    n_params = count_trainable_parameters(model)
    print(f"Trainable parameters: {n_params/1e6:.2f}M")
    assert n_params < 200_000_000, "Model exceeds 200M trainable params!"
    model.to(device)

    optimizer = build_optimizer(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)

    start_epoch = 0
    best_ap50 = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_ap50 = ckpt.get("best_ap50", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best AP50={best_ap50:.4f})")

    history: List[Dict] = []
    global_step = [0]

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, args, global_step,
        )

        record = {"epoch": epoch, **train_metrics}
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            val_metrics = validate(model, val_set, list(range(len(val_set))), device)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            print(f"  >> val AP={val_metrics['AP']:.4f}  AP50={val_metrics['AP50']:.4f}  "
                  f"AP75={val_metrics['AP75']:.4f}")

            if val_metrics["AP50"] > best_ap50:
                best_ap50 = val_metrics["AP50"]
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "best_ap50": best_ap50},
                           os.path.join(args.output_dir, "best_ap50_model.pth"))
                print(f"  >> new best AP50={best_ap50:.4f}, checkpoint saved")

        history.append(record)
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_ap50": best_ap50},
                os.path.join(args.output_dir, "last_model.pth"),
            )

    print(f"Training done. Best AP50={best_ap50:.4f}")


if __name__ == "__main__":
    main()
