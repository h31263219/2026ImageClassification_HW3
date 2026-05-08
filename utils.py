"""Utility helpers: RLE encoding, COCO-style evaluation, training meters."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils


def encode_mask(binary_mask: np.ndarray) -> Dict:
    """Encode a binary HxW mask into the COCO RLE format expected by the
    competition (uncompressed counts string)."""
    binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {"size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
            "counts": rle["counts"]}


def predictions_to_coco(
    pred: Dict[str, torch.Tensor],
    image_id: int,
    score_thresh: float = 0.05,
    mask_thresh: float = 0.5,
) -> List[Dict]:
    """Convert a single torchvision Mask R-CNN prediction dict into the
    COCO result list (one entry per instance)."""
    results: List[Dict] = []
    if pred["boxes"].numel() == 0:
        return results

    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()  # N x 1 x H x W

    for i in range(boxes.shape[0]):
        score = float(scores[i])
        if score < score_thresh:
            continue
        m = masks[i, 0] if masks.ndim == 4 else masks[i]
        binary = m >= mask_thresh
        if not binary.any():
            continue
        x1, y1, x2, y2 = boxes[i].tolist()
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        results.append({
            "image_id": int(image_id),
            "bbox": bbox,
            "score": score,
            "category_id": int(labels[i]),
            "segmentation": encode_mask(binary),
        })
    return results


def save_results(results: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f)


def make_gt_coco(
    dataset,
    indices: Iterable[int],
) -> Tuple[Dict, Dict[int, int]]:
    """Build a COCO-format ground-truth dictionary from a subset of a
    ``CellInstanceDataset``.

    Returns
    -------
    coco_dict : dict
        Standard COCO dataset dict (images, annotations, categories).
    sample_to_image_id : dict
        Maps the in-memory sample index to the assigned COCO image id.
    """
    images: List[Dict] = []
    annotations: List[Dict] = []
    sample_to_image_id: Dict[int, int] = {}
    ann_id = 1

    for image_id, idx in enumerate(indices, start=1):
        _, target = dataset[idx]
        sample_to_image_id[int(idx)] = image_id
        masks = target["masks"].numpy()  # N x H x W
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        if masks.size == 0:
            h = w = 0
        else:
            h, w = masks.shape[1], masks.shape[2]
        images.append({"id": image_id, "file_name": f"sample_{image_id}.tif",
                        "height": int(h), "width": int(w)})
        for n in range(masks.shape[0]):
            seg = encode_mask(masks[n])
            x1, y1, x2, y2 = boxes[n].tolist()
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(labels[n]),
                "segmentation": seg,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float(mask_utils.area(seg)),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [
        {"id": 1, "name": "class1"},
        {"id": 2, "name": "class2"},
        {"id": 3, "name": "class3"},
        {"id": 4, "name": "class4"},
    ]
    return {"images": images, "annotations": annotations, "categories": categories}, sample_to_image_id


def evaluate_ap50(
    coco_gt_dict: Dict,
    coco_results: List[Dict],
) -> Dict[str, float]:
    """Run COCO segmentation evaluation and return AP / AP50 / AP75."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not coco_results:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    gt_path = "_tmp_gt.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt_dict, f)
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    os.remove(gt_path)
    stats = coco_eval.stats
    return {"AP": float(stats[0]), "AP50": float(stats[1]), "AP75": float(stats[2])}


class AverageMeter:
    """Tracks an exponentially-weighted moving average for loss reporting."""

    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum = momentum
        self.value = None

    def update(self, x: float) -> None:
        if self.value is None:
            self.value = x
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * x

    def __float__(self) -> float:
        return float(self.value if self.value is not None else 0.0)
