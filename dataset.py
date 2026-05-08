"""Dataset utilities for HW3 instance segmentation.

The training data is organized as

    train/
        <image_id>/
            image.tif
            class1.tif   # optional, instance mask (each unique non-zero
            class2.tif   # value is one instance of that class)
            class3.tif
            class4.tif

The test data is

    test_release/
        <image_name>.tif
    test_image_name_to_ids.json   # maps file_name -> id

Each per-class mask is a single channel image where every distinct
non-zero pixel value represents one instance of that class. Background
is encoded as 0.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import skimage.io as sio
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


CLASS_NAMES: Tuple[str, ...] = ("class1", "class2", "class3", "class4")
NUM_CLASSES: int = len(CLASS_NAMES) + 1  # +1 for background


def _read_image(path: str) -> np.ndarray:
    """Read a TIFF image and return an HxWx3 uint8 RGB array."""
    img = sio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        if img.max() > 0:
            img = img / img.max() * 255.0
        img = img.astype(np.uint8)
    return img


def _read_mask(path: str) -> np.ndarray:
    """Read an instance mask. Returns a 2D int32 array."""
    mask = sio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int32)


def _masks_from_label_map(label_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a label map (each unique value = one instance) into a stack
    of binary masks and their corresponding bounding boxes.

    Returns
    -------
    masks : (N, H, W) uint8
    boxes : (N, 4) float32, format [x1, y1, x2, y2]
    """
    instance_ids = np.unique(label_map)
    instance_ids = instance_ids[instance_ids != 0]

    masks: List[np.ndarray] = []
    boxes: List[List[float]] = []
    for inst_id in instance_ids:
        m = (label_map == inst_id).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        # require positive area
        if x2 <= x1 or y2 <= y1:
            continue
        masks.append(m)
        boxes.append([x1, y1, x2 + 1.0, y2 + 1.0])

    if not masks:
        return (
            np.zeros((0, *label_map.shape), dtype=np.uint8),
            np.zeros((0, 4), dtype=np.float32),
        )
    return np.stack(masks, axis=0), np.asarray(boxes, dtype=np.float32)


def list_train_samples(data_dir: str) -> List[str]:
    """Return the sorted list of training sample folder names."""
    train_dir = os.path.join(data_dir, "train")
    samples = [
        d for d in sorted(os.listdir(train_dir))
        if os.path.isdir(os.path.join(train_dir, d))
    ]
    return samples


def split_train_val(
    samples: Sequence[str],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val


class CellInstanceDataset(Dataset):
    """Instance segmentation dataset for cell images."""

    def __init__(
        self,
        data_dir: str,
        sample_ids: Sequence[str],
        augment: bool = False,
    ) -> None:
        self.data_dir = data_dir
        self.sample_ids = list(sample_ids)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_targets(self, sample_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_masks: List[np.ndarray] = []
        all_boxes: List[np.ndarray] = []
        all_labels: List[int] = []
        for class_idx, class_name in enumerate(CLASS_NAMES, start=1):
            mask_path = os.path.join(sample_dir, f"{class_name}.tif")
            if not os.path.exists(mask_path):
                continue
            label_map = _read_mask(mask_path)
            masks, boxes = _masks_from_label_map(label_map)
            if masks.shape[0] == 0:
                continue
            all_masks.append(masks)
            all_boxes.append(boxes)
            all_labels.extend([class_idx] * masks.shape[0])

        if not all_masks:
            # Empty annotation - return zero-sized arrays.
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 1, 1), dtype=np.uint8),
            )
        masks = np.concatenate(all_masks, axis=0)
        boxes = np.concatenate(all_boxes, axis=0)
        labels = np.asarray(all_labels, dtype=np.int64)
        return boxes, labels, masks

    def _augment(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            if masks.size:
                masks = np.ascontiguousarray(masks[:, :, ::-1])
            if boxes.size:
                w = image.shape[1]
                x1 = boxes[:, 0].copy()
                x2 = boxes[:, 2].copy()
                boxes[:, 0] = w - x2
                boxes[:, 2] = w - x1
        # Random vertical flip
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1, :, :])
            if masks.size:
                masks = np.ascontiguousarray(masks[:, ::-1, :])
            if boxes.size:
                h = image.shape[0]
                y1 = boxes[:, 1].copy()
                y2 = boxes[:, 3].copy()
                boxes[:, 1] = h - y2
                boxes[:, 3] = h - y1
        # Mild brightness/contrast jitter
        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, masks, boxes

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.data_dir, "train", sample_id)
        image = _read_image(os.path.join(sample_dir, "image.tif"))
        boxes, labels, masks = self._load_targets(sample_dir)

        if self.augment and masks.shape[0] > 0:
            image, masks, boxes = self._augment(image, masks, boxes)

        # Drop instances that may have collapsed to zero-area after flips.
        if boxes.shape[0] > 0:
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
            masks = masks[keep]

        image_tensor = TF.to_tensor(image)  # CxHxW float in [0, 1]
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor(
                (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                if boxes.size
                else np.zeros((0,), dtype=np.float32),
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "sample_id": sample_id,
        }
        return image_tensor, target


class CellTestDataset(Dataset):
    """Test-time dataset that yields image + metadata."""

    def __init__(self, data_dir: str, mapping_json: Optional[str] = None) -> None:
        self.data_dir = data_dir
        if mapping_json is None:
            mapping_json = os.path.join(data_dir, "test_image_name_to_ids.json")
        with open(mapping_json, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        entry = self.entries[idx]
        path = os.path.join(self.data_dir, "test_release", entry["file_name"])
        image = _read_image(path)
        image_tensor = TF.to_tensor(image)
        meta = {
            "image_id": int(entry["id"]),
            "file_name": entry["file_name"],
            "height": int(entry["height"]),
            "width": int(entry["width"]),
            "orig_height": image.shape[0],
            "orig_width": image.shape[1],
        }
        return image_tensor, meta


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def test_collate_fn(batch):
    images, metas = list(zip(*batch))
    return list(images), list(metas)
