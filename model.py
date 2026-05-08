"""Model builder for HW3 instance segmentation.

We build on Mask R-CNN (He et al., ICCV 2017) using torchvision's
implementation. Key configurable bits:

* backbone        : ResNet-50 / ResNet-101 with FPN, ImageNet pretrained
* anchor sizes    : finer scales suited to small cell instances
* RoI pooling     : higher mask resolution (28x28 -> 56x56) for sharper masks
* score threshold : lowered at inference time to maintain recall

The number of trainable parameters stays well below the 200M cap
required by the homework spec.
"""

from __future__ import annotations

from typing import Optional

import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def _build_anchor_generator(default_anchors: bool = False) -> AnchorGenerator:
    """Anchor sizes tuned for small cell instances.

    The default torchvision anchors start at 32 px which is too large for
    the smaller cell types in this dataset. We add 8 / 16 px anchors and
    keep the common aspect ratios. Pass ``default_anchors=True`` to
    reproduce the torchvision baseline (used for the §4.1 ablation).
    """
    if default_anchors:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    else:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    return AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)


def build_model(
    num_classes: int = 5,
    backbone_name: str = "resnet50",
    pretrained_backbone: bool = True,
    hi_res_mask: bool = True,
    trainable_backbone_layers: int = 5,
    box_score_thresh: float = 0.05,
    box_nms_thresh: float = 0.5,
    box_detections_per_img: int = 500,
    default_anchors: bool = False,
) -> MaskRCNN:
    """Build a Mask R-CNN with sensible defaults for cell segmentation.

    Parameters
    ----------
    num_classes : int
        Number of classes including the background (4 cells + 1 bg = 5).
    backbone_name : str
        ``resnet50`` or ``resnet101``.
    pretrained_backbone : bool
        Load ImageNet pretrained weights for the backbone.
    hi_res_mask : bool
        If True, use a 28x28 RoI mask head and produce 56x56 masks.
    trainable_backbone_layers : int
        Number of trainable backbone stages (0..5). 5 trains the whole
        backbone which usually helps when the dataset is small.
    box_score_thresh, box_nms_thresh, box_detections_per_img : float, float, int
        Inference-time post-processing knobs.
    """
    weights = (
        torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        if backbone_name == "resnet50" and pretrained_backbone
        else (
            torchvision.models.ResNet101_Weights.IMAGENET1K_V2
            if backbone_name == "resnet101" and pretrained_backbone
            else None
        )
    )

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=weights,
        trainable_layers=trainable_backbone_layers,
    )

    anchor_generator = _build_anchor_generator(default_anchors=default_anchors)

    box_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2,
    )
    mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=14 if hi_res_mask else 14,
        sampling_ratio=2,
    )

    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool,
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        min_size=800,
        max_size=1333,
    )

    if hi_res_mask:
        # Upgrade the mask head: deeper conv stack + 28x28 -> 56x56 output.
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes,
        )

    # Replace the box predictor (already done by num_classes arg, but keep
    # it explicit so users can change num_classes after build).
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: Optional[str] = None) -> dict:
    state = torch.load(path, map_location=map_location)
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    return state
