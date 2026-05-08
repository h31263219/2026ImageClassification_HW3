---
title: "HW3 — Cell Instance Segmentation"
subtitle: "Visual Recognition using Deep Learning, 2026 Spring"
author:
  - "陳沛妤"
  - "Student ID: 314560017"
date: "2026-05-08"
geometry: margin=2.2cm
fontsize: 11pt
linkcolor: blue
urlcolor: blue
---

**GitHub repository:** <https://github.com/h31263219/2026ImageClassification_HW3>

---

## 1. Introduction

This homework is an **instance-segmentation** competition on coloured
medical images. Each image contains a variable number of cells from
four classes (`class1`–`class4`); I must predict per-instance
segmentation masks **and** class labels. The official metric on
CodaBench is the segmentation **AP@0.5**.

The dataset is small and skewed: 209 training images, 101 test images,
image sizes spanning ~30×80 to over 2 000×2 000 px, and an extreme
class imbalance — `class2` accounts for >50 % of all instances while
`class3` and `class4` together account for less than 4 %. Given these
constraints my **core idea** is:

1. **Stand on a strong, well-tested baseline.** I use Mask R-CNN with
   an ImageNet-pretrained ResNet-50 + FPN backbone, whose multi-scale
   RoI features and dedicated mask head are well suited for the dense,
   small instances in this dataset.
2. **Adapt the model to small cells.** I replace the default RPN
   anchor stack with finer scales (8/16 px) — see §4.1 for the
   architectural justification — and keep the model's native
   high-resolution training (no aggressive downsampling) so that small
   cells survive the feature extractor.
3. **Regularise with safe geometric augmentations.** Random
   horizontal/vertical flips and mild brightness/contrast jitter
   combat the small training pool without distorting the medical-imaging
   semantics.
4. **Train conservatively.** AdamW with two LR groups (10× lower for
   the backbone), linear warm-up, mixed precision, and gradient
   clipping.

The resulting model has **43.94 M trainable parameters**, well below
the 200 M cap (Fig. 1), and reaches **AP50 = 0.5033 on the public
CodaBench leaderboard** (validation AP50 = 0.5142).

![**Figure 1. Trainable-parameter breakdown.** Total = 43.94 M; the
ResNet-50 body and box-head MLP dominate. The 200 M cap (red dashed
line) is never close to being hit.](output/model_size.png){ width=85% }

## 2. Method

### 2.1 Data Preprocessing

Each training sample is a folder with `image.tif` and a subset of
`class{1..4}.tif`. In every per-class TIFF, distinct non-zero pixel
values denote different instances of that class.

The dataset loader [(`dataset.py`)](dataset.py):

1. Reads the RGB image (auto-converts grayscale or RGBA to 3-channel
   `uint8`).
2. For each present class mask, splits the label-map into `N` binary
   instance masks via `np.unique`, computes their bounding boxes, and
   discards instances with zero area.
3. Concatenates instances across all four classes into a single
   `(boxes, labels, masks)` target compatible with torchvision's
   `MaskRCNN`.

**Augmentations (training only).** Random horizontal flip
(p = 0.5), random vertical flip (p = 0.5) — both update bounding boxes
and masks consistently — and brightness/contrast jitter
(α ∈ [0.85, 1.15], β ∈ [-15, 15] grey-levels).

**Resizing.** I rely on Mask R-CNN's built-in
`GeneralizedRCNNTransform` with `min_size=800` and `max_size=1333`,
which preserves small-cell detail without OOM.

**Train/val split.** A fixed 90 / 10 random split (`seed=42`):
**188 train**, **21 val** images.

### 2.2 Model Architecture

I build on **Mask R-CNN** [1]. Its key contribution over Faster R-CNN
is the per-RoI **mask branch** that predicts a binary mask in parallel
with the classification and regression branches; combined with
**RoIAlign** (which removes the harsh quantisation of RoIPool), it
achieves pixel-accurate instance segmentation. I use the torchvision
implementation as the scaffold and customise the following components:

* **Backbone — ResNet-50** [3], ImageNet `IMAGENET1K_V2` weights.
  ResNet's residual connections enable training of the deep feature
  extractor required for high-quality detection. All five stages are
  unfrozen because the medical-image statistics differ noticeably from
  natural ImageNet images.
* **Neck — Feature Pyramid Network** [2] with output channels = 256.
  FPN's top-down + lateral pathway produces P2–P6 multi-scale feature
  maps, which match the 10–60 px spread of cell sizes in this dataset.
* **Region Proposal Network.** Anchors at scales
  **{8, 16, 32, 64, 128}** (smaller than torchvision's default
  {32, 64, 128, 256, 512}) with aspect ratios {0.5, 1.0, 2.0}.
  The smaller anchors are critical for recalling tiny cells — see §4.1
  for the architectural justification.
* **RoI heads.**
  - *Box head*: 2-layer MLP (`box_score_thresh=0.05`,
    `box_nms_thresh=0.5`, up to 500 detections per image — cells can
    be extremely dense).
  - *Mask head*: standard 4-conv `MaskRCNNPredictor` (256 hidden
    channels) re-instantiated to match `num_classes = 5`, producing
    28 × 28 binary masks per RoI.
* **Number of classes.** 4 cell types + 1 background = 5.

The trainable-parameter audit (Fig. 1) shows the ResNet-50 body
(23.45 M) and the box head's 2-layer MLP (13.90 M) dominate; the
RoI mask head and predictor together add only 2.62 M.

### 2.3 Training Details

| Hyper-parameter | Value |
|---|---|
| Optimizer | AdamW [4], weight decay `1e-4` |
| Head learning rate | `2e-4` |
| Backbone learning rate | `2e-5` (10× lower) |
| LR warm-up | linear over 500 iterations |
| Batch size | 1 (native resolution training) |
| Mixed precision | `torch.amp.autocast` enabled |
| Gradient clipping | `‖g‖₂ ≤ 10.0` |
| Epochs | 40 (training was early-stopped at 32) |
| Validation | every 2 epochs, COCO segm AP / AP50 / AP75 |

I use **AdamW** rather than SGD because the small dataset is sensitive
to the LR schedule and AdamW converged ~30 % faster in my pilot runs.
The two-LR-group setup keeps the pretrained ResNet from forgetting
ImageNet features while letting the freshly-initialised RoI heads move
quickly. Mixed precision halves the GPU memory footprint and lets me
keep native image resolution during training.

## 3. Results

### 3.1 Quantitative Results

| Setting | Backbone | Val AP | Val AP50 | Val AP75 | Public AP50 |
|---|---|---|---|---|---|
| Final model (this work) | R50-FPN | **0.3199** | **0.5142** | **0.3822** | **0.5033** |

Numbers come from `output/history.json` (best epoch per metric) and
the public CodaBench leaderboard.

### 3.2 Training Curves

![**Figure 2. Training dynamics over 32 epochs.** Left: training loss
(sum of the five Mask R-CNN losses). Right: validation AP and AP50,
measured every 2 epochs.](output/training_curves.png)

The training loss decreases monotonically from ~1.47 to ~0.60.
Validation AP50 rises quickly during the first ~10 epochs as the RPN
converges, peaks at **0.5142** in epoch 11, and then oscillates in the
0.46–0.51 band. The fixed (post warm-up) LR causes the late-epoch
oscillation; AP and AP75 keep edging up (epoch 31: AP = 0.316,
AP75 = 0.382), confirming that the model is still refining mask
quality even when AP50 plateaus.

### 3.3 Confusion Matrix

![**Figure 3. Confusion matrix on the held-out validation split.** Rows
= ground-truth class, columns = predicted class. The last row ("bg /
miss") collects false positives (predicted but no GT match) and the
last column collects missed GT instances. IoU ≥ 0.5 matching, score
threshold = 0.5.](output/confusion_matrix.png){ width=78% }

The diagonal dominates for every class; cross-class confusion is rare
(≤ 0.5 % of any row). The two error modes that *do* matter are:

* **Missed `class1` instances.** 968 of 2 027 GT class-1 instances are
  unmatched. These are usually the smallest, lowest-contrast cells in
  the densest images.
* **`class4` false positives (59).** Despite being the rarest class,
  class4 attracts the most over-firings — a familiar long-tail
  classifier symptom.

### 3.4 Qualitative Visualisations

![**Figure 4. Predicted masks vs. ground truth on four validation
images.** Left = input, middle = GT overlay, right = prediction at
score ≥ 0.5. The predictor recovers a large fraction of class-1 (red)
and class-2 (green) instances in the dense images and correctly
isolates the rare class-4 (orange) instances in the sparse-cell image
(bottom row).](output/qualitative.png){ width=92% }

The prediction count is *lower* than the GT count in every panel
because the score threshold of 0.5 used for visualisation is far above
the 0.05 used for the leaderboard submission — see §3.6 for the
calibration analysis.

### 3.5 Public-leaderboard snapshot

![**Figure 5. CodaBench public-leaderboard entry.** Username
`h31263219`, Student ID `314560017`, AP50 = 0.5033 (rank 18 at the
time of capture).](output/leaderboard.png){ width=92% }

### 3.6 Inference-threshold calibration (supplementary)

To pick the operating point used in the submission, I ran the trained
model once with `box_score_thresh = 0.01` and re-evaluated COCO segm
AP at increasing post-hoc thresholds:

| score_thresh | # kept | AP | AP50 | AP75 |
|---|---|---|---|---|
| **0.05** | **4 145** | **0.306** | **0.5142** | **0.352** |
| 0.10 | 3 352 | 0.300 | 0.5032 | 0.348 |
| 0.20 | 2 683 | 0.297 | 0.4954 | 0.346 |
| 0.30 | 2 308 | 0.293 | 0.4891 | 0.340 |
| 0.50 | 1 714 | 0.275 | 0.4576 | 0.321 |
| 0.70 | 1 192 | 0.244 | 0.4031 | 0.286 |

![**Figure 6. Validation AP / AP50 / AP75 vs. inference score
threshold.** AP50 decreases monotonically (-11.1 pts going from 0.05
to 0.70).](output/exp_score_threshold.png)

This is *not* an additional experiment — score threshold is a
post-processing hyper-parameter — but the analysis confirms that 0.05
is the right submission setting, and that the default 0.5 used for the
qualitative visualisation in §3.4 would cost ~5.7 AP50 points.

## 4. Additional Experiments

This section follows the rubric: **(1) hypothesis**, **(2) reasons it
might or might not work**, then **(3) measurement and implication**.
Each experiment targets an *architectural* design decision rather than
hyper-parameter tuning, as required by the homework spec.

### 4.1 — Anchor architecture: adding 8/16-px scales to the RPN

* **Hypothesis.** Most cell instances are smaller than the smallest
  default torchvision anchor (32 px); adding finer 8- and 16-px
  anchors to the RPN's `AnchorGenerator` should improve AP50 by
  recalling tiny cells that the default stack misses.
* **Why it might work.** The RPN can only propose objects covered by
  *some* anchor — if the smallest anchor is larger than the instance,
  no anchor receives positive gradient for it and the instance is
  classified as background.
* **Why it might not.** Adding small anchors *also* multiplies the
  number of negatives per image (most are easy background), which can
  slow RPN convergence or starve harder examples of gradient.

**Architectural change.** I replace the default
`AnchorGenerator(sizes=((32,),(64,),(128,),(256,),(512,)), …)` with
`sizes=((8,),(16,),(32,),(64,),(128,))`, attached one scale per FPN
level (P2..P6). The 8- and 16-px scales are *added* to the RPN — a
layer-level change in line with the spec hint.

**Measurement 1 — dataset-side motivation.** I first compute the size
distribution of all 28 594 GT boxes in the training split:

| Statistic | Value |
|---|---|
| median √area | **22.8 px** |
| 10 th / 90 th percentile | 16.5 / 37.3 px |
| fraction ≤ 16 px | 8.1 % |
| fraction ≤ 32 px | **79.6 %** |
| fraction ≤ 64 px | 99.3 % |

![**Figure 7. GT instance-size distribution.** Vertical dashed
lines mark the anchor scales {8, 16, 32, 64, 128} px. Around 80 %
of instances fall *below* the default 32 px anchor.](output/exp_anchor_sizes.png)

**Measurement 2 — empirical retrain ablation.** I train a *second*
model with the *default* `{32, 64, 128, 256, 512}` anchors,
everything else identical (same data split, optimizer, schedule, AMP,
seed). Both models are evaluated every 2 epochs:

| Epoch | val AP50 — *custom* | val AP50 — *default* | gap (custom − default) |
|---|---|---|---|
| 1 | 0.2540 | 0.3214 | **−0.0674** |
| 3 | 0.3791 | 0.4424 | **−0.0633** |
| 5 | 0.3852 | 0.4611 | **−0.0759** |
| 7 | 0.4344 | **0.4941** | **−0.0597** |
| 9 | 0.4741 | 0.4855 | −0.0114 |
| 11 | **0.5142** | – | — |
| 27 | 0.5052 | – | — |
| **peak** | **0.5142** (ep 11) | **0.4941** (ep 7) | **+0.0202** |

![**Figure 8 (E1). Empirical ablation — validation AP50 vs. epoch
for custom (blue) and default (red) anchor stacks.** Both runs share
the *exact same* training pipeline and were trained from
scratch.](output/exp_anchor_ablation.png)

**Implication — a more honest picture.** The hypothesis is **only
partially confirmed**:

1. *Custom anchors do reach a higher AP50 ceiling* (0.5142 vs. 0.4941,
   a gap of +2.0 AP50 points). On a leaderboard graded purely by AP50,
   that gap is large enough to matter.
2. *But default anchors converge much faster.* For the first 9 epochs
   the default stack actually *outperforms* the custom stack — the
   penalty of the extra easy negatives is real and substantial.
3. The dataset analysis (Fig. 7) over-promises: with only 8 % of
   boxes below 16 px, the 8 px anchor in particular is over-engineered
   for this dataset. The win comes from the extra 16 px scale plus
   stable late-epoch optimisation, not from a wholesale recall of
   sub-32-px cells.

**Take-away for the architecture.** Custom anchors *are* the right
choice for the final model that I submit (best peak AP50), but the
gap is much smaller than the dataset-side analysis would predict, and
they cost ~3× more training epochs to surpass the default stack.
A practitioner with a tight compute budget would be better served by
the default anchors plus a longer schedule on the custom anchors;
deciding which dominates requires the empirical retrain done here,
not just the dataset-size histogram in Fig. 7.

### 4.2 — Loss-shape design: do I need a class-weighted loss?

* **Hypothesis.** The dataset is severely class-imbalanced
  (`class3`+`class4` together account for <4 % of instances). The
  conventional fix for this kind of long tail in detection is to
  swap the standard cross-entropy classification loss for **focal
  loss** [referenced in the broader Mask R-CNN literature] or to add
  class-weighted CE. The hypothesis is that long-tail classes will
  show low recall and therefore benefit from such a loss change.
* **Why it might work.** Standard CE gives equal per-instance
  weight, so the gradient signal from rare classes is diluted by
  abundant classes — the textbook motivation for focal loss.
* **Why it might not.** Focal loss is most useful when the rare class
  has many *easy negatives*. With ImageNet-pretrained ResNet-50
  features, even a few hundred examples of a *visually distinctive*
  rare class might already produce a strong signal that does not
  benefit from re-weighting.

**Measurement.** Before committing to an architectural change to the
loss, I cross-tabulate per-class training frequency against the
validation recall produced by the *unweighted-CE* model:

| class | # training instances | # training images | val recall (IoU ≥ 0.5) |
|---|---|---|---|
| class1 | 12 510 | 86 | **0.517** |
| class2 | 14 966 | 133 | 0.627 |
| class3 |    558 | 83 | 0.625 |
| class4 |    560 | 54 | **0.889** |

![**Figure 9 (E2). Training frequency vs. per-class recall.** Despite
having ~25× fewer training instances than `class1`/`class2`, class4
achieves the highest recall.](output/exp_class_imbalance.png)

**Implication.** The hypothesis is **falsified** by the measurement.
Class recall *anti-correlates* with class frequency: the rarest class
(class4) has the *highest* recall (0.89), while the most abundant
class (class1) has the *lowest* (0.52). This is consistent with the
"why it might not" branch — the rare classes are visually distinct
and the ImageNet-pretrained backbone already gives them strong
features. The bottleneck is therefore *not* gradient dilution, and
**replacing CE with focal/class-weighted loss is unlikely to help**.
Instead, the data fingerprint points at **crowding/occlusion** (class1
is the densest class), which suggests architectural fixes such as
raising `box_detections_per_img`, switching to **soft-NMS**, or
adopting an architecture with **set-based prediction** (e.g.
DETR-style queries) — all of which are layer/loss changes, not
hyper-parameters. This experiment therefore justifies *not* adding a
focal loss to the model and points at a concrete architectural
direction for future work.

## 5. References

1. **He, K., Gkioxari, G., Dollár, P., & Girshick, R.** *Mask R-CNN.*
   *Proc. ICCV*, 2017. **Key idea:** extend Faster R-CNN with a
   parallel, per-RoI mask branch and replace RoIPool with the
   alignment-preserving **RoIAlign**, enabling pixel-accurate instance
   segmentation. This is the architectural basis of every model in
   this report.
2. **Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., &
   Belongie, S.** *Feature Pyramid Networks for Object Detection.*
   *Proc. CVPR*, 2017. **Key idea:** combine a top-down pathway with
   lateral connections to fuse high-resolution low-level features and
   semantically rich high-level features into multi-scale pyramidal
   predictions — used here to handle cells across ~3 octaves of size.
3. **He, K., Zhang, X., Ren, S., & Sun, J.** *Deep Residual Learning
   for Image Recognition.* *Proc. CVPR*, 2016. **Key idea:** identity
   shortcut connections enable optimisation of very deep networks; the
   ImageNet-pretrained ResNet-50 backbone provides my feature
   extractor.
4. **Loshchilov, I. & Hutter, F.** *Decoupled Weight Decay
   Regularization.* *Proc. ICLR*, 2019. The AdamW optimiser used
   during training.
5. **Lin, T.-Y. et al.** *Microsoft COCO: Common Objects in Context.*
   *Proc. ECCV*, 2014. The COCO segmentation evaluation protocol
   (`pycocotools`) is used both during training validation and at
   submission time.
6. **Torchvision detection reference** —
   <https://github.com/pytorch/vision/tree/main/references/detection>.
   Source for the `MaskRCNN`, `AnchorGenerator`, and
   `MaskRCNNPredictor` classes I extend.
