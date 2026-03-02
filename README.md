# DenseUAV-Homo

Cross-view geo-localisation model that matches UAV drone images to satellite imagery using a shared ViT-S backbone with a network-predicted homography alignment stage.

---

## Requirements

```bash
pip install torch torchvision timm kornia tqdm pyyaml pillow
```

Tested with Python 3.10, PyTorch 2.x, timm 1.0.x, kornia 0.7.x.

---

## Dataset Setup

Download the [DenseUAV dataset](https://github.com/Dmmm1997/DenseUAV) and place it at a path of your choice. The expected layout:

```
DenseUAV/
├── Dense_GPS_train.txt
├── Dense_GPS_test.txt
├── Dense_GPS_ALL.txt
├── train/
│   ├── drone/
│   │   ├── 000000/
│   │   │   ├── H80.JPG
│   │   │   ├── H90.JPG
│   │   │   └── H100.JPG
│   │   └── ...
│   └── satellite/
│       ├── 000000/
│       │   ├── H80.tif
│       │   └── H80_old.tif
│       └── ...
└── test/
    ├── query_drone/
    └── gallery_satellite/
```

Set the path in `configs/denseuav_v1.yaml`:

```yaml
data_root: "/abs/path/to/DenseUAV"
```

Or pass `--data_root` on the command line to override it at runtime.

---

## Quick Start — Sanity Check

Run this before training to verify shapes end-to-end without loading any data:

```bash
cd denseuav-homo
python scripts/sanity_check_shapes.py
```

Expected output ends with:

```
  All model assertions PASSED.
```

---

## Training

All training is launched from the repo root (`denseuav-homo/`):

```bash
# Default config (120 epochs, AdamW, cosine LR, AMP on CUDA):
python scripts/train.py

# Override data root and output directory:
python scripts/train.py \
    --data_root  /path/to/DenseUAV \
    --output_dir outputs/my_run

# Override individual hyperparameters:
python scripts/train.py \
    --epochs     60 \
    --batch_size 16 \
    --lr         5e-5

# Resume from a checkpoint:
python scripts/train.py \
    --resume outputs/denseuav_v1/epoch_0010.pt

# CPU training (AMP disabled automatically):
python scripts/train.py --no_amp

# Random init (no ImageNet pretrained backbone):
python scripts/train.py --no_pretrained
```

Checkpoints are saved to `outputs/denseuav_v1/epoch_NNNN.pt` by default (configurable via `output_dir` and `save_interval` in the YAML).

Training logs go to `outputs/denseuav_v1/train.log`.

### Homography diagnostics

At the end of every epoch the trainer logs gate and delta statistics:

```
[homo] gate_mean=0.1234 gate_min=0.1043 gate_max=0.2187 | delta_norm_mean=0.4521 delta_abs_max=1.2043
```

`gate_mean` should rise above ~0.15 within the first 10 epochs when homography supervision is active. If it stays near 0.12, check that `loss.w_homo > 0` in the config.

---

## Evaluation

Two evaluation modes are supported via `--split`:

| `--split` | Dataset | Protocol |
|-----------|---------|----------|
| `train` (default) | `DenseUAVPairs` (2256 pairs) | Closed-set proxy — measures memorisation |
| `test` | `DenseUAVQuery` (777) + `DenseUAVGallery` (3033) | **Official DenseUAV benchmark** |

```bash
# Official test-split evaluation (777 queries vs 3033 gallery):
python scripts/eval.py \
    --checkpoint outputs/denseuav_v1/epoch_0120.pt \
    --split      test

# Training-split proxy (quick sanity check):
python scripts/eval.py \
    --checkpoint outputs/denseuav_v1/epoch_0120.pt \
    --split      train

# Explicit options:
python scripts/eval.py \
    --checkpoint  outputs/denseuav_v1/epoch_0120.pt \
    --config      configs/denseuav_v1.yaml \
    --data_root   /path/to/DenseUAV \
    --split       test \
    --batch_size  16 \
    --device      cuda

# CPU evaluation:
python scripts/eval.py \
    --checkpoint outputs/denseuav_v1/epoch_0120.pt \
    --device cpu
```

Reported metrics:

| Metric      | Description                                            |
|-------------|--------------------------------------------------------|
| Recall@1    | Fraction of queries whose top-1 retrieval is correct   |
| Recall@5    | Fraction of queries with a correct match in top-5      |
| Recall@10   | Fraction of queries with a correct match in top-10     |
| SDM@1       | Soft distance match score at K=1 (GPS-weighted)        |
| SDM@5       | Soft distance match score at K=5                       |
| SDM@10      | Soft distance match score at K=10                      |

SDM@K requires GPS coordinates in `Dense_GPS_test.txt`; a warning is logged
and SDM is skipped automatically if GPS is unavailable.

Evaluation log is appended to `<checkpoint_dir>/eval.log`.

---

## Configuration

All settings live in `configs/denseuav_v1.yaml`.  Key parameters:

```yaml
# Paths
data_root: "../DenseUAV"         # relative to repo root
output_dir: "outputs/denseuav_v1"

# Model
embed_dim: 384                   # ViT-S dimension
homo_hidden: 256                 # HomographyNet CNN width
gate_bias_init: -2.0             # initial warp gate ≈ sigmoid(-2) ≈ 0.12

# Training
epochs: 120
batch_size: 32
lr: 1.0e-4
warmup_epochs: 5

# Classifier head
head:
  scale: 30.0                    # initial logit scale for CosineClassifier
  learnable_scale: true

# Loss weights
loss:
  w_ce:       1.0
  w_triplet:  1.0
  w_kl:       1.0
  w_homo:     0.5                # homography feature-alignment weight
  lambda_reg: 0.01               # delta L2 regularisation inside HomoAlignLoss
  temperature: 4.0               # KL softmax temperature
  margin:      0.3               # triplet margin

# Contrastive learning (InfoNCE with memory queue)
contrastive:
  use_contrastive: true
  queue_size:  4096              # cross-batch negatives kept in FIFO queue
  temperature: 0.07              # InfoNCE temperature
  w_contrastive: 1.0

# Evaluation
recall_k: [1, 5, 10]
sdm_k:    [1, 5, 10]
```

---

## Architecture Overview

See `Model.md` for the full architecture description.  Brief summary:

| Component | Details |
|-----------|---------|
| Backbone | `vit_small_patch16_224` (timm), shared UAV+SAT, img_size=512, embed_dim=384 |
| Pooling | GeM (p=3.0, learnable) |
| Classifier | `CosineClassifier`: logits = s·(emb @ norm(W)ᵀ), s=30.0 learnable |
| Homography | `HomographyNet` → delta (B,8) + gate_logit (B,1) → `HomographyWarpLayer` |
| Augmentation | `PairedTransform`: shared random flip/rotation for UAV+SAT; color jitter UAV-only |
| Total params | ~23.2 M |

### Loss design

```
L = w_ce · CE  +  w_triplet · SWTriplet  +  w_kl · BiKL
  + w_homo · HomoAlign(Fu_warped, Fs, gate, delta)
  + w_con  · InfoNCE(emb_uav, emb_sat, queue)
```

`HomoAlign` provides direct gradient signal to `HomographyNet`:

```
gate    = sigmoid(gate_logit)
L_align = mean( gate × |Fu_warped − Fs.detach()| )   # gate-weighted L1
L_reg   = mean( delta² )                               # prevents extreme warps
L_homo  = L_align + λ_reg × L_reg
```

---

## Project Structure

```
denseuav-homo/
├── configs/
│   └── denseuav_v1.yaml             # all hyperparameters
├── data/
│   ├── denseuav_dataset.py          # DenseUAVPairs / DenseUAVQuery / DenseUAVGallery
│   ├── paired_transforms.py         # PairedTransform — shared geometric augmentation
│   ├── samplers.py                  # PairPerClassBatchSampler
│   └── transforms.py               # single-image transform pipelines (eval use)
├── engine/
│   ├── evaluator.py                 # Evaluator: Recall@K + SDM@K
│   ├── hooks.py                     # periodic eval + checkpoint helpers
│   └── trainer.py                   # one-epoch training loop (AMP, grad clip, homo logging)
├── losses/
│   ├── ce.py                        # LabelSmoothingCE
│   ├── contrastive.py               # InfoNCELoss
│   ├── homography_loss.py           # HomographyAlignmentLoss
│   ├── kl.py                        # BidirectionalKLLoss (T=4.0, T²-scaled)
│   ├── sw_triplet.py                # SoftWeightedTripletLoss
│   └── total_loss.py                # DenseUAVLoss (combined)
├── metrics/
│   ├── recall.py                    # recall_at_k
│   └── sdm.py                       # sdm_at_k, haversine_distance
├── models/
│   ├── cosine_head.py               # CosineClassifier (weight-normalised)
│   ├── heads.py                     # GeM pooling head
│   ├── homography_net.py            # HomographyNet (delta + gate prediction)
│   ├── homography_warp.py           # HomographyWarpLayer (kornia-based)
│   └── vit_siamese.py               # SiameseViT — main model
├── scripts/
│   ├── eval.py                      # standalone evaluation entry point
│   ├── sanity_check_shapes.py       # shape + metric correctness verification
│   ├── sanity_check_batch.py        # dataloader batch inspection
│   └── train.py                     # training entry point
├── tests/
│   └── test_warp_identity.py        # delta=0 → warp is identity
└── utils/
    ├── checkpoint.py                # save / load / resume helpers
    ├── logger.py                    # stdout + file logger
    ├── memory_queue.py              # MemoryQueue — FIFO embedding store
    ├── meters.py                    # AverageMeter, MetricCollection
    └── seed.py                      # deterministic seed helper
```
