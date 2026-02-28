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

---

## Evaluation

```bash
# Evaluate a checkpoint on the training split:
python scripts/eval.py \
    --checkpoint outputs/denseuav_v1/epoch_0120.pt

# Explicit options:
python scripts/eval.py \
    --checkpoint  outputs/denseuav_v1/epoch_0120.pt \
    --config      configs/denseuav_v1.yaml \
    --data_root   /path/to/DenseUAV \
    --split       train \
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

Evaluation log is appended to `<checkpoint_dir>/eval.log`.

> **Note:** The proper test-split evaluation (777 query UAV vs 3033 gallery
> satellite images) requires `DenseUAVQuery` and `DenseUAVGallery` dataset
> classes, which are marked as TODO in `data/denseuav_dataset.py`. Once
> implemented, use `evaluator.evaluate_split(model, query_loader, gallery_loader)`.

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

# Loss
loss:
  w_ce: 1.0
  w_triplet: 1.0
  w_kl: 1.0
  temperature: 0.07
  margin: 0.3

# Evaluation
recall_k: [1, 5, 10]
sdm_k:    [1, 5, 10]
```

---

## Hooks (engine/hooks.py)

`engine/hooks.py` provides helpers for wiring periodic evaluation and
checkpointing into a custom training loop:

```python
from engine.hooks import maybe_evaluate, maybe_save_checkpoint, BestCheckpointTracker

best_tracker = BestCheckpointTracker(metric_key="train/Recall@1")

for epoch in range(start_epoch, total_epochs):
    # ... training ...

    metrics = maybe_evaluate(
        epoch=epoch + 1, eval_interval=10, total_epochs=total_epochs,
        model=model, evaluator=evaluator, dataloader=val_loader, logger=logger,
    )
    maybe_save_checkpoint(
        epoch=epoch + 1, save_interval=10, total_epochs=total_epochs,
        state=state, output_dir=output_dir, logger=logger,
    )
    if metrics is not None:
        best_tracker.update(metrics, state, output_dir, logger)
```

---

## Project Structure

```
denseuav-homo/
├── configs/
│   └── denseuav_v1.yaml        # all hyperparameters
├── data/
│   ├── denseuav_dataset.py     # DenseUAVPairs dataset
│   ├── samplers.py             # PairPerClassBatchSampler
│   └── transforms.py          # train / val augmentation pipelines
├── engine/
│   ├── evaluator.py            # Evaluator: Recall@K + SDM@K
│   ├── hooks.py                # periodic eval + checkpoint helpers
│   └── trainer.py              # one-epoch training loop (AMP, grad clip)
├── losses/
│   ├── ce.py                   # LabelSmoothingCE
│   ├── kl.py                   # BidirectionalKLLoss
│   ├── sw_triplet.py           # SoftWeightedTripletLoss
│   └── total_loss.py           # DenseUAVLoss (combined)
├── metrics/
│   ├── recall.py               # recall_at_k
│   └── sdm.py                  # sdm_at_k, haversine_distance
├── models/
│   ├── heads.py                # GeM pooling head
│   ├── homography_net.py       # HomographyNet (delta + gate prediction)
│   ├── homography_warp.py      # HomographyWarpLayer (kornia-based)
│   └── vit_siamese.py          # SiameseViT — main model
├── scripts/
│   ├── eval.py                 # standalone evaluation entry point
│   ├── sanity_check_shapes.py  # shape + metric correctness verification
│   ├── sanity_check_batch.py   # dataloader batch inspection
│   └── train.py                # training entry point
├── tests/
│   └── test_warp_identity.py   # delta=0 → warp is identity
└── utils/
    ├── checkpoint.py           # save / load / resume helpers
    ├── logger.py               # stdout + file logger
    ├── meters.py               # AverageMeter, MetricCollection
    └── seed.py                 # deterministic seed helper
```
