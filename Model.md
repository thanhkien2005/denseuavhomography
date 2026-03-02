# Model Architecture — DenseUAV-Homo

This document describes the full architecture of `SiameseViT`, the cross-view
geo-localisation model defined in `models/vit_siamese.py`.

---

## Overview

The model takes a pair of images — one from a UAV drone and one from a satellite
— and produces L2-normalised embeddings for both.  At retrieval time, the cosine
similarity between a UAV query embedding and every satellite gallery embedding
ranks the candidates.

The key design element beyond a standard Siamese retrieval model is a
**network-predicted homography** that geometrically aligns the UAV feature map
toward the satellite viewpoint before pooling.  This compensates for the large
viewpoint change between nadir satellite imagery and oblique/near-nadir UAV
imagery at the same ground location.

---

## Full Forward Pipeline

```
uav_img  (B, 3, 512, 512)           sat_img  (B, 3, 512, 512)
       │                                      │
       ▼                                      ▼
  ViT-S backbone ──────── shared weights ── ViT-S backbone
  forward_features()                         forward_features()
       │                                      │
  reshape_patch_tokens()                reshape_patch_tokens()
       │                                      │
  Fu_raw (B, 384, 32, 32)             Fs (B, 384, 32, 32)
       │                                      │
       └──────────┐                           │
                  ▼                           │
          HomographyNet                       │
          (uav_img, sat_img)                  │
                  │                           │
          delta (B, 8)                        │
          gate_logit (B, 1)                   │
                  │                           │
          HomographyWarpLayer                 │
          Fu_warped (B, 384, 32, 32) ─────── ─┤ ← HomographyAlignmentLoss
                  │                           │   supervises alignment
          gate = σ(gate_logit)                │
          Fu = gate·Fu_warped                 │
             + (1−gate)·Fu_raw               │
                  │                           │
  ┌───────────────┘                           │
  │                                           │
  ▼                                           ▼
GeM pooling + L2-norm               GeM pooling + L2-norm
emb_uav (B, 384)                    emb_sat (B, 384)
  │                                           │
  ▼                                           ▼
CosineClassifier(384→C)       CosineClassifier(384→C)  ← shared weights
logit_uav (B, C)                    logit_sat (B, C)
```

**Output dict keys returned by `forward()`:**

| Key | Shape | Description |
|-----|-------|-------------|
| `emb_uav` | (B, 384) | L2-normalised UAV embedding |
| `emb_sat` | (B, 384) | L2-normalised satellite embedding |
| `logit_uav` | (B, C) | UAV classifier logits |
| `logit_sat` | (B, C) | satellite classifier logits |
| `gate_logit` | (B, 1) | raw gate logit; `gate = sigmoid(gate_logit)` |
| `delta` | (B, 8) | predicted corner displacements |
| `Fu_raw` | (B, 384, 32, 32) | UAV feature map before warp |
| `Fu_warped` | (B, 384, 32, 32) | UAV feature map after warp |
| `Fs` | (B, 384, 32, 32) | satellite feature map |

`Fu_raw`, `Fu_warped`, and `Fs` are exposed for `HomographyAlignmentLoss`;
they are not used at inference time.

---

## 1. Shared ViT-S Backbone

**File:** `models/vit_siamese.py` — `SiameseViT.forward_features()`

Both the UAV and satellite branches run through the **same** Vision Transformer
weights (vit_small_patch16_224 from timm):

- Input: `(B, 3, 512, 512)`
- Patch size: 16 → 32 × 32 patch grid → 1024 patch tokens + 1 CLS token
- Output: `(B, 1025, 384)` — full token sequence

Sharing weights forces a common embedding space across modalities without
doubling parameter count. The backbone is initialised from ImageNet-21k
pretrained weights; the positional embedding is bicubically interpolated from
the 14 × 14 (224 px) grid to the 32 × 32 (512 px) grid.

After `forward_features`, the CLS token is dropped and the 1024 patch tokens
are reshaped into a 2-D spatial feature map:

```
tokens  (B, 1025, 384)
  → drop CLS → (B, 1024, 384)
  → transpose → (B, 384, 1024)
  → reshape  → (B, 384, 32, 32)   ← Fu_raw or Fs
```

---

## 2. HomographyNet

**File:** `models/homography_net.py`

A lightweight CNN that takes both images as input and predicts:
- `delta (B, 8)`: four 2-D corner displacements in 32 × 32 feature-grid pixel units
- `gate_logit (B, 1)`: raw logit for the sigmoid blend gate

### Architecture

```
uav_img (B, 3, 512, 512) ─┐
                            ├─ resize to 128×128 ─ concat → (B, 6, 128, 128)
sat_img (B, 3, 512, 512) ─┘
    ↓
Conv(6→32,  s=2) + ReLU   → (B,  32, 64, 64)
Conv(32→64, s=2) + ReLU   → (B,  64, 32, 32)
Conv(64→128,s=2) + ReLU   → (B, 128, 16, 16)
Conv(128→256,s=2)+ ReLU   → (B, 256,  8,  8)
AdaptiveAvgPool(1)         → (B, 256,  1,  1)
Flatten                    → (B, 256)
Linear(256→256) + ReLU    → (B, 256)
    ├─ head_delta: Linear(256→8)  → delta      (B, 8)
    └─ head_gate:  Linear(256→1)  → gate_logit (B, 1)
```

### Initialisation

| Parameter     | Init strategy              | Effect at t=0                    |
|---------------|----------------------------|-----------------------------------|
| head_delta W  | zeros                      | delta = 0 → identity warp        |
| head_delta b  | zeros                      | delta = 0 → identity warp        |
| head_gate W   | zeros                      | gate_logit = bias (not input-dep) |
| head_gate b   | −2.0                       | g = σ(−2) ≈ 0.12                 |

This guarantees that at the start of training the homography branch is nearly
inactive (g ≈ 0.12), so the model can first learn a good embedding without
being destabilised by a random warp.  Once `HomographyAlignmentLoss` provides
gradient signal, the gate rises as the predicted warp improves.

---

## 3. HomographyWarpLayer

**File:** `models/homography_warp.py`

Converts the predicted corner displacements into a 3 × 3 homography matrix and
applies it to the UAV feature map using differentiable bilinear sampling.

### Steps

```
delta (B, 8)
  reshape → (B, 4, 2)       ← per-corner (dx, dy)
  + src_corners              ← fixed {TL, TR, BR, BL} in grid coords (0..31)
  = dst_corners (B, 4, 2)

kornia.get_perspective_transform(src, dst) → H (B, 3, 3)

kornia.warp_perspective(Fu_raw, H, dsize=(32,32),
                        mode="bilinear", align_corners=True)
  → Fu_warped (B, 384, 32, 32)
```

- **Differentiable:** gradients flow through `delta → dst → H → warp_perspective → Fu_warped`.
- **Out-of-bounds:** regions that land outside the 32 × 32 grid are filled with 0.
- **Identity test:** when `delta = 0`, `H = I`, and `Fu_warped ≡ Fu_raw`
  (verified in `tests/test_warp_identity.py`).

---

## 4. Gated Blending

```python
gate      = sigmoid(gate_logit).reshape(B, 1, 1, 1)   # ∈ (0, 1)
Fu_warped = warp_layer(Fu_raw, delta)
Fu        = gate * Fu_warped + (1 - gate) * Fu_raw
```

The gate is a per-sample scalar learned end-to-end.  It lets the model
smoothly interpolate between the original UAV feature map and the
geometrically aligned version.

The gate gradient has two opposing forces that naturally balance:

| Source | Effect on gate |
|--------|---------------|
| `HomographyAlignmentLoss` | **pushes gate down** when `Fu_warped` is far from `Fs` (penalises opening a bad gate) |
| CE / triplet / KL losses | **pushes gate up** when `Fu_warped ≈ Fs` (warped features improve classification) |

This synergy means: once the HomographyNet learns to produce a useful warp,
it becomes beneficial to open the gate — a self-reinforcing loop.

Expected progression with homography supervision active:

| Epoch | gate_mean | delta_norm_mean |
|-------|-----------|----------------|
| 0     | ≈ 0.12    | ≈ 0            |
| ~20   | rising    | non-zero, stabilising |
| ~60+  | 0.4 – 0.7 | geometry-dependent |

---

## 5. GeM Pooling

**File:** `models/heads.py`

Generalised Mean (GeM) pooling compresses the 2-D feature map into a 1-D descriptor:

```
feat_map (B, 384, 32, 32)
  → clamp(min=1e-6)
  → mean(x^p, dim=[H,W])^(1/p)   p is learnable (init 3.0)
  → (B, 384)
  → L2-normalise → unit-sphere embedding (B, 384)
```

Compared to global average pooling (p=1), GeM with p > 1 emphasises larger
activations and discards background clutter, which is beneficial for
instance-level retrieval.

---

## 6. Classifier Head — CosineClassifier

**File:** `models/cosine_head.py`

The classifier is a **weight-normalised cosine head** rather than a plain
`nn.Linear`:

```
logit = s · (L2_norm(emb) @ L2_norm(W)^T)
```

where `W ∈ ℝ^{C × D}` are the class prototypes and `s` is a learnable
temperature scale (init 30.0).

**Why not plain Linear?**  With L2-normalised embeddings all inputs sit on the
unit hypersphere.  A plain `nn.Linear` produces logits bounded by the weight
norm, which varies per class and makes cross-entropy saturate at `ln(C) ≈ 7.7`
for C = 2256 classes — preventing meaningful gradient flow.  The cosine head
maps all logits to `[−s, s]`, giving uniform, well-scaled gradients regardless
of class.

Both UAV and SAT branches share the same `CosineClassifier` weights.
The scale parameter `s` is updated by the main AdamW optimiser.

---

## 7. Training Losses

**File:** `losses/total_loss.py` — `DenseUAVLoss`

### Combined loss formula

```
L = w_ce      · (CE(logit_uav, y) + CE(logit_sat, y)) / 2
  + w_triplet  · SWTriplet(emb_uav, emb_sat, y)
  + w_kl       · BiKL(logit_uav, logit_sat)
  + w_homo     · HomoAlign(Fu_warped, Fs, gate_logit, delta)
  + w_con      · InfoNCE(emb_uav, emb_sat, queue_uav, queue_sat)
```

### Loss components

| Component | Class | Purpose |
|-----------|-------|---------|
| Cross-Entropy (CE) | `LabelSmoothingCE` | Supervise each branch with ground-truth location label |
| Soft Triplet | `SoftWeightedTripletLoss` | Push same-class UAV/SAT embeddings together; pull apart different-class pairs |
| Bi-directional KL | `BidirectionalKLLoss` | Mutual learning: make UAV and SAT logit distributions consistent. Temperature T=4.0 with T²-scaled gradient magnitude |
| **Homography Alignment** | `HomographyAlignmentLoss` | Explicit supervision for HomographyNet so the branch is not a no-op |
| **InfoNCE (contrastive)** | `InfoNCELoss` + `MemoryQueue` | Cross-batch negatives prevent metric collapse; queue of 4096 past embeddings, T=0.07 |

### HomographyAlignmentLoss — detail

**File:** `losses/homography_loss.py`

```
gate      = sigmoid(gate_logit).reshape(B, 1, 1, 1)

L_align   = mean( gate × |Fu_warped − Fs.detach()| )   # gate-weighted L1
L_reg     = mean( delta² )                               # delta L2 regularisation

L_homo    = L_align + λ_reg × L_reg
```

- `.detach()` on `Fs`: only the UAV→satellite direction is supervised; the
  satellite backbone is not pulled toward the UAV prediction.
- Gate weighting: the alignment penalty scales with how much the warped branch
  actually contributes to the output — no gradient waste when the gate is near zero.
- Delta regularisation (`λ_reg = 0.01`): prevents degenerate homographies
  (collapsed or flipped quads) early in training.

### Default loss weights

| Weight | Default | Config key |
|--------|---------|-----------|
| `w_ce` | 1.0 | `loss.w_ce` |
| `w_triplet` | 1.0 | `loss.w_triplet` |
| `w_kl` | 1.0 | `loss.w_kl` |
| `w_homo` | 0.5 | `loss.w_homo` |
| `λ_reg` | 0.01 | `loss.lambda_reg` |
| KL temperature | 4.0 | `loss.temperature` |
| `w_con` | 1.0 | `contrastive.w_contrastive` |
| InfoNCE temperature | 0.07 | `contrastive.temperature` |

---

## 8. Data Augmentation — Paired Transforms

**File:** `data/paired_transforms.py` — `PairedTransform`

Because HomographyNet must learn a *relative* geometric transformation between
the UAV and satellite views, the two images must share the same global
orientation at training time.  Applying independent random flips would give the
network inconsistent targets (e.g. UAV flipped, SAT not) and produce zero
usable gradient.

`PairedTransform` draws **one** random decision per geometric operation and
applies it to **both** images:

| Transform | Shared? | Notes |
|-----------|---------|-------|
| Resize to 512 × 512 | — | both |
| RandomHorizontalFlip (p=0.5) | **yes** | same coin flip |
| RandomVerticalFlip (p=0.5) | **yes** | same coin flip |
| RandomRotation ±15° | **yes** | same angle |
| ColorJitter | UAV only | photometric; does not break geometry |
| ToTensor + Normalize (ImageNet) | — | both |

---

## 9. Retrieval at Inference

No losses or classifier head are used at retrieval time.  For a query set of
UAV images and a gallery of satellite images:

```
sim[i, j] = emb_uav[i] · emb_sat[j]   (cosine similarity, unit sphere)
rank[i]   = argsort(sim[i], descending=True)
```

The model is evaluated with:
- **Recall@K** — fraction of queries whose ground-truth gallery item appears in the top-K.
- **SDM@K** — GPS-weighted soft distance match score at rank K; rewards near-misses.

---

## 10. Training Diagnostics

The trainer logs the following homography-branch statistics at the end of every epoch:

| Metric | Description |
|--------|-------------|
| `gate_mean` | Epoch-average of sigmoid(gate_logit) across all samples |
| `gate_min` | Epoch-minimum gate value (true extreme, not smoothed) |
| `gate_max` | Epoch-maximum gate value |
| `delta_norm_mean` | Epoch-average per-sample L2 norm of delta (B, 8) |
| `delta_abs_max` | Epoch-maximum absolute corner displacement |

Log line format:
```
[homo] gate_mean=0.1234 gate_min=0.1043 gate_max=0.2187 | delta_norm_mean=0.4521 delta_abs_max=1.2043
```

If the gate does not rise above ~0.15 after 10 epochs, the alignment signal is
not flowing; check that `w_homo > 0` and that `PairedTransform` is in use.

---

## Parameter Count (approximate)

| Component | Parameters |
|-----------|-----------|
| ViT-S backbone | ~21.7 M |
| HomographyNet CNN | ~0.6 M |
| GeM + CosineClassifier | ~0.9 M |
| **Total** | **~23.2 M** |

(Exact count printed by `repr(model)` or `scripts/sanity_check_shapes.py`.)
