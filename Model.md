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
          Fu_warped (B, 384, 32, 32)          │
                  │                           │
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
Linear(384, C)                      Linear(384, C)   ← shared weights
logit_uav (B, C)                    logit_sat (B, C)
```

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
being destabilised by a random warp.

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
geometrically aligned version:

- Early training: g ≈ 0.12 → alignment contributes little; main embedding branch learns freely.
- Late training: g grows as the homography prediction improves; alignment contributes more.

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

## 6. Classifier Head

A single shared `nn.Linear(384, num_classes, bias=False)` is applied to both
`emb_uav` and `emb_sat` to produce classification logits used during training.
`bias=False` is important because the embeddings are L2-normalised (on the unit
hypersphere), and a bias would break the symmetry expected by cosine-similarity
metrics.

---

## 7. Training Losses

**File:** `losses/total_loss.py` — `DenseUAVLoss`

```
L = w_ce · (CE(logit_uav, y) + CE(logit_sat, y)) / 2
  + w_triplet · SWTriplet(emb_uav, emb_sat, y)
  + w_kl · BiKL(logit_uav, logit_sat)
```

| Component            | Class                    | Purpose                                               |
|----------------------|--------------------------|-------------------------------------------------------|
| Cross-Entropy (CE)   | `LabelSmoothingCE`       | Supervise each branch with ground-truth location label |
| Soft Triplet         | `SoftWeightedTripletLoss`| Push same-class UAV/SAT embeddings together; pull apart different-class pairs |
| Bi-directional KL    | `BidirectionalKLLoss`    | Mutual learning: make UAV and SAT logit distributions consistent |

Default weights: `w_ce = w_triplet = w_kl = 1.0` (configurable in YAML).

---

## 8. Retrieval at Inference

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

## Parameter Count (approximate)

| Component          | Parameters |
|--------------------|-----------|
| ViT-S backbone     | ~21.7 M   |
| HomographyNet CNN  | ~0.6 M    |
| GeM + classifier   | ~0.9 M    |
| **Total**          | **~23.2 M** |

(Exact count printed by `repr(model)` or `scripts/sanity_check_shapes.py`.)
