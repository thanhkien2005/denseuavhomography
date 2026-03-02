# denseuavhomography — Claude Code Project Context (claude.md)

> **Purpose:** This file gives Claude Code the full project context, current status, confirmed issues, and the exact fix roadmap.  
> **Repo:** `https://github.com/thanhkien2005/denseuavhomography`

---

## 0) One-sentence summary

We are building a **DenseUAV-style cross-view geo-localization** model (UAV ↔ Satellite) with an added **homography/warp branch**; the current repo runs but **evaluation is only on train-pairs** and several training signals (CE / KL / triplet / homography) are structurally weak, causing misleading “perfect” train metrics and high CE loss.

---

## 1) Current architecture & training setup (as implemented)

### 1.1 Model
- **Backbone:** Siamese shared-weight ViT-S (patch tokens reshaped to feature map).
- **Pooling:** GeM pooling.
- **Embedding:** L2-normalized embedding vector (dim=384).
- **Classifier:** currently a plain `Linear(384 → 2256, bias=False)` on the normalized embedding.
- **Homography branch:** `HomographyNet` predicts:
  - `delta` (8 params) initialized to 0 (identity warp)
  - `gate_logit` initialized to -2 ⇒ `sigmoid(gate_logit) ≈ 0.12`
  - forward path computes warp via kornia and blends:  
    `Fu = gate * Fu_warped + (1-gate) * Fu_raw`

Key files:
- `models/vit_siamese.py`
- `models/homography_net.py`

### 1.2 Loss (DenseUAV-style composition)
Total loss:
- **CE** on both views (UAV and SAT), averaged
- **Soft-weighted triplet** cross-view (bi-directional)
- **Bi-directional KL** mutual learning on logits

Key files:
- `losses/total_loss.py`
- `losses/sw_triplet.py`
- `losses/kl.py`

### 1.3 Data (current state)
- Training dataset class: `DenseUAVPairs`
- For altitude `H80`, it builds **exactly 1 pair per location**:
  - `num_classes == num_samples == 2256`
  - each location has exactly one UAV image and one SAT image
- **Critical detail:** `uav_gps` is copied from `sat_gps` (identical coordinates for the matched pair)

Key file:
- `data/denseuav_dataset.py`

### 1.4 Eval (current state)
- `scripts/eval.py` supports only `--split train` (`choices=["train"]`).
- It evaluates **train-pairs retrieval**: query embeddings and gallery embeddings both come from the same paired dataloader (N=2256 vs N=2256).
- The **true DenseUAV protocol** (Q=777 query UAV vs G=3033 gallery SAT) exists as an evaluator method `evaluate_split()` but is unreachable because datasets are missing.

Key files:
- `scripts/eval.py`
- `engine/evaluator.py`
- `metrics/recall.py`
- `metrics/sdm.py`

---

## 2) What we confirmed (Checklist report findings)

### 2.1 Evaluation is NOT the real test protocol (CONFIRMED FAIL)
- `scripts/eval.py`: `--split` is hardcoded to `choices=["train"]`
- test split path raises `NotImplementedError`
- Current metrics are computed on train-pairs retrieval only.

**Impact:** `Recall@1 = 1.000` is **not meaningful** for true test generalization.

### 2.2 SDM is inflated on training split (CONFIRMED)
- In `DenseUAVPairs`, `uav_gps == sat_gps` for the matched pair ⇒ distance=0 ⇒ SDM proximity score=1.0 whenever the correct pair is retrieved.
- So **SDM@1 becomes trivially implied** by Recall@1 on train-pairs.

### 2.3 Cross-Entropy is structurally “stuck near ln(C)” (CONFIRMED FAIL)
- `models/vit_siamese.py`:
  - `emb = F.normalize(...)` (unit norm)
  - `logits = Linear(emb)` with no scale, no weight norm
- With unit-norm inputs and small-initialized weights, logits are near 0 ⇒ softmax ~ uniform ⇒ CE ~ ln(2256)=7.72 at init.
- Logs show: CE starts at ~7.7217 and ends at ~6.0758 at epoch 120.

**Impact:** CE cannot converge properly under this head design.

### 2.4 KL mutual learning is effectively dead (CONFIRMED FAIL)
- `losses/kl.py` uses `T=0.07` (CLIP-style contrastive temperature).
- This makes distributions near one-hot ⇒ KL ~ 0 quickly.
- Logs: `loss_kl` goes ~0.0182 (epoch 1) → ~0.0002 (epoch 120).

**Impact:** Mutual learning provides almost no gradient signal.

### 2.5 Triplet collapses very early (CONFIRMED FAIL)
- Logs: triplet ~0.2841 (epoch 1) → ~0.0186 (epoch 5) → ~0.0017 (epoch 120).
- Root cause: batch has unique labels; with 1-to-1 pairs and a pretrained backbone, negatives become trivially separable within-batch; there is no cross-batch hard negative mining.

**Impact:** Metric learning signal turns off early.

### 2.6 Homography branch likely becomes a no-op (high risk; partly UNCERTAIN)
- Warp is called, but:
  - gate starts ~0.12 (bias -2), delta starts 0
  - there is **no explicit homography supervision**
  - downstream losses feeding gradients into homography become weak (CE stuck, triplet dead, KL dead)
- Net effect: homography likely stays near identity and/or gate stays low.

**Impact:** the key architectural contribution likely does nothing.

### 2.7 Augmentation breaks geometry pairing (risk)
- UAV and SAT transforms are applied independently.
- Independent flips/rotations can destroy consistent geometric relations that a homography branch tries to learn.

### 2.8 Hyperparams differ from paper baseline
- AdamW + warmup/cosine, input size 512 vs paper’s SGD + step schedule, 224.

**Note:** This is not necessarily “wrong,” but it makes reproduction harder and can change convergence behavior.

---

## 3) Logs summary (what they actually mean)

From `train.log`:
- Epoch 1: `loss_total=8.0240 | loss_ce=7.7217 | loss_triplet=0.2841 | loss_kl=0.0182`
- Epoch 120: `loss_total=6.0777 | loss_ce=6.0758 | loss_triplet=0.0017 | loss_kl=0.0002`

Interpretation:
- Training becomes almost purely a weak CE objective with tiny logits.

From `eval.log`:
- `split: train`
- `train/Recall@1 = 1.0000`
- `train/SDM@1 = 1.0000`

Interpretation:
- This is **train-pairs memorization** + SDM inflation due to GPS identity on matched pairs.

---

## 4) Fix roadmap (order matters)

> **Rule:** Fix “illusory evaluation” first, then fix training signals.

### Step 1 — Implement TRUE TEST protocol (highest priority)
Goal:
- Add `DenseUAVQuery` (777 query UAV) and `DenseUAVGallery` (3033 gallery SAT)
- Wire `scripts/eval.py --split test` to call `evaluator.evaluate_split()`
- Only compute SDM if both sides have real GPS coords; otherwise skip with warning.

Acceptance:
- `python -m scripts.eval --config configs/denseuav_v1.yaml --split test` runs (or fails with clear “missing data dirs” message)
- Test metrics are logged under `test/*`

### Step 2 — Fix the classifier head (CE must be learnable)
Goal:
- Replace plain Linear-on-unit-emb with cosine classifier + scale (NormFace/ArcFace style):
  - `logits = s * F.linear(emb, normalize(W))`
  - `s` configurable and preferably learnable (init 30)

Acceptance:
- CE should drop significantly below ln(C) early in training
- logits range increases meaningfully

### Step 3 — Fix KL mutual learning temperature
Goal:
- Use distillation temperature for KL (e.g., T=4.0)
- Multiply KL by `T*T` to keep gradient magnitude

Acceptance:
- `loss_kl` remains non-trivial (does not collapse to ~0 immediately)

### Step 4 — Fix metric learning collapse
Goal:
- Replace/augment SWTriplet with cross-batch negatives (memory queue) using InfoNCE, OR add a memory bank for hard mining.

Acceptance:
- metric/contrastive loss remains >0 beyond epoch 5
- better generalization on true test split

### Step 5 — Make homography branch learnable
Goal:
- Add explicit homography supervision (feature alignment / consistency / regularization)
- Make augmentations paired/synchronous to preserve geometry

Acceptance:
- log `gate_mean`, `delta_norm` change meaningfully from init
- ablation shows enabling warp improves retrieval under true test protocol

### Optional Step 6 — Paper-like baseline
Goal:
- Reproduce paper baseline: 224px + SGD + milestones (70/110)
- Use it as a stable reference before adding homography complexity.

---

## 5) Critical “DO NOT DO” list (to avoid wasted runs)
- Do NOT trust any Recall/SDM from train split as “final performance”.
- Do NOT tune homography branch before true test protocol exists.
- Do NOT keep KL temperature at 0.07 for mutual learning on classification logits.
- Do NOT keep independent geometric augmentations for UAV vs SAT if homography alignment is desired.

---

## 6) Where to work (files map)
Top struggling files (must be addressed in roadmap):
1) `scripts/eval.py` — test split blocked
2) `models/vit_siamese.py` — CE head stuck due to no scale
3) `data/denseuav_dataset.py` — missing query/gallery datasets; GPS identity issue
4) `losses/kl.py` — T=0.07 kills KL
5) `losses/sw_triplet.py` — triplet collapse due to easy negatives

Related:
- `engine/evaluator.py` — `evaluate_split()` exists but currently unused
- `data/transforms.py` — independent aug may break geometry

---

## 7) Commands & sanity checks
Recommended commands:
- Train:
  - `python -m scripts.train --config configs/denseuav_v1.yaml`
- Eval (train proxy):
  - `python -m scripts.eval --config configs/denseuav_v1.yaml --split train`
- Eval (real test, after Step 1):
  - `python -m scripts.eval --config configs/denseuav_v1.yaml --split test`

Must-log values after fixes:
- CE: `loss_ce`, also average |logits| magnitude
- KL: `loss_kl`
- metric loss: triplet or InfoNCE loss
- homography: `gate_mean`, `gate_min/max`, `delta_norm_mean`, `delta_abs_max`

---

## 8) Notes about the “Homography .xlsx”
If present in the workspace, it may be located at:
- `/mnt/data/Homography .xlsx`

Use it only if it contains experiment plans or parameter grids; it is not required for core training/eval fixes.

---

## 9) Expected outcome after Step 1–3 (minimum viable correctness)
Once Step 1–3 are done:
- We will have **real test metrics** (not train proxy).
- CE will become trainable (no longer stuck near ln(C)).
- KL will contribute a meaningful mutual-learning signal.

Only after that should we invest in Step 4–5 (hard negatives + homography supervision).
