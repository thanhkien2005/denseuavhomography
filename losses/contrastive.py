"""
losses/contrastive.py — Symmetric InfoNCE loss with memory-queue negatives.

Formula
───────
For a batch of B paired samples (emb_uav, emb_sat):

    UAV → SAT direction
    ───────────────────
    Gallery = [emb_sat  (B, D),    ← current batch (positive at col i)
               queue_sat (K, D)]   ← past-batch negatives (no grad)

    logits_i = emb_uav[i] · Gallery^T / τ    (1, B+K)
    targets  = i   (positive is always the i-th column)

    L_u2s = CrossEntropy(logits, targets)    summed over all i → batchmean

    SAT → UAV direction is symmetric (swap roles of uav/sat).

    L_total = (L_u2s + L_s2u) / 2

Why T = 0.07 here
─────────────────
InfoNCE with a large negative bank is a (B+K)-way classification.  With
K = 4096 negatives the effective number of classes is large and a *small*
temperature (sharper distribution) is needed to maintain high entropy in the
denominator — making the positive stand out requires a steep gradient.
This is the CLIP / MoCo / SimCLR regime (T ≈ 0.07), distinct from the
knowledge-distillation KL temperature (T = 4.0 in losses/kl.py).

Warmup
──────
When the queue is empty (K = 0) the gallery reduces to `emb_sat` alone and
the loss is standard in-batch InfoNCE (B-way classification).  This is a
valid lower bound on the full loss, so no special handling is needed.

AMP compatibility
─────────────────
Queue embeddings are stored as float32; current-batch embeddings may be
float16 under AMP.  The gallery is cast to match the query dtype inside
`forward()` before the matrix multiply, so no NaN/Inf issues arise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE with optional memory-queue negatives.

    Args:
        temperature: Logit scale before cross-entropy.  Use a small value
                     (e.g. 0.07) — this is the *contrastive* temperature,
                     NOT the knowledge-distillation temperature.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        assert temperature > 0.0, f"temperature must be > 0, got {temperature}"
        self.temperature = temperature

    # ──────────────────────────────────────────────────────────────────────
    # Internal: one retrieval direction
    # ──────────────────────────────────────────────────────────────────────

    def _one_direction(
        self,
        queries:  Tensor,   # (B, D) requires-grad embeddings
        pos_keys: Tensor,   # (B, D) current-batch positive keys
        neg_keys: Tensor,   # (K, D) queue negatives (may be empty, K=0)
    ) -> Tensor:
        """InfoNCE for one query→gallery direction.

        Args:
            queries:  (B, D) — current batch query embeddings.
            pos_keys: (B, D) — positive key for queries[i] is pos_keys[i].
            neg_keys: (K, D) — queue negatives; K=0 gives in-batch InfoNCE.

        Returns:
            Scalar cross-entropy loss (batchmean).

        Shape trace:
            gallery   : (B+K, D)   pos_keys stacked before neg_keys
            logits    : (B, B+K)   queries @ gallery.T / τ
            targets   : (B,)       arange(B) — positive always at col i
            loss      : ()         F.cross_entropy(logits, targets)
        """
        B   = queries.shape[0]
        τ   = self.temperature

        # Build gallery: positives first (so target = arange(B)), negatives after
        if neg_keys.shape[0] > 0:
            gallery = torch.cat(
                [pos_keys, neg_keys.to(dtype=queries.dtype)], dim=0
            )   # (B+K, D)
        else:
            gallery = pos_keys   # (B, D) — in-batch InfoNCE fallback

        logits  = queries @ gallery.T / τ                          # (B, B+K)
        targets = torch.arange(B, device=queries.device)           # (B,)
        return F.cross_entropy(logits, targets)                    # scalar

    # ──────────────────────────────────────────────────────────────────────
    # Public forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        emb_uav:  Tensor,   # (B, D)  current batch UAV   embeddings
        emb_sat:  Tensor,   # (B, D)  current batch SAT   embeddings
        q_uav:    Tensor,   # (K, D)  queue       UAV   embeddings (detached)
        q_sat:    Tensor,   # (K, D)  queue       SAT   embeddings (detached)
    ) -> Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            emb_uav: (B, D) L2-normalised UAV embeddings (current batch).
            emb_sat: (B, D) L2-normalised SAT embeddings (current batch).
            q_uav:   (K, D) queue UAV embeddings; K may be 0 during warmup.
            q_sat:   (K, D) queue SAT embeddings; K may be 0 during warmup.

        Returns:
            Scalar: (L_u2s + L_s2u) / 2.
        """
        loss_u2s = self._one_direction(emb_uav, emb_sat, q_sat)
        loss_s2u = self._one_direction(emb_sat, emb_uav, q_uav)
        return (loss_u2s + loss_s2u) * 0.5

    def __repr__(self) -> str:
        return f"InfoNCELoss(temperature={self.temperature})"
