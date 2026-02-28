"""
losses/sw_triplet.py — Soft Weighted Triplet Loss (cross-view, bi-directional).

Motivation
──────────
Standard hard-negative triplet mining is non-differentiable and requires a
separate mining step.  Soft-weighted triplet loss instead:
  1. Computes margin violations for ALL negative pairs in the batch.
  2. Weights each violation by the softmax of the corresponding negative
     similarity — harder negatives (higher similarity) get larger gradients.
  3. No mining step needed; the whole batch is used end-to-end.

Cross-view setup
────────────────
Given B paired samples (emb_uav[i], emb_sat[i]) with unique labels:

    Direction 1 — UAV anchor → SAT gallery:
        positive for anchor_i  = sat_i   (same label)
        negatives for anchor_i = sat_j   (j ≠ i, since labels are unique)

    Direction 2 — SAT anchor → UAV gallery (symmetric):
        positive for anchor_i  = uav_i
        negatives for anchor_i = uav_j   (j ≠ i)

    Total loss = (loss_dir1 + loss_dir2) / 2

Soft-weighted formula (per direction)
──────────────────────────────────────
    S[i,j]  = anchor_i · gallery_j        (B×B cosine similarity, L2-normed)
    pos_sim[i] = S[i,i]                   (B,)  diagonal = positive pairs

    neg_mask[i,j] = (label[i] ≠ label[j])
    w[i,j]  = softmax_j({ S[i,k] | neg_mask[i,k] })   soft negative weight
    v[i,j]  = max(0, S[i,j] − pos_sim[i] + margin)    margin violation

    loss_i  = Σ_j  w[i,j] · v[i,j]       (weighted sum over negatives)
    L       = mean_i(loss_i)

Shape:
    emb_uav  : (B, D)  L2-normalised UAV embeddings
    emb_sat  : (B, D)  L2-normalised SAT embeddings
    labels   : (B,)    integer class labels
    output   : ()      scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SoftWeightedTripletLoss(nn.Module):
    """Bi-directional soft-weighted triplet loss for cross-view retrieval.

    Args:
        margin: Cosine-space margin between positive and negative similarities.
                Typical value: 0.3 for L2-normalised embeddings in [-1, 1].
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        assert margin > 0, f"margin must be > 0, got {margin}"
        self.margin = margin

    # ──────────────────────────────────────────────────────────────────────
    # Internal: one retrieval direction
    # ──────────────────────────────────────────────────────────────────────

    def _one_direction(
        self,
        anchor:  Tensor,   # (B, D)  L2-normalised anchor embeddings
        gallery: Tensor,   # (B, D)  L2-normalised gallery embeddings
        labels:  Tensor,   # (B,)    integer class labels
    ) -> Tensor:
        """Soft-weighted triplet loss for one anchor→gallery direction.

        Args:
            anchor:  (B, D) — each row is an anchor embedding.
            gallery: (B, D) — gallery[i] is the POSITIVE for anchor[i].
            labels:  (B,)  — used to build the negative mask.

        Returns:
            Scalar loss.

        Shape trace:
            sim          : (B, B)    anchor[i]·gallery[j]
            sim_pos      : (B, 1)    diagonal of sim (positive pairs)
            neg_mask     : (B, B)    True where labels[i] != labels[j]
            neg_only_sim : (B, B)    sim with positive positions set to -inf
            soft_weights : (B, B)    softmax over negatives; 0 at positives
            violations   : (B, B)    max(0, sim − sim_pos + margin)
            loss_per_anc : (B,)      weighted sum over negatives
            loss         : ()        mean over anchors
        """
        B = anchor.shape[0]

        if B <= 1:
            # Cannot form any triplet without at least 2 samples
            return anchor.new_zeros(()).requires_grad_(True)

        # Cosine similarity matrix (embeddings are L2-normalised)
        sim = anchor @ gallery.T    # (B, B)

        # Positive similarity: diagonal element for each anchor
        sim_pos = sim.diagonal().unsqueeze(1)   # (B, 1) — broadcast over gallery dim

        # Negative mask: True where the two items have different labels
        neg_mask = labels.unsqueeze(1) != labels.unsqueeze(0)   # (B, B) bool

        if not neg_mask.any():
            # Degenerate batch: all labels are the same → no negatives
            return anchor.new_zeros(()).requires_grad_(True)

        # ── Soft weights (softmax over negative similarities) ─────────────
        # Mask positive positions with -inf so softmax assigns them weight 0
        neg_only_sim = sim.masked_fill(~neg_mask, float("-inf"))  # (B, B)
        soft_weights = F.softmax(neg_only_sim, dim=1)             # (B, B)
        # Note: where ~neg_mask, exp(-inf)=0 → weight=0; sum is over negatives only

        # ── Margin violations ─────────────────────────────────────────────
        # violation[i,j] = max(0, S[i,j] - S[i,i] + margin)
        violations = (sim - sim_pos + self.margin).clamp(min=0.0)  # (B, B)

        # ── Weighted loss per anchor ──────────────────────────────────────
        # soft_weights already sums to 1 over negatives; positive weight = 0
        loss_per_anchor = (soft_weights * violations).sum(dim=1)   # (B,)

        return loss_per_anchor.mean()   # scalar

    # ──────────────────────────────────────────────────────────────────────
    # Public forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        emb_uav: Tensor,   # (B, D)  L2-normalised UAV embeddings
        emb_sat: Tensor,   # (B, D)  L2-normalised SAT embeddings
        labels:  Tensor,   # (B,)    integer class labels
    ) -> Tensor:
        """Compute bi-directional soft-weighted triplet loss.

        Args:
            emb_uav: (B, D) — L2-normalised UAV embeddings.
            emb_sat: (B, D) — L2-normalised satellite embeddings.
            labels:  (B,)  — class label per pair (unique within batch).

        Returns:
            Scalar loss averaged over both retrieval directions.

        Shape trace:
            loss_uav2sat : ()   UAV anchor → SAT gallery
            loss_sat2uav : ()   SAT anchor → UAV gallery (symmetric)
            output       : ()   (loss_uav2sat + loss_sat2uav) / 2
        """
        assert emb_uav.ndim == 2, f"emb_uav must be (B,D), got {emb_uav.shape}"
        assert emb_sat.ndim == 2, f"emb_sat must be (B,D), got {emb_sat.shape}"
        assert emb_uav.shape == emb_sat.shape, (
            f"emb_uav {emb_uav.shape} and emb_sat {emb_sat.shape} must match"
        )
        assert labels.shape == (emb_uav.shape[0],), \
            f"labels must be (B,), got {labels.shape}"

        # Direction 1: UAV as anchor, SAT as gallery
        loss_uav2sat = self._one_direction(emb_uav, emb_sat, labels)

        # Direction 2: SAT as anchor, UAV as gallery (symmetric)
        loss_sat2uav = self._one_direction(emb_sat, emb_uav, labels)

        return (loss_uav2sat + loss_sat2uav) * 0.5
