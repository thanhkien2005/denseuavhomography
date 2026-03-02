"""
losses/total_loss.py — Combined DenseUAV training loss.

Total loss (DenseUAV style):
    L = w_ce      · (CE(logit_uav, y) + CE(logit_sat, y)) / 2
      + w_triplet  · SWTriplet(emb_uav, emb_sat, y)
      + w_kl       · BiKL(logit_uav, logit_sat)
      + w_homo     · HomographyAlignmentLoss(Fu_warped, Fs, gate_logit, delta)

All weights and temperature come from configs/denseuav_v1.yaml → loss section.

forward() input:
    outputs : dict returned by SiameseViT.forward()
        "emb_uav"    : (B, 384)
        "emb_sat"    : (B, 384)
        "logit_uav"  : (B, num_classes)
        "logit_sat"  : (B, num_classes)
        "gate_logit" : (B, 1)
        "delta"      : (B, 8)
        "Fu_raw"     : (B, 384, 32, 32)
        "Fu_warped"  : (B, 384, 32, 32)
        "Fs"         : (B, 384, 32, 32)
    labels  : (B,)  integer class labels

forward() output:
    total_loss : scalar Tensor
    loss_dict  : dict[str, Tensor] with individual (weighted) components
        "total"      : same as total_loss
        "ce"         : weighted CE component
        "triplet"    : weighted triplet component
        "kl"         : weighted KL component
        "homo"       : weighted homography alignment component
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.ce               import LabelSmoothingCE
from losses.homography_loss  import HomographyAlignmentLoss
from losses.kl               import BidirectionalKLLoss
from losses.sw_triplet       import SoftWeightedTripletLoss


class DenseUAVLoss(nn.Module):
    """Combined CE + SoftWeightedTriplet + Bi-directional KL + HomographyAlignment loss.

    Args:
        w_ce:            Weight for cross-entropy term.
        w_triplet:       Weight for soft-weighted triplet term.
        w_kl:            Weight for bi-directional KL term.
        w_homo:          Weight for homography alignment term.
        temperature:     Temperature for KL softmax scaling.
        margin:          Cosine-space triplet margin.
        label_smoothing: Label smoothing for CE (0 = no smoothing).
        lambda_reg:      Delta regularisation weight inside HomographyAlignmentLoss.
    """

    def __init__(
        self,
        w_ce:            float = 1.0,
        w_triplet:       float = 1.0,
        w_kl:            float = 1.0,
        w_homo:          float = 0.5,
        temperature:     float = 0.07,
        margin:          float = 0.3,
        label_smoothing: float = 0.0,
        lambda_reg:      float = 0.01,
    ) -> None:
        super().__init__()

        self.w_ce      = w_ce
        self.w_triplet = w_triplet
        self.w_kl      = w_kl
        self.w_homo    = w_homo

        self.ce      = LabelSmoothingCE(label_smoothing=label_smoothing)
        self.triplet = SoftWeightedTripletLoss(margin=margin)
        self.kl      = BidirectionalKLLoss(temperature=temperature)
        self.homo    = HomographyAlignmentLoss(lambda_reg=lambda_reg)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        outputs: Dict[str, Tensor],
        labels:  Tensor,               # (B,)
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute total DenseUAV loss.

        Args:
            outputs: Dict from SiameseViT.forward() with keys:
                "emb_uav"    (B, D)
                "emb_sat"    (B, D)
                "logit_uav"  (B, C)
                "logit_sat"  (B, C)
                "gate_logit" (B, 1)
                "delta"      (B, 8)
                "Fu_warped"  (B, C, H, W)
                "Fs"         (B, C, H, W)
            labels: (B,) integer class labels.

        Returns:
            total_loss : scalar Tensor (for .backward())
            loss_dict  : {
                "total"   : scalar — total weighted loss
                "ce"      : scalar — CE component (already weighted)
                "triplet" : scalar — triplet component (already weighted)
                "kl"      : scalar — KL component (already weighted)
                "homo"    : scalar — homography alignment component (already weighted)
            }
        """
        emb_uav   = outputs["emb_uav"]    # (B, D)
        emb_sat   = outputs["emb_sat"]    # (B, D)
        logit_uav = outputs["logit_uav"]  # (B, C)
        logit_sat = outputs["logit_sat"]  # (B, C)

        # ── CE loss: average over both modalities ─────────────────────────
        loss_ce = self.w_ce * (
            self.ce(logit_uav, labels) + self.ce(logit_sat, labels)
        ) * 0.5                         # scalar

        # ── Soft-weighted triplet loss ─────────────────────────────────────
        loss_triplet = self.w_triplet * self.triplet(emb_uav, emb_sat, labels)

        # ── Bi-directional KL (mutual learning) ───────────────────────────
        loss_kl = self.w_kl * self.kl(logit_uav, logit_sat)

        # ── Homography alignment: push HomographyNet to actually warp ─────
        loss_homo_raw, _, _ = self.homo(
            outputs["Fu_warped"],
            outputs["Fs"],
            outputs["gate_logit"],
            outputs["delta"],
        )
        loss_homo = self.w_homo * loss_homo_raw

        # ── Total ─────────────────────────────────────────────────────────
        total = loss_ce + loss_triplet + loss_kl + loss_homo

        loss_dict: Dict[str, Tensor] = {
            "total":   total,
            "ce":      loss_ce,
            "triplet": loss_triplet,
            "kl":      loss_kl,
            "homo":    loss_homo,
        }

        return total, loss_dict

    def __repr__(self) -> str:
        return (
            f"DenseUAVLoss("
            f"w_ce={self.w_ce}, "
            f"w_triplet={self.w_triplet}, "
            f"w_kl={self.w_kl}, "
            f"w_homo={self.w_homo}, "
            f"T={self.kl.temperature}, "
            f"margin={self.triplet.margin})"
        )
