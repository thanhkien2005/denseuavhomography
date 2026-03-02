"""
losses/homography_loss.py — Feature-alignment loss for the homography branch.

HomographyAlignmentLoss provides explicit gradient signal to HomographyNet
so the branch is not a no-op.

Loss formula
────────────
Given:
    Fu_warped  : (B, C, H, W)  — UAV feature map warped by predicted delta
    Fs         : (B, C, H, W)  — satellite feature map (detached target)
    gate_logit : (B, 1)        — raw gate logit; gate = sigmoid(gate_logit)
    delta      : (B, 8)        — predicted corner displacements

Step 1 — gate-weighted L1 feature alignment:
    gate      = sigmoid(gate_logit).reshape(B, 1, 1, 1)
    L_align   = mean(gate * |Fu_warped - Fs.detach()|)

    Rationale:
      • Gradient through Fu_warped → delta: pushes HomographyNet to produce a
        warp that aligns UAV features with satellite features.
      • .detach() on Fs: satellite branch is not pulled toward UAV; only the
        UAV→satellite direction is supervised.
      • Gate weighting: loss magnitude is proportional to how much the warped
        branch actually contributes to the final embedding.  Gradient through
        gate_logit: once alignment improves, opening the gate reduces CE/KL
        losses (warped features ≈ satellite features → classification easier),
        reinforcing gate increase.

Step 2 — delta L2 regularisation:
    L_reg     = delta.pow(2).mean()

    Rationale: keeps corner displacements small to prevent degenerate
    homographies (collapsed corners, flipped quads) early in training.

Total:
    L_homo = L_align + lambda_reg * L_reg

Expected training behaviour after this loss is active
─────────────────────────────────────────────────────
    • gate_mean should rise from ~0.12 toward 0.4–0.7 over 10–30 epochs.
    • delta_norm_mean should increase as the network learns non-trivial warps,
      stabilising once the homography is approximately correct.
    • L_align should decrease monotonically as alignment improves.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class HomographyAlignmentLoss(nn.Module):
    """Gate-weighted feature-alignment + delta regularisation loss.

    See module docstring for the full formula and rationale.

    Args:
        lambda_reg: Weight of the delta L2 regularisation term.
    """

    def __init__(self, lambda_reg: float = 0.01) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        Fu_warped:  Tensor,   # (B, C, H, W)
        Fs:         Tensor,   # (B, C, H, W)
        gate_logit: Tensor,   # (B, 1)
        delta:      Tensor,   # (B, 8)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute L_homo, L_align, and L_reg.

        Args:
            Fu_warped:  Warped UAV feature map (B, C, H, W).
            Fs:         Satellite feature map; used as target (detached internally).
            gate_logit: Raw gate logit from HomographyNet (B, 1).
            delta:      Corner displacements from HomographyNet (B, 8).

        Returns:
            loss_homo  : scalar — L_align + lambda_reg * L_reg
            loss_align : scalar — gate-weighted L1 alignment component
            loss_reg   : scalar — delta L2 regularisation component

        Shape trace:
            gate       : (B, 1, 1, 1)    broadcast over C, H, W
            diff       : (B, C, H, W)    |Fu_warped - Fs.detach()|
            loss_align : ()
            loss_reg   : ()
        """
        gate = torch.sigmoid(gate_logit).reshape(-1, 1, 1, 1)   # (B,1,1,1)

        # Gate-weighted L1 alignment; satellite features are the frozen target
        loss_align = (gate * (Fu_warped - Fs.detach()).abs()).mean()

        # Delta L2 regularisation: prevent extreme / degenerate homographies
        loss_reg = delta.pow(2).mean()

        loss_homo = loss_align + self.lambda_reg * loss_reg

        return loss_homo, loss_align, loss_reg

    def __repr__(self) -> str:
        return f"HomographyAlignmentLoss(lambda_reg={self.lambda_reg})"
