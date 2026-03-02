"""
models/cosine_head.py — Weight-normalised cosine classifier.

Replaces the plain nn.Linear head when embeddings are L2-normalised.

    logits = s * (emb @ normalize(W, dim=1)^T)

where s is a configurable logit scale (learnable or fixed).

Motivation
──────────
With L2-normalised embeddings the plain nn.Linear head gives logits ≈ 0.05
(kaiming-init magnitude on the unit sphere), so softmax ≈ uniform and
CE ≈ ln(C) throughout training.  A logit scale s≈30 lifts the logit range
to [−30, 30], matching NormFace / CosFace / ArcFace practice and allowing
the cross-entropy loss to actually converge.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CosineClassifier(nn.Module):
    """Weight-normalised cosine classifier with configurable logit scale.

    Args:
        in_features:      Embedding dimension (e.g. 384).
        num_classes:      Number of output classes (e.g. 2256).
        scale:            Initial logit scale s.  With L2-normalised inputs
                          logits ∈ [−s, s]; s=30.0 is a common default.
        learnable_scale:  If True, s is an ``nn.Parameter`` updated by the
                          main optimiser.  If False, s is a fixed buffer.

    Forward:
        x (B, D)  — L2-normalised embeddings
        → (B, C)  — logits ∈ [−scale, scale]
    """

    def __init__(
        self,
        in_features:     int,
        num_classes:     int,
        scale:           float = 30.0,
        learnable_scale: bool  = True,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.num_classes  = num_classes

        # Weight matrix — same shape as nn.Linear weight (C, D), bias=False:
        # L2-normalised embeddings make a bias term redundant.
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: Tensor) -> Tensor:
        """Cosine-scaled logits.

        Args:
            x: (B, D) L2-normalised embeddings.

        Returns:
            logits: (B, C), values in [−scale, scale].
        """
        w_norm = F.normalize(self.weight, p=2, dim=1)   # (C, D)
        return self.scale * F.linear(x, w_norm)          # (B, C)

    def extra_repr(self) -> str:
        learnable = isinstance(self.scale, nn.Parameter)
        return (
            f"in_features={self.in_features}, "
            f"num_classes={self.num_classes}, "
            f"scale={self.scale.item():.2f}, "
            f"learnable_scale={learnable}"
        )
