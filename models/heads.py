"""
models/heads.py — pooling heads used by the Siamese model.

GeM (Generalized Mean) Pooling
───────────────────────────────
A learnable generalisation of average pooling:

    GeM(x) = ( 1/(H·W) · Σ_{i} x_i^p )^(1/p)

Special cases:
    p → 1  :  average pooling
    p → ∞  :  max pooling
    p = 3  :  de-emphasises near-zero activations; standard for image retrieval

Reference: Radenovic et al., "Fine-tuning CNN Image Retrieval
           with No Human Annotation", TPAMI 2019.

Shape:
    input  : (B, C, H, W)   feature map
    output : (B, C)         pooled descriptor

Note: L2 normalisation is NOT applied here; it is done in the model
      after pooling so the raw GeM output can also be used for debugging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GeM(nn.Module):
    """Generalized Mean pooling layer.

    Args:
        p:          Initial power parameter.  p=3.0 is standard for retrieval.
        eps:        Small floor applied before pow to avoid gradient issues
                    near zero.  1e-6 is safe for float32.
        learnable:  If True, p is an nn.Parameter optimised by the main
                    optimizer.  If False, p is a fixed float constant.
    """

    def __init__(
        self,
        p:          float = 3.0,
        eps:        float = 1e-6,
        learnable:  bool  = True,
    ) -> None:
        super().__init__()

        if learnable:
            # 0-d trainable scalar; initialised to p
            self.p: nn.Parameter | float = nn.Parameter(torch.tensor(float(p)))
        else:
            self.p = float(p)

        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Pool a (B, C, H, W) feature map to (B, C).

        Shape trace:
            x              : (B, C, H, W)
            x.clamp(...)   : (B, C, H, W)   values in [eps, +∞)
            .pow(p)        : (B, C, H, W)   element-wise ^p
            .mean([-2,-1]) : (B, C)         spatial average
            .pow(1/p)      : (B, C)         ^(1/p) — final descriptor

        Args:
            x: (B, C, H, W) feature map.  Values should be non-negative
               (typical after ReLU / GELU activations).

        Returns:
            (B, C) GeM-pooled descriptor.
        """
        assert x.ndim == 4, f"GeM expects (B,C,H,W) input, got shape {x.shape}"

        return (
            x.clamp(min=self.eps)   # (B, C, H, W)  guard against near-zero
            .pow(self.p)            # (B, C, H, W)  ^p  (differentiable via self.p)
            .mean(dim=[-2, -1])     # (B, C)         spatial mean
            .pow(1.0 / self.p)      # (B, C)         ^(1/p)
        )

    def extra_repr(self) -> str:
        p_val     = self.p.item() if isinstance(self.p, nn.Parameter) else self.p
        learnable = isinstance(self.p, nn.Parameter)
        return f"p={p_val:.3f}, eps={self.eps}, learnable={learnable}"
