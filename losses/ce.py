"""
losses/ce.py — Cross-entropy loss wrapper.

Applies standard cross-entropy to classifier logits for one modality.
The DenseUAVLoss calls this twice (once for UAV, once for satellite)
and averages the result.

Shape:
    logits : (B, num_classes)  float32
    labels : (B,)              int64
    output : scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with optional label smoothing.

    Args:
        label_smoothing: Smoothing factor in [0, 1).  0 = standard CE.
                         Small values (0.0–0.1) often improve generalisation.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        assert 0.0 <= label_smoothing < 1.0, (
            f"label_smoothing must be in [0,1), got {label_smoothing}"
        )
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: (B, C)  — raw classifier outputs (no temperature applied).
            labels: (B,)    — integer class indices in [0, C).

        Returns:
            Scalar loss averaged over the batch.

        Shape trace:
            logits : (B, C)
            labels : (B,)
            output : ()     scalar
        """
        assert logits.ndim == 2, \
            f"logits must be (B, C), got {logits.shape}"
        assert labels.ndim == 1, \
            f"labels must be (B,), got {labels.shape}"
        assert logits.shape[0] == labels.shape[0], (
            f"Batch size mismatch: logits {logits.shape[0]} vs "
            f"labels {labels.shape[0]}"
        )

        return F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.label_smoothing,
            reduction="mean",
        )
