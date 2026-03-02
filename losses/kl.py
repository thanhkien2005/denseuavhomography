"""
losses/kl.py — Bi-directional KL divergence for mutual learning.

Mutual learning (DenseUAV style)
─────────────────────────────────
Each branch (UAV / satellite) acts as both student and teacher for the other.
Instead of using hard labels alone, the model also minimises the KL divergence
between the softened predictions of the two views.

Temperature scaling
───────────────────
Dividing logits by T before softmax controls distribution sharpness:
    T → 0 : one-hot (very confident) → KL gradient vanishes
    T = 1 : standard softmax
    T >> 1 : uniform → KL gradient is dominated by noise

T = 0.07 (contrastive-learning default) produces near-one-hot distributions
from the start.  Once both branches agree on the top class, KL → 0 and the
term contributes nothing to training.  T = 4.0 keeps distributions soft
enough for continued mutual-learning gradient throughout training.

T² scaling (Hinton et al. 2015)
────────────────────────────────
When logits are divided by T the gradient of KL w.r.t. the logits is also
divided by T (chain rule through softmax).  Multiplying the loss by T² restores
the gradient magnitude to the T = 1 scale, preventing the KL term from being
swamped by CE/triplet terms just because T is large.

Formula
───────
    P_u = softmax(logit_u / T)           (B, C)
    P_s = softmax(logit_s / T)           (B, C)

    KL(P_u ‖ P_s) = Σ_c P_u_c · (log P_u_c − log P_s_c)
    KL(P_s ‖ P_u) = Σ_c P_s_c · (log P_s_c − log P_u_c)

    L_kl = T² · [KL(P_u ‖ P_s) + KL(P_s ‖ P_u)] / 2

Shape:
    logit_u : (B, C)   UAV classifier logits
    logit_s : (B, C)   satellite classifier logits
    output  : ()       scalar
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BidirectionalKLLoss(nn.Module):
    """Bi-directional KL divergence between temperature-scaled softmax outputs.

    Args:
        temperature: Logit scaling before softmax.  Must be > 0.
    """

    def __init__(self, temperature: float = 4.0) -> None:
        super().__init__()
        assert temperature > 0.0, \
            f"temperature must be > 0, got {temperature}"
        self.temperature = temperature

    def forward(self, logit_u: Tensor, logit_s: Tensor) -> Tensor:
        """Compute symmetric KL divergence.

        Args:
            logit_u: (B, C)  UAV classifier logits (raw, no prior softmax).
            logit_s: (B, C)  Satellite classifier logits.

        Returns:
            Scalar loss: T² · mean of KL(P_u ‖ P_s) and KL(P_s ‖ P_u).

        Shape trace:
            logit_u / T      : (B, C)
            log_p_u          : (B, C)    log-probabilities for UAV branch
            log_p_s          : (B, C)    log-probabilities for SAT branch
            kl_u_s           : ()        KL(P_u ‖ P_s)  using log_target=True
            kl_s_u           : ()        KL(P_s ‖ P_u)  using log_target=True
            output           : ()        T² · (kl_u_s + kl_s_u) / 2

        Note on F.kl_div convention:
            F.kl_div(input=log_q, target=log_p, log_target=True)
                = Σ exp(log_p) · (log_p − log_q)
                = Σ p · (log p − log q)
                = KL(p ‖ q)
            So: F.kl_div(log_p_s, log_p_u, log_target=True) = KL(P_u ‖ P_s).
        """
        assert logit_u.ndim == 2, \
            f"logit_u must be (B, C), got {logit_u.shape}"
        assert logit_s.ndim == 2, \
            f"logit_s must be (B, C), got {logit_s.shape}"
        assert logit_u.shape == logit_s.shape, (
            f"logit_u {logit_u.shape} and logit_s {logit_s.shape} must match"
        )

        T = self.temperature

        # Log-probabilities for both branches: numerically stable via log_softmax
        log_p_u = F.log_softmax(logit_u / T, dim=1)   # (B, C)
        log_p_s = F.log_softmax(logit_s / T, dim=1)   # (B, C)

        # KL(P_u ‖ P_s): how different is P_u from P_s
        # input = log_p_s (log Q), target = log_p_u (log P), log_target=True
        kl_u_s = F.kl_div(
            log_p_s, log_p_u,
            reduction="batchmean",
            log_target=True,
        )   # scalar  KL(P_u ‖ P_s)

        # KL(P_s ‖ P_u): symmetric direction
        kl_s_u = F.kl_div(
            log_p_u, log_p_s,
            reduction="batchmean",
            log_target=True,
        )   # scalar  KL(P_s ‖ P_u)

        # T² factor restores gradient magnitude to the T=1 scale
        # (Hinton et al. 2015 — classic knowledge distillation scaling).
        return (kl_u_s + kl_s_u) * 0.5 * (T * T)
