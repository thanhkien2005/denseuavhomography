"""
models/homography_net.py — lightweight CNN that predicts homography parameters.

HomographyNet
─────────────
Inputs : uav_img (B, 3, H, W)  +  sat_img (B, 3, H, W)
Outputs: delta      (B, 8)   — 4-corner displacements in 32×32 feature-grid pixels
         gate_logit (B, 1)   — raw logit for sigmoid gate g = σ(gate_logit)

Architecture
────────────
1. Resize both images to in_size × in_size (default 128) for efficiency.
2. Concat along channel → (B, 6, in_size, in_size).
3. Four conv+ReLU blocks with stride-2 downsampling.
4. AdaptiveAvgPool(1,1) + Flatten → (B, homo_hidden).
5. One FC+ReLU hidden layer.
6. Two separate linear heads:
      head_delta : outputs (B, 8)  — zero-initialized → identity warp at start
      head_gate  : outputs (B, 1)  — bias = gate_bias_init ≈ -2 → g ≈ 0.12

Shape trace (in_size=128, homo_hidden=256):
    concat      : (B,  6, 128, 128)
    conv1 s=2   : (B, 32,  64,  64)
    conv2 s=2   : (B, 64,  32,  32)
    conv3 s=2   : (B,128,  16,  16)
    conv4 s=2   : (B,256,   8,   8)
    gap+flatten : (B,256)
    fc+relu     : (B,256)
    head_delta  : (B,  8)
    head_gate   : (B,  1)

Initialization guarantees
─────────────────────────
• head_delta weights + bias = 0  → delta = 0 at init → identity warp
• head_gate  weights = 0, bias = gate_bias_init (default -2.0)
  → g = sigmoid(-2) ≈ 0.12  → small warp contribution at the start of training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HomographyNet(nn.Module):
    """Lightweight CNN predicting homography corner deltas and a blend gate.

    Args:
        homo_hidden:    Width of the CNN output and FC hidden layer (default 256).
        gate_bias_init: Initial bias for gate logit head; sigmoid of this value
                        is the initial gate g.  Negative → small gate at start.
        in_size:        Spatial size to which both input images are resized
                        before the CNN.  128 keeps memory small while retaining
                        enough detail for geometric estimation.
    """

    def __init__(
        self,
        homo_hidden:    int   = 256,
        gate_bias_init: float = -2.0,
        in_size:        int   = 128,
    ) -> None:
        super().__init__()

        self.in_size = in_size

        # ── CNN backbone ──────────────────────────────────────────────────
        # 4 × (Conv3×3-s2 + ReLU); input channels = 6 (concat UAV+SAT)
        # Output spatial size: in_size / 2^4  (128→8 for in_size=128)
        self.cnn = nn.Sequential(
            # (B,  6, 128, 128)
            nn.Conv2d(6,           32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B, 32,  64,  64)
            nn.Conv2d(32,          64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B, 64,  32,  32)
            nn.Conv2d(64,         128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B,128,  16,  16)
            nn.Conv2d(128, homo_hidden, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # (B,256,   8,   8)
            nn.AdaptiveAvgPool2d(1),    # (B, 256, 1, 1)
            nn.Flatten(),               # (B, 256)
        )

        # ── FC hidden layer ───────────────────────────────────────────────
        self.fc = nn.Sequential(
            nn.Linear(homo_hidden, homo_hidden),
            nn.ReLU(inplace=True),
        )   # (B, 256)

        # ── Output heads ─────────────────────────────────────────────────
        # delta: 4 corners × 2 (dx, dy) = 8 values
        self.head_delta = nn.Linear(homo_hidden, 8)   # (B, 8)
        # gate: scalar logit; g = sigmoid(gate_logit)
        self.head_gate  = nn.Linear(homo_hidden, 1)   # (B, 1)

        # ── Init: zero-out delta head → identity warp at start ────────────
        nn.init.zeros_(self.head_delta.weight)
        nn.init.zeros_(self.head_delta.bias)

        # ── Init: small gate at start → warp barely active initially ─────
        nn.init.zeros_(self.head_gate.weight)
        nn.init.constant_(self.head_gate.bias, gate_bias_init)
        # sigmoid(gate_bias_init) = sigmoid(-2.0) ≈ 0.12

    def forward(
        self,
        uav_img: Tensor,   # (B, 3, H, W)  — typically (B, 3, 512, 512)
        sat_img: Tensor,   # (B, 3, H, W)
    ) -> tuple[Tensor, Tensor]:
        """Predict homography corner displacements and blend gate.

        Args:
            uav_img: (B, 3, H, W) UAV image.
            sat_img: (B, 3, H, W) satellite image.

        Returns:
            delta      : (B, 8)   corner displacements [dx0,dy0,dx1,dy1,...,dx3,dy3]
                         in 32×32 feature-grid pixel units.
                         delta = 0 → identity (no warp).
            gate_logit : (B, 1)   raw gate logit; apply sigmoid to get g ∈ (0,1).

        Shape trace:
            resize uav  : (B, 3, in_size, in_size)   e.g. (B,3,128,128)
            resize sat  : (B, 3, in_size, in_size)
            concat      : (B, 6, in_size, in_size)
            cnn output  : (B, homo_hidden)            e.g. (B,256)
            fc output   : (B, homo_hidden)
            head_delta  : (B, 8)
            head_gate   : (B, 1)
        """
        # Resize for efficiency (bilinear, no align_corners for consistency)
        if uav_img.shape[-1] != self.in_size or uav_img.shape[-2] != self.in_size:
            uav_img = F.interpolate(
                uav_img, size=(self.in_size, self.in_size),
                mode="bilinear", align_corners=False,
            )   # (B, 3, in_size, in_size)
            sat_img = F.interpolate(
                sat_img, size=(self.in_size, self.in_size),
                mode="bilinear", align_corners=False,
            )   # (B, 3, in_size, in_size)

        x = torch.cat([uav_img, sat_img], dim=1)   # (B, 6, in_size, in_size)
        x = self.cnn(x)                             # (B, homo_hidden)
        x = self.fc(x)                              # (B, homo_hidden)

        delta      = self.head_delta(x)             # (B, 8)
        gate_logit = self.head_gate(x)              # (B, 1)

        return delta, gate_logit
