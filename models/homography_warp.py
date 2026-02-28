"""
models/homography_warp.py — differentiable homography warp on feature maps.

HomographyWarpLayer
────────────────────
Given a feature map and 8-value corner-displacement delta, this layer:
  1. Builds dst corners = src corners + delta.reshape(B,4,2)
  2. Computes H (B,3,3) via kornia.get_perspective_transform(src, dst)
     H maps the source corner coordinates to the destination corners.
  3. Warps the feature map with kornia.warp_perspective(feat, H, dsize,
     align_corners=True).
  4. Returns the warped feature map (same shape as input).

The layer is fully differentiable:
  • Gradients flow through delta → dst → H → warp_perspective → warped.
  • Gradients also flow through feat → warp_perspective.

Corner coordinate convention
────────────────────────────
For a G×G feature map (G=32), the four corners in (x=col, y=row) order are:
    top-left     : (0,   0  )
    top-right    : (G-1, 0  )
    bottom-right : (G-1, G-1)
    bottom-left  : (0,   G-1)

delta[b] = [dx0, dy0, dx1, dy1, dx2, dy2, dx3, dy3]
           displacement of corner i by (dxi, dyi) in feature-grid pixels.

Verified: delta = 0 → H = identity → warped = feat (max_abs_diff = 0.0).

Shape:
    feat_map : (B, C, G, G)   e.g. (B, 384, 32, 32)
    delta    : (B, 8)
    H        : (B, 3, 3)      homography matrix
    output   : (B, C, G, G)   warped feature map
"""

from __future__ import annotations

import kornia.geometry.transform as KGT
import torch
import torch.nn as nn
from torch import Tensor


class HomographyWarpLayer(nn.Module):
    """Differentiable feature-map warp via predicted 4-corner homography.

    Args:
        grid_size: Spatial size of the feature map (G).  Default 32 matches
                   ViT-S with img_size=512, patch_size=16: 512/16 = 32.
    """

    def __init__(self, grid_size: int = 32) -> None:
        super().__init__()

        self.G = grid_size
        G      = float(grid_size - 1)   # max pixel coordinate (0-indexed)

        # Source corners in (x, y) = (col, row) order: TL, TR, BR, BL
        # Registered as a buffer so it moves with .to(device) automatically.
        src = torch.tensor(
            [[0., 0.],          # top-left
             [G,  0.],          # top-right
             [G,  G ],          # bottom-right
             [0., G ]],         # bottom-left
            dtype=torch.float32,
        )   # (4, 2)
        self.register_buffer("src_corners", src)   # lives on same device as model

    def delta_to_H(self, delta: Tensor) -> Tensor:
        """Compute homography H from corner displacements delta.

        Args:
            delta: (B, 8) — [dx0,dy0, dx1,dy1, dx2,dy2, dx3,dy3]
                   displacement for each of the 4 corners.

        Returns:
            H: (B, 3, 3)  homography mapping src corners → dst corners.

        Shape trace:
            src : (B, 4, 2)   broadcast of src_corners
            dst : (B, 4, 2)   src + delta.reshape(B,4,2)
            H   : (B, 3, 3)   from kornia.get_perspective_transform
        """
        B   = delta.shape[0]
        src = self.src_corners.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 2)
        dst = src + delta.reshape(B, 4, 2)                     # (B, 4, 2)

        # kornia convention:
        #   get_perspective_transform(src_pts, dst_pts) → H such that dst = H·src
        #   warp_perspective(image, H, dsize) applies H^{-1} for inverse sampling
        H = KGT.get_perspective_transform(src, dst)            # (B, 3, 3)
        return H

    def forward(self, feat_map: Tensor, delta: Tensor) -> Tensor:
        """Warp feature map according to the homography induced by delta.

        Args:
            feat_map: (B, C, G, G)  — e.g. (B, 384, 32, 32)
            delta:    (B, 8)        — corner displacements in feature-grid pixels.
                      delta = 0 ↔ identity transform (output == input).

        Returns:
            warped: (B, C, G, G)  — same shape as feat_map.

        Shape trace:
            H      : (B, 3, 3)      from delta_to_H
            warped : (B, C, G, G)   from kornia.warp_perspective
        """
        assert feat_map.ndim == 4, \
            f"feat_map must be (B,C,H,W), got {feat_map.shape}"
        assert delta.ndim == 2 and delta.shape[1] == 8, \
            f"delta must be (B,8), got {delta.shape}"
        assert feat_map.shape[0] == delta.shape[0], \
            "Batch size mismatch between feat_map and delta"

        B, C, H_f, W_f = feat_map.shape

        H_mat = self.delta_to_H(delta)   # (B, 3, 3)

        # warp_perspective uses inverse mapping:
        # dst[u,v] = bilinear_sample(src, H_mat^{-1} · [u,v,1]^T)
        # align_corners=True: pixel 0 → normalised -1, pixel G-1 → normalised +1
        # padding_mode='zeros': out-of-bounds regions filled with 0.0
        warped = KGT.warp_perspective(
            feat_map.float(),       # ensure float32 (safe for AMP contexts)
            H_mat,
            dsize=(H_f, W_f),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )   # (B, C, G, G)

        assert warped.shape == feat_map.shape, \
            f"warped shape {warped.shape} != input shape {feat_map.shape}"

        return warped
