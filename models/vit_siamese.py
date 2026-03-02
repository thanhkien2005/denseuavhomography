"""
models/vit_siamese.py — Siamese ViT-S model for UAV geo-localisation.

Architecture overview
─────────────────────
Both UAV and satellite images pass through the SAME (shared) backbone weights.
The UAV branch additionally receives a network-predicted homography warp before
pooling, allowing the model to align the UAV feature map toward the satellite
viewpoint.

Full forward pipeline
─────────────────────
    uav_img (B,3,512,512)          sat_img (B,3,512,512)
         │                                  │
    ViT-S backbone (shared)          ViT-S backbone (shared)
    forward_features()               forward_features()
         │                                  │
    reshape_patch_tokens()          reshape_patch_tokens()
         │                                  │
    Fu_raw (B,384,32,32)            Fs (B,384,32,32)
         │           ↖ both images             │
         │        HomographyNet                │
         │        delta (B,8)                  │
         │        gate_logit (B,1)             │
         │              │                      │
         │        HomographyWarpLayer          │
         │        Fu_warped (B,384,32,32)      │
         │              │                      │
         │  Fu = gate * Fu_warped              │
         │     + (1-gate) * Fu_raw             │
         │                                     │
    GeM pooling (B,384)              GeM pooling (B,384)
    L2 normalise                     L2 normalise
    emb_uav (B,384)                  emb_sat (B,384)
         │                                  │
    Linear → logit_uav (B,C)    Linear → logit_sat (B,C)

Shared weights
──────────────
One backbone + one GeM + one classifier are used for BOTH modalities.

forward() output dict
─────────────────────
{
    "emb_uav"    : (B, 384)         — L2-normalised UAV embedding
    "emb_sat"    : (B, 384)         — L2-normalised SAT embedding
    "logit_uav"  : (B, num_classes) — UAV classifier logits
    "logit_sat"  : (B, num_classes) — SAT classifier logits
    "gate_logit" : (B, 1)           — raw gate logit (sigmoid → gate g)
    "delta"      : (B, 8)           — predicted corner displacements
    "Fu_raw"     : (B, 384, 32, 32) — UAV feature map before warp (for homo loss)
    "Fu_warped"  : (B, 384, 32, 32) — UAV feature map after warp  (for homo loss)
    "Fs"         : (B, 384, 32, 32) — SAT feature map             (for homo loss)
}
"""

from __future__ import annotations

import math
import os
import sys

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Allow running this file directly (python models/vit_siamese.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cosine_head    import CosineClassifier
from models.heads          import GeM
from models.homography_net import HomographyNet
from models.homography_warp import HomographyWarpLayer


# ─────────────────────────────────────────────────────────────────────────────
# Pos-embed interpolation safety helper
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate_pos_embed(
    backbone:   nn.Module,
    img_size:   int,
    patch_size: int,
) -> None:
    """Bicubic-resize backbone.pos_embed in-place if it doesn't match img_size.

    timm 1.0.x handles this automatically; this function is a safe fallback
    for older timm versions or edge cases.

    Args:
        backbone:   timm VisionTransformer instance.
        img_size:   Target image size (e.g. 512).
        patch_size: Patch size (e.g. 16).
    """
    expected_len = (img_size // patch_size) ** 2 + 1   # +1 for cls token
    pos_embed    = backbone.pos_embed                   # (1, actual_len, C)
    actual_len   = pos_embed.shape[1]

    if actual_len == expected_len:
        return   # already correct — nothing to do

    C        = pos_embed.shape[2]
    cls_pe   = pos_embed[:, :1, :]                      # (1, 1, C)
    patch_pe = pos_embed[:, 1:, :]                      # (1, old_N, C)

    old_N    = patch_pe.shape[1]
    old_size = int(math.sqrt(old_N))
    assert old_size * old_size == old_N, (
        f"pos_embed patch count {old_N} is not a perfect square; "
        "cannot interpolate automatically."
    )
    new_size = img_size // patch_size                   # e.g. 32

    # (1, old_N, C) → (1, C, old_size, old_size) for interpolate()
    patch_pe = (
        patch_pe
        .reshape(1, old_size, old_size, C)
        .permute(0, 3, 1, 2)                            # (1, C, old_size, old_size)
    )
    patch_pe = F.interpolate(
        patch_pe,
        size=(new_size, new_size),
        mode="bicubic",
        align_corners=False,
    )                                                   # (1, C, new_size, new_size)

    # (1, C, new_size, new_size) → (1, new_N, C)
    patch_pe = (
        patch_pe
        .permute(0, 2, 3, 1)
        .reshape(1, new_size * new_size, C)
    )

    new_pos_embed = torch.cat([cls_pe, patch_pe], dim=1)  # (1, 1+new_N, C)
    backbone.pos_embed = nn.Parameter(new_pos_embed)

    print(
        f"[SiameseViT] pos_embed interpolated: "
        f"{old_size}x{old_size}→{new_size}x{new_size}  "
        f"({old_N+1} tokens → {new_size**2+1} tokens)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class SiameseViT(nn.Module):
    """Shared-weight ViT-S Siamese model for cross-view geo-localisation.

    The UAV branch predicts a homography from both images via HomographyNet,
    warps the UAV feature map, then blends the warped and original maps using a
    learned gate before pooling.  The satellite branch is pooled unchanged.

    Args:
        num_classes:    Number of training location classes (e.g. 2256).
        embed_dim:      ViT-S embed dimension (384); must match backbone.
        img_size:       Input image spatial size (512 → 32×32 patch grid).
        patch_size:     ViT patch size (16).
        gem_p:          Initial GeM power parameter.
        gem_learnable:  Whether to learn GeM p during training.
        pretrained:          If True, load ImageNet-pretrained timm weights.
        homo_hidden:         Hidden width of HomographyNet (default 256).
        gate_bias_init:      Initial gate bias; sigmoid of this → initial g.
        head_scale:          Initial logit scale s for CosineClassifier (30.0).
        head_learnable_scale: If True, s is learned; otherwise fixed.
    """

    _BACKBONE_NAME = "vit_small_patch16_224"

    def __init__(
        self,
        num_classes:    int,
        embed_dim:      int   = 384,
        img_size:       int   = 512,
        patch_size:     int   = 16,
        gem_p:          float = 3.0,
        gem_learnable:  bool  = True,
        pretrained:     bool  = True,
        homo_hidden:         int   = 256,
        gate_bias_init:      float = -2.0,
        head_scale:          float = 30.0,
        head_learnable_scale: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim   = embed_dim             # 384
        self.img_size    = img_size              # 512
        self.patch_size  = patch_size            # 16
        self.grid_size   = img_size // patch_size  # 32
        self.n_patches   = self.grid_size ** 2    # 1024
        self.num_classes = num_classes

        # ── Backbone (shared weights for UAV and SAT) ──────────────────────
        # timm 1.0.x: img_size=512 triggers automatic pos_embed interpolation
        # from 224-pretrained weights (14×14 → 32×32 grid).
        # global_pool='' → forward_features() returns ALL tokens (B, N+1, C).
        # num_classes=0  → removes timm's classification head entirely.
        self.backbone = timm.create_model(
            self._BACKBONE_NAME,
            pretrained  = pretrained,
            img_size    = img_size,
            num_classes = 0,
            global_pool = "",
        )

        # Safety: explicit bicubic fallback for timm versions that don't
        # auto-interpolate pos_embed when img_size != pretraining size.
        _interpolate_pos_embed(self.backbone, img_size=img_size, patch_size=patch_size)

        # Sanity: confirm backbone embed_dim matches requested embed_dim
        actual_dim = self.backbone.embed_dim
        assert actual_dim == embed_dim, (
            f"Backbone '{self._BACKBONE_NAME}' has embed_dim={actual_dim} "
            f"but SiameseViT was configured with embed_dim={embed_dim}. "
            "Adjust embed_dim or choose a different backbone."
        )

        # Sanity: confirm pos_embed is correctly sized after creation
        actual_len   = self.backbone.pos_embed.shape[1]
        expected_len = self.n_patches + 1                  # 1025
        assert actual_len == expected_len, (
            f"pos_embed has {actual_len} tokens after construction; "
            f"expected {expected_len} for img_size={img_size}, patch_size={patch_size}."
        )

        # ── HomographyNet: predicts delta (B,8) and gate_logit (B,1) ──────
        self.homo_net = HomographyNet(
            homo_hidden    = homo_hidden,
            gate_bias_init = gate_bias_init,
        )

        # ── HomographyWarpLayer: warps UAV feature map ─────────────────────
        self.warp_layer = HomographyWarpLayer(grid_size=self.grid_size)

        # ── Shared pooling + classifier ───────────────────────────────────
        # GeM: (B, 384, 32, 32) → (B, 384)
        self.gem        = GeM(p=gem_p, learnable=gem_learnable)
        # Cosine classifier: logits = s * (emb @ normalize(W)^T)
        # With L2-normalised inputs logits ∈ [−s, s]; s≈30 prevents
        # the CE ≈ ln(C) ceiling caused by plain Linear on a unit sphere.
        self.classifier = CosineClassifier(
            in_features     = embed_dim,
            num_classes     = num_classes,
            scale           = head_scale,
            learnable_scale = head_learnable_scale,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public feature-extraction helpers
    # ──────────────────────────────────────────────────────────────────────

    def forward_features(self, x: Tensor) -> Tensor:
        """Run the ViT backbone and return the full token sequence.

        Args:
            x: (B, 3, img_size, img_size) — e.g. (B, 3, 512, 512)

        Returns:
            tokens: (B, 1025, 384)
                    Index 0      = cls token
                    Index 1-1024 = patch tokens in raster order
        """
        tokens = self.backbone.forward_features(x)   # (B, 1025, 384)

        assert tokens.shape[1] == self.n_patches + 1, (
            f"forward_features returned {tokens.shape[1]} tokens; "
            f"expected {self.n_patches + 1}."
        )
        assert tokens.shape[2] == self.embed_dim, (
            f"forward_features embed dim {tokens.shape[2]} != {self.embed_dim}."
        )
        return tokens   # (B, 1025, 384)

    def reshape_patch_tokens(self, tokens: Tensor) -> Tensor:
        """Convert patch token sequence into a 2-D spatial feature map.

        Args:
            tokens: (B, 1025, 384) — full token sequence (cls + patches)

        Returns:
            feat_map: (B, 384, 32, 32)
        """
        B, L, C = tokens.shape
        assert L == self.n_patches + 1, (
            f"reshape_patch_tokens: expected {self.n_patches+1} tokens, got {L}."
        )

        patch_tokens = tokens[:, 1:, :]                            # (B, 1024, 384)
        feat_map = (
            patch_tokens
            .transpose(1, 2)                                        # (B, 384, 1024)
            .reshape(B, C, self.grid_size, self.grid_size)          # (B, 384, 32, 32)
        )
        return feat_map   # (B, 384, 32, 32)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        uav_img: Tensor,   # (B, 3, 512, 512)
        sat_img: Tensor,   # (B, 3, 512, 512)
    ) -> dict:
        """Encode both branches; warp UAV features with predicted homography.

        Pipeline
        ────────
        1. Extract Fu_raw (B,384,32,32) from UAV tokens via shared backbone.
        2. Extract Fs     (B,384,32,32) from SAT tokens via shared backbone.
        3. HomographyNet(uav_img, sat_img) → delta (B,8), gate_logit (B,1).
        4. gate = sigmoid(gate_logit).reshape(B,1,1,1)          ∈ (0,1)
        5. Fu_warped = HomographyWarpLayer(Fu_raw, delta)       (B,384,32,32)
        6. Fu = gate * Fu_warped + (1 - gate) * Fu_raw          blended map
        7. emb_uav = L2_norm(GeM(Fu))                           (B,384)
           emb_sat = L2_norm(GeM(Fs))                           (B,384)
        8. logit_uav = classifier(emb_uav)                      (B,num_classes)
           logit_sat = classifier(emb_sat)                      (B,num_classes)

        Args:
            uav_img: (B, 3, 512, 512) UAV image batch.
            sat_img: (B, 3, 512, 512) satellite image batch.

        Returns dict:
            "emb_uav"    : (B, 384)          L2-normalised UAV embedding
            "emb_sat"    : (B, 384)          L2-normalised satellite embedding
            "logit_uav"  : (B, num_classes)  UAV classifier logits
            "logit_sat"  : (B, num_classes)  satellite classifier logits
            "gate_logit" : (B, 1)            raw gate logit; sigmoid → g
            "delta"      : (B, 8)            predicted corner displacements
            "Fu_raw"     : (B, 384, 32, 32)  UAV feature map before warp
            "Fu_warped"  : (B, 384, 32, 32)  UAV feature map after warp
            "Fs"         : (B, 384, 32, 32)  satellite feature map
        """
        # ── 1-2. Extract feature maps via shared backbone ──────────────────
        tokens_uav = self.forward_features(uav_img)        # (B, 1025, 384)
        tokens_sat = self.forward_features(sat_img)        # (B, 1025, 384)

        Fu_raw = self.reshape_patch_tokens(tokens_uav)     # (B, 384, 32, 32)
        Fs     = self.reshape_patch_tokens(tokens_sat)     # (B, 384, 32, 32)

        # ── 3. Predict homography parameters ──────────────────────────────
        delta, gate_logit = self.homo_net(uav_img, sat_img)
        # delta      : (B, 8)   corner displacements in feature-grid pixels
        # gate_logit : (B, 1)   raw logit; g = sigmoid(gate_logit) ∈ (0,1)

        # ── 4-6. Warp + blend UAV feature map ─────────────────────────────
        gate      = torch.sigmoid(gate_logit).reshape(-1, 1, 1, 1)  # (B,1,1,1)
        Fu_warped = self.warp_layer(Fu_raw, delta)                   # (B,384,32,32)
        Fu        = gate * Fu_warped + (1.0 - gate) * Fu_raw        # (B,384,32,32)

        # ── 7. GeM pooling + L2 normalisation ─────────────────────────────
        emb_uav = F.normalize(self.gem(Fu), p=2, dim=1)    # (B, 384)
        emb_sat = F.normalize(self.gem(Fs), p=2, dim=1)    # (B, 384)

        # ── 8. Classifier logits ───────────────────────────────────────────
        logit_uav = self.classifier(emb_uav)               # (B, num_classes)
        logit_sat = self.classifier(emb_sat)               # (B, num_classes)

        return {
            "emb_uav":    emb_uav,       # (B, 384)
            "emb_sat":    emb_sat,       # (B, 384)
            "logit_uav":  logit_uav,     # (B, num_classes)
            "logit_sat":  logit_sat,     # (B, num_classes)
            "gate_logit": gate_logit,    # (B, 1)
            "delta":      delta,         # (B, 8)
            "Fu_raw":     Fu_raw,        # (B, 384, 32, 32)
            "Fu_warped":  Fu_warped,     # (B, 384, 32, 32)
            "Fs":         Fs,            # (B, 384, 32, 32)
        }

    def __repr__(self) -> str:
        n_param = sum(p.numel() for p in self.parameters()) / 1e6
        return (
            f"SiameseViT("
            f"backbone={self._BACKBONE_NAME}, "
            f"img_size={self.img_size}, "
            f"grid={self.grid_size}x{self.grid_size}, "
            f"embed_dim={self.embed_dim}, "
            f"num_classes={self.num_classes}, "
            f"params={n_param:.1f}M)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

# Keys that exist in the new model but not in checkpoints trained before the
# CosineClassifier upgrade.  These are safe to initialise from the default
# CosineClassifier.__init__ rather than from the checkpoint.
_COSINE_HEAD_NEW_KEYS: frozenset[str] = frozenset({"classifier.scale"})


def load_checkpoint_compat(
    model:      nn.Module,
    state_dict: dict,
    logger=None,
) -> None:
    """Load a state dict with backward compatibility for the cosine-head upgrade.

    When loading a checkpoint trained with the old ``nn.Linear`` classifier
    the key ``classifier.scale`` will be absent.  This function:

    - Loads with ``strict=False``.
    - Silently accepts ``classifier.scale`` as missing (keeps init value)
      and logs a single warning.
    - Raises ``RuntimeError`` for any other missing or unexpected keys.

    Args:
        model:      The *unwrapped* SiameseViT (pass ``unwrap_model(model)``
                    when using DataParallel).
        state_dict: The ``"model"`` dict from a saved checkpoint.
        logger:     Optional Python logger; falls back to ``print``.
    """
    _warn = (lambda msg: logger.warning(msg)) if logger else print

    res = model.load_state_dict(state_dict, strict=False)

    truly_missing = [k for k in res.missing_keys if k not in _COSINE_HEAD_NEW_KEYS]
    truly_extra   = list(res.unexpected_keys)

    if res.missing_keys and not truly_missing:
        _warn(
            "Pre-cosine-head checkpoint: 'classifier.scale' absent — "
            "keeping CosineClassifier init value (s=30.0).  "
            "Fine-tune from this checkpoint to learn the scale."
        )

    if truly_missing or truly_extra:
        raise RuntimeError(
            f"Checkpoint mismatch — "
            f"missing: {truly_missing}, unexpected: {truly_extra}"
        )
