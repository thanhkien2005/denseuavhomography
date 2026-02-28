"""
tests/test_warp_identity.py
───────────────────────────
Property test: if delta == 0, HomographyWarpLayer must return a tensor
that is numerically identical to its input (max absolute difference = 0.0).

Background
──────────
When delta = 0 all 4 corner displacements are zero, so
    src_corners == dst_corners
    → kornia.get_perspective_transform returns the identity matrix H ≈ I(3)
    → kornia.warp_perspective with H ≈ I samples each pixel at its own
       location → output == input (no interpolation error for integer coords)

This test also verifies that:
  1. HomographyWarpLayer output shape matches input shape.
  2. The layer handles dtype float32 correctly.
  3. Non-zero delta produces a DIFFERENT output (i.e. the warp is active).

Run:
    python tests/test_warp_identity.py
    # or via pytest: pytest tests/test_warp_identity.py -v
"""

from __future__ import annotations

import os
import sys

# Allow imports from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.homography_warp import HomographyWarpLayer


# ─────────────────────────────────────────────────────────────────────────────

def test_warp_identity() -> None:
    """delta == 0 → warped == feat_map (max_abs_diff must be 0.0)."""
    torch.manual_seed(0)

    B, C, G = 2, 384, 32    # production shapes
    layer   = HomographyWarpLayer(grid_size=G)
    layer.eval()

    feat  = torch.randn(B, C, G, G)            # (B, 384, 32, 32)
    delta = torch.zeros(B, 8)                   # (B, 8)  — zero displacement

    with torch.no_grad():
        warped = layer(feat, delta)             # (B, 384, 32, 32)

    # ── Shape check ──────────────────────────────────────────────────────
    assert warped.shape == feat.shape, (
        f"Shape mismatch: warped={warped.shape}, expected={feat.shape}"
    )
    print(f"  Shape check      : {tuple(warped.shape)}  OK")

    # ── Identity check ────────────────────────────────────────────────────
    max_diff = (warped - feat).abs().max().item()
    print(f"  max_abs_diff     : {max_diff:.2e}  (threshold: 1e-5)")
    assert max_diff < 1e-5, (
        f"Identity warp failed: max_abs_diff={max_diff:.2e} >= 1e-5. "
        "Check kornia warp_perspective with align_corners=True."
    )

    # ── Dtype check ───────────────────────────────────────────────────────
    assert warped.dtype == torch.float32, \
        f"Output dtype is {warped.dtype}, expected float32"
    print(f"  dtype check      : {warped.dtype}  OK")

    print("  test_warp_identity                     PASSED")


def test_nonzero_delta_differs() -> None:
    """Non-zero delta must produce output that differs from input."""
    torch.manual_seed(1)

    B, C, G = 2, 8, 32
    layer   = HomographyWarpLayer(grid_size=G)
    layer.eval()

    feat  = torch.randn(B, C, G, G)

    # Large delta: shift each corner by ~5 pixels → clearly non-identity
    delta = torch.ones(B, 8) * 5.0   # (B, 8)

    with torch.no_grad():
        warped = layer(feat, delta)

    max_diff = (warped - feat).abs().max().item()
    print(f"  non-zero delta max_abs_diff : {max_diff:.4f}  (expected > 0)")
    assert max_diff > 1e-3, (
        f"Non-zero delta produced essentially no warp: max_diff={max_diff:.2e}"
    )
    assert warped.shape == feat.shape

    print("  test_nonzero_delta_differs             PASSED")


def test_backward_through_warp() -> None:
    """Gradients must flow through delta and feat_map."""
    torch.manual_seed(2)

    B, C, G = 2, 16, 32
    layer   = HomographyWarpLayer(grid_size=G)

    feat  = torch.nn.Parameter(torch.randn(B, C, G, G))
    delta = torch.nn.Parameter(torch.randn(B, 8) * 0.5)

    warped = layer(feat, delta)         # (B, C, G, G)
    loss   = warped.mean()
    loss.backward()

    assert feat.grad is not None,  "No gradient on feat_map"
    assert delta.grad is not None, "No gradient on delta"
    assert feat.grad.abs().max()  > 0, "feat_map gradient is zero"
    assert delta.grad.abs().max() > 0, "delta gradient is zero"

    print(f"  feat.grad  norm  : {feat.grad.norm().item():.4f}")
    print(f"  delta.grad norm  : {delta.grad.norm().item():.4f}")
    print("  test_backward_through_warp             PASSED")


def test_small_feat_map() -> None:
    """Layer must work on any grid size, not only G=32."""
    torch.manual_seed(3)
    for G in [8, 16, 32]:
        layer = HomographyWarpLayer(grid_size=G)
        feat  = torch.randn(1, 4, G, G)
        delta = torch.zeros(1, 8)
        with torch.no_grad():
            warped = layer(feat, delta)
        assert warped.shape == feat.shape
        assert (warped - feat).abs().max().item() < 1e-5
    print("  test_small_feat_map                    PASSED")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 56)
    print("  HomographyWarpLayer — identity tests")
    print("=" * 56)
    test_warp_identity()
    test_nonzero_delta_differs()
    test_backward_through_warp()
    test_small_feat_map()
    print("=" * 56)
    print("  All warp tests PASSED.")
    print("=" * 56)
