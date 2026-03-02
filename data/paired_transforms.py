"""
data/paired_transforms.py — geometrically consistent paired UAV+SAT transforms.

The geometric augmentations (horizontal flip, vertical flip, rotation) are
applied with the SAME random parameters to both the UAV and satellite images.
This preserves their spatial correspondence, which is required for the
HomographyNet branch to receive a meaningful gradient signal.

Color jitter is applied to UAV only (photometric; does not break geometry).

Usage:
    from data.paired_transforms import PairedTransform
    tf = PairedTransform(img_size=512, is_train=True)
    uav_tensor, sat_tensor = tf(uav_pil, sat_pil)
"""

import random

import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


class PairedTransform:
    """Apply shared random geometric transforms to a (UAV, SAT) PIL-image pair.

    Shared (same random draw for both images):
        - RandomHorizontalFlip  (p=0.5)
        - RandomVerticalFlip    (p=0.5)
        - RandomRotation        (±rotation_deg degrees)

    UAV-only (photometric; does not affect geometry):
        - ColorJitter (brightness, contrast, saturation, hue)

    Both images: Resize → ToTensor → Normalize (ImageNet stats).

    Args:
        img_size:     Target spatial resolution (square).
        is_train:     If True, apply random augmentations.
        rotation_deg: Maximum rotation angle in degrees.
    """

    def __init__(
        self,
        img_size:     int   = 512,
        is_train:     bool  = True,
        rotation_deg: float = 15.0,
    ) -> None:
        self.img_size     = img_size
        self.is_train     = is_train
        self.rotation_deg = rotation_deg
        self._jitter = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
        )

    def __call__(self, uav_pil, sat_pil):
        """Transform a paired (UAV, SAT) PIL image pair into tensors.

        Args:
            uav_pil: PIL Image — UAV drone image.
            sat_pil: PIL Image — satellite image.

        Returns:
            uav_tensor: FloatTensor (3, img_size, img_size)
            sat_tensor: FloatTensor (3, img_size, img_size)
        """
        uav = TF.resize(uav_pil, [self.img_size, self.img_size])
        sat = TF.resize(sat_pil, [self.img_size, self.img_size])

        if self.is_train:
            # ── Shared random horizontal flip ──────────────────────────────
            if random.random() < 0.5:
                uav = TF.hflip(uav)
                sat = TF.hflip(sat)

            # ── Shared random vertical flip ────────────────────────────────
            if random.random() < 0.5:
                uav = TF.vflip(uav)
                sat = TF.vflip(sat)

            # ── Shared random rotation ─────────────────────────────────────
            angle = random.uniform(-self.rotation_deg, self.rotation_deg)
            uav = TF.rotate(uav, angle)
            sat = TF.rotate(sat, angle)

            # ── UAV-only colour jitter (photometric; does not break geometry)
            uav = self._jitter(uav)

        uav_t = TF.normalize(TF.to_tensor(uav), mean=_MEAN, std=_STD)
        sat_t = TF.normalize(TF.to_tensor(sat), mean=_MEAN, std=_STD)

        return uav_t, sat_t

    def __repr__(self) -> str:
        return (
            f"PairedTransform(img_size={self.img_size}, "
            f"is_train={self.is_train}, "
            f"rotation_deg={self.rotation_deg})"
        )
