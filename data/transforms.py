"""
data/transforms.py — build torchvision transform pipelines.

Design notes:
  - UAV (drone) images benefit from stronger spatial / colour augmentation
    because the camera pose and lighting vary with each flight.
  - Satellite images are top-down orthophotos; moderate augmentation only.
  - Both modalities share the same ImageNet normalisation statistics.

Usage:
    from data.transforms import build_transforms
    tf_uav = build_transforms(img_size=512, is_train=True,  modality="uav")
    tf_sat = build_transforms(img_size=512, is_train=True,  modality="satellite")
    tf_val = build_transforms(img_size=512, is_train=False)   # same for both
"""

from torchvision import transforms

# ImageNet statistics (used for both modalities)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def build_transforms(
    img_size: int = 512,
    is_train: bool = True,
    modality: str = "uav",          # "uav" | "satellite"
) -> transforms.Compose:
    """Return a torchvision Compose pipeline.

    Args:
        img_size:  Target spatial resolution (square).  Matches model input.
        is_train:  If True, apply data-augmentation steps.
        modality:  Controls augmentation intensity; ignored when is_train=False.

    Returns:
        A Compose pipeline that accepts a PIL Image and returns
        a float32 tensor of shape (3, img_size, img_size), normalised.
    """
    if not is_train:
        # Deterministic pipeline: resize → tensor → normalise
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    if modality == "uav":
        # UAV images: full augmentation suite
        #   - Random horizontal flip  (UAV can orbit either direction)
        #   - Random vertical flip    (less common but valid)
        #   - Colour jitter           (exposure / white-balance differences)
        #   - Small random rotation   (UAV yaw drift)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    elif modality == "satellite":
        # Satellite images: only flips (orthophotos are radiometrically stable)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])

    else:
        raise ValueError(
            f"Unknown modality '{modality}'. Expected 'uav' or 'satellite'."
        )
