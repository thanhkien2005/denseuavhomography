"""
utils/seed.py — reproducibility helpers.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all relevant RNG seeds and (optionally) CUDA deterministic flags.

    Args:
        seed:         Integer seed value.
        deterministic: If True, force cuDNN into deterministic mode.
                       Slightly slower but fully reproducible across runs.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)   # for multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True can speed up training when input shapes are fixed
        torch.backends.cudnn.benchmark = True
