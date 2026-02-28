"""
utils/checkpoint.py — pure-PyTorch checkpoint save / load helpers.
"""

from __future__ import annotations

import os
from typing import Union

import torch
import torch.nn as nn


def save_checkpoint(
    state: dict,
    path: str,
    logger=None,
) -> None:
    """Serialize *state* dict to *path* with torch.save.

    Parent directories are created automatically if they do not exist.

    Args:
        state:   Arbitrary dict (e.g. {"epoch": 5, "model": sd, "optim": sd}).
        path:    Destination file path (absolute or relative).
        logger:  Optional logger; if provided a one-line info message is emitted.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)
    if logger is not None:
        keys = list(state.keys())
        logger.info(f"Checkpoint saved → {path}  (keys: {keys})")


def load_checkpoint(
    path: str,
    map_location: Union[str, torch.device] = "cpu",
    logger=None,
) -> dict:
    """Load and return the checkpoint dict stored at *path*.

    Args:
        path:         File to load.
        map_location: Passed directly to torch.load (e.g. "cpu", "cuda:0").
        logger:       Optional logger for an info message on success.

    Returns:
        The deserialized dict.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{path}'. "
            "Check --resume path or output_dir."
        )
    ckpt = torch.load(path, map_location=map_location)
    if logger is not None:
        epoch = ckpt.get("epoch", "?")
        logger.info(f"Checkpoint loaded ← {path}  (epoch={epoch})")
    return ckpt


def resume_if_possible(
    path: str | None,
    map_location: Union[str, torch.device] = "cpu",
    logger=None,
) -> tuple[dict | None, int]:
    """Attempt to load a checkpoint; return gracefully if nothing to resume.

    Args:
        path:         Path to checkpoint file, or None.
        map_location: Forwarded to torch.load.
        logger:       Optional logger.

    Returns:
        (ckpt_dict, start_epoch) where:
          - ckpt_dict   is None  if no checkpoint was loaded.
          - start_epoch is 0     if no checkpoint was loaded,
                        else ckpt_dict.get("epoch", 0).
    """
    if path is None:
        if logger is not None:
            logger.info("No resume path provided; starting from scratch.")
        return None, 0

    if not os.path.isfile(path):
        if logger is not None:
            logger.info(
                f"Resume path '{path}' not found; starting from scratch."
            )
        return None, 0

    ckpt = load_checkpoint(path, map_location=map_location, logger=logger)
    start_epoch = ckpt.get("epoch", 0)
    return ckpt, start_epoch


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping DDP if necessary.

    Args:
        model: A plain nn.Module or a DistributedDataParallel wrapper.

    Returns:
        model.module if the model was wrapped in DDP, else model itself.
    """
    return model.module if hasattr(model, "module") else model
