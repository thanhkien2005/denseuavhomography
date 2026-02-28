"""
engine/hooks.py — periodic evaluation and checkpoint hooks for the train loop.

Two standalone functions and one stateful class:

    maybe_evaluate(...)         Run the Evaluator every eval_interval epochs.
    maybe_save_checkpoint(...)  Save a checkpoint every save_interval epochs.
    BestCheckpointTracker       Tracks a scalar metric and saves best.pt.

Usage in scripts/train.py (after each epoch):

    from engine.hooks import maybe_evaluate, maybe_save_checkpoint, BestCheckpointTracker

    best_tracker = BestCheckpointTracker(metric_key="train/Recall@1")

    for epoch in range(start_epoch, total_epochs):
        # ... training ...

        metrics = maybe_evaluate(
            epoch        = epoch + 1,
            eval_interval= cfg.get("eval_interval", 10),
            total_epochs = total_epochs,
            model        = model,
            evaluator    = evaluator,
            dataloader   = val_loader,
            logger       = logger,
        )

        state = {"epoch": epoch+1, "model": ..., "optimizer": ...}

        maybe_save_checkpoint(
            epoch        = epoch + 1,
            save_interval= cfg.get("save_interval", 10),
            total_epochs = total_epochs,
            state        = state,
            output_dir   = output_dir,
            logger       = logger,
        )

        if metrics is not None:
            best_tracker.update(metrics, state, output_dir, logger)
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.checkpoint import save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Periodic evaluation hook
# ─────────────────────────────────────────────────────────────────────────────

def maybe_evaluate(
    epoch:         int,
    eval_interval: int,
    total_epochs:  int,
    model:         nn.Module,
    evaluator,                       # engine.evaluator.Evaluator
    dataloader:    DataLoader,
    logger         = None,
    prefix:        str = "train/",
) -> Optional[Dict[str, float]]:
    """Run evaluation if this epoch falls on an eval checkpoint.

    Evaluation is triggered when:
        epoch % eval_interval == 0   (every N epochs)
        OR epoch == total_epochs     (always on the final epoch)

    Args:
        epoch:         Current epoch number (1-based, i.e. epoch after training).
        eval_interval: Run eval every this many epochs.  0 disables eval.
        total_epochs:  Total number of training epochs.
        model:         Model with forward(uav_img, sat_img) -> dict interface.
        evaluator:     Evaluator instance from engine/evaluator.py.
        dataloader:    Paired DataLoader (yields uav_img, sat_img, label, gps).
        logger:        Optional logger; prints each metric key = value.
        prefix:        Metric key prefix passed to evaluator.evaluate().

    Returns:
        Dict of metric names → float values if eval ran, else None.
    """
    if eval_interval <= 0:
        return None
    if epoch % eval_interval != 0 and epoch != total_epochs:
        return None

    if logger:
        logger.info(f"[eval hook] Running evaluation at epoch {epoch} …")

    metrics = evaluator.evaluate(model, dataloader, prefix=prefix)

    if logger:
        for key in sorted(metrics):
            logger.info(f"  {key:<24} = {metrics[key]:.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Periodic checkpoint hook
# ─────────────────────────────────────────────────────────────────────────────

def maybe_save_checkpoint(
    epoch:         int,
    save_interval: int,
    total_epochs:  int,
    state:         dict,
    output_dir:    str,
    logger         = None,
) -> Optional[str]:
    """Save a checkpoint if this epoch falls on a save checkpoint.

    Checkpointing is triggered when:
        epoch % save_interval == 0   (every N epochs)
        OR epoch == total_epochs     (always on the final epoch)

    The file is written as <output_dir>/epoch_NNNN.pt.

    Args:
        epoch:         Current epoch number (1-based).
        save_interval: Save every this many epochs.  0 disables periodic saves.
        total_epochs:  Total number of training epochs.
        state:         Dict to serialize (e.g. {"epoch":…, "model":…, "optimizer":…}).
        output_dir:    Directory for checkpoint files (created if absent).
        logger:        Optional logger.

    Returns:
        Absolute path of the saved file if a checkpoint was written, else None.
    """
    if save_interval <= 0 and epoch != total_epochs:
        return None
    if save_interval > 0 and epoch % save_interval != 0 and epoch != total_epochs:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"epoch_{epoch:04d}.pt")
    save_checkpoint(state, path, logger=logger)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Best-checkpoint tracker
# ─────────────────────────────────────────────────────────────────────────────

class BestCheckpointTracker:
    """Save best.pt whenever a scalar metric improves.

    Typical usage::

        tracker = BestCheckpointTracker(metric_key="train/Recall@1")

        for epoch in range(...):
            metrics = maybe_evaluate(...)
            if metrics is not None:
                tracker.update(metrics, state, output_dir, logger)

    Args:
        metric_key:       Key in the metrics dict to track (e.g. "train/Recall@1").
        higher_is_better: If True (default), higher values are better (Recall, SDM).
                          Set False for loss-based metrics.
    """

    def __init__(
        self,
        metric_key:       str  = "train/Recall@1",
        higher_is_better: bool = True,
    ) -> None:
        self.metric_key       = metric_key
        self.higher_is_better = higher_is_better
        self.best_value: Optional[float] = None

    def is_better(self, value: float) -> bool:
        """Return True if *value* is strictly better than the stored best."""
        if self.best_value is None:
            return True
        return value > self.best_value if self.higher_is_better else value < self.best_value

    def update(
        self,
        metrics:    Dict[str, float],
        state:      dict,
        output_dir: str,
        logger      = None,
    ) -> bool:
        """Check metric and save best.pt if it improved.

        Args:
            metrics:    Dict returned by Evaluator.evaluate().
            state:      Checkpoint state dict to serialize.
            output_dir: Directory for best.pt.
            logger:     Optional logger.

        Returns:
            True if a new best was found and saved.
        """
        value = metrics.get(self.metric_key)
        if value is None:
            if logger:
                logger.warning(
                    f"[best tracker] Metric key '{self.metric_key}' not found in "
                    f"metrics dict (keys: {list(metrics.keys())}). Skipping."
                )
            return False

        if not self.is_better(value):
            return False

        prev = self.best_value
        self.best_value = value

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "best.pt")
        save_checkpoint(state, path, logger=logger)

        if logger:
            prev_str = f"{prev:.4f}" if prev is not None else "—"
            logger.info(
                f"[best tracker] New best {self.metric_key}: "
                f"{prev_str} → {value:.4f}  saved to {path}"
            )
        return True

    def __repr__(self) -> str:
        best_str = f"{self.best_value:.4f}" if self.best_value is not None else "none"
        return (
            f"BestCheckpointTracker("
            f"key={self.metric_key!r}, "
            f"higher_is_better={self.higher_is_better}, "
            f"best={best_str})"
        )
