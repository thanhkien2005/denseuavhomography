"""
engine/trainer.py — one-epoch training loop with AMP + gradient clipping.

Design
──────
Trainer is a stateless helper: it holds shared config (device, criterion,
AMP flag, logger) but does NOT own the model, optimizer, or scaler.
Those are passed as arguments to train_one_epoch() so the caller
(scripts/train.py) controls their lifecycle.

AMP notes (PyTorch 2.x, CPU-only fallback)
────────────────────────────────────────────
- AMP is automatically disabled when running on CPU (CUDA required for
  float16 autocast and GradScaler).
- When use_amp=True and CUDA is available, torch.amp.autocast reduces
  memory and speeds up forward pass.
- GradScaler prevents gradient underflow in float16 training; it is
  skipped on CPU (caller passes scaler=None).

NaN guard
─────────
An assert fires immediately on NaN loss, before .backward() is called.
This avoids corrupting model weights with NaN gradients.

Usage:
    trainer = Trainer(criterion, device, use_amp=True, logger=logger,
                      log_interval=50, grad_clip_norm=1.0)
    meters  = trainer.train_one_epoch(model, loader, optimizer, scaler, epoch=0)
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.memory_queue import MemoryQueue
from utils.meters       import MetricCollection


class Trainer:
    """Stateless training helper.

    Args:
        criterion:      DenseUAVLoss (or any callable returning (loss, dict)).
        device:         torch.device for inference and label placement.
        use_amp:        Enable Automatic Mixed Precision.
                        Automatically disabled on CPU even if True.
        logger:         Optional logger from utils.logger.get_logger().
        log_interval:   Log metrics every N batches.
        grad_clip_norm: Max gradient norm for clipping (0 = disabled).
    """

    def __init__(
        self,
        criterion,
        device:         torch.device,
        use_amp:        bool          = True,
        logger                        = None,
        log_interval:   int           = 50,
        grad_clip_norm: float         = 1.0,
    ) -> None:
        self.criterion      = criterion
        self.device         = device
        self.logger         = logger
        self.log_interval   = log_interval
        self.grad_clip_norm = grad_clip_norm

        # AMP requires CUDA; disable gracefully on CPU
        self.use_amp = use_amp and (device.type == "cuda")
        if use_amp and not self.use_amp:
            if logger:
                logger.info(
                    "AMP requested but device is CPU — AMP disabled."
                )

    # ──────────────────────────────────────────────────────────────────────
    # One epoch
    # ──────────────────────────────────────────────────────────────────────

    def train_one_epoch(
        self,
        model:          nn.Module,
        loader:         DataLoader,
        optimizer:      torch.optim.Optimizer,
        scaler:         Optional[GradScaler],
        epoch:          int          = 0,
        queue:          Optional[MemoryQueue] = None,
        contrastive     = None,   # Optional[InfoNCELoss]
        w_contrastive:  float        = 1.0,
    ) -> MetricCollection:
        """Run one full pass over the dataloader.

        Args:
            model:         SiameseViT (or any module with forward(uav,sat)->dict).
            loader:        DataLoader yielding batches with "uav_img", "sat_img",
                           "label" keys.
            optimizer:     Gradient-descent optimizer (e.g. AdamW).
            scaler:        GradScaler for AMP; pass None when AMP is disabled.
            epoch:         Current epoch index (0-based), used for logging.
            queue:         Optional MemoryQueue for cross-batch negatives.
            contrastive:   Optional InfoNCELoss; used only when queue is given.
            w_contrastive: Weight applied to the InfoNCE term in the total loss.

        Returns:
            MetricCollection with keys "loss_total", "loss_ce", "loss_triplet",
            "loss_kl", "loss_contrastive".  Access averages via .avg.

        Shape trace (per batch):
            uav_img        : (B, 3, H, W)  → device
            sat_img        : (B, 3, H, W)  → device
            labels         : (B,)           → device
            emb_uav        : (B, D)         from model
            emb_sat        : (B, D)         from model
            logit_uav      : (B, C)         from model
            logit_sat      : (B, C)         from model
            loss_contrastive: ()            InfoNCE (0 when queue/contrastive absent)
            total_loss     : ()             backward target
        """
        use_contrastive = (queue is not None) and (contrastive is not None)

        model.train()
        meters = MetricCollection(
            ["loss_total", "loss_ce", "loss_triplet", "loss_kl",
             "loss_contrastive", "loss_homo", "gate_mean", "delta_norm_mean"]
        )
        device_type = self.device.type    # "cuda" or "cpu"

        # Epoch-level min/max trackers (not smoothed — true extremes over epoch)
        gate_epoch_min    = float("inf")
        gate_epoch_max    = float("-inf")
        delta_abs_max_ep  = 0.0

        for batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        ):
            # ── Move data to device ────────────────────────────────────────
            uav_img: Tensor = batch["uav_img"].to(
                self.device, non_blocking=True
            )   # (B, 3, H, W)
            sat_img: Tensor = batch["sat_img"].to(
                self.device, non_blocking=True
            )   # (B, 3, H, W)
            labels: Tensor  = batch["label"].to(
                self.device, non_blocking=True
            )   # (B,)

            B = uav_img.shape[0]

            # ── Forward (with optional AMP) ────────────────────────────────
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device_type,
                enabled=self.use_amp,
            ):
                outputs    = model(uav_img, sat_img)
                # outputs: {emb_uav:(B,D), emb_sat:(B,D),
                #           logit_uav:(B,C), logit_sat:(B,C)}

                total_loss, loss_dict = self.criterion(outputs, labels)
                # total_loss: scalar   loss_dict: {total,ce,triplet,kl}

                # ── InfoNCE (batch + queue negatives) ─────────────────────
                loss_con_val = 0.0
                if use_contrastive:
                    loss_con = contrastive(
                        outputs["emb_uav"], outputs["emb_sat"],
                        queue.uav_embeddings,   # (K, D) — past batches, detached
                        queue.sat_embeddings,   # (K, D) — past batches, detached
                    )
                    total_loss   = total_loss + w_contrastive * loss_con
                    loss_con_val = loss_con.item()

            # ── Gate / delta diagnostic stats (no grad; float32 for accuracy) ──
            with torch.no_grad():
                gate_b  = torch.sigmoid(outputs["gate_logit"].detach().float())  # (B,1)
                delta_b = outputs["delta"].detach().float()                        # (B,8)
                gate_mean_val     = gate_b.mean().item()
                gate_min_val      = gate_b.min().item()
                gate_max_val      = gate_b.max().item()
                delta_norm_val    = delta_b.norm(dim=1).mean().item()
                delta_abs_max_val = delta_b.abs().max().item()

            gate_epoch_min   = min(gate_epoch_min,   gate_min_val)
            gate_epoch_max   = max(gate_epoch_max,   gate_max_val)
            delta_abs_max_ep = max(delta_abs_max_ep, delta_abs_max_val)
            meters["gate_mean"].update(gate_mean_val, n=B)
            meters["delta_norm_mean"].update(delta_norm_val, n=B)

            # ── NaN guard ─────────────────────────────────────────────────
            # Check BEFORE .backward() to avoid poisoning model weights.
            assert not torch.isnan(total_loss), (
                f"NaN loss at epoch={epoch}, batch={batch_idx}. "
                "Possible causes: extreme LR, bad data normalisation, "
                "near-zero GeM activations, or collapsed embeddings."
            )
            assert not torch.isinf(total_loss), (
                f"Inf loss at epoch={epoch}, batch={batch_idx}. "
                "Check logit magnitude (temperature too small?) or weight init."
            )

            # ── Backward + optimise ────────────────────────────────────────
            if scaler is not None:
                # AMP path: scale → backward → unscale → clip → step → update
                scaler.scale(total_loss).backward()

                if self.grad_clip_norm > 0:
                    # Must unscale before clipping so clip sees real gradients
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.grad_clip_norm
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                # Non-AMP path (CPU or AMP disabled)
                total_loss.backward()

                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.grad_clip_norm
                    )

                optimizer.step()

            # ── Enqueue current batch AFTER optimizer step ─────────────────
            # Standard MoCo-style: embeddings are stale by one iteration,
            # which prevents the queue from seeing its own gradients.
            if use_contrastive:
                queue.enqueue(
                    outputs["emb_uav"].detach().float(),
                    outputs["emb_sat"].detach().float(),
                )

            # ── Accumulate meters ──────────────────────────────────────────
            meters["loss_total"].update(total_loss.item(),              n=B)
            meters["loss_ce"].update(loss_dict["ce"].item(),            n=B)
            meters["loss_triplet"].update(loss_dict["triplet"].item(),  n=B)
            meters["loss_kl"].update(loss_dict["kl"].item(),            n=B)
            meters["loss_contrastive"].update(loss_con_val,             n=B)
            meters["loss_homo"].update(loss_dict.get("homo", torch.tensor(0.0)).item(), n=B)

            # ── Periodic logging ───────────────────────────────────────────
            if self.logger and (batch_idx + 1) % self.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                log_keys = ["loss_total", "loss_ce", "loss_triplet", "loss_kl", "loss_homo"]
                if use_contrastive:
                    log_keys.append("loss_contrastive")
                self.logger.info(
                    f"epoch={epoch}  batch={batch_idx+1}/{len(loader)}  "
                    f"lr={lr:.2e}  "
                    + "  ".join(
                        f"{k}={meters[k].avg:.4f}" for k in log_keys
                    )
                )

        # ── Epoch-end homography diagnostics ──────────────────────────────
        if self.logger:
            self.logger.info(
                f"  [homo] gate_mean={meters['gate_mean'].avg:.4f} "
                f"gate_min={gate_epoch_min:.4f} "
                f"gate_max={gate_epoch_max:.4f} | "
                f"delta_norm_mean={meters['delta_norm_mean'].avg:.4f} "
                f"delta_abs_max={delta_abs_max_ep:.4f}"
            )

        return meters
