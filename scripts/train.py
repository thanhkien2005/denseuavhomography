"""
scripts/train.py — minimal but complete training entry point.

Run from repo root (denseuav-homo/):
    python scripts/train.py
    python scripts/train.py --config configs/denseuav_v1.yaml
    python scripts/train.py --data_root /abs/path/to/DenseUAV --epochs 1

What this script does:
    1. Load YAML config (override via CLI flags)
    2. Set random seed for reproducibility
    3. Build transforms → DenseUAVPairs dataset → PairPerClassBatchSampler
    4. Build SiameseViT model (pretrained=True)
    5. Build AdamW optimizer + cosine-with-warmup LR scheduler
    6. Build DenseUAVLoss criterion
    7. Build AMP GradScaler (CUDA only; None on CPU)
    8. Resume from checkpoint if --resume / config.resume is set
    9. Train for config.epochs epochs, saving a checkpoint each epoch
   10. (Evaluator slot: metrics printed per epoch once eval datasets added)
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on sys.path so all intra-project imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from data.paired_transforms import PairedTransform
from data.denseuav_dataset  import DenseUAVPairs
from data.samplers         import PairPerClassBatchSampler
from engine.trainer        import Trainer
from losses.contrastive    import InfoNCELoss
from losses.total_loss     import DenseUAVLoss
from models.vit_siamese    import SiameseViT, load_checkpoint_compat
from utils.checkpoint      import resume_if_possible, save_checkpoint, unwrap_model
from utils.logger          import get_logger
from utils.memory_queue    import MemoryQueue
from utils.meters          import MetricCollection
from utils.seed            import set_seed


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DenseUAV-Homo training script")
    p.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "denseuav_v1.yaml",
        ),
        help="Path to YAML config file.",
    )
    p.add_argument("--data_root",  default=None,
                   help="Override data_root from config.")
    p.add_argument("--output_dir", default=None,
                   help="Override output_dir from config.")
    p.add_argument("--resume",     default=None,
                   help="Path to checkpoint to resume from.")
    p.add_argument("--epochs",     type=int, default=None,
                   help="Override number of training epochs.")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch_size from config.")
    p.add_argument("--lr",         type=float, default=None,
                   help="Override learning rate from config.")
    p.add_argument("--no_amp",     action="store_true",
                   help="Disable Automatic Mixed Precision.")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do not load ImageNet pretrained weights.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg: dict, cli_data_root: str | None) -> str:
    """Resolve data_root: CLI > config, relative paths anchored to repo root."""
    raw = cli_data_root or cfg["data_root"]
    if not os.path.isabs(raw):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw = os.path.normpath(os.path.join(repo_root, raw))
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# LR scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_scheduler(
    optimizer:      optim.Optimizer,
    warmup_epochs:  int,
    total_epochs:   int,
    min_lr:         float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup → cosine annealing LR schedule.

    Args:
        optimizer:     The optimizer whose LR is scaled.
        warmup_epochs: Number of epochs for linear warm-up from ~0 to base_lr.
        total_epochs:  Total training epochs.
        min_lr:        Minimum LR at end of cosine phase.

    Returns:
        SequentialLR combining warmup + cosine.
    """
    # Linear ramp: 1/1000 of base_lr → base_lr over warmup_epochs steps
    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=max(1, warmup_epochs),
    )
    # Cosine anneal for the remainder
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup_epochs),
        eta_min=min_lr,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # ── CLI overrides ─────────────────────────────────────────────────────
    data_root  = resolve_data_root(cfg, args.data_root)
    output_dir = args.output_dir  or cfg["output_dir"]
    epochs     = args.epochs      or cfg["epochs"]
    batch_size = args.batch_size  or cfg["batch_size"]
    lr         = args.lr          or cfg["lr"]
    resume_path = args.resume     or cfg.get("resume", None)
    use_amp    = not args.no_amp
    pretrained = not args.no_pretrained

    # Make output_dir relative to repo root if not absolute
    if not os.path.isabs(output_dir):
        repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(repo_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = get_logger(
        "train",
        log_file=os.path.join(output_dir, "train.log"),
    )
    logger.info("=" * 60)
    logger.info("  DenseUAV-Homo training")
    logger.info("=" * 60)
    logger.info(f"  config     : {args.config}")
    logger.info(f"  data_root  : {data_root}")
    logger.info(f"  output_dir : {output_dir}")
    logger.info(f"  epochs     : {epochs}")
    logger.info(f"  batch_size : {batch_size}")
    logger.info(f"  lr         : {lr}")
    logger.info(f"  pretrained : {pretrained}")
    logger.info(f"  use_amp    : {use_amp}")

    # ── Seed ──────────────────────────────────────────────────────────────
    seed = cfg["seed"]
    set_seed(seed, deterministic=True)
    logger.info(f"  seed       : {seed}")

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  device     : {device}")

    # ── Transforms ────────────────────────────────────────────────────────
    # PairedTransform applies the SAME random flip/rotation to both UAV and
    # SAT images, preserving their spatial correspondence so HomographyNet
    # receives a valid gradient signal.
    img_size  = cfg["img_size"]
    tf_paired = PairedTransform(img_size=img_size, is_train=True)

    # ── Dataset + Sampler + DataLoader ────────────────────────────────────
    logger.info("Building dataset ...")
    dataset = DenseUAVPairs(
        data_root        = data_root,
        gps_file         = cfg["gps_train"],
        drone_altitude   = cfg.get("drone_altitude", "H80"),
        paired_transform = tf_paired,
    )
    logger.info(f"  {dataset}")

    labels  = [s[2] for s in dataset.samples]
    sampler = PairPerClassBatchSampler(labels, batch_size=batch_size)
    logger.info(f"  {sampler}")

    loader = DataLoader(
        dataset,
        batch_sampler = sampler,
        num_workers   = cfg.get("num_workers", 4),
        pin_memory    = cfg.get("pin_memory", True) and device.type == "cuda",
        persistent_workers = cfg.get("num_workers", 4) > 0,
    )
    logger.info(f"  loader: {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building model ...")
    head_cfg = cfg.get("head", {})
    model = SiameseViT(
        num_classes          = dataset.num_classes,
        embed_dim            = cfg.get("embed_dim", 384),
        img_size             = img_size,
        patch_size           = cfg.get("patch_size", 16),
        gem_p                = cfg.get("gem_p", 3.0),
        gem_learnable        = cfg.get("gem_learnable", True),
        pretrained           = pretrained,
        head_scale           = head_cfg.get("scale", 30.0),
        head_learnable_scale = head_cfg.get("learnable_scale", True),
    ).to(device)
    logger.info(f"  {model}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = cfg.get("weight_decay", 1e-4),
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs = cfg.get("warmup_epochs", 5),
        total_epochs  = epochs,
        min_lr        = cfg.get("min_lr", 1e-6),
    )

    # ── Loss criterion ────────────────────────────────────────────────────
    loss_cfg  = cfg.get("loss", {})
    criterion = DenseUAVLoss(
        w_ce         = loss_cfg.get("w_ce",        1.0),
        w_triplet    = loss_cfg.get("w_triplet",   1.0),
        w_kl         = loss_cfg.get("w_kl",        1.0),
        w_homo       = loss_cfg.get("w_homo",      0.5),
        temperature  = loss_cfg.get("temperature", 0.07),
        margin       = loss_cfg.get("margin",      0.3),
        lambda_reg   = loss_cfg.get("lambda_reg",  0.01),
    )
    logger.info(f"  {criterion}")

    # ── AMP GradScaler (CUDA only) ────────────────────────────────────────
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("  AMP GradScaler enabled.")
    else:
        logger.info("  AMP disabled (CPU or --no_amp flag).")

    # ── Contrastive memory queue + InfoNCE loss ───────────────────────────
    con_cfg        = cfg.get("contrastive", {})
    use_contrastive = con_cfg.get("use_contrastive", False)
    queue          = None
    contrastive    = None
    w_contrastive  = 0.0

    if use_contrastive:
        embed_dim     = cfg.get("embed_dim", 384)
        queue_size    = con_cfg.get("queue_size", 4096)
        con_temp      = con_cfg.get("temperature", 0.07)
        w_contrastive = con_cfg.get("w_contrastive", 1.0)

        queue       = MemoryQueue(queue_size=queue_size, embed_dim=embed_dim, device=device)
        contrastive = InfoNCELoss(temperature=con_temp)
        logger.info(
            f"  InfoNCE enabled — queue_size={queue_size}, "
            f"T={con_temp}, w={w_contrastive}"
        )

    # ── Resume ────────────────────────────────────────────────────────────
    ckpt, start_epoch = resume_if_possible(
        resume_path,
        map_location=str(device),
        logger=logger,
    )
    if ckpt is not None:
        load_checkpoint_compat(unwrap_model(model), ckpt["model"], logger=logger)
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if queue is not None and ckpt.get("queue") is not None:
            queue.load_state_dict(ckpt["queue"])
            logger.info(f"  Restored queue ({len(queue)} entries).")
        logger.info(f"Resumed from epoch {start_epoch}.")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        criterion      = criterion,
        device         = device,
        use_amp        = use_amp,
        logger         = logger,
        log_interval   = cfg.get("log_interval", 50),
        grad_clip_norm = 1.0,
    )

    # ── Training loop ─────────────────────────────────────────────────────
    save_interval = cfg.get("save_interval", 10)
    logger.info(f"Starting training from epoch {start_epoch} to {epochs} ...")

    for epoch in range(start_epoch, epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        # Reshuffle sampler for this epoch (optional determinism)
        sampler.set_epoch(epoch)

        # One training epoch
        meters = trainer.train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            queue=queue,
            contrastive=contrastive,
            w_contrastive=w_contrastive,
        )

        # Step LR scheduler once per epoch
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log epoch summary
        summary_keys = ["loss_total", "loss_ce", "loss_triplet", "loss_kl", "loss_homo"]
        if use_contrastive:
            summary_keys.append("loss_contrastive")
        logger.info(
            f"Epoch {epoch+1} summary | lr={current_lr:.2e} | "
            + " | ".join(f"{k}={meters[k].avg:.4f}" for k in summary_keys)
        )

        # Log cosine-head scale if it is learnable
        _cls = getattr(unwrap_model(model), "classifier", None)
        if _cls is not None and isinstance(getattr(_cls, "scale", None), torch.nn.Parameter):
            logger.info(f"  cosine_scale : {_cls.scale.item():.4f}")

        # TODO: add evaluation here once DenseUAVQuery/Gallery datasets exist
        # if (epoch + 1) % eval_interval == 0:
        #     evaluator = Evaluator(recall_k=[1,5,10], sdm_k=[1,5,10])
        #     metrics = evaluator.evaluate(model, val_loader)
        #     logger.info(f"Val metrics: {metrics}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(output_dir, f"epoch_{epoch+1:04d}.pt")
            save_checkpoint(
                state={
                    "epoch":     epoch + 1,
                    "model":     unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict() if scaler is not None else None,
                    "queue":     queue.state_dict() if queue is not None else None,
                    "metrics":   meters.summary(),
                    "config":    cfg,
                },
                path=ckpt_path,
                logger=logger,
            )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
