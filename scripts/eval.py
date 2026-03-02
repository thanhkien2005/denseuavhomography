"""
scripts/eval.py — standalone evaluation entry point.

Loads a saved checkpoint, rebuilds the model, and runs the Evaluator to
print Recall@K and SDM@K metrics.

Run from repo root (denseuav-homo/):

    # Evaluate on the training split (proxy — measures memorisation):
    python scripts/eval.py --checkpoint outputs/denseuav_v1/epoch_0120.pt

    # Evaluate on the official test split (777 queries vs 3033 gallery):
    python scripts/eval.py \\
        --checkpoint outputs/denseuav_v1/epoch_0120.pt \\
        --split      test

    # Explicit options:
    python scripts/eval.py \\
        --checkpoint outputs/denseuav_v1/epoch_0120.pt \\
        --config     configs/denseuav_v1.yaml \\
        --data_root  /path/to/DenseUAV \\
        --split      train \\
        --batch_size 16 \\
        --device     cuda

Output
------
Metrics are printed to stdout and appended to <output_dir>/eval.log.

Notes
-----
- --split train uses DenseUAVPairs (2256 paired locations, closed-set proxy).
- --split test uses DenseUAVQuery (777 drone queries) + DenseUAVGallery
  (3033 satellite gallery) — the official DenseUAV retrieval benchmark.
  Requires test/query_drone/ and test/gallery_satellite/ under data_root;
  a FileNotFoundError is raised with a clear message if they are absent.
- SDM@K is skipped with a warning when GPS is unavailable for query or gallery.
- The evaluator uses UAV images as queries and satellite images as the gallery.
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on sys.path so all intra-project imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from data.denseuav_dataset import DenseUAVGallery, DenseUAVPairs, DenseUAVQuery
from data.transforms       import build_transforms
from engine.evaluator      import Evaluator
from models.vit_siamese    import SiameseViT, load_checkpoint_compat
from utils.checkpoint      import load_checkpoint, unwrap_model
from utils.logger          import get_logger


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DenseUAV-Homo evaluation script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint saved by scripts/train.py.",
    )
    p.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "denseuav_v1.yaml",
        ),
        help="Path to YAML config (used for model + dataset settings).",
    )
    p.add_argument("--data_root", default=None,
                   help="Override data_root from config.")
    p.add_argument(
        "--split", default="train", choices=["train", "test"],
        help=(
            "Which split to evaluate on.  "
            "'train' uses DenseUAVPairs (2256 paired locations, proxy metric).  "
            "'test'  uses DenseUAVQuery (777 queries) + DenseUAVGallery "
            "(3033 gallery items) — the official DenseUAV test protocol."
        ),
    )
    p.add_argument("--batch_size", type=int, default=None,
                   help="Inference batch size (default: from config).")
    p.add_argument("--num_workers", type=int, default=None,
                   help="DataLoader workers (default: from config).")
    p.add_argument("--device", default=None,
                   help="torch device string, e.g. 'cuda' or 'cpu'.  "
                        "Auto-detected if not set.")
    p.add_argument("--output_dir", default=None,
                   help="Directory for eval.log.  Defaults to checkpoint dir.")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Do not load ImageNet weights (weights come from ckpt).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg: dict, cli_data_root: str | None) -> str:
    raw = cli_data_root or cfg["data_root"]
    if not os.path.isabs(raw):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw = os.path.normpath(os.path.join(repo_root, raw))
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    # ── Resolve settings (CLI overrides config) ───────────────────────────
    data_root   = resolve_data_root(cfg, args.data_root)
    batch_size  = args.batch_size  or cfg.get("batch_size", 16)
    num_workers = args.num_workers or cfg.get("num_workers", 4)
    output_dir  = args.output_dir  or os.path.dirname(os.path.abspath(args.checkpoint))

    if args.device is not None:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    os.makedirs(output_dir, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────
    logger = get_logger(
        "eval",
        log_file=os.path.join(output_dir, "eval.log"),
    )
    logger.info("=" * 60)
    logger.info("  DenseUAV-Homo evaluation")
    logger.info("=" * 60)
    logger.info(f"  checkpoint : {args.checkpoint}")
    logger.info(f"  config     : {args.config}")
    logger.info(f"  data_root  : {data_root}")
    logger.info(f"  split      : {args.split}")
    logger.info(f"  batch_size : {batch_size}")
    logger.info(f"  device     : {device}")

    # ── Dataset + DataLoader ──────────────────────────────────────────────
    img_size = cfg.get("img_size", 512)
    logger.info(f"Building {args.split} dataset …")

    # Inference transforms (no augmentation)
    tf_uav = build_transforms(img_size=img_size, is_train=False, modality="uav")
    tf_sat = build_transforms(img_size=img_size, is_train=False, modality="satellite")

    is_split_eval: bool = False   # True for test split (separate q/g loaders)
    compute_sdm:   bool = True    # set False when GPS is unavailable

    if args.split == "train":
        dataset = DenseUAVPairs(
            data_root      = data_root,
            gps_file       = cfg.get("gps_train", "Dense_GPS_train.txt"),
            drone_altitude = cfg.get("drone_altitude", "H80"),
            transform_uav  = tf_uav,
            transform_sat  = tf_sat,
        )
        logger.info(f"  {dataset}")

        loader = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = device.type == "cuda",
            drop_last   = False,
        )
        logger.info(f"  {len(loader)} batches")

    else:  # "test" — official DenseUAV protocol: Q=777 queries vs G=3033 gallery
        is_split_eval = True

        query_dataset = DenseUAVQuery(
            data_root      = data_root,
            gps_file       = cfg.get("gps_test", "Dense_GPS_test.txt"),
            drone_altitude = cfg.get("drone_altitude", "H80"),
            transform_uav  = tf_uav,
        )
        gallery_dataset = DenseUAVGallery(
            data_root      = data_root,
            gps_file       = cfg.get("gps_test", "Dense_GPS_test.txt"),
            drone_altitude = cfg.get("drone_altitude", "H80"),
            transform_sat  = tf_sat,
        )
        logger.info(f"  {query_dataset}")
        logger.info(f"  {gallery_dataset}")

        compute_sdm = query_dataset.has_gps and gallery_dataset.has_gps
        if not compute_sdm:
            logger.warning(
                "GPS coordinates are unavailable for query or gallery — "
                "SDM@K will be skipped."
            )

        query_loader = DataLoader(
            query_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = device.type == "cuda",
            drop_last   = False,
        )
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = device.type == "cuda",
            drop_last   = False,
        )
        logger.info(
            f"  {len(query_loader)} query batches, "
            f"{len(gallery_loader)} gallery batches"
        )

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building model …")

    # num_classes must match the training checkpoint; read from config.
    num_classes = cfg.get("num_classes", 2256)
    head_cfg    = cfg.get("head", {})

    # When loading a checkpoint the weights override the backbone anyway;
    # setting pretrained=False avoids a redundant download.
    model = SiameseViT(
        num_classes          = num_classes,
        embed_dim            = cfg.get("embed_dim", 384),
        img_size             = img_size,
        patch_size           = cfg.get("patch_size", 16),
        gem_p                = cfg.get("gem_p", 3.0),
        gem_learnable        = cfg.get("gem_learnable", True),
        pretrained           = False,
        homo_hidden          = cfg.get("homo_hidden", 256),
        gate_bias_init       = cfg.get("gate_bias_init", -2.0),
        head_scale           = head_cfg.get("scale", 30.0),
        head_learnable_scale = head_cfg.get("learnable_scale", True),
    )

    # ── Load checkpoint ───────────────────────────────────────────────────
    logger.info(f"Loading checkpoint …")
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu", logger=logger)

    state_dict = ckpt.get("model", ckpt)   # support both wrapped and bare state dicts
    load_checkpoint_compat(unwrap_model(model), state_dict, logger=logger)
    ckpt_epoch = ckpt.get("epoch", "?")
    logger.info(f"  checkpoint epoch : {ckpt_epoch}")

    model = model.to(device)
    model.eval()
    logger.info(f"  {model}")

    # ── Evaluator ─────────────────────────────────────────────────────────
    recall_k = cfg.get("recall_k", [1, 5, 10])
    sdm_k    = cfg.get("sdm_k",    [1, 5, 10])
    sdm_s    = cfg.get("sdm_s",    5e3)

    evaluator = Evaluator(
        recall_k = recall_k,
        sdm_k    = sdm_k,
        device   = device_str,
        sdm_s    = sdm_s,
    )
    logger.info(f"  {evaluator}")

    # ── Run evaluation ────────────────────────────────────────────────────
    logger.info(f"\nRunning {args.split}-split evaluation …")
    prefix = f"{args.split}/"

    if is_split_eval:
        metrics = evaluator.evaluate_split(
            model, query_loader, gallery_loader,
            prefix      = prefix,
            compute_sdm = compute_sdm,
        )
    else:
        metrics = evaluator.evaluate(model, loader, prefix=prefix)

    # ── Print results ─────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  Results  (checkpoint epoch {ckpt_epoch})")
    logger.info("=" * 60)

    # Print Recall@K first, then SDM@K (SDM absent when GPS unavailable)
    for section, keys in [
        ("Recall@K", [k for k in metrics if "Recall" in k]),
        ("SDM@K",    [k for k in metrics if "SDM"    in k]),
    ]:
        if not keys:
            continue
        logger.info(f"  {section}:")
        for key in sorted(keys, key=lambda s: int(s.split("@")[1])):
            logger.info(f"    {key:<22} = {metrics[key]:.4f}")

    logger.info("=" * 60)
    logger.info(f"  Full log saved to: {os.path.join(output_dir, 'eval.log')}")


if __name__ == "__main__":
    main()
