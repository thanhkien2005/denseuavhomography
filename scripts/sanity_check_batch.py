"""
scripts/sanity_check_batch.py
─────────────────────────────
Load config → build dataset + sampler + DataLoader → pull exactly ONE batch →
print and assert every tensor shape.

Run from repo root (denseuav-homo/):
    python scripts/sanity_check_batch.py
    python scripts/sanity_check_batch.py --config configs/denseuav_v1.yaml
    python scripts/sanity_check_batch.py --data_root /abs/path/to/DenseUAV

No model is imported; this script validates the data pipeline only.
"""

import argparse
import os
import sys
import time

# Allow imports from repo root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from torch.utils.data import DataLoader

from data.transforms       import build_transforms
from data.denseuav_dataset import DenseUAVPairs
from data.samplers         import PairPerClassBatchSampler
from utils.seed            import set_seed


# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DenseUAV batch shape sanity check")
    p.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "denseuav_v1.yaml",
        ),
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--data_root",
        default=None,
        help="Override data_root from config (absolute or relative to cwd).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch_size from config.",
    )
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args   = parse_args()
    cfg    = load_config(args.config)

    # ── Resolve data_root ────────────────────────────────────────────────────
    data_root = args.data_root or cfg["data_root"]
    if not os.path.isabs(data_root):
        # Resolve relative path from the repo root (parent of scripts/)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.normpath(os.path.join(repo_root, data_root))

    batch_size = args.batch_size or cfg["batch_size"]
    img_size   = cfg["img_size"]
    seed       = cfg["seed"]

    print("=" * 60)
    print("  DenseUAV — batch sanity check")
    print("=" * 60)
    print(f"  config     : {args.config}")
    print(f"  data_root  : {data_root}")
    print(f"  img_size   : {img_size}")
    print(f"  batch_size : {batch_size}")
    print(f"  seed       : {seed}")
    print()

    set_seed(seed, deterministic=True)

    # ── Build transforms ─────────────────────────────────────────────────────
    tf_uav = build_transforms(img_size=img_size, is_train=True,  modality="uav")
    tf_sat = build_transforms(img_size=img_size, is_train=True,  modality="satellite")
    print(f"  [transforms] UAV  pipeline : {len(tf_uav.transforms)} steps")
    print(f"  [transforms] SAT  pipeline : {len(tf_sat.transforms)} steps")

    # ── Build dataset ─────────────────────────────────────────────────────────
    t0 = time.time()
    dataset = DenseUAVPairs(
        data_root      = data_root,
        gps_file       = cfg["gps_train"],
        drone_altitude = cfg.get("drone_altitude", "H80"),
        transform_uav  = tf_uav,
        transform_sat  = tf_sat,
    )
    print(f"  [dataset]    {dataset}")
    print(f"               built in {time.time()-t0:.1f}s")

    # ── Build sampler + DataLoader ────────────────────────────────────────────
    labels  = [s[2] for s in dataset.samples]   # extract label per sample
    sampler = PairPerClassBatchSampler(labels, batch_size=batch_size)
    print(f"  [sampler]    {sampler}")

    loader = DataLoader(
        dataset,
        batch_sampler = sampler,
        num_workers   = 0,      # 0 avoids multiprocessing issues on Windows
        pin_memory    = False,
    )
    print(f"  [loader]     {len(loader)} batches per epoch")
    print()

    # ── Pull exactly one batch ────────────────────────────────────────────────
    print("  Pulling first batch from DataLoader ...")
    t0    = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")
    print()

    # ── Print + assert shapes ─────────────────────────────────────────────────
    B = batch_size

    uav_img = batch["uav_img"]    # expected (B, 3, 512, 512)
    sat_img = batch["sat_img"]    # expected (B, 3, 512, 512)
    label   = batch["label"]      # expected (B,)
    uav_gps = batch["uav_gps"]    # expected (B, 2)
    sat_gps = batch["sat_gps"]    # expected (B, 2)

    rows = [
        ("uav_img",  uav_img,  (B, 3, img_size, img_size), torch.float32),
        ("sat_img",  sat_img,  (B, 3, img_size, img_size), torch.float32),
        ("label",    label,    (B,),                        torch.int64),
        ("uav_gps",  uav_gps,  (B, 2),                     torch.float32),
        ("sat_gps",  sat_gps,  (B, 2),                     torch.float32),
    ]

    all_ok = True
    print(f"  {'key':<10}  {'actual shape':<25}  {'expected':<25}  {'dtype':<12}  status")
    print(f"  {'-'*9}  {'-'*24}  {'-'*24}  {'-'*11}  {'-'*6}")
    for name, tensor, expected_shape, expected_dtype in rows:
        shape_ok = tuple(tensor.shape) == expected_shape
        dtype_ok = tensor.dtype       == expected_dtype
        ok       = shape_ok and dtype_ok
        all_ok   = all_ok and ok
        status   = "OK" if ok else "FAIL"
        print(
            f"  {name:<10}  "
            f"{str(tuple(tensor.shape)):<25}  "
            f"{str(expected_shape):<25}  "
            f"{str(tensor.dtype):<12}  "
            f"{status}"
        )

    print()

    # Extra content checks
    label_unique = label.unique()
    print(f"  Unique labels in batch : {len(label_unique)}/{B} "
          f"(expected {B} — all distinct)")
    assert len(label_unique) == B, (
        f"Batch has duplicate labels! {label.tolist()}"
    )

    # Pixel range check: after ImageNet normalisation values should span roughly [-3, 3]
    for name, tensor, _, _ in [("uav_img", uav_img, None, None),
                                 ("sat_img", sat_img, None, None)]:
        vmin, vmax = tensor.min().item(), tensor.max().item()
        print(f"  {name} value range : [{vmin:.3f}, {vmax:.3f}]  "
              f"(post-normalisation, expected approx [-3, 3])")
        assert -5.0 < vmin < 5.0 and -5.0 < vmax < 5.0, (
            f"{name} values out of expected range: [{vmin}, {vmax}]"
        )

    print()

    # Hard-assert after printing so we see all rows before a potential crash
    for name, tensor, expected_shape, expected_dtype in rows:
        assert tuple(tensor.shape) == expected_shape, (
            f"{name}: shape {tuple(tensor.shape)} != {expected_shape}"
        )
        assert tensor.dtype == expected_dtype, (
            f"{name}: dtype {tensor.dtype} != {expected_dtype}"
        )

    print("=" * 60)
    print("  All shape + dtype + content assertions PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
