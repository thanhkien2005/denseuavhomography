"""
engine/evaluator.py — collect embeddings and compute Recall@K + SDM@K.

Design
──────
The Evaluator is model-agnostic: it calls

    outputs = model(uav_img, sat_img)

and expects the return value to be a dict containing at minimum:

    outputs["emb_uav"] : (B, D)  L2-normalised UAV embeddings
    outputs["emb_sat"] : (B, D)  L2-normalised satellite embeddings

Both must be on CPU or GPU; the evaluator moves them to CPU for
concatenation and metric computation.

DataLoader expectation
──────────────────────
The dataloader should yield batches with keys:
    "uav_img"  : (B, 3, H, W)  float32
    "sat_img"  : (B, 3, H, W)  float32
    "label"    : (B,)           int64
    "uav_gps"  : (B, 2)         float32  [lon, lat]
    "sat_gps"  : (B, 2)         float32  [lon, lat]

This matches DenseUAVPairs.__getitem__ exactly.

Retrieval setup
───────────────
After collecting all embeddings from the dataloader:
    queries  = all UAV embeddings,  shape (N, D)
    gallery  = all SAT embeddings,  shape (N, D)

The correct gallery match for query i is every gallery j where
    label[i] == label[j].

When using DenseUAVPairs (one pair per location), N_query == N_gallery
and each label appears exactly once in both sets, so query i's only
correct match is gallery i.

For the proper test split (777 query UAV vs 3033 gallery SAT items),
pass two separate DataLoaders via evaluate_split().

Usage:
    ev = Evaluator(recall_k=[1,5,10], sdm_k=[1,5,10], device="cuda")
    metrics = ev.evaluate(model, train_loader)
    # {"Recall@1": 0.83, "Recall@5": 0.96, "SDM@1": 0.71, ...}
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow import from repo root when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.recall import recall_at_k
from metrics.sdm    import sdm_at_k_multi


class Evaluator:
    """Collect embeddings from a DataLoader and compute retrieval metrics.

    Args:
        recall_k: List of K values for Recall@K  (e.g. [1, 5, 10]).
        sdm_k:    List of K values for SDM@K     (e.g. [1, 5, 10]).
        device:   Torch device string for model inference.
        sdm_s:    Distance scale (metres) for SDM proximity score.
    """

    def __init__(
        self,
        recall_k: List[int] = (1, 5, 10),
        sdm_k:    List[int] = (1, 5, 10),
        device:   str       = "cuda",
        sdm_s:    float     = 5e3,
    ) -> None:
        self.recall_k = list(recall_k)
        self.sdm_k    = list(sdm_k)
        self.device   = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sdm_s    = sdm_s

    # ──────────────────────────────────────────────────────────────────────
    # Internal: embedding extraction
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_embeddings(
        self,
        model:      nn.Module,
        dataloader: DataLoader,
        desc:       str = "Extracting",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward all batches through model; collect and return tensors.

        Returns:
            emb_uav    : (N, D)   UAV embeddings (L2-normalised, CPU)
            emb_sat    : (N, D)   SAT embeddings (L2-normalised, CPU)
            labels     : (N,)     class labels (CPU)
            uav_gps    : (N, 2)   UAV GPS [lon, lat] (CPU)
            sat_gps    : (N, 2)   SAT GPS [lon, lat] (CPU)
        """
        model.eval()
        model.to(self.device)

        all_emb_uav: List[Tensor] = []
        all_emb_sat: List[Tensor] = []
        all_labels:  List[Tensor] = []
        all_uav_gps: List[Tensor] = []
        all_sat_gps: List[Tensor] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            # Move images to device; keep GPS/labels on CPU
            uav_img: Tensor = batch["uav_img"].to(self.device)  # (B,3,H,W)
            sat_img: Tensor = batch["sat_img"].to(self.device)  # (B,3,H,W)
            labels:  Tensor = batch["label"]                    # (B,)
            uav_gps: Tensor = batch["uav_gps"]                  # (B,2)
            sat_gps: Tensor = batch["sat_gps"]                  # (B,2)

            # Model forward
            # Expected interface: model(uav_img, sat_img) -> dict
            #   dict["emb_uav"] : (B, D)  L2-normalised UAV embedding
            #   dict["emb_sat"] : (B, D)  L2-normalised SAT embedding
            # All other outputs (logits, homo params) are ignored here.
            outputs = model(uav_img, sat_img)

            assert "emb_uav" in outputs and "emb_sat" in outputs, (
                "Model must return a dict with keys 'emb_uav' and 'emb_sat'. "
                f"Got keys: {list(outputs.keys())}"
            )

            emb_uav: Tensor = outputs["emb_uav"]   # (B, D)
            emb_sat: Tensor = outputs["emb_sat"]   # (B, D)

            assert emb_uav.ndim == 2, f"emb_uav must be (B,D), got {emb_uav.shape}"
            assert emb_sat.ndim == 2, f"emb_sat must be (B,D), got {emb_sat.shape}"
            assert emb_uav.shape == emb_sat.shape, (
                f"emb_uav {emb_uav.shape} and emb_sat {emb_sat.shape} must match"
            )

            all_emb_uav.append(emb_uav.cpu())
            all_emb_sat.append(emb_sat.cpu())
            all_labels.append(labels.cpu())
            all_uav_gps.append(uav_gps.cpu())
            all_sat_gps.append(sat_gps.cpu())

        # Concatenate across batches
        emb_uav  = torch.cat(all_emb_uav,  dim=0)   # (N, D)
        emb_sat  = torch.cat(all_emb_sat,  dim=0)   # (N, D)
        labels   = torch.cat(all_labels,   dim=0)   # (N,)
        uav_gps  = torch.cat(all_uav_gps,  dim=0)   # (N, 2)
        sat_gps  = torch.cat(all_sat_gps,  dim=0)   # (N, 2)

        N, D = emb_uav.shape
        assert emb_sat.shape  == (N, D),   f"emb_sat shape mismatch: {emb_sat.shape}"
        assert labels.shape   == (N,),     f"labels shape mismatch: {labels.shape}"
        assert uav_gps.shape  == (N, 2),   f"uav_gps shape mismatch: {uav_gps.shape}"
        assert sat_gps.shape  == (N, 2),   f"sat_gps shape mismatch: {sat_gps.shape}"

        return emb_uav, emb_sat, labels, uav_gps, sat_gps

    # ──────────────────────────────────────────────────────────────────────
    # Internal: single-modality helpers for evaluate_split()
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_uav_embeddings(
        self,
        model:      nn.Module,
        dataloader: DataLoader,
        desc:       str = "Extracting UAV",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Collect UAV embeddings from a query-only dataloader.

        Each batch must contain "uav_img", "label", and "uav_gps".
        A zeros tensor is used as a dummy satellite input so that the model
        forward can execute; emb_sat is discarded.  The UAV embedding is
        mildly affected by the HomographyNet receiving zeros for sat_img
        (warp gate ≈ 0.12 at init, gradient signal was minimal during
        training), which is an accepted approximation for test evaluation.

        Returns:
            emb_uav : (Q, D)  L2-normalised UAV embeddings (CPU)
            labels  : (Q,)    class labels (CPU)
            uav_gps : (Q, 2)  GPS [lon, lat] (CPU); zeros if unavailable
        """
        model.eval()
        model.to(self.device)

        all_emb_uav: List[Tensor] = []
        all_labels:  List[Tensor] = []
        all_uav_gps: List[Tensor] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            uav_img: Tensor = batch["uav_img"].to(self.device)
            # Dummy zeros for sat_img — emb_uav is the only output used here.
            dummy_sat = torch.zeros_like(uav_img)
            outputs   = model(uav_img, dummy_sat)

            all_emb_uav.append(outputs["emb_uav"].cpu())
            all_labels.append(batch["label"].cpu())
            all_uav_gps.append(batch["uav_gps"].cpu())

        return (
            torch.cat(all_emb_uav, dim=0),
            torch.cat(all_labels,  dim=0),
            torch.cat(all_uav_gps, dim=0),
        )

    @torch.no_grad()
    def _collect_sat_embeddings(
        self,
        model:      nn.Module,
        dataloader: DataLoader,
        desc:       str = "Extracting SAT",
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Collect SAT embeddings from a gallery-only dataloader.

        Each batch must contain "sat_img", "label", and "sat_gps".
        emb_sat is computed solely from sat_img (the satellite branch does
        not depend on uav_img), so zeros are safe as the dummy UAV input.

        Returns:
            emb_sat : (G, D)  L2-normalised SAT embeddings (CPU)
            labels  : (G,)    class labels (CPU)
            sat_gps : (G, 2)  GPS [lon, lat] (CPU); zeros if unavailable
        """
        model.eval()
        model.to(self.device)

        all_emb_sat: List[Tensor] = []
        all_labels:  List[Tensor] = []
        all_sat_gps: List[Tensor] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            sat_img: Tensor = batch["sat_img"].to(self.device)
            # emb_sat depends only on sat_img; zeros for uav_img are safe.
            dummy_uav = torch.zeros_like(sat_img)
            outputs   = model(dummy_uav, sat_img)

            all_emb_sat.append(outputs["emb_sat"].cpu())
            all_labels.append(batch["label"].cpu())
            all_sat_gps.append(batch["sat_gps"].cpu())

        return (
            torch.cat(all_emb_sat, dim=0),
            torch.cat(all_labels,  dim=0),
            torch.cat(all_sat_gps, dim=0),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public: paired evaluate (one dataloader, UAV=query, SAT=gallery)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        model:      nn.Module,
        dataloader: DataLoader,
        prefix:     str = "",
    ) -> Dict[str, float]:
        """Evaluate model on a single paired dataloader.

        UAV embeddings are queries; satellite embeddings are the gallery.
        The correct match for query i is every gallery j with the same label.

        Args:
            model:      Model with forward(uav_img, sat_img) -> dict interface.
            dataloader: Yields batches with "uav_img", "sat_img", "label",
                        "uav_gps", "sat_gps".
            prefix:     Optional string prefix for metric keys (e.g. "val/").

        Returns:
            Dict of metric names → float values.
            Keys: "{prefix}Recall@K", "{prefix}SDM@K" for each configured K.

        Shape trace:
            emb_uav   : (N, D)   all UAV query embeddings
            emb_sat   : (N, D)   all SAT gallery embeddings
            labels    : (N,)     class label per item
            sim       : (N, N)   pairwise cosine similarity matrix
        """
        emb_uav, emb_sat, labels, uav_gps, sat_gps = self._collect_embeddings(
            model, dataloader, desc=f"{prefix}eval"
        )

        # queries  = UAV embeddings  (N, D)
        # gallery  = SAT embeddings  (N, D)
        # For each query i, correct gallery items = those j where labels[i]==labels[j]
        q_labels = labels   # (N,) — query labels
        g_labels = labels   # (N,) — gallery labels (same tensor, different semantics)

        results: Dict[str, float] = {}

        # ── Recall@K ──────────────────────────────────────────────────────
        recall = recall_at_k(
            query_emb      = emb_uav,
            gallery_emb    = emb_sat,
            query_labels   = q_labels,
            gallery_labels = g_labels,
            k_list         = self.recall_k,
        )
        for k, v in recall.items():
            results[f"{prefix}Recall@{k}"] = v

        # ── SDM@K ─────────────────────────────────────────────────────────
        # Retrieve top-max(sdm_k) for all queries, then slice per K.
        # GPS: queries use uav_gps, gallery items use sat_gps.
        sdm = sdm_at_k_multi(
            query_emb   = emb_uav,
            gallery_emb = emb_sat,
            q_gps       = uav_gps,
            g_gps       = sat_gps,
            k_list      = self.sdm_k,
            s           = self.sdm_s,
        )
        for k, v in sdm.items():
            results[f"{prefix}SDM@{k}"] = v

        return results

    # ──────────────────────────────────────────────────────────────────────
    # Public: split evaluate (separate query and gallery dataloaders)
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_split(
        self,
        model:              nn.Module,
        query_dataloader:   DataLoader,   # yields UAV query batches
        gallery_dataloader: DataLoader,   # yields SAT gallery batches
        prefix:             str  = "test/",
        compute_sdm:        bool = True,
    ) -> Dict[str, float]:
        """Evaluate on disjoint query and gallery sets (proper test protocol).

        Used for the DenseUAV test split:
            query_dataloader   → DenseUAVQuery   (777  drone images)
            gallery_dataloader → DenseUAVGallery (3033 satellite images)

        Args:
            model:              Siamese model with forward(uav_img, sat_img).
            query_dataloader:   Yields batches with "uav_img", "label", "uav_gps".
            gallery_dataloader: Yields batches with "sat_img", "label", "sat_gps".
            prefix:             Metric key prefix (default "test/").
            compute_sdm:        If False, skip SDM (use when GPS is unavailable).

        Returns:
            Dict of metric names → float values.
            Keys: "{prefix}Recall@K" always;
                  "{prefix}SDM@K"    only when compute_sdm=True.

        Shape trace:
            emb_uav_q : (Q, D)   Q = 777
            emb_sat_g : (G, D)   G = 3033
            sim       : (Q, G)
        """
        emb_uav_q, q_labels, uav_gps = self._collect_uav_embeddings(
            model, query_dataloader, desc=f"{prefix}query"
        )
        emb_sat_g, g_labels, sat_gps = self._collect_sat_embeddings(
            model, gallery_dataloader, desc=f"{prefix}gallery"
        )

        results: Dict[str, float] = {}

        # ── Recall@K ──────────────────────────────────────────────────────
        recall = recall_at_k(
            query_emb      = emb_uav_q,
            gallery_emb    = emb_sat_g,
            query_labels   = q_labels,
            gallery_labels = g_labels,
            k_list         = self.recall_k,
        )
        for k, v in recall.items():
            results[f"{prefix}Recall@{k}"] = v

        # ── SDM@K (optional — requires valid GPS for both sets) ───────────
        if compute_sdm:
            sdm = sdm_at_k_multi(
                query_emb   = emb_uav_q,
                gallery_emb = emb_sat_g,
                q_gps       = uav_gps,
                g_gps       = sat_gps,
                k_list      = self.sdm_k,
                s           = self.sdm_s,
            )
            for k, v in sdm.items():
                results[f"{prefix}SDM@{k}"] = v

        return results

    def __repr__(self) -> str:
        return (
            f"Evaluator("
            f"recall_k={self.recall_k}, "
            f"sdm_k={self.sdm_k}, "
            f"device={self.device}, "
            f"sdm_s={self.sdm_s:.0f}m)"
        )
