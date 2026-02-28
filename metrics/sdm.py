"""
metrics/sdm.py — Score-based Distance Metric (SDM@K) from the DenseUAV paper.

SDM@K
─────
Unlike binary Recall@K, SDM@K gives credit for *nearly-correct* retrievals
by weighting each retrieved item by (a) its rank and (b) its geographic
proximity to the query.

Formula (per query q):
    SDM@K(q) = [Σ_{i=1}^{K}  w_i · score(d_i)] / Z

where:
    w_i    = K - i + 1              rank weight (rank-1 gets weight K)
    d_i    = Haversine distance (m) between query GPS and i-th retrieved GPS
    score  = max(0, 1 - d_i / s)   linear proximity in [0,1], zero beyond s
    s      = 5000 m                 distance scale (default from DenseUAV paper)
    Z      = Σ_{i=1}^{K} w_i = K·(K+1)/2    normalisation constant

Final SDM@K = mean over all queries.

GPS format:
    All GPS tensors store [longitude, latitude] in decimal degrees,
    matching the DenseUAV annotation file convention (E<lon> N<lat>).

Usage:
    # topk_indices: (Q, K) gallery indices from sim.topk(K, dim=1)
    score = sdm_at_k(topk_indices, q_gps, g_gps, K=10, s=5e3)
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# Haversine distance
# ─────────────────────────────────────────────────────────────────────────────

_EARTH_R_M = 6_371_000.0   # mean Earth radius in metres


def haversine_distance(
    lon1: Tensor,   # (...) decimal degrees
    lat1: Tensor,   # (...) decimal degrees
    lon2: Tensor,   # (...) decimal degrees
    lat2: Tensor,   # (...) decimal degrees
) -> Tensor:
    """Vectorised Haversine great-circle distance in metres.

    Accepts any broadcastable shape; returns a tensor of the same shape.

    Shape trace (example):
        lon1, lat1 : (Q, 1)
        lon2, lat2 : (Q, K)
        return     : (Q, K)
    """
    # Convert degrees → radians
    lat1_r = torch.deg2rad(lat1)
    lat2_r = torch.deg2rad(lat2)
    dlat   = torch.deg2rad(lat2 - lat1)
    dlon   = torch.deg2rad(lon2 - lon1)

    a = (
        torch.sin(dlat / 2.0) ** 2
        + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2.0) ** 2
    )
    # Clamp a to [0,1] to guard against floating-point rounding above 1
    a = torch.clamp(a, 0.0, 1.0)
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))
    return _EARTH_R_M * c   # metres


# ─────────────────────────────────────────────────────────────────────────────
# SDM@K — single K
# ─────────────────────────────────────────────────────────────────────────────

def sdm_at_k(
    topk_indices: Tensor,   # (Q, K)  gallery indices for each query
    q_gps:        Tensor,   # (Q, 2)  query GPS  [lon, lat] in decimal degrees
    g_gps:        Tensor,   # (G, 2)  gallery GPS [lon, lat] in decimal degrees
    K:            int,
    s:            float = 5e3,   # distance scale in metres (default 5 km)
) -> float:
    """Compute mean SDM@K over all Q queries.

    Args:
        topk_indices: (Q, K) int tensor of gallery indices, ranked by
                      descending similarity (index 0 = rank-1 best match).
        q_gps:        (Q, 2) float tensor [lon, lat] per query.
        g_gps:        (G, 2) float tensor [lon, lat] per gallery item.
        K:            Number of top retrievals to consider.
        s:            Distance scale in metres; score = 0 for d >= s.

    Returns:
        Scalar float in [0, 1].  Higher is better.

    Shape trace:
        topk_gps   : (Q, K, 2)   GPS of the K retrieved gallery items
        q_lon/lat  : (Q, 1)      query lon/lat, broadcast over K
        g_lon/lat  : (Q, K)      retrieved gallery lon/lat
        dist       : (Q, K)      Haversine distances in metres
        prox_score : (Q, K)      linear proximity scores in [0, 1]
        weights    : (K,)        rank weights [K, K-1, ..., 1]
        w_scores   : (Q,)        weighted score per query
        Z          : scalar      K*(K+1)/2
    """
    assert topk_indices.ndim == 2, \
        f"topk_indices must be (Q, K), got {topk_indices.shape}"
    assert q_gps.ndim == 2 and q_gps.shape[1] == 2, \
        f"q_gps must be (Q, 2), got {q_gps.shape}"
    assert g_gps.ndim == 2 and g_gps.shape[1] == 2, \
        f"g_gps must be (G, 2), got {g_gps.shape}"

    Q, actual_K = topk_indices.shape
    assert actual_K >= K, (
        f"topk_indices has only {actual_K} columns but K={K} requested."
    )
    # Use only the first K columns (caller may pass K_max columns)
    idx = topk_indices[:, :K]   # (Q, K)

    # GPS coordinates of the K retrieved gallery items per query
    topk_gps = g_gps[idx]      # (Q, K, 2)

    # Expand query GPS for broadcasting: (Q, 1)
    q_lon = q_gps[:, 0].unsqueeze(1)   # (Q, 1)
    q_lat = q_gps[:, 1].unsqueeze(1)   # (Q, 1)
    g_lon = topk_gps[:, :, 0]          # (Q, K)
    g_lat = topk_gps[:, :, 1]          # (Q, K)

    # Haversine distances in metres between query and each retrieved item
    dist = haversine_distance(q_lon, q_lat, g_lon, g_lat)  # (Q, K)

    # Linear proximity score: 1.0 at distance 0, 0.0 at distance >= s
    prox_score = torch.clamp(1.0 - dist / s, min=0.0)      # (Q, K)  ∈ [0, 1]

    # Rank weights: w_i = K - i + 1  →  [K, K-1, ..., 1]  for ranks 1..K
    weights = torch.arange(K, 0, -1, dtype=torch.float32)  # (K,)
    # Move to same device as prox_score
    weights = weights.to(prox_score.device)

    Z = weights.sum()   # K*(K+1)/2  — normalisation constant

    # Weighted score per query
    w_scores = (prox_score * weights).sum(dim=1)    # (Q,)

    sdm_per_query = w_scores / Z                    # (Q,)  ∈ [0, 1]
    return sdm_per_query.mean().item()              # scalar


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: compute SDM for multiple K values at once
# ─────────────────────────────────────────────────────────────────────────────

def sdm_at_k_multi(
    query_emb:   Tensor,        # (Q, D)  L2-normalised query embeddings
    gallery_emb: Tensor,        # (G, D)  L2-normalised gallery embeddings
    q_gps:       Tensor,        # (Q, 2)  query GPS [lon, lat]
    g_gps:       Tensor,        # (G, 2)  gallery GPS [lon, lat]
    k_list:      List[int],     # e.g. [1, 5, 10]
    s:           float = 5e3,
) -> Dict[int, float]:
    """Compute SDM@K for all K values in k_list from raw embeddings.

    Retrieves top-max(k_list) gallery items once, then slices for each K.

    Args:
        query_emb:   (Q, D) L2-normalised.
        gallery_emb: (G, D) L2-normalised.
        q_gps:       (Q, 2) decimal-degree GPS [lon, lat].
        g_gps:       (G, 2) decimal-degree GPS [lon, lat].
        k_list:      List of K values.
        s:           Distance scale in metres.

    Returns:
        Dict {K: sdm_score} with values in [0, 1].

    Shape trace:
        sim          : (Q, G)
        topk_indices : (Q, K_max)
    """
    K_max = max(k_list)
    sim   = query_emb @ gallery_emb.T                          # (Q, G)
    topk_indices = sim.topk(K_max, dim=1, largest=True).indices # (Q, K_max)

    return {
        k: sdm_at_k(topk_indices, q_gps, g_gps, K=k, s=s)
        for k in sorted(k_list)
    }
