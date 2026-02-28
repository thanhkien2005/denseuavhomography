"""
metrics/recall.py — Recall@K for cross-view geo-localisation.

Recall@K (UAV → Satellite)
──────────────────────────
For each UAV query embedding, retrieve the top-K most similar satellite
gallery embeddings by cosine similarity.  A query is considered "correct"
if at least one of the top-K gallery items shares the same location label.

    Recall@K = (# queries with correct match in top-K) / (# total queries)

Inputs are assumed to be **L2-normalised** so that cosine similarity
reduces to a dot product: sim(q, g) = q · g  ∈ [-1, 1].

Usage:
    results = recall_at_k(q_emb, g_emb, q_labels, g_labels, k_list=[1,5,10])
    # results = {1: 0.823, 5: 0.961, 10: 0.982}
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor


def recall_at_k(
    query_emb:      Tensor,       # (Q, D)  L2-normalised query embeddings
    gallery_emb:    Tensor,       # (G, D)  L2-normalised gallery embeddings
    query_labels:   Tensor,       # (Q,)    integer class label per query
    gallery_labels: Tensor,       # (G,)    integer class label per gallery item
    k_list:         List[int],    # e.g. [1, 5, 10]
) -> Dict[int, float]:
    """Compute Recall@K for each K in k_list.

    Args:
        query_emb:      (Q, D) float tensor, L2-normalised.
        gallery_emb:    (G, D) float tensor, L2-normalised.
        query_labels:   (Q,)  long tensor with class indices.
        gallery_labels: (G,)  long tensor with class indices.
        k_list:         List of K values to evaluate.

    Returns:
        Dict mapping K → recall value in [0, 1].

    Shape trace:
        sim          : (Q, G)   cosine similarity matrix
        topk_idx     : (Q, K)   gallery indices sorted by similarity
        topk_labels  : (Q, K)   labels of retrieved gallery items
        match        : (Q, K)   bool: retrieved label == query label
        hit          : (Q,)     bool: any match in top-K
        recall       : scalar
    """
    assert query_emb.ndim  == 2, f"query_emb must be (Q,D), got {query_emb.shape}"
    assert gallery_emb.ndim == 2, f"gallery_emb must be (G,D), got {gallery_emb.shape}"
    assert query_emb.shape[1] == gallery_emb.shape[1], (
        f"Embedding dim mismatch: query {query_emb.shape[1]} vs "
        f"gallery {gallery_emb.shape[1]}"
    )
    assert query_labels.shape   == (query_emb.shape[0],),   \
        f"query_labels shape {query_labels.shape} != ({query_emb.shape[0]},)"
    assert gallery_labels.shape == (gallery_emb.shape[0],), \
        f"gallery_labels shape {gallery_labels.shape} != ({gallery_emb.shape[0]},)"

    Q, D = query_emb.shape      # Q queries
    G    = gallery_emb.shape[0] # G gallery items
    K_max = max(k_list)

    assert K_max <= G, (
        f"Requested K={K_max} but gallery only has G={G} items."
    )

    # ── cosine similarity (dot product of L2-normalised vectors) ──────────
    # sim[q, g] = query_emb[q] · gallery_emb[g]
    sim = query_emb @ gallery_emb.T           # (Q, G)

    # ── retrieve top-K_max gallery indices per query ──────────────────────
    # largest=True because higher cosine similarity = more similar
    topk_idx    = sim.topk(K_max, dim=1, largest=True).indices  # (Q, K_max)
    topk_labels = gallery_labels[topk_idx]                      # (Q, K_max)

    # ── compute Recall@K for each K ───────────────────────────────────────
    # query_labels[:,None] broadcasts: (Q,1) vs (Q,K_max)
    match = topk_labels == query_labels.unsqueeze(1)            # (Q, K_max) bool

    results: Dict[int, float] = {}
    for k in sorted(k_list):
        # hit[q] = True if any of the first k retrieved items is correct
        hit         = match[:, :k].any(dim=1)                  # (Q,)  bool
        results[k]  = hit.float().mean().item()                 # scalar in [0,1]

    return results
