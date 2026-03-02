"""
utils/memory_queue.py — FIFO memory queue for cross-batch negatives.

Stores L2-normalised embeddings from past iterations on CPU so the GPU
memory footprint stays bounded.  Embeddings are moved to the target device
only when accessed.

Usage in a training loop
────────────────────────
    queue = MemoryQueue(queue_size=4096, embed_dim=384, device=device)

    for batch in loader:
        emb_uav, emb_sat = model(...)      # (B, D), L2-normalised

        # --- loss uses queue BEFORE this batch is enqueued ---
        q_uav = queue.uav_embeddings       # (K, D) — previous batches
        q_sat = queue.sat_embeddings       # (K, D)
        loss = infonce(emb_uav, emb_sat, q_uav, q_sat)
        loss.backward()
        optimizer.step()

        # --- enqueue AFTER update (standard stale-by-one, MoCo style) ---
        queue.enqueue(emb_uav.detach().float(), emb_sat.detach().float())

Warmup
──────
When the queue is empty (first iteration) `uav_embeddings` and
`sat_embeddings` return zero-row tensors (shape 0×D).  InfoNCELoss handles
this naturally and degrades to in-batch contrastive until the queue fills.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor


class MemoryQueue:
    """Circular FIFO buffer of paired (uav, sat) L2-normalised embeddings.

    Args:
        queue_size: Maximum number of stored embedding vectors.
        embed_dim:  Embedding dimension (must match model output, e.g. 384).
        device:     torch.device to move embeddings to on read.
                    Pass None to keep on CPU.
    """

    def __init__(
        self,
        queue_size: int,
        embed_dim:  int,
        device: Optional[torch.device] = None,
    ) -> None:
        self.queue_size = queue_size
        self.embed_dim  = embed_dim
        self._device    = device

        # CPU buffers — shape (queue_size, embed_dim)
        self._uav = torch.zeros(queue_size, embed_dim)
        self._sat = torch.zeros(queue_size, embed_dim)

        self._ptr    = 0   # next write position
        self._filled = 0   # number of valid entries (< queue_size during warmup)

    # ──────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────

    def enqueue(self, emb_uav: Tensor, emb_sat: Tensor) -> None:
        """Add a batch of paired embeddings to the queue.

        Args:
            emb_uav: (B, D) detached float32 UAV embeddings.
            emb_sat: (B, D) detached float32 SAT embeddings.

        Wraps around (circular FIFO) when the queue is full.
        """
        B   = emb_uav.shape[0]
        ptr = self._ptr
        end = ptr + B

        if end <= self.queue_size:
            self._uav[ptr:end].copy_(emb_uav.cpu())
            self._sat[ptr:end].copy_(emb_sat.cpu())
        else:
            # Batch spans the circular boundary — split into two writes
            tail = self.queue_size - ptr
            self._uav[ptr:].copy_(emb_uav[:tail].cpu())
            self._sat[ptr:].copy_(emb_sat[:tail].cpu())
            self._uav[:B - tail].copy_(emb_uav[tail:].cpu())
            self._sat[:B - tail].copy_(emb_sat[tail:].cpu())

        self._ptr    = end % self.queue_size
        self._filled = min(self._filled + B, self.queue_size)

    # ──────────────────────────────────────────────────────────────────────
    # Read
    # ──────────────────────────────────────────────────────────────────────

    @property
    def uav_embeddings(self) -> Tensor:
        """(K, D) float32 — valid UAV embeddings, moved to target device."""
        t = self._uav[:self._filled]
        return t.to(self._device) if self._device is not None else t

    @property
    def sat_embeddings(self) -> Tensor:
        """(K, D) float32 — valid SAT embeddings, moved to target device."""
        t = self._sat[:self._filled]
        return t.to(self._device) if self._device is not None else t

    def __len__(self) -> int:
        """Number of valid (filled) entries in the queue."""
        return self._filled

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────

    def state_dict(self) -> Dict:
        return {
            "uav":    self._uav,
            "sat":    self._sat,
            "ptr":    self._ptr,
            "filled": self._filled,
        }

    def load_state_dict(self, d: Dict) -> None:
        self._uav    = d["uav"]
        self._sat    = d["sat"]
        self._ptr    = d["ptr"]
        self._filled = d["filled"]

    def __repr__(self) -> str:
        return (
            f"MemoryQueue(queue_size={self.queue_size}, "
            f"embed_dim={self.embed_dim}, "
            f"filled={self._filled})"
        )
