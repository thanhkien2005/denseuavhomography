"""
data/samplers.py — batch sampler that guarantees unique labels per batch.

PairPerClassBatchSampler
------------------------
Yields batches of exactly `batch_size` indices where every index has a
DISTINCT label.  This ensures:
  - Each batch contains `batch_size` unique locations.
  - The triplet loss has maximally diverse negatives within every batch.
  - Incomplete final batches (fewer than batch_size unique labels remaining)
    are dropped to keep tensor stacking trivial.

Epoch behaviour
---------------
On each call to __iter__ (i.e. each DataLoader epoch), the label pool is
re-shuffled using Python's random module.  Reproducibility is controlled by
calling utils.seed.set_seed() before training; the sampler itself is
stateless w.r.t. epochs.

Usage:
    from data.samplers import PairPerClassBatchSampler
    sampler = PairPerClassBatchSampler(dataset.labels, batch_size=32)
    loader  = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterator, List, Sequence

from torch.utils.data import Sampler


class PairPerClassBatchSampler(Sampler[List[int]]):
    """Yields batches of indices, one per unique label, each batch size P.

    Args:
        labels:     Sequence of integer labels, one per dataset item.
                    Must have at least `batch_size` unique values.
        batch_size: Number of samples per batch  (= number of unique labels
                    per batch, denoted P in the paper).

    Raises:
        AssertionError: If the number of unique labels < batch_size.
    """

    def __init__(self, labels: Sequence[int], batch_size: int) -> None:
        super().__init__()

        self.batch_size = batch_size

        # Group dataset indices by label  {label_int: [idx, idx, ...]}
        self._label_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self._label_to_indices[lbl].append(idx)

        self._unique_labels: List[int] = sorted(self._label_to_indices.keys())
        n_unique = len(self._unique_labels)

        assert n_unique >= batch_size, (
            f"PairPerClassBatchSampler: batch_size={batch_size} but only "
            f"{n_unique} unique labels exist in the dataset.  "
            "Reduce batch_size or use a larger dataset split."
        )

    # ------------------------------------------------------------------
    # Properties (useful for logging)
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Total number of unique classes known to this sampler."""
        return len(self._unique_labels)

    # ------------------------------------------------------------------
    # Sampler interface
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices for one epoch.

        Algorithm:
          1. Shuffle the pool of unique labels.
          2. Walk through the pool in steps of `batch_size`.
          3. For each label in a step, pick ONE random index from that label's
             index list.
          4. Yield the resulting batch of `batch_size` indices.
          5. Drop any trailing labels that don't fill a complete batch.
        """
        # Fresh shuffle every epoch (reproducibility via global random seed)
        label_pool = self._unique_labels.copy()
        random.shuffle(label_pool)

        batch: List[int] = []
        for label in label_pool:
            # Pick a random sample for this label
            idx = random.choice(self._label_to_indices[label])
            batch.append(idx)

            if len(batch) == self.batch_size:
                yield batch
                batch = []
        # Drop any trailing incomplete batch (batch list is discarded here)

    def __len__(self) -> int:
        """Number of complete batches per epoch (drops incomplete tail)."""
        return len(self._unique_labels) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        """Seed the random state for this epoch (optional; DDP-friendly).

        Calling this before each epoch gives fully deterministic shuffling
        when combined with a fixed global seed.  Without it, shuffling uses
        the current random state (still reproducible if set_seed() was called).

        Args:
            epoch: Current epoch index (0-based).
        """
        # Re-seed with a hash of (global_seed XOR epoch) pattern.
        # We don't store the global seed here; callers who want strict
        # reproducibility should call utils.seed.set_seed() + this method.
        random.seed(epoch)

    def __repr__(self) -> str:
        return (
            f"PairPerClassBatchSampler("
            f"num_classes={self.num_classes}, "
            f"batch_size={self.batch_size}, "
            f"batches_per_epoch={len(self)})"
        )
