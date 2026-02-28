"""
utils/meters.py — lightweight running-average accumulators.
"""

from __future__ import annotations
from typing import Dict


class AverageMeter:
    """Track the running mean of a scalar (e.g. loss, accuracy).

    Example::
        meter = AverageMeter("loss")
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item(), n=batch_size)
        print(meter)   # "loss: 0.4231 (avg)"
    """

    def __init__(self, name: str = "", fmt: str = ".4f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0
        self.avg: float = 0.0

    def update(self, val: float, n: int = 1) -> None:
        """Accumulate *val* weighted by *n* samples."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        fmt_str = f"{{:{self.fmt}}}"
        return (
            f"{self.name}: "
            f"{fmt_str.format(self.val)} "
            f"(avg {fmt_str.format(self.avg)})"
        )


class MetricCollection:
    """A named dictionary of AverageMeters for convenient batch logging.

    Example::
        mc = MetricCollection(["loss", "loss_ce", "loss_trip", "loss_kl"])
        mc["loss"].update(total_loss.item(), n=B)
        mc.log(logger, step=100)
        mc.reset()
    """

    def __init__(self, names: list[str]) -> None:
        self._meters: Dict[str, AverageMeter] = {
            n: AverageMeter(n) for n in names
        }

    def __getitem__(self, key: str) -> AverageMeter:
        return self._meters[key]

    def reset(self) -> None:
        for m in self._meters.values():
            m.reset()

    def log(self, logger, step: int) -> None:
        """Write all meter summaries via *logger.info*."""
        parts = [f"step={step}"] + [repr(m) for m in self._meters.values()]
        logger.info("  ".join(parts))

    def summary(self) -> Dict[str, float]:
        """Return {name: avg} dict (useful for checkpoint metadata)."""
        return {name: m.avg for name, m in self._meters.items()}
