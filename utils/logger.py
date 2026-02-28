"""
utils/logger.py — minimal stdout logger with optional file sink.

Usage:
    logger = get_logger("train", log_file="outputs/run.log")
    logger.info("epoch 1 started")
"""

import logging
import os
import sys
from typing import Optional


def get_logger(name: str = "denseuav", log_file: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """Return (or create) a named logger.

    The logger writes to stdout.  If *log_file* is provided, a second
    FileHandler is added and the parent directory is created automatically.

    Calling this function multiple times with the same *name* returns the
    same logger (idempotent — handlers are not duplicated).
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- stdout handler ---
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # --- optional file handler ---
    if log_file is not None:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Prevent propagation to the root logger (avoids duplicate output).
    logger.propagate = False
    return logger
