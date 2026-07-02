"""Device selection for experiment scripts."""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Prefer CUDA, then MPS, fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
