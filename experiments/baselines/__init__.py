"""Baselines for Gnome benchmarks.

Currently the SOAP optimizer (https://arxiv.org/abs/2409.11321). AdamW comes
from ``torch.optim`` and is not re-exported.
"""

from experiments.baselines.soap import SOAP

__all__ = ["SOAP"]
