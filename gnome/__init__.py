"""Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition."""

from gnome.blocks import stack_residuals
from gnome.diagnostics import (
    DEFAULT_METRICS,
    CollectDiagnostics,
    JsonlDiagnostics,
    PrintDiagnostics,
    multi,
)
from gnome.optimizer import Gnome

__all__ = [
    "Gnome",
    "stack_residuals",
    "DEFAULT_METRICS",
    "PrintDiagnostics",
    "JsonlDiagnostics",
    "CollectDiagnostics",
    "multi",
]
__version__ = "0.1.0"
