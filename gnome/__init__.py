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
from gnome.rho import format_records, measure_rho

__all__ = [
    "Gnome",
    "stack_residuals",
    "DEFAULT_METRICS",
    "PrintDiagnostics",
    "JsonlDiagnostics",
    "CollectDiagnostics",
    "multi",
    "measure_rho",
    "format_records",
]
__version__ = "0.1.0"
