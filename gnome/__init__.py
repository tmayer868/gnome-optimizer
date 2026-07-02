"""Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition."""

from gnome.blocks import stack_residuals
from gnome.optimizer import Gnome

__all__ = ["Gnome", "stack_residuals"]
__version__ = "0.1.0"
