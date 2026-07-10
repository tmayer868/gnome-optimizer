"""Gnome optimizer, JAX port (MSE / PINN scope — see jax_port_plan.md).

Public API:

- :func:`gnome` — optimizer factory returning ``(init, update, step)``.
- :class:`GnomeState`, :class:`GnomeOptimizer` — state / bundle types.
- :func:`stack_residuals` — multi-block PINN residual stacking.
- :func:`build_surrogate_mse`, :func:`compute_main_loss` — the pieces
  ``step`` composes, exported for callers using the pure ``update`` core.
"""

from .blocks import stack_residuals
from .gnome import GnomeOptimizer, GnomeState, gnome
from .surrogate import build_surrogate_mse, compute_main_loss

__all__ = [
    "gnome",
    "GnomeOptimizer",
    "GnomeState",
    "stack_residuals",
    "build_surrogate_mse",
    "compute_main_loss",
]
