"""Multi-block residual stacking for PINN-style losses.

A PINN loss is a weighted sum of per-block mean-squared residuals,
``L = Σ_j λ_j · mean_i(r_{j,i}²)`` — typically one block for the PDE
residual on collocation points, one for the IC, one for each BC, etc.
This helper folds those blocks into a single flat residual vector whose
plain ``mean(·²)`` reproduces ``L`` exactly. Feed the result through
Gnome's ``loss='mse'`` path: the automatic ``√2·ε`` Rademacher surrogate
on the stacked vector decomposes into the independent-Rademacher-per-block
GGN estimator, with the right λ_j weighting and cross-block independence
guaranteed by the per-coordinate ε draw.

Algebra. With per-block element count ``N_j`` and total ``N = Σ N_j``,
scaling block j by ``α_j = sqrt(λ_j · N / N_j)`` gives

    mean(stacked²) = (1/N) Σ_j α_j² · sum_i r_{j,i}²
                  = Σ_j λ_j · mean_i(r_{j,i}²).

For the surrogate ``S = (sqrt(2)/sqrt(N)) Σ ε · stacked`` we get

    g_s = ∂S/∂θ = Σ_j sqrt(2 λ_j / N_j) · J_j^T ε_{(j)},
    E[g_s g_s^T] = Σ_j 2 λ_j · (1/N_j) · sum_k J_{j,k}^T J_{j,k},

which is the true multi-block GGN with the right λ_j weights. Cross-block
expectations vanish exactly because ε is drawn per coordinate (no shared
probes across blocks).

Higher-order autograd flows through unchanged. The helper only does scalar
multiplies and a ``torch.cat``; if a residual was built with
``create_graph=True`` on input-derivative operators (``u_t``, ``u_xx``,
...), the stacked tensor still differentiates back to θ through those
operators.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


def stack_residuals(
    residuals: Sequence[torch.Tensor],
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Rescale and concatenate per-block PINN residuals into one MSE target.

    Args:
        residuals: Per-block residual tensors. Each is reshaped to 1-D, so
            shapes ``(N_j,)``, ``(N_j, 1)``, and ``(N_j, d_j)`` are all
            accepted — what matters is the element count.
        weights: Per-block scalar weights ``λ_j``. Defaults to all-ones
            (equal weighting). Must match ``residuals`` in length.

    Returns:
        A flat 1-D tensor of length ``Σ_j N_j`` such that
        ``mean(out**2) == Σ_j λ_j · mean(r_j**2)``.

    The returned tensor preserves the autograd graph of every input
    residual, including higher-order input derivatives built with
    ``create_graph=True``. Pass the result as ``y_hat`` against a zeros
    target through ``Gnome.step`` (or any standard MSE loop).
    """
    if not residuals:
        raise ValueError("stack_residuals requires at least one residual block")
    if weights is None:
        weights = [1.0] * len(residuals)
    if len(weights) != len(residuals):
        raise ValueError(
            f"weights and residuals length mismatch: "
            f"{len(weights)} weights vs {len(residuals)} residuals"
        )
    flats = [r.reshape(-1) for r in residuals]
    sizes = [f.numel() for f in flats]
    n_total = sum(sizes)
    scaled = [
        f * math.sqrt(w * n_total / s)
        for f, w, s in zip(flats, weights, sizes)
    ]
    return torch.cat(scaled)
