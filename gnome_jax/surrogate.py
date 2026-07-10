"""Main-loss and GGN-surrogate construction for Gnome (JAX port, MSE only).

The optimizer owns both of these so their scales stay mutually calibrated:
the ``eps`` / ``clip`` contract (docs/method.md §5.1) assumes the main loss
uses the fixed ``sum() / B`` reduction and the surrogate carries the
``sqrt(2)`` intrinsic-Hessian factor and ``1/sqrt(K)`` normalization. User
closures return raw ``(y_hat, y)`` and never pick a reduction.

CCE / CCE-Hutchinson surrogates (for the WikiText LM benchmark) are not
ported in v1 — see jax_port_plan.md Phase 7.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

_SQRT_TWO = math.sqrt(2.0)


def compute_main_loss(y_hat: jax.Array, y: jax.Array) -> jax.Array:
    """MSE main loss with Gnome's fixed reduction.

    ``((y_hat - y) ** 2).sum() / B`` — sum over output dim, mean over the
    batch (dim 0). This is the reduction the surrogate scaling is calibrated
    against; do not swap it for ``mean()``.
    """
    return ((y_hat - y) ** 2).sum() / y_hat.shape[0]


def build_surrogate_mse(y_hat_aux: jax.Array, key: jax.Array) -> jax.Array:
    """Scalar S whose parameter gradient is an unbiased GGN factor (MSE).

    The intrinsic output Hessian of the MSE loss is ``L'' = 2`` per element,
    so ``S = (sqrt(2) · ε · y_hat_aux).sum() / sqrt(K)`` with Rademacher
    signs ``ε`` satisfies ``E[(dS/dθ)(dS/dθ)^T] = (1/K) Σ_aux 2·J^T J`` — an
    unbiased estimator of the GGN, independent of the aux batch size ``K``
    (batch dim 0 of ``y_hat_aux``).
    """
    K = y_hat_aux.shape[0]
    signs = jax.random.rademacher(key, y_hat_aux.shape, dtype=y_hat_aux.dtype)
    signs = jax.lax.stop_gradient(signs)
    return (_SQRT_TWO * signs * y_hat_aux).sum() / math.sqrt(K)
