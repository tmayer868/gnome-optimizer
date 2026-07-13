"""Pure-JAX ModifiedMlp with RWF + optional periodic/Fourier embeddings.

The plain-architecture backbone of the jaxpi benchmarks (Wang et al.,
arXiv:2502.00604), reimplemented from the published methods — jaxpi's own
code is under a no-redistribution license and is not copied:

* **ModifiedMlp** (Wang, Teng & Perdikaris 2021): two input encoders
  ``u, v``; each hidden layer's activation is gated as
  ``x = act(Dense(x)); x = x·u + (1−x)·v``.
* **Random weight factorization** (RWF; Wang et al. 2022,
  arXiv:2210.01274): every kernel is stored as ``g ⊙ V`` with
  ``g = exp(N(mean, stddev))`` per output column factored out of a Glorot
  init, ``V = W/g``; biases start at zero.
* **Periodic embeddings**: coordinates listed in ``period_axes`` are
  replaced by ``[cos(p·x), sin(p·x)]``, making the network exactly
  periodic in them (used to enforce periodic BCs by construction).
* **Random Fourier features** (Tancik et al. 2020): the embedded input is
  mapped through ``[cos(zB), sin(zB)]`` with ``B ~ N(0, scale²)`` of shape
  ``(dim_in, fourier_dim/2)``. Matching jaxpi, ``B`` is a trainable
  parameter (it receives gradients like any other weight).

Params are a plain dict pytree — no flax. ``make_modified_mlp`` returns
``(init_fn, apply_fn)`` where ``apply_fn(params, z)`` maps a single point
``z`` of shape ``(in_dim,)`` to ``(out_dim,)``; vmap for batches, and
``jax.grad`` through it for PDE residuals.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


def _rwf_dense_init(key, fan_in, fan_out, mean, stddev):
    kw, kg = jax.random.split(key)
    w = jax.nn.initializers.glorot_normal()(kw, (fan_in, fan_out))
    g = jnp.exp(mean + stddev * jax.random.normal(kg, (fan_out,)))
    return {"g": g, "v": w / g, "b": jnp.zeros(fan_out)}


def _dense(p, x):
    return x @ (p["g"] * p["v"]) + p["b"]


def make_modified_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int,
    num_layers: int,
    rwf_mean: float = 1.0,
    rwf_stddev: float = 0.1,
    period: Optional[Sequence[float]] = None,
    period_axes: Optional[Sequence[int]] = None,
    fourier_scale: Optional[float] = None,
    fourier_dim: Optional[int] = None,
) -> Tuple[Callable, Callable]:
    """Build ``(init_fn, apply_fn)`` for a ModifiedMlp.

    Args:
        in_dim: Raw coordinate dimension (e.g. 2 for ``(t, x)``).
        hidden / out_dim / num_layers: ModifiedMlp shape (num_layers is the
            number of gated hidden layers, matching jaxpi's ``num_layers``).
        rwf_mean / rwf_stddev: RWF init distribution for ``g``.
        period / period_axes: Periodic-embedding periods and the coordinate
            axes they apply to (e.g. ``period=(pi,), period_axes=(1,)``
            makes the net 2π/p-periodic in coordinate 1). None disables.
        fourier_scale / fourier_dim: Random-Fourier-feature init scale and
            output width (must be even). None disables.
    """
    if (period is None) != (period_axes is None):
        raise ValueError("period and period_axes must be given together")
    if (fourier_scale is None) != (fourier_dim is None):
        raise ValueError("fourier_scale and fourier_dim must be given together")
    if fourier_dim is not None and fourier_dim % 2:
        raise ValueError("fourier_dim must be even")

    period_axes = tuple(period_axes) if period_axes is not None else ()
    period = tuple(period) if period is not None else ()

    # Coordinate dim after the periodic embedding (each periodic axis
    # becomes a cos/sin pair).
    embed_dim = in_dim + len(period_axes)
    # Input width the encoders / first hidden layer actually see.
    net_in = fourier_dim if fourier_dim is not None else embed_dim

    def _embed(params, z):
        if period_axes:
            parts = []
            for i in range(in_dim):
                if i in period_axes:
                    p = period[period_axes.index(i)]
                    parts.extend([jnp.cos(p * z[i]), jnp.sin(p * z[i])])
                else:
                    parts.append(z[i])
            z = jnp.stack(parts)
        if fourier_dim is not None:
            zb = z @ params["fourier"]
            z = jnp.concatenate([jnp.cos(zb), jnp.sin(zb)])
        return z

    def init_fn(key):
        keys = jax.random.split(key, num_layers + 4)
        params = {}
        if fourier_dim is not None:
            params["fourier"] = fourier_scale * jax.random.normal(
                keys[-2], (embed_dim, fourier_dim // 2)
            )
        dims = [net_in] + [hidden] * num_layers
        params["enc_u"] = _rwf_dense_init(
            keys[0], net_in, hidden, rwf_mean, rwf_stddev
        )
        params["enc_v"] = _rwf_dense_init(
            keys[1], net_in, hidden, rwf_mean, rwf_stddev
        )
        params["hidden"] = [
            _rwf_dense_init(keys[2 + i], dims[i], hidden, rwf_mean, rwf_stddev)
            for i in range(num_layers)
        ]
        params["out"] = _rwf_dense_init(
            keys[-1], hidden, out_dim, rwf_mean, rwf_stddev
        )
        return params

    def apply_fn(params, z):
        x = _embed(params, z)
        u = jnp.tanh(_dense(params["enc_u"], x))
        v = jnp.tanh(_dense(params["enc_v"], x))
        for layer in params["hidden"]:
            x = jnp.tanh(_dense(layer, x))
            x = x * u + (1 - x) * v
        return _dense(params["out"], x)

    return init_fn, apply_fn
