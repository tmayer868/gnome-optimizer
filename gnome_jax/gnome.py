"""Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition (JAX port).

Gnome is inspired by SOAP_ but makes two changes that turn the SOAP machinery
into a second-order method:

1. **GGN eigenbasis.** SOAP builds its Kronecker factors from the loss
   gradient (an empirical-Fisher proxy). Gnome builds them from a Hutchinson
   estimate of the Generalized Gauss-Newton matrix — the true GGN, not the
   empirical Fisher. The eigenbases are therefore aligned with curvature
   directions of the loss surface rather than with past gradients.

2. **Newton step in the rotated basis, clip as trust region.** SOAP runs an
   Adam update inside the rotated basis (gradient divided by ``sqrt`` of a
   second-moment EMA). Gnome runs a Newton step instead — the rotated
   gradient divided by the *un-square-rooted* curvature EMA — bounded by
   ``clip`` in both the rotated and rotated-back bases.

This port follows the two-level API from jax_port_plan.md: a pure
``update(g_main, g_aux, state, params)`` core plus a closure-driving
``step(params, state, key, main_fn, aux_fn)`` wrapper. Both are pure
functions of their inputs; the whole ``step`` — user closures with their
nested ``jax.grad`` PDE derivatives included — compiles under one
``jax.jit``. RNG is an explicit ``key`` argument, threaded and returned.

Structure and jit patterns (the ``lax.cond`` init/refresh branches, the
``Preconditioner`` pytree) are adapted from SOAP_JAX_, the unofficial optax
port of SOAP; the algorithm itself matches the PyTorch reference in
``gnome/optimizer.py``. Two documented deviations from PyTorch (see
jax_port_plan.md "Decisions"): warmup is a clean linear ramp without the
incidental ``0.01`` floor / ``-1`` offset, and no bit-exact parity is
attempted (RNG streams differ by construction).

.. _SOAP: https://arxiv.org/abs/2409.11321
.. _SOAP_JAX: https://github.com/haydn-jones/SOAP_JAX

Usage::

    import gnome_jax

    opt = gnome_jax.gnome(lr=1e-3)
    state = opt.init(params)
    key = jax.random.PRNGKey(0)

    @jax.jit
    def train_step(params, state, key, batch):
        x_main, y_main, x_aux, y_aux = batch
        return opt.step(
            params, state, key,
            main_fn=lambda p: (model_apply(p, x_main), y_main),
            aux_fn=lambda p: (model_apply(p, x_aux), y_aux),
        )

    params, loss, state, key = train_step(params, state, key, batch)
"""

from __future__ import annotations

from itertools import chain
from typing import Any, Callable, Iterable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .surrogate import build_surrogate_mse, compute_main_loss

PreconditionerMatrix = Union[jax.Array, None]
Params = Any  # arbitrary pytree of arrays
ClosureFn = Callable[[Params], Tuple[jax.Array, jax.Array]]  # p -> (y_hat, y)

_QR_DTYPE = jnp.float32  # working precision for GG / Q, matching the
# PyTorch implementation's float32 eigh path (independent of param dtype,
# including under jax_enable_x64).
_PRECISION = jax.lax.Precision.HIGHEST


@jtu.register_pytree_node_class
class Preconditioner:
    """Per-parameter Kronecker-factor container.

    One entry per tensor mode; ``None`` marks modes that are not
    preconditioned (1-D params with ``precondition_1d=False``, or modes
    larger than ``max_precond_dim``). The ``None``/array pattern is part of
    the static pytree structure, so the per-mode Python loops in
    ``_project`` / ``_project_back`` trace cleanly under ``jax.jit``.
    """

    __slots__ = ("matrices",)

    def __init__(self, matrices: Iterable[PreconditionerMatrix]):
        self.matrices = tuple(matrices)

    def tree_flatten(self):
        return (self.matrices, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    def map(self, fn):
        return Preconditioner(fn(m) for m in self.matrices)


def _is_preconditioner(value: object) -> bool:
    return isinstance(value, Preconditioner)


class GnomeState(NamedTuple):
    """Optimizer state. All fields are pytrees mirroring the params.

    ``count`` is a single global step counter (vs PyTorch's per-param
    ``state["step"]``); this assumes every parameter receives a gradient
    every step, which holds for the closure API — ``jax.grad`` returns
    zeros, not None, for unused parameters.
    """

    count: jax.Array  # int32 scalar; step 1 is the basis-init step
    exp_avg: Params  # first-moment EMA of the main grad, rotated basis
    exp_avg_sq: Params  # curvature EMA of the squared surrogate grad ("gnd_m")
    GG: Params  # pytree of Preconditioner (Kronecker factors)
    Q: Params  # pytree of Preconditioner (eigenbases)


class GnomeOptimizer(NamedTuple):
    """Bundle of pure functions returned by :func:`gnome`.

    ``init(params) -> GnomeState``

    ``update(g_main, g_aux, state, params) -> (new_params, new_state)`` —
    the pure core, for callers that compute their own gradients (build
    ``g_aux`` from :func:`gnome_jax.build_surrogate_mse`).

    ``step(params, state, key, main_fn, aux_fn) ->
    (new_params, loss, new_state, new_key)`` — the closure API matching the
    PyTorch optimizer: each closure maps params to ``(y_hat, y)`` and the
    optimizer owns the loss reduction and surrogate construction.
    """

    init: Callable[[Params], GnomeState]
    update: Callable[..., Tuple[Params, GnomeState]]
    step: Callable[..., Tuple[Params, jax.Array, GnomeState, jax.Array]]


# ----------------------------------------------------------------------
# Preconditioner machinery (SOAP lineage, adapted from SOAP_JAX)
# ----------------------------------------------------------------------


def _init_conditioner(
    p: jax.Array,
    max_precond_dim: int,
    precondition_1d: bool,
) -> Preconditioner:
    if p.ndim == 1:
        if not precondition_1d or p.shape[0] > max_precond_dim:
            return Preconditioner([None])
        return Preconditioner(
            [jnp.zeros((p.shape[0], p.shape[0]), dtype=_QR_DTYPE)]
        )
    return Preconditioner(
        [
            jnp.zeros((s, s), dtype=_QR_DTYPE) if s <= max_precond_dim else None
            for s in p.shape
        ]
    )


def _lerp(start: jax.Array, end: jax.Array, weight: float) -> jax.Array:
    return start + weight * (end - start)


def _update_preconditioner(
    grad: jax.Array,
    GG: Preconditioner,
    beta: float,
) -> Preconditioner:
    """EMA the per-mode outer products of ``grad`` into the Kronecker
    factors. For Gnome ``grad`` is the *surrogate* gradient ``g_aux`` — the
    defining change vs SOAP, which feeds the loss gradient here.
    """
    if grad.ndim == 1:
        if GG.matrices[0] is None:
            return GG
        outer = jnp.matmul(
            grad[:, None], grad[None, :], precision=_PRECISION
        )
        return Preconditioner(
            [_lerp(GG.matrices[0], outer.astype(GG.matrices[0].dtype), 1 - beta)]
        )

    new_GG = []
    for idx, gg in enumerate(GG.matrices):
        if gg is None:
            new_GG.append(None)
            continue
        outer = jnp.tensordot(
            grad,
            grad,
            axes=[[*chain(range(idx), range(idx + 1, grad.ndim))]] * 2,
            precision=_PRECISION,
        )
        new_GG.append(_lerp(gg, outer.astype(gg.dtype), 1 - beta))
    return Preconditioner(new_GG)


def _project(grad: jax.Array, Q: Preconditioner) -> jax.Array:
    """Rotate ``grad`` into the eigenbasis. Each preconditioned mode is
    contracted against its Q; unpreconditioned modes are cycled to the back
    so the mode order is preserved overall.
    """
    for mat in Q.matrices:
        if mat is not None:
            grad = jnp.tensordot(
                grad, mat.astype(grad.dtype), axes=((0,), (0,)),
                precision=_PRECISION,
            )
        else:
            grad = jnp.moveaxis(grad, 0, -1)
    return grad


def _project_back(grad: jax.Array, Q: Preconditioner) -> jax.Array:
    for mat in Q.matrices:
        if mat is not None:
            grad = jnp.tensordot(
                grad, mat.astype(grad.dtype), axes=((0,), (1,)),
                precision=_PRECISION,
            )
        else:
            grad = jnp.moveaxis(grad, 0, -1)
    return grad


def _get_orthogonal_matrix(gg: PreconditionerMatrix) -> PreconditionerMatrix:
    """Initial eigenbasis via full ``eigh``, eigenvectors sorted by
    descending eigenvalue.

    Matches the PyTorch ``_eigh_safe``: relative jitter (1e-6 of the mean
    absolute diagonal, floored at 1e-6) keeps rank-deficient factors
    tractable, and a non-finite result falls back to the identity — the
    ``jnp.where`` traceable analogue of the reference's try/except.
    """
    if gg is None:
        return None
    m = gg.astype(_QR_DTYPE)
    n = m.shape[0]
    scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(m))), 1.0)
    eye = jnp.eye(n, dtype=_QR_DTYPE)
    _, evecs = jnp.linalg.eigh(m + (1e-6 * scale) * eye)
    q = jnp.flip(evecs, axis=1)
    return jnp.where(jnp.all(jnp.isfinite(q)), q, eye)


def _qr_refresh_eigenbasis(
    GG: Preconditioner,
    Q: Preconditioner,
    exp_avg_sq: jax.Array,
) -> Tuple[Preconditioner, jax.Array]:
    """One power-iteration + QR refresh of the eigenbasis.

    Rayleigh-estimate the eigenvalues from ``diag(Q^T GG Q)``, sort the
    columns descending, permute ``exp_avg_sq`` along the corresponding mode
    to stay aligned (it is a per-coordinate variance in the rotated basis),
    then orthonormalize ``GG @ Q``. Converges to the true eigenbasis over
    refresh cycles at a fraction of ``eigh``'s cost.
    """
    new_Q = []
    for ind, (m, o) in enumerate(zip(GG.matrices, Q.matrices)):
        if m is None or o is None:
            new_Q.append(None)
            continue
        m_f = m.astype(_QR_DTYPE)
        o_f = o.astype(_QR_DTYPE)
        est_eig = jnp.diag(
            jnp.matmul(
                jnp.matmul(o_f.T, m_f, precision=_PRECISION),
                o_f,
                precision=_PRECISION,
            )
        )
        sort_idx = jnp.argsort(est_eig, descending=True)
        exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=ind)
        o_f = o_f[:, sort_idx]
        power_iter = jnp.matmul(m_f, o_f, precision=_PRECISION)
        q_new, _ = jnp.linalg.qr(power_iter)
        new_Q.append(q_new)
    return Preconditioner(new_Q), exp_avg_sq


# ----------------------------------------------------------------------
# Optimizer factory
# ----------------------------------------------------------------------


def gnome(
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    shampoo_beta: float = 0.95,
    eps: float = 1e-4,
    weight_decay: float = 0.01,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    clip: Optional[float] = 1.0,
    warmup: int = 200,
    precondition_1d: bool = False,
) -> GnomeOptimizer:
    """Build a Gnome optimizer (MSE loss; defaults match the PyTorch class).

    Args:
        lr: Learning rate.
        betas: ``(beta1, beta2)`` for the gradient and curvature EMAs in the
            rotated basis.
        shampoo_beta: EMA decay for the Kronecker factors. If negative,
            ``betas[1]`` is used.
        eps: Second-order damping added to the curvature denominator. Not a
            numerical-safety term: larger values pull the update toward
            gradient descent, smaller toward a pure Newton step.
        weight_decay: Decoupled weight decay coefficient.
        precondition_frequency: Steps between eigenbasis refreshes.
        max_precond_dim: Modes larger than this get no Kronecker factor.
        clip: Clamp the per-coordinate update to ``[-clip, +clip]`` in both
            the rotated and rotated-back bases (trust region). ``None``
            disables.
        warmup: Linear lr ramp length: effective lr at update ``k`` is
            ``lr * min(k / warmup, 1.0)``. ``0`` disables.
        precondition_1d: Build Kronecker factors for 1-D params too.

    Returns:
        A :class:`GnomeOptimizer` bundle of ``(init, update, step)``.
    """
    if lr < 0.0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError(f"Invalid beta1: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError(f"Invalid beta2: {betas[1]}")
    if eps < 0.0:
        raise ValueError(f"Invalid eps: {eps}")
    if clip is not None and clip <= 0.0:
        raise ValueError(f"Invalid clip: {clip}")
    if warmup < 0:
        raise ValueError(f"Invalid warmup: {warmup}")
    if precondition_frequency <= 0:
        raise ValueError(
            f"Invalid precondition_frequency: {precondition_frequency}"
        )

    beta1, beta2 = betas
    gg_beta = shampoo_beta if shampoo_beta >= 0 else beta2

    def init_fn(params: Params) -> GnomeState:
        zeros = jtu.tree_map(jnp.zeros_like, params)
        conditioners = jtu.tree_map(
            lambda p: _init_conditioner(p, max_precond_dim, precondition_1d),
            params,
        )
        # Q starts as zero matrices with the same static structure the eigh
        # init produces, so both lax.cond branches in update_fn agree.
        return GnomeState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=zeros,
            exp_avg_sq=zeros,
            GG=conditioners,
            Q=jtu.tree_map(
                lambda p: _init_conditioner(p, max_precond_dim, precondition_1d),
                params,
            ),
        )

    def update_fn(
        g_main: Params,
        g_aux: Params,
        state: GnomeState,
        params: Params,
    ) -> Tuple[Params, GnomeState]:
        count_inc = state.count + 1
        state = state._replace(count=count_inc)

        def init_step() -> Tuple[Params, GnomeState]:
            # First step: seed the Kronecker factors with the first
            # surrogate gradient and build the eigenbasis; no parameter
            # update (matching the PyTorch reference).
            new_GG = jtu.tree_map(
                lambda g, gg: _update_preconditioner(g, gg, gg_beta),
                g_aux,
                state.GG,
                is_leaf=_is_preconditioner,
            )
            new_Q = jtu.tree_map(
                lambda gg: gg.map(_get_orthogonal_matrix),
                new_GG,
                is_leaf=_is_preconditioner,
            )
            return params, state._replace(GG=new_GG, Q=new_Q)

        def update_step() -> Tuple[Params, GnomeState]:
            # Effective update index: count 1 was the basis-init step.
            eff_step = (count_inc - 1).astype(jnp.float32)

            # Rotate the loss gradient and the surrogate gradient into the
            # current eigenbasis.
            g_rot = jtu.tree_map(
                lambda g, q: _project(g, q),
                g_main,
                state.Q,
                is_leaf=_is_preconditioner,
            )
            gs_rot = jtu.tree_map(
                lambda g, q: _project(g, q),
                g_aux,
                state.Q,
                is_leaf=_is_preconditioner,
            )

            # EMAs in the rotated basis: first moment from the loss grad,
            # curvature ("gnd_m") from the squared surrogate grad.
            exp_avg = jtu.tree_map(
                lambda m, g: beta1 * m + (1.0 - beta1) * g,
                state.exp_avg,
                g_rot,
            )
            exp_avg_sq = jtu.tree_map(
                lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g),
                state.exp_avg_sq,
                gs_rot,
            )

            # Bias correction on each moment separately — the eps damping
            # sits between them, so the usual combined factor would change
            # the update.
            bc1 = 1.0 - beta1**eff_step
            bc2 = 1.0 - beta2**eff_step

            # Newton step in the rotated basis: un-square-rooted curvature
            # denominator (the Gnome change vs SOAP), clipped as a trust
            # region in both bases.
            def newton_update(m, v, q):
                upd = (m / bc1) / (v / bc2 + eps)
                if clip is not None:
                    upd = jnp.clip(upd, -clip, clip)
                upd = _project_back(upd, q)
                if clip is not None:
                    upd = jnp.clip(upd, -clip, clip)
                return upd

            updates = jtu.tree_map(
                newton_update,
                exp_avg,
                exp_avg_sq,
                state.Q,
                is_leaf=_is_preconditioner,
            )

            if warmup > 0:
                lr_eff = lr * jnp.minimum(eff_step / warmup, 1.0)
            else:
                lr_eff = jnp.asarray(lr, jnp.float32)

            if weight_decay > 0.0:
                apply_update = lambda p, u: p - lr_eff * u - lr_eff * weight_decay * p
            else:
                apply_update = lambda p, u: p - lr_eff * u
            new_params = jtu.tree_map(apply_update, params, updates)

            # Refresh the Kronecker factors with the new surrogate gradient;
            # refresh the eigenbasis on schedule (translating exp_avg into
            # the new basis and permuting exp_avg_sq alongside Q's columns).
            new_GG = jtu.tree_map(
                lambda g, gg: _update_preconditioner(g, gg, gg_beta),
                g_aux,
                state.GG,
                is_leaf=_is_preconditioner,
            )

            def refresh():
                q_and_v = jtu.tree_map(
                    lambda gg, q, v: _qr_refresh_eigenbasis(gg, q, v),
                    new_GG,
                    state.Q,
                    exp_avg_sq,
                    is_leaf=_is_preconditioner,
                )
                new_Q = jtu.tree_map(lambda _, x: x[0], g_main, q_and_v)
                new_v = jtu.tree_map(lambda _, x: x[1], g_main, q_and_v)
                new_m = jtu.tree_map(
                    lambda m, old_q, nq: _project(_project_back(m, old_q), nq),
                    exp_avg,
                    state.Q,
                    new_Q,
                    is_leaf=_is_preconditioner,
                )
                return new_Q, new_v, new_m

            def keep():
                return state.Q, exp_avg_sq, exp_avg

            new_Q, exp_avg_sq_out, exp_avg_out = jax.lax.cond(
                (count_inc - 1) % precondition_frequency == 0,
                refresh,
                keep,
            )

            return new_params, GnomeState(
                count=count_inc,
                exp_avg=exp_avg_out,
                exp_avg_sq=exp_avg_sq_out,
                GG=new_GG,
                Q=new_Q,
            )

        return jax.lax.cond(count_inc == 1, init_step, update_step)

    def step_fn(
        params: Params,
        state: GnomeState,
        key: jax.Array,
        main_fn: ClosureFn,
        aux_fn: ClosureFn,
    ) -> Tuple[Params, jax.Array, GnomeState, jax.Array]:
        key_aux, key_next = jax.random.split(key)

        def main_loss(p):
            y_hat, y = main_fn(p)
            return compute_main_loss(y_hat, jax.lax.stop_gradient(y))

        loss, g_main = jax.value_and_grad(main_loss)(params)

        def surrogate(p):
            y_hat_aux, _ = aux_fn(p)
            return build_surrogate_mse(y_hat_aux, key_aux)

        g_aux = jax.grad(surrogate)(params)

        new_params, new_state = update_fn(g_main, g_aux, state, params)
        return new_params, loss, new_state, key_next

    return GnomeOptimizer(init=init_fn, update=update_fn, step=step_fn)
