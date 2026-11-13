"""Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition.

Gnome is inspired by SOAP_ but makes two changes that turn the SOAP machinery
into a second-order method:

1. **GGN eigenbasis.** SOAP builds its Kronecker factors from the loss
   gradient (an empirical-Fisher proxy). Gnome builds them from a
   Hutchinson estimate of the Generalized Gauss-Newton matrix — the true
   GGN, not the empirical Fisher. The eigenbases ``Q_L`` and ``Q_R`` are
   therefore aligned with curvature directions of the loss surface rather
   than with the directions of past gradients.

2. **Newton step in the rotated basis, with an l2 trust region.** SOAP runs
   an Adam update inside the rotated basis (gradient divided by ``sqrt`` of
   a second-moment EMA). Gnome runs a Newton step instead — the rotated
   gradient divided by the *un-square-rooted* curvature EMA — which is the
   diagonal Gauss-Newton update inside the eigenbasis. A pure Newton step
   has no built-in step-size control, so Gnome damps it with a
   Levenberg-Marquardt ``lambda`` chosen as the smallest value satisfying
   ``||m / (v + lambda)||_2 <= trust_radius * sqrt(||p||² + S0)``, where
   ``S0`` is the tensor's squared norm on its first optimizer step.
   Because ``Q_L`` and ``Q_R`` are
   orthonormal the l2 norm survives the rotation back, so the bound holds
   in parameter space too. It is a trust region, not a band-aid: when
   curvature is well-conditioned it never binds, but it prevents blow-ups
   from small denominators while the eigenbasis is still warming up.

Gnome supports four explicit loss types — chosen via the ``loss`` constructor
arg — so that the curvature scaling is unambiguous and reduction-independent:

  * ``loss='mse'`` — mean-squared error for regression. The intrinsic output
    Hessian is ``L''=2`` per element; the Rademacher surrogate is
    ``S = (sqrt(2) * eps * y_hat).sum() / sqrt(K)`` over the aux samples.
    Internally the main loss uses ``((y_hat - y)**2).sum() / B`` (sum over
    output dim, mean over batch).
  * ``loss='cce'`` — softmax cross-entropy for classification, Fisher
    sampling variant. The output Hessian is non-diagonal so the surrogate
    draws ``y_tilde ~ Categorical(softmax(y_hat))`` and uses
    ``F.cross_entropy(y_hat, y_tilde, reduction='sum') / sqrt(K)``.
    Internally the main loss uses ``F.cross_entropy(reduction='mean')``.
  * ``loss='cce_hutchinson'`` — same task and main loss as ``cce``, but a
    Rademacher Hutchinson factorization of the softmax Hessian
    ``H = diag(p) - p p^T = A A^T`` with ``A = diag(sqrt(p))(I - sqrt(p) sqrt(p)^T)``.
    Drawing ``R ~ Rademacher`` per (sample, class) and forming
    ``v = sqrt(p) ⊙ R - (sqrt(p)·R) · p`` gives ``E[v v^T] = H`` exactly,
    so ``S = <y_hat, detach(v)>.sum() / sqrt(K)`` estimates the same GGN
    as ``cce`` without a discrete label draw. Lower per-step variance at
    the same aux batch size ``K``.
  * ``loss='bce_hutchinson'`` — binary cross-entropy with logits and a
    Rademacher Hutchinson factorization of its diagonal output Hessian.
    With ``p = sigmoid(logits)``, the detached probe is
    ``sqrt(p * (1 - p)) * R`` and the main loss is summed over output
    coordinates then averaged over the batch.

Because the optimizer owns both the loss and the surrogate, your two
closures just return ``(y_hat, y)`` for the main batch and the aux batch
respectively — no need to pick a reduction or write a per-loss helper.
The two batches should be disjoint (typically a random split of one
mini-batch into a B-K main slice and a K aux slice):

    opt = Gnome(model.parameters(), lr=1e-3, loss='mse')

    for x, y in loader:
        perm = torch.randperm(x.shape[0])
        x_main, y_main = x[perm[K:]], y[perm[K:]]
        x_aux,  y_aux  = x[perm[:K]], y[perm[:K]]

        def main_closure(): return model(x_main), y_main
        def aux_closure():  return model(x_aux),  y_aux

        loss = opt.step(main_closure, aux_closure)

Each closure runs an independent forward. By default each backward releases
its graph immediately. Optional ``step(..., surrogate_partition_sizes=[...])``
splits the auxiliary batch into separate surrogates whose gradients are computed
in one batched autograd call using vmap. Curvature contributions
are summed without cross-partition outer products. The optimizer never writes
into ``p.grad``; gradients are obtained via ``torch.autograd.grad`` and applied
directly.

**Learning-rate schedules.** Gnome does not own a schedule. ``group["lr"]``
is read fresh on every step and multiplies the update *after* the
trust-region solve, so it scales step length linearly and any stock
``torch.optim.lr_scheduler`` drives Gnome exactly as it drives AdamW::

    opt = Gnome(model.parameters(), lr=1e-3, loss='mse')
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=200),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - 200)],
        milestones=[200],
    )

    for ...:
        opt.step(main_closure, aux_closure)
        sched.step()

On MSE the Gauss-Newton step self-anneals as the residual shrinks, so a
decay schedule is optional there in a way it is not for gradient-RMS
methods; a warmup is still usually worth having while the eigenbasis is
cold.

.. _SOAP: https://arxiv.org/abs/2409.11321
"""

from __future__ import annotations

from itertools import chain
from numbers import Integral
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import math
import random
import numpy as np

ClosureReturn = Tuple[torch.Tensor, torch.Tensor]
MergeDims = Union[bool, str, Sequence[Sequence[int]]]


def _normalize_merge_dims(
    merge_dims: MergeDims,
) -> Union[bool, Tuple[Tuple[int, ...], ...]]:
    """Normalize the dimension-merging configuration.

    ``False`` disables merging, while ``True`` and ``"greedy"`` select the
    historical size-bounded greedy grouping. An explicit partition is a
    sequence of axis groups, for example ``((0,), (1,), (2, 3))`` for
    ``[O][I][HW]``. Explicit partitions must cover axes ``0..N-1`` exactly
    once; they apply only to N-dimensional tensors, leaving parameters of
    other ranks unmerged.
    """
    if merge_dims is False or merge_dims is None:
        return False
    if merge_dims is True or merge_dims == "greedy":
        return True
    if isinstance(merge_dims, str):
        raise ValueError(
            f"Invalid merge_dims mode {merge_dims!r}; expected False, True, "
            "'greedy', or an explicit axis partition"
        )

    try:
        partition = tuple(tuple(group) for group in merge_dims)
    except TypeError as exc:
        raise ValueError(
            "merge_dims partition must be a sequence of non-empty axis groups"
        ) from exc
    if not partition or any(not group for group in partition):
        raise ValueError("merge_dims partition groups must be non-empty")

    axes = tuple(axis for group in partition for axis in group)
    if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axes):
        raise ValueError("merge_dims partition axes must be integers")
    if sorted(axes) != list(range(len(axes))):
        raise ValueError(
            "merge_dims partition must cover axes 0..N-1 exactly once; "
            f"got {partition}"
        )
    return partition


def _prefer_cpu(t: torch.Tensor) -> bool:
    """True if ``t`` lives on a backend whose linalg kernels are slower than
    a CPU round-trip. MPS qr/eigh are either unimplemented or much slower
    than running on CPU and copying back, so we route them through CPU.
    """
    return t.device.type == "mps"


def _rademacher_like(t: torch.Tensor) -> torch.Tensor:
    """Independent ±1 Rademacher samples, same shape/dtype/device as ``t``.

    Uses ``rand_like(...) < 0.5`` rather than ``empty_like(...).bernoulli_(0.5)``
    because the in-place Bernoulli kernel on MPS crashes with
    "MPSGraph encodeToCommandBuffer ... Method cache corrupted" on large
    tensors (~10M+ elements), which the CCE-Hutchinson surrogate hits with
    LM-scale vocabularies. ``rand_like`` + comparison goes through a
    different, stable MPS kernel.
    """
    return (torch.rand_like(t) < 0.5).to(t.dtype).mul_(2.0).sub_(1.0)


def _linalg_work_dtype(t: torch.Tensor) -> torch.dtype:
    """Internal precision for preconditioner factors and decompositions.

    Half and bfloat16 inputs are promoted to float32 because PyTorch's
    decomposition support and numerical accuracy are poor at those dtypes.
    Float64 is deliberately preserved: casting a double-precision run to
    float32 here computes a low-precision eigenbasis and merely relabels it as
    float64 when it is cast back, defeating the purpose of the double run.
    """
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _eigh_safe(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh`` with CPU routing for MPS and relative jitter.

    On MPS ``eigh`` is not implemented and would error; on CUDA/CPU it runs
    natively. A small relative jitter keeps rank-deficient factors
    numerically tractable.
    """
    work_dtype = _linalg_work_dtype(M)
    M_f = M.to(dtype=work_dtype)
    n = M_f.shape[0]
    scale = M_f.diag().abs().mean().clamp(min=1.0)
    jitter = 1e-6 * scale
    eye = torch.eye(n, device=M_f.device, dtype=M_f.dtype)
    M_j = M_f + jitter * eye
    try:
        if _prefer_cpu(M_j):
            evals_cpu, evecs_cpu = torch.linalg.eigh(M_j.cpu())
            evals = evals_cpu.to(M_f.device)
            evecs = evecs_cpu.to(M_f.device)
        else:
            evals, evecs = torch.linalg.eigh(M_j)
    except Exception:
        evals = torch.ones(n, device=M_f.device, dtype=M_f.dtype)
        evecs = eye
    return evals, evecs


class Gnome(Optimizer):
    """Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition.

    SOAP-style preconditioning with two key changes: the Kronecker factors
    are built from a Hutchinson estimate of the **Generalized Gauss-Newton**
    matrix (not the loss-gradient outer products SOAP uses), and the update
    inside the rotated basis is a **Newton step** — rotated gradient divided
    by the curvature EMA, no ``sqrt`` — bounded by an **l2 trust region**
    rather than tuned by an Adam-style epsilon. See the module docstring for
    the full design.

    Args:
        params: Iterable of parameters or parameter groups.
            Shared/tied parameters must appear only once.
        lr: Learning rate.
        betas: ``(beta1, beta2)`` for the gradient and curvature EMAs in
            the rotated basis.
        shampoo_beta: EMA decay for the Kronecker factors L and R. If
            negative, ``betas[1]`` is used.
        eps: Second-order damping added to the curvature denominator. Unlike
            Adam's epsilon, this is not a numerical-safety term: it controls
            how aggressively curvature is used. Larger values pull the update
            toward gradient descent; smaller values toward a pure Newton step.
        weight_decay: Decoupled weight decay coefficient.
        norm_free: Normalize each squared-surrogate contribution to unit norm
            before it enters the curvature EMA, so ``betas[1]`` controls
            recency rather than being overridden by a decaying ``||g_s||``.
            Off by default: it also removes the residual-driven self-annealing
            that lets Gnome hold a fixed lr on MSE. Turn it on for
            fast-learning objectives (classification), where the loss moves a
            long way inside one EMA window and there is no annealing to lose.
            Note it rescales ``v_hat``, so ``lr`` generally needs retuning
            when toggled.
        precondition_frequency: Steps between eigenbasis refreshes.
        use_soap_basis: If True, initialize and update the Kronecker basis
            factors from the main loss gradient instead of the surrogate.
            Uses the accumulated, optionally clipped main gradient. The
            rotated curvature EMA still uses the surrogate gradient, so this
            changes only the basis source. Default False; can be overridden
            per parameter group and is saved in optimizer checkpoints.
        basis_transport: How EMA coordinates follow basis updates. "overlap"
            transports curvature with squared overlaps and momentum with
            unsquared overlaps. "soap" uses the previous SOAP-style curvature
            permutation and momentum projection. Default "overlap". Independent
            of use_soap_basis; saved per parameter group in checkpoints.
        max_precond_dim: Modes larger than this are skipped (no Kronecker
            factor maintained along that dimension).
        trust_radius: l2 trust region on the update in the rotated basis.
            ``lambda`` is set to the smallest value (floored at ``eps``) for
            which ``||m / (v + lambda)||_2 <= trust_radius * sqrt(||p||² + S0)``.
            ``S0`` is the squared parameter norm captured on its first step
            and saved in ordinary optimizer state. An entirely zero tensor
            uses ``S0 = 1``, the expected squared norm of a length-N vector
            with Xavier variance ``1/N`` under a hypothetical square layer.
            Larger = weaker bound = longer steps; ``None`` disables the solve
            and falls back to plain ``m / (v + eps)`` damping.

            Because the bound is on the l2 norm and the ``Q`` matrices are
            orthonormal, it transfers exactly to the parameter basis. A
            uniform ``lambda`` also shrinks the step along the
            Newton-to-gradient-descent continuum without distorting its
            direction, unlike a per-coordinate clamp, and no single
            coordinate with a spuriously small ``v_i`` can set the damping
            for a whole tensor.
        max_grad_norm: If not None, clip the global L2 norm of the main
            gradient (across all parameters, à la
            ``torch.nn.utils.clip_grad_norm_``) to this value each step before
            it enters the update. Applies only to the main loss gradient, not
            the curvature surrogate. Default None; the ``trust_radius`` region
            is the usual safeguard, but a global norm clip can help on steps
            where the raw gradient spikes.
        loss: One of ``"mse"`` (mean-squared error for regression),
            ``"bce_hutchinson"`` (binary cross-entropy with logits and a
            Rademacher Hutchinson factorization of the diagonal output
            Hessian),
            ``"cce"`` (softmax cross-entropy with Fisher-sampling surrogate),
            or ``"cce_hutchinson"`` (softmax cross-entropy with a Rademacher
            Hutchinson factorization of the output Hessian; same main loss
            as ``"cce"``, different surrogate). The optimizer constructs
            both the main loss and the surrogate internally so that the
            scaling is consistent; the user's closure only needs to return
            ``(y_hat, y)``.
        merge_dims: Dimension grouping used before forming Kronecker factors.
            ``False`` (default) leaves every tensor axis separate. ``True`` or
            ``"greedy"`` uses the historical greedy grouping, combining
            contiguous axes until the next product would exceed
            ``max_precond_dim``. An explicit partition such as
            ``((0,), (1,), (2, 3))`` gives a convolution weight the grouping
            ``[O][I][HW]``. A partition covering N axes applies only to
            N-dimensional parameters; parameters of other ranks stay
            unmerged. For ``channels_last``, explicit axis numbers refer to
            the internally normalized channels-first order.
        precondition_1d: Build a Kronecker factor for 1D parameters as well.
        data_format: ``"channels_first"`` or ``"channels_last"`` for the
            ``merge_dims`` layout convention.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-4,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        max_grad_norm: Optional[float] = None,
        trust_radius: Optional[float] = 1.0,
        norm_free: bool = False,
        loss: str = "mse",
        merge_dims: MergeDims = False,
        precondition_1d: bool = False,
        data_format: str = "channels_first",
        use_soap_basis: bool = False,
        basis_transport: str = "overlap",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if trust_radius is not None and trust_radius <= 0.0:
            raise ValueError(f"Invalid trust_radius: {trust_radius}")
        if loss not in ("mse", "bce_hutchinson", "cce", "cce_hutchinson"):
            raise ValueError(
                f"Invalid loss mode: {loss!r}; "
                f"expected 'mse', 'bce_hutchinson', 'cce', "
                "or 'cce_hutchinson'."
            )
        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(f"Invalid data_format: {data_format!r}")
        merge_dims = _normalize_merge_dims(merge_dims)

        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
            trust_radius=trust_radius,
            use_soap_basis=use_soap_basis,
            basis_transport=basis_transport,
        )
        super().__init__(params, defaults)
        self._data_format = data_format
        self._loss_mode = loss
        # Global (all-parameter) main-gradient norm clip; None disables it.
        self._max_grad_norm = max_grad_norm
        # Optimizer-level (not per-group): normalize each squared-surrogate
        # contribution to unit norm before it enters the curvature EMA. See
        # _param_step for why, and why it is off by default.
        self._norm_free = norm_free
        # Global step counter: one increment per step() call, regardless of how
        # many micro-batch closures it was given. Distinct from the per-tensor
        # state["step"], which lags it by one (the first step builds the
        # eigenbasis and returns without updating).
        self._step_count = 0

    @staticmethod
    def _validate_param_group(group: dict) -> None:
        if group["basis_transport"] not in ("overlap", "soap"):
            raise ValueError("basis_transport must be 'overlap' or 'soap'")
        if not isinstance(group["use_soap_basis"], bool):
            raise ValueError("use_soap_basis must be a bool")
        # Accept old all-Gnome groups without retaining the removed selector.
        if group.pop("update_rule", "gnome") != "gnome":
            raise ValueError("Gnome no longer supports alternate update rules")

    def add_param_group(self, param_group: dict) -> None:
        group = {**self.defaults, **param_group}
        self._validate_param_group(group)
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("Optimizer parameters must have a deterministic order, not a set")
        else:
            params = list(params)
        # PyTorch rejects overlap across groups, but only warns about repeats
        # within one group. A tensor must never receive two updates here.
        tensors = [p[1] if isinstance(p, tuple) else p for p in params]
        if len({id(p) for p in tensors}) != len(tensors):
            raise ValueError("A parameter may appear only once in an optimizer group")
        group["params"] = params
        super().add_param_group(group)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_diagnostics(
        self,
        metrics: Optional[Sequence[Callable[[dict], dict]]] = None,
        *,
        params: Optional[Sequence[int]] = None,
    ) -> list[dict]:
        """Read per-parameter state without changing the optimizer.

        Returns records with step, param index, shape, numel, lr, and eps.
        By default, each record also contains lam: the stored damping from
        the last update, or None before the first update. Parameters without
        initialized state are omitted. Indices follow trainable parameter order.

        metrics is a sequence of callables receiving a context with p, state,
        group, step, and param. Each returns a dictionary of named scalar
        tensors or Python scalars. Pass [] for metadata only. Callables must
        treat the context as read-only. Tensor results are detached and
        converted to Python values for logging.

        params optionally selects parameter indices before metrics run.
        Scheduling and writing the returned records belong to the caller.
        No temporary update tensors are retained for diagnostics.
        """
        from gnome.diagnostics import lm_damping

        metrics = (lm_damping,) if metrics is None else tuple(metrics)
        if any(not callable(metric) for metric in metrics):
            raise TypeError("diagnostics metrics must be callable")
        selected = None if params is None else set(params)
        records = []
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                param_idx = idx
                idx += 1
                state = self.state.get(p)
                if not state or (selected is not None and param_idx not in selected):
                    continue
                context = dict(p=p, state=state, group=group,
                               step=self._step_count, param=param_idx)
                record = dict(step=self._step_count, param=param_idx,
                              shape=tuple(p.shape), numel=p.numel(),
                              lr=group["lr"], eps=group["eps"])
                for metric in metrics:
                    values = metric(context)
                    if not isinstance(values, dict):
                        raise TypeError("diagnostics metrics must return dictionaries")
                    for name, value in values.items():
                        if name in record:
                            raise ValueError(f"Duplicate diagnostics field: {name!r}")
                        if torch.is_tensor(value):
                            if value.ndim != 0:
                                raise ValueError(f"Diagnostics field {name!r} must be scalar")
                            value = value.detach().item()
                        elif value is not None and not isinstance(value, (bool, int, float)):
                            raise TypeError(f"Diagnostics field {name!r} must be scalar")
                        record[name] = value
                records.append(record)
        return records

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """``Optimizer.state_dict`` plus Gnome's optimizer-level counters.

        ``_step_count`` is a plain instance attribute rather than per-parameter
        state, so the base implementation would silently drop it and a resumed
        run would restart the counter from zero.
        """
        sd = super().state_dict()
        sd["gnome"] = {"step_count": self._step_count}
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        extra = state_dict.get("gnome")
        if extra is not None:
            # Hand the base implementation only the keys it knows about.
            state_dict = {k: v for k, v in state_dict.items() if k != "gnome"}
        # Legacy checkpoints use Gnome groups and a surrogate-derived basis.
        state_dict = dict(state_dict)
        state_dict["param_groups"] = [
            # Accept checkpoints written with the original misspelled option.
            {"basis_transport": self.defaults["basis_transport"],
             "use_soap_basis": group.get("use_soap_basi", False),
             **{key: value for key, value in group.items() if key != "use_soap_basi"}}
            for group in state_dict["param_groups"]
        ]
        for group in state_dict["param_groups"]:
            self._validate_param_group(group)
        super().load_state_dict(state_dict)
        if extra is not None:
            self._step_count = int(extra["step_count"])

    # ------------------------------------------------------------------
    # SOAP machinery: dim merging, Kronecker factor maintenance, projection
    # ------------------------------------------------------------------

    def _reshape_for_preconditioner(
        self,
        tensor: torch.Tensor,
        max_precond_dim: int,
        merge_dims: MergeDims,
    ) -> Tuple[torch.Tensor, dict]:
        """Group tensor axes and return metadata needed to undo the reshape."""
        spec = _normalize_merge_dims(merge_dims)
        original_shape = tuple(tensor.shape)
        identity_meta = {
            "applied": False,
            "original_shape": original_shape,
        }
        if spec is False:
            return tensor, identity_meta

        # An explicit N-axis partition is intended for N-dimensional
        # parameters (for example conv kernels) and leaves other parameter
        # ranks alone. This lets one optimizer group conv weights without
        # accidentally flattening its Linear weights.
        if spec is not True:
            partition_rank = sum(len(group) for group in spec)
            if tensor.dim() != partition_rank:
                return tensor, identity_meta

        channels_last = self._data_format == "channels_last" and tensor.dim() == 4
        if channels_last:
            tensor = tensor.permute(0, 3, 1, 2)
        canonical_shape = tuple(tensor.shape)

        if spec is True:
            merged_sizes = []
            current_size = 1
            for size in canonical_shape:
                next_size = current_size * size
                if next_size > max_precond_dim:
                    if current_size > 1:
                        merged_sizes.append(current_size)
                        current_size = size
                    else:
                        merged_sizes.append(size)
                        current_size = 1
                else:
                    current_size = next_size
            if current_size > 1 or not merged_sizes:
                merged_sizes.append(current_size)
            axis_order = tuple(range(tensor.dim()))
            merged_shape = tuple(merged_sizes)
        else:
            partition = spec
            axis_order = tuple(axis for group in partition for axis in group)
            if axis_order != tuple(range(tensor.dim())):
                tensor = tensor.permute(axis_order)
            merged_shape = tuple(
                math.prod(canonical_shape[axis] for axis in group)
                for group in partition
            )
        metadata = {
            "applied": True,
            "original_shape": original_shape,
            "canonical_shape": canonical_shape,
            "axis_order": axis_order,
            "channels_last": channels_last,
        }
        return tensor.reshape(merged_shape), metadata

    def _restore_merged_dims(
        self,
        tensor: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        """Undo :meth:`_reshape_for_preconditioner`, including axis permutation."""
        if not metadata["applied"]:
            return tensor

        canonical_shape = metadata["canonical_shape"]
        axis_order = metadata["axis_order"]
        permuted_shape = tuple(canonical_shape[axis] for axis in axis_order)
        tensor = tensor.reshape(permuted_shape)

        if axis_order != tuple(range(len(axis_order))):
            inverse_order = [0] * len(axis_order)
            for current_axis, original_axis in enumerate(axis_order):
                inverse_order[original_axis] = current_axis
            tensor = tensor.permute(inverse_order)
        if metadata["channels_last"]:
            tensor = tensor.permute(0, 2, 3, 1)
        return tensor.reshape(metadata["original_shape"])

    def _merge_dims(
        self,
        grad: torch.Tensor,
        max_precond_dim: int,
        merge_dims: MergeDims = True,
    ) -> torch.Tensor:
        """Return the grouped view; ``True`` preserves historical greedy behavior."""
        return self._reshape_for_preconditioner(
            grad, max_precond_dim, merge_dims
        )[0]

    def _init_preconditioner(
        self,
        grad: torch.Tensor,
        state: dict,
        precondition_frequency: int,
        shampoo_beta: float,
        max_precond_dim: int,
        precondition_1d: bool,
        merge_dims: MergeDims,
    ) -> None:
        state["GG"] = []
        # Keep the factors in the same precision used by eigh/QR: float64 for
        # a double run, otherwise float32 (including half/bfloat16 models).
        factor_dtype = _linalg_work_dtype(grad)
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(
                        grad.shape[0], grad.shape[0],
                        device=grad.device, dtype=factor_dtype,
                    )
                )
        else:
            ref = (
                self._merge_dims(grad, max_precond_dim, merge_dims)
                if merge_dims else grad
            )
            for sh in ref.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(
                        sh, sh, device=grad.device, dtype=factor_dtype,
                    ))

        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def _project(
        self,
        grad: torch.Tensor,
        state: dict,
        merge_dims: MergeDims,
        max_precond_dim: int,
    ) -> torch.Tensor:
        if merge_dims:
            grad, merge_metadata = self._reshape_for_preconditioner(
                grad, max_precond_dim, merge_dims
            )
        for mat in state["Q"]:
            if len(mat) > 0:
                # Q is stored in the decomposition's working precision
                # (float64 for double runs, otherwise float32). Align it to
                # grad's dtype so the projection is accepted for half models.
                grad = torch.tensordot(grad, mat.to(grad.dtype), dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            grad = self._restore_merged_dims(grad, merge_metadata)
        return grad

    def _project_back(
        self,
        grad: torch.Tensor,
        state: dict,
        merge_dims: MergeDims,
        max_precond_dim: int,
    ) -> torch.Tensor:
        if merge_dims:
            grad, merge_metadata = self._reshape_for_preconditioner(
                grad, max_precond_dim, merge_dims
            )
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat.to(grad.dtype), dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            grad = self._restore_merged_dims(grad, merge_metadata)
        return grad

    def _update_preconditioner(
        self,
        G_s: Union[torch.Tensor, Sequence[torch.Tensor]],
        state: dict,
        max_precond_dim: int,
        merge_dims: MergeDims,
        precondition_1d: bool,
        basis_transport: str = "overlap",
    ) -> None:
        """Update factors once from the sum of basis-gradient outer products.

        Multiple surrogate gradients contribute separately, without cross
        terms. With use_soap_basis, the input is just the main gradient.
        Refresh the eigenbasis Q on the usual per-step schedule.

        On basis refresh, transport the gradient EMA with basis overlaps and
        the diagonal curvature EMA with squared overlaps. Between refreshes
        both EMAs stay in the current basis. The "soap" policy instead retains
        the previous permutation-only curvature handling and momentum round trip.
        """
        if basis_transport == "soap" and state["Q"] is not None:
            state["grad_m"] = self._project_back(
                state["grad_m"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )

        gradients = [G_s] if isinstance(G_s, torch.Tensor) else G_s
        if gradients[0].dim() == 1:
            if precondition_1d and gradients[0].shape[0] <= max_precond_dim:
                gg = state["GG"][0]
                # GG uses the preconditioner's working precision (float64 for
                # double runs, otherwise float32, matching _eigh_safe). Cast
                # the contribution to that dtype before accumulating it.
                outer = None
                for grad in gradients:
                    contribution = (grad.unsqueeze(1) @ grad.unsqueeze(0)).to(gg.dtype)
                    outer = contribution if outer is None else outer.add_(contribution)
                gg.lerp_(outer, 1 - state["shampoo_beta"])
        else:
            refs = [
                self._merge_dims(grad, max_precond_dim, merge_dims)
                if merge_dims else grad
                for grad in gradients
            ]
            for idx, sh in enumerate(refs[0].shape):
                if sh <= max_precond_dim:
                    gg = state["GG"][idx]
                    outer = None
                    for ref in refs:
                        contribution = torch.tensordot(
                            ref, ref,
                            dims=[
                                [*chain(range(idx), range(idx + 1, len(ref.shape)))]
                            ] * 2,
                        ).to(gg.dtype)
                        outer = contribution if outer is None else outer.add_(contribution)
                    gg.lerp_(outer, 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self._eigvecs_descending(state["GG"])
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            self._qr_refresh_eigenbasis(state, max_precond_dim, merge_dims, basis_transport)

        if basis_transport == "soap" and state["step"] > 0:
            state["grad_m"] = self._project(
                state["grad_m"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )

    def _eigvecs_descending(self, mats):
        """Initial eigenvector matrices via full ``eigh``, sorted descending.

        Used once per parameter when the eigenbasis is built for the first
        time. Subsequent refreshes use ``_qr_refresh_eigenbasis`` (cheaper).
        """
        out = []
        for m in mats:
            if len(m) == 0:
                out.append([])
                continue
            _, Q = _eigh_safe(m.data)
            Q = torch.flip(Q, [1])
            if m.data.dtype != torch.float:
                Q = Q.to(dtype=m.data.dtype)
            out.append(Q)
        return out

    def _qr_refresh_eigenbasis(
        self,
        state: dict,
        max_precond_dim: int,
        merge_dims: MergeDims,
        basis_transport: str = "overlap",
    ) -> None:
        """Incrementally refresh the eigenbasis via one power iteration + QR.

        Equivalent to one step of subspace iteration on each Kronecker
        factor ``GG[idx]`` starting from the current ``Q[idx]``: estimate
        eigenvalues from the Rayleigh diagonal ``diag(Q^T GG Q)``, sort
        descending, do ``power = GG @ Q``, then QR-orthonormalize. Converges
        to the true eigenbasis over multiple refresh cycles, but each
        refresh is far cheaper than a full ``eigh`` — and avoids the
        CPU↔MPS bounce that ``eigh`` requires on Apple Silicon.

        Transport each EMA along every refreshed tensor axis. For an overlap
        C = Q_new.T @ Q_old, the gradient EMA uses C and the diagonal curvature
        EMA uses C.square(). This assumes historical coordinates were
        uncorrelated in the old basis, preserves nonnegativity and total
        curvature mass, and includes eigenvector permutations and sign flips.
        No separate permutation of the curvature EMA is applied in overlap
        mode. In soap mode, only reorder curvature coordinates; momentum
        projection is handled by _update_preconditioner.
        """
        precond_list = state["GG"]
        orth_list = state["Q"]
        grad_m, gnd_m = state["grad_m"], state["gnd_m"]
        grad_dtype, gnd_dtype = grad_m.dtype, gnd_m.dtype
        if merge_dims:
            grad_m, merge_metadata = self._reshape_for_preconditioner(
                grad_m, max_precond_dim, merge_dims
            )
            gnd_m, _ = self._reshape_for_preconditioner(
                gnd_m, max_precond_dim, merge_dims
            )

        new_Q = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0 or len(o) == 0:
                new_Q.append([])
                continue
            work_dtype = _linalg_work_dtype(m)
            m_f = m.data.to(dtype=work_dtype)
            o_f = o.to(dtype=work_dtype)

            est_eig = torch.diag(o_f.T @ m_f @ o_f)
            sort_idx = torch.argsort(est_eig, descending=True)
            power_iter = m_f @ o_f[:, sort_idx]

            if _prefer_cpu(power_iter):
                Q_cpu, _ = torch.linalg.qr(power_iter.cpu())
                Q = Q_cpu.to(power_iter.device)
            else:
                Q, _ = torch.linalg.qr(power_iter)

            if m.data.dtype != torch.float:
                Q = Q.to(dtype=m.data.dtype)
            new_Q.append(Q)

            if basis_transport == "soap":
                gnd_m = gnd_m.index_select(ind, sort_idx)
            else:
                # Contract the old coordinate with the overlap's old-basis axis,
                # then put the new coordinate back in the same tensor position.
                overlap = Q.to(work_dtype).T @ o_f
                grad_m = torch.tensordot(
                    overlap, grad_m.to(work_dtype), dims=([1], [ind])
                ).movedim(0, ind)
                gnd_m = torch.tensordot(
                    overlap.square(), gnd_m.to(work_dtype), dims=([1], [ind])
                ).movedim(0, ind)

        if merge_dims:
            grad_m = self._restore_merged_dims(grad_m, merge_metadata)
            gnd_m = self._restore_merged_dims(gnd_m, merge_metadata)

        state["grad_m"] = grad_m.to(grad_dtype)
        state["gnd_m"] = gnd_m.to(gnd_dtype)
        state["Q"] = new_Q

    # ------------------------------------------------------------------
    # Loss + surrogate construction
    # ------------------------------------------------------------------

    _SQRT_TWO = 2.0 ** 0.5

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the main loss with a fixed reduction so the main gradient
        scale matches the surrogate's intrinsic scale.

        MSE: ``((y_hat - y) ** 2).sum() / B`` — sum over output dim, mean
        over batch. Equivalent to ``F.mse_loss(reduction='sum') / B``.

        BCE-Hutchinson: ``binary_cross_entropy_with_logits(..., sum) / B``.

        CCE: ``F.cross_entropy(y_hat, y, reduction='mean')``.
        """
        if self._loss_mode == "mse":
            return ((y_hat - y) ** 2).sum() / y_hat.shape[0]
        if self._loss_mode == "bce_hutchinson":
            return F.binary_cross_entropy_with_logits(
                y_hat, y, reduction="sum"
            ) / y_hat.shape[0]
        # cce, cce_hutchinson — both train against the same CCE main loss;
        # only the surrogate construction differs between them.
        return F.cross_entropy(y_hat, y, reduction="mean")

    def _build_surrogate(
        self,
        y_hat: torch.Tensor,
        aux_idx: torch.Tensor,
        *,
        normalization_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Return a scalar S whose gradient w.r.t. parameters is an unbiased
        Hutchinson-style factor of the GGN.

        All branches divide their summed surrogate by ``sqrt(K)`` so that
        ``E[(dS/dθ)²] = (1/K) · sum_aux L''·J²`` is an unbiased estimator
        of ``E_data[L''·J²]``, independent of ``K``. For a partition, pass
        the full auxiliary batch size as ``normalization_size`` to retain
        the global normalization while selecting only that partition's rows.

        MSE: the intrinsic output Hessian is ``L'' = 2`` per element, so
        ``S = (sqrt(2) · eps · y_hat[aux_idx]).sum() / sqrt(K)`` with
        Rademacher signs ``eps``.

        CCE: Fisher sampling — draw ``y_tilde ~ Categorical(softmax(y_hat))``
        per aux sample and use ``F.cross_entropy(y_hat[aux_idx], y_tilde,
        reduction='sum') / sqrt(K)``. ``(softmax(z) - onehot(y_tilde))`` is
        an unbiased ``sqrt(H_L)·eps`` factor for the non-diagonal softmax
        Hessian.

        CCE-Hutchinson: Rademacher-based factorization of the softmax
        Hessian ``H = diag(p) - p p^T``. Writing ``H = A A^T`` with
        ``A = diag(sqrt(p)) (I - sqrt(p) sqrt(p)^T)`` and drawing Rademacher
        ``R`` per (sample, class), the vector
        ``v = sqrt(p) ⊙ R - (sqrt(p)·R) · p`` satisfies ``E[v v^T] = H``,
        and ``S = <logits, detach(v)>.sum() / sqrt(K)`` gives the same GGN
        estimator as the Fisher-sampling branch without drawing a discrete
        label. Lower per-step variance for the same K.

        BCE-Hutchinson: for logits ``z`` and ``p = sigmoid(z)``, the output
        Hessian is diagonal with entries ``p(1-p)``. Drawing per-element
        Rademacher ``R`` gives ``v = sqrt(p(1-p)) * R`` and
        ``S = <z, detach(v)>.sum() / sqrt(K)``.
        """
        K = aux_idx.shape[0] if normalization_size is None else normalization_size
        inv_sqrt_K = K ** -0.5

        if self._loss_mode == "cce":
            logits = y_hat[aux_idx]
            flat = logits.reshape(-1, logits.shape[-1])
            with torch.no_grad():
                probs = F.softmax(flat.detach(), dim=-1)
                y_tilde = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return F.cross_entropy(flat, y_tilde, reduction="sum") * inv_sqrt_K

        if self._loss_mode == "cce_hutchinson":
            logits = y_hat[aux_idx]
            with torch.no_grad():
                probs = F.softmax(logits.detach(), dim=-1)
                sqrt_p = probs.sqrt()
                R = _rademacher_like(sqrt_p)
                c = (sqrt_p * R).sum(dim=-1, keepdim=True)
                v = sqrt_p * R - c * probs
            return (logits * v).sum() * inv_sqrt_K

        if self._loss_mode == "bce_hutchinson":
            logits = y_hat[aux_idx]
            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                scale = (probs * (1.0 - probs)).sqrt()
                v = scale * _rademacher_like(scale)
            return (logits * v).sum() * inv_sqrt_K

        # MSE: known intrinsic L'' = 2, no double-backward needed.
        aux_y_hat = y_hat[aux_idx]
        eps_signs = _rademacher_like(aux_y_hat)
        return (self._SQRT_TWO * eps_signs * aux_y_hat).sum() * inv_sqrt_K

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    @staticmethod
    def _lm_lambda(m, v, T, eps, lam_prev=None, iters=3):
        """Smallest lambda >= eps with ``|| m / (v + lambda) ||_2 <= T``.

        The l2 trust-region solve. ``T`` is ``trust_radius * sqrt(||p||² + S0)``
        (see ``trust_radius`` for the initialization reference); the caller is
        ``_param_step``, which runs this once per parameter tensor in the
        rotated basis.

        ``phi(lambda) = || m / (v + lambda) ||_2`` is strictly decreasing, so
        the root is unique when ``phi(eps) > T``; otherwise the constraint is
        slack and the bracket floor keeps ``lambda = eps`` (an undamped
        Newton step). Newton is run on Hebden's reciprocal form
        ``1/T - 1/phi``, which is nearly linear in lambda and often converges
        in 2-3 iterations, warm-started from the previous step's lambda inside
        an analytic bracket derived from ``min(v)`` and ``max(v)``. All ops
        stay on device -- no ``.item()``, no convergence check, no sync.

        Aggregating over all P coordinates is what keeps ``lambda(T)``
        smooth: by the implicit function theorem there are no argmax kinks,
        and no single coordinate with a spuriously small ``v_i`` can set the
        damping for a whole tensor. (An l-infinity bound would give a closed
        form instead of this solve, but has exactly that outlier
        sensitivity; ``l_p`` for large p is the middle ground, and is this
        solver with the exponent changed.)

        Compute ``slope = -phi'/phi`` from normalized update components.
        Forming ``sum(m² / (v + lambda)³)`` first can overflow in FP32 for
        small damping, making the derivative infinite and freezing Newton
        at the bracket floor. The normalized correction also avoids forming
        a ``phi²`` numerator.
        """
        gn = m.norm()
        ratio = gn / T

        # Analytic bracket from vmin <= v_i <= vmax.
        lo = (ratio - v.max()).clamp_(min=eps)
        hi = (ratio - v.min()).clamp_(min=eps)
        hi = torch.maximum(hi, lo)  # guard against v.min ~ v.max

        lam = lo if lam_prev is None else torch.as_tensor(
            lam_prev, device=lo.device, dtype=lo.dtype
        ).clamp(lo, hi)

        for _ in range(iters):
            d = v + lam
            u = m / d
            phi = u.norm().clamp_min(1e-30)
            slope = ((u / phi).square() / d).sum().clamp_min(1e-30)
            lam = (lam + (phi / T - 1.0) / slope).clamp(lo, hi)

        return lam

    def step(  # type: ignore[override]
        self,
        main_closure: Union[
            Callable[[], ClosureReturn],
            Sequence[Callable[[], ClosureReturn]],
        ],
        aux_closure: Optional[Callable[[], ClosureReturn]] = None,
        *,
        surrogate_partition_sizes: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Perform a single optimization step.

        Args:
            main_closure: Forward pass on the main batch. Returns
                ``(y_hat, y)`` — the model's output (batch dim 0) and the
                corresponding targets. Used to compute the main loss
                gradient.

                To accumulate gradients across several micro-batches before
                a single update (e.g. when the full main batch would not fit
                in memory), pass a **list/tuple of closures** instead. Each
                is run in its own independent forward+backward, and their
                gradients are combined into one logical step as a
                **size-weighted average** — weighted by each micro-batch's
                sample count (``y_hat.shape[0]``) so the result matches a
                single forward over the concatenation of all micro-batches,
                regardless of how they were chunked (uneven final chunk is
                fine). No EMA rescaling is needed: the eigenbasis ``Q`` and
                all EMAs update exactly once per logical step. The aux
                surrogate still runs once per step (see ``aux_closure``).
            aux_closure: Forward pass on the auxiliary batch. Returns
                ``(y_hat, y)`` — same shape contract as ``main_closure``,
                but typically on a smaller, disjoint slice of samples.
                Used to compute the Hutchinson surrogate gradient. Runs once
                per logical step even when the main batch is accumulated.
            surrogate_partition_sizes: Optional positive integer sizes of
                contiguous partitions along the auxiliary output's batch axis.
                Must sum to ``y_hat_aux.shape[0]``. For a stacked PINN residual,
                use the auxiliary blocks' element counts in stacking order,
                e.g. ``[n_pde, n_ic, n_bc]``. Partition gradients are computed
                together using ``autograd.grad(is_grads_batched=True)``,
                retaining the full batch's ``1/sqrt(K)`` scaling.
                Factor outer products and rotated squares are summed, then
                each EMA is updated once. None uses one surrogate as before.
                Ignored when there are no trainable Gnome parameters.

        Main and auxiliary closures run independent forward passes. Multiple
        surrogate partitions share one batched backward, which releases the
        auxiliary graph. Partition gradients are held until the parameter
        updates; more partitions cost more backward work and memory.

        Returns:
            The scalar main loss (detached) for logging. When multiple
            micro-batch closures are passed, this is their size-weighted
            mean, i.e. the loss over the concatenated main batch.
        """
        if main_closure is None:
            raise RuntimeError(
                "Gnome.step requires a main_closure."
            )

        # Normalize to a list of micro-batch closures. A single callable is
        # wrapped in a 1-element list; its code path below is arithmetically
        # identical to the non-accumulated case (no rescaling is applied when
        # there is exactly one closure), so existing callers are unaffected.
        if isinstance(main_closure, (list, tuple)):
            main_closures = list(main_closure)
            if not main_closures:
                raise RuntimeError("main_closure list must be non-empty.")
        else:
            main_closures = [main_closure]
        single = len(main_closures) == 1

        params, groups_for_params = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
                    groups_for_params.append(group)
        if params and aux_closure is None:
            raise RuntimeError("Gnome.step requires aux_closure.")
        self._step_count += 1

        # MAIN: loop over micro-batch closures, accumulating a size-weighted
        # average gradient. Each closure runs an independent forward+backward
        # (no retain_graph). ``g_accum`` entries stay None until a micro-batch
        # first contributes a grad for that param, so the None-skip semantics
        # in the apply loop below match the single-closure path exactly.
        g_accum: list = [None] * len(params)
        total_n = 0
        weighted_loss: Optional[torch.Tensor] = None
        for mc in main_closures:
            with torch.enable_grad():
                main_result = mc()
            if not (isinstance(main_result, tuple) and len(main_result) == 2):
                raise RuntimeError(
                    "main_closure must return (y_hat, y); got "
                    f"{type(main_result).__name__}"
                )
            y_hat_main, y_main = main_result
            if not y_hat_main.requires_grad:
                raise RuntimeError(
                    "y_hat from main_closure must have requires_grad=True."
                )
            n = y_hat_main.shape[0]
            loss = self._compute_loss(y_hat_main, y_main)
            if not params:
                return loss.detach()
            g_main = torch.autograd.grad(loss, params, allow_unused=True)
            # this micro-batch's graph is now freed
            del y_hat_main, main_result

            total_n += n
            if not single:
                wl = loss.detach() * n
                weighted_loss = (
                    wl if weighted_loss is None else weighted_loss + wl
                )
            for i, g in enumerate(g_main):
                if g is None:
                    continue
                # Weight by micro-batch size; the /total_n normalization is
                # applied once after the loop. For a single closure we skip
                # the scaling entirely to keep the update bit-identical to the
                # non-accumulated path.
                # For a single closure contrib is g itself (one iteration, no
                # accumulation), so ``g_accum`` ends up bit-identical to the
                # non-accumulated path. For multiple closures contrib is a
                # fresh ``g * n`` tensor, safe to accumulate in place.
                contrib = g if single else g * n
                if g_accum[i] is None:
                    g_accum[i] = contrib
                else:
                    g_accum[i].add_(contrib)

        if single:
            mean_loss = loss.detach()  # the one closure's loss, unchanged
        else:
            inv = 1.0 / total_n
            for i in range(len(g_accum)):
                if g_accum[i] is not None:
                    g_accum[i].mul_(inv)
            mean_loss = weighted_loss * inv

        # Optional global gradient-norm clip on the (accumulated) main gradient,
        # matching torch.nn.utils.clip_grad_norm_: rescale every parameter's
        # gradient so their combined L2 norm is at most max_grad_norm. The
        # curvature surrogate (g_aux) is intentionally left untouched.
        if self._max_grad_norm is not None:
            grads = [g for g in g_accum if g is not None]
            if grads:
                total_norm = torch.sqrt(
                    sum(g.detach().pow(2).sum() for g in grads)
                )
                clip_coef = self._max_grad_norm / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for g in grads:
                        g.mul_(clip_coef)

        # AUX: independent graph for the curvature surrogate gradients.
        g_aux = [None] * len(params)
        if params:
            with torch.enable_grad():
                aux_result = aux_closure()
            if not (isinstance(aux_result, tuple) and len(aux_result) == 2):
                raise RuntimeError(
                    "aux_closure must return (y_hat, y); got "
                    f"{type(aux_result).__name__}"
                )
            y_hat_aux, _y_aux = aux_result
            if not y_hat_aux.requires_grad:
                raise RuntimeError(
                    "y_hat from aux_closure must have requires_grad=True."
                )
            K = y_hat_aux.shape[0]
            sizes = [K] if surrogate_partition_sizes is None else list(surrogate_partition_sizes)
            if not sizes or any(
                isinstance(size, bool) or not isinstance(size, Integral) or size <= 0
                for size in sizes
            ):
                raise ValueError("surrogate_partition_sizes must contain positive integers")
            if sum(sizes) != K:
                raise ValueError(
                    f"surrogate_partition_sizes must sum to auxiliary batch size {K}; "
                    f"got {sum(sizes)}"
                )
            surrogates = []
            start = 0
            for size in sizes:
                aux_idx = torch.arange(start, start + size, device=y_hat_aux.device)
                surrogates.append(
                    self._build_surrogate(y_hat_aux, aux_idx, normalization_size=K)
                )
                start += size
            if len(surrogates) == 1:
                # Preserve the ordinary single-surrogate path without vmap overhead.
                surrogate_grads = torch.autograd.grad(
                    surrogates[0], params, allow_unused=True,
                )
                for i, grad in enumerate(surrogate_grads):
                    g_aux[i] = [] if grad is None else [grad.detach()]
            else:
                # Identity seeds request one VJP per scalar surrogate. The vmap
                # backend batches those VJPs in one autograd call, returning
                # [partitions, *parameter.shape] without mixing the gradients.
                S = torch.stack(surrogates)
                surrogate_grads = torch.autograd.grad(
                    S, params,
                    grad_outputs=torch.eye(len(surrogates), device=S.device, dtype=S.dtype),
                    is_grads_batched=True, allow_unused=True,
                )
                for i, grad in enumerate(surrogate_grads):
                    g_aux[i] = [] if grad is None else grad.detach().unbind(0)
                del S
            del y_hat_aux, aux_result, surrogates

        with torch.no_grad():
            for p, g, G_s, group in zip(params, g_accum, g_aux, groups_for_params):
                if g is None:
                    continue
                if not G_s:
                    G_s = torch.zeros_like(g)
                self._param_step(p, g.detach(), G_s, group)

        return mean_loss

    def _param_step(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        G_s: Union[torch.Tensor, Sequence[torch.Tensor]],
        group: dict,
    ) -> None:
        state = self.state[p]
        surrogate_grads = [G_s] if isinstance(G_s, torch.Tensor) else G_s
        basis_grad = g if group["use_soap_basis"] else surrogate_grads

        if "initial_sq_norm" not in state:
            norm_sq = p.detach().square().sum()
            if norm_sq == 0:
                norm_sq = norm_sq.new_tensor(1.0)
            state["initial_sq_norm"] = norm_sq

        if "step" not in state:
            state["step"] = 0
        if "grad_m" not in state:
            state["grad_m"] = torch.zeros_like(g)
            state["gnd_m"] = torch.zeros_like(g)

        if "Q" not in state:
            shampoo_beta = (
                group["shampoo_beta"]
                if group["shampoo_beta"] >= 0
                else group["betas"][1]
            )
            self._init_preconditioner(
                g, state,
                precondition_frequency=group["precondition_frequency"],
                shampoo_beta=shampoo_beta,
                max_precond_dim=group["max_precond_dim"],
                precondition_1d=group["precondition_1d"],
                merge_dims=group["merge_dims"],
            )
            self._update_preconditioner(
                basis_grad, state,
                max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
                precondition_1d=group["precondition_1d"],
                basis_transport=group["basis_transport"],
            )
            return  # first step: build the basis, skip the update.

        state["step"] += 1
        beta1, beta2 = group["betas"]
        # Read fresh each step, so an external torch.optim.lr_scheduler that
        # mutates group["lr"] takes effect immediately. Gnome applies no
        # schedule of its own.
        lr = group["lr"]
        grad_m, gnd_m = state["grad_m"], state["gnd_m"]

        # Project loss gradient and surrogate into the selected eigenbasis.
        g_rot = self._project(
            g, state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )
        # Rotate and square each partition separately: summing gradients
        # before squaring would reintroduce the cross-partition noise.
        gs_sq = None
        for surrogate_grad in surrogate_grads:
            Gs_rot = self._project(
                surrogate_grad, state,
                merge_dims=group["merge_dims"],
                max_precond_dim=group["max_precond_dim"],
            )
            contribution = Gs_rot.square()
            gs_sq = contribution if gs_sq is None else gs_sq.add_(contribution)
        # Normalize the aggregate once, preserving the partitions' relative weights.
        if self._norm_free:
            gs_sq = gs_sq / gs_sq.norm().clamp_min(1e-30)
        gnd_m.mul_(beta2).add_(gs_sq, alpha=(1.0 - beta2))

        # Gradient EMA in the rotated basis.
        grad_m.mul_(beta1).add_(g_rot, alpha=(1.0 - beta1))

        # Bias correction.
        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]
        grad_hat = grad_m / bc1
        gnd_hat = gnd_m / bc2

        eps = group["eps"]
        trust_radius = group["trust_radius"]
        if trust_radius is None:
            lam = eps
        else:
            T = trust_radius * math.sqrt(
                10 + p.square().sum() # 10 + p.square().sum()  p.numel()
            )
            lam = self._lm_lambda(grad_hat, gnd_hat, T, eps,
                                  lam_prev=state.get("lm_lambda"))
        state["lm_lambda"] = lam
        # Newton step in the rotated basis, damped by lam.

        update_unorm = grad_hat / gnd_hat.add(lam)
        damping = (
            update_unorm
            .abs()
            .clamp(min=0.0, max=1.0)
        )

        update_rot = torch.sign(grad_hat) * damping
        # Zero-initialized projections can block gradients upstream. Keep a
        # zero update at zero instead of dividing by zero during normalization.
        # Rotate back into the parameter basis. Orthonormal Q means the l2
        # norm — and hence the trust-region bound — carries over exactly, so
        # subsequent coordinate clipping can only reduce that norm.
        update = self._project_back(
            update_rot, state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )

        p.add_(update, alpha=-lr)
        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=-lr * group["weight_decay"])

        # Refresh Kronecker factors with the selected gradient; eigenbasis is
        # recomputed on schedule.
        self._update_preconditioner(
            basis_grad, state,
            max_precond_dim=group["max_precond_dim"],
            merge_dims=group["merge_dims"],
            precondition_1d=group["precondition_1d"],
            basis_transport=group["basis_transport"],
        )
