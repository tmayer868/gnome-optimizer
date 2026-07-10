"""Gnome: Gauss-Newton Optimizer via Matrix Eigen-decomposition.

Gnome is inspired by SOAP_ but makes two changes that turn the SOAP machinery
into a second-order method:

1. **GGN eigenbasis.** SOAP builds its Kronecker factors from the loss
   gradient (an empirical-Fisher proxy). Gnome builds them from a
   Hutchinson estimate of the Generalized Gauss-Newton matrix — the true
   GGN, not the empirical Fisher. The eigenbases ``Q_L`` and ``Q_R`` are
   therefore aligned with curvature directions of the loss surface rather
   than with the directions of past gradients.

2. **Newton step in the rotated basis, clip as trust region.** SOAP runs an
   Adam update inside the rotated basis (gradient divided by ``sqrt`` of a
   second-moment EMA). Gnome runs a Newton step instead — the rotated
   gradient divided by the *un-square-rooted* curvature EMA — which is the
   diagonal Gauss-Newton update inside the eigenbasis. A pure Newton step
   has no built-in step-size control, so Gnome bounds the per-coordinate
   update with ``clip`` (default ``1.0``) in both the rotated and
   rotated-back bases. The clip acts as a trust region, not a band-aid:
   when curvature is well-conditioned it never binds, but it prevents
   blow-ups from small denominators while the eigenbasis is still warming
   up.

Gnome supports three explicit loss types — chosen via the ``loss`` constructor
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

Each closure runs an independent forward, and each backward releases its
own graph immediately — no ``retain_graph``, no held activations between
passes. The optimizer never writes into ``p.grad``; gradients are
obtained via ``torch.autograd.grad`` and applied directly.

.. _SOAP: https://arxiv.org/abs/2409.11321
"""

from __future__ import annotations

from itertools import chain
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

ClosureReturn = Tuple[torch.Tensor, torch.Tensor]


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


def _eigh_safe(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch.linalg.eigh`` with CPU routing for MPS and relative jitter.

    On MPS ``eigh`` is not implemented and would error; on CUDA/CPU it runs
    natively. A small relative jitter keeps rank-deficient factors
    numerically tractable.
    """
    M_f = M.float()
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
    by the curvature EMA, no ``sqrt`` — bounded by ``clip`` as a
    trust-region safeguard rather than tuned by an Adam-style epsilon. See
    the module docstring for the full design.

    Args:
        params: Iterable of parameters or parameter groups.
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
        precondition_frequency: Steps between eigenbasis refreshes.
        max_precond_dim: Modes larger than this are skipped (no Kronecker
            factor maintained along that dimension).
        clip: If not None, clamp the per-coordinate update magnitude to
            ``[-clip, +clip]`` in both the rotated and the rotated-back
            bases. Acts as a trust-region safeguard: at sensible learning
            rates it does nothing, but it prevents blow-ups from pathological
            denominators (e.g. early steps with a noisy eigenbasis estimate).
            Pass ``None`` to disable.
        max_grad_norm: If not None, clip the global L2 norm of the main
            gradient (across all parameters, à la
            ``torch.nn.utils.clip_grad_norm_``) to this value each step before
            it enters the update. Applies only to the main loss gradient, not
            the curvature surrogate. Default None; the per-coordinate ``clip``
            trust region is the usual safeguard, but a global norm clip can
            help on steps where the raw gradient spikes.
        warmup: Number of steps over which the learning rate is linearly
            ramped from zero to ``lr``. At step ``k`` (1-indexed) the
            effective learning rate is ``lr * min(k / warmup, 1.0)``. Pass
            ``0`` to disable internal warmup (e.g. when using an external
            scheduler).
        loss: One of ``"mse"`` (mean-squared error for regression),
            ``"cce"`` (softmax cross-entropy with Fisher-sampling surrogate),
            or ``"cce_hutchinson"`` (softmax cross-entropy with a Rademacher
            Hutchinson factorization of the output Hessian; same main loss
            as ``"cce"``, different surrogate). The optimizer constructs
            both the main loss and the surrogate internally so that the
            scaling is consistent; the user's closure only needs to return
            ``(y_hat, y)``.
        merge_dims: Whether to merge conv-layer dimensions before forming
            Kronecker factors.
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
        clip: Optional[float] = 1.0,
        max_grad_norm: Optional[float] = None,
        warmup: int = 200,
        loss: str = "mse",
        merge_dims: bool = False,
        precondition_1d: bool = False,
        data_format: str = "channels_first",
    ):
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
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError(f"Invalid max_grad_norm: {max_grad_norm}")
        if warmup < 0:
            raise ValueError(f"Invalid warmup: {warmup}")
        if loss not in ("mse", "cce", "cce_hutchinson"):
            raise ValueError(
                f"Invalid loss mode: {loss!r}; "
                f"expected 'mse', 'cce', or 'cce_hutchinson'."
            )
        if data_format not in ("channels_first", "channels_last"):
            raise ValueError(f"Invalid data_format: {data_format!r}")

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
            clip=clip,
            warmup=warmup,
        )
        super().__init__(params, defaults)
        self._data_format = data_format
        self._loss_mode = loss
        # Global (all-parameter) main-gradient norm clip; None disables it.
        self._max_grad_norm = max_grad_norm
        self._step_count = 0

    # ------------------------------------------------------------------
    # SOAP machinery: dim merging, Kronecker factor maintenance, projection
    # ------------------------------------------------------------------

    def _merge_dims(self, grad: torch.Tensor, max_precond_dim: int) -> torch.Tensor:
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []
        curr = 1
        for sh in shape:
            nxt = curr * sh
            if nxt > max_precond_dim:
                if curr > 1:
                    new_shape.append(curr)
                    curr = sh
                else:
                    new_shape.append(sh)
                    curr = 1
            else:
                curr = nxt
        if curr > 1 or not new_shape:
            new_shape.append(curr)
        return grad.reshape(new_shape)

    def _init_preconditioner(
        self,
        grad: torch.Tensor,
        state: dict,
        precondition_frequency: int,
        shampoo_beta: float,
        max_precond_dim: int,
        precondition_1d: bool,
        merge_dims: bool,
    ) -> None:
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device)
                )
        else:
            ref = self._merge_dims(grad, max_precond_dim) if merge_dims else grad
            for sh in ref.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def _project(
        self,
        grad: torch.Tensor,
        state: dict,
        merge_dims: bool,
        max_precond_dim: int,
    ) -> torch.Tensor:
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                # Q is stored in the eigen-decomposition's float32 working
                # precision; align it to grad's dtype (a no-op for float32
                # params, an upcast for float64) so the projection matches.
                grad = torch.tensordot(grad, mat.to(grad.dtype), dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _project_back(
        self,
        grad: torch.Tensor,
        state: dict,
        merge_dims: bool,
        max_precond_dim: int,
    ) -> torch.Tensor:
        original_shape = grad.shape
        permuted_shape = None
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self._merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat.to(grad.dtype), dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)
        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def _update_preconditioner(
        self,
        G_s: torch.Tensor,
        state: dict,
        max_precond_dim: int,
        merge_dims: bool,
        precondition_1d: bool,
    ) -> None:
        """Update the Kronecker factors from the surrogate gradient G_s and,
        on schedule, refresh the eigenbasis Q.

        ``grad_m`` is a first-moment (vector) quantity so it is translated
        out of the old basis and back into the new basis when Q is
        refreshed. ``gnd_m`` is a diagonal-variance quantity in the rotated
        basis; rotating it as a vector can produce negative entries, so it
        is left in place (same approximation as SOAP).
        """
        if state["Q"] is not None:
            state["grad_m"] = self._project_back(
                state["grad_m"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )

        if G_s.dim() == 1:
            if precondition_1d and G_s.shape[0] <= max_precond_dim:
                gg = state["GG"][0]
                # GG is kept in float32 (the preconditioner's internal working
                # precision, matching _eigh_safe); cast the contribution to its
                # dtype so higher-precision (e.g. float64) grads accumulate. For
                # float32 grads this .to() is a no-op and behaviour is unchanged.
                gg.lerp_(
                    (G_s.unsqueeze(1) @ G_s.unsqueeze(0)).to(gg.dtype),
                    1 - state["shampoo_beta"],
                )
        else:
            ref = self._merge_dims(G_s, max_precond_dim) if merge_dims else G_s
            for idx, sh in enumerate(ref.shape):
                if sh <= max_precond_dim:
                    outer = torch.tensordot(
                        ref, ref,
                        dims=[
                            [*chain(range(idx), range(idx + 1, len(ref.shape)))]
                        ] * 2,
                    )
                    gg = state["GG"][idx]
                    gg.lerp_(outer.to(gg.dtype), 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self._eigvecs_descending(state["GG"])
        elif state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            self._qr_refresh_eigenbasis(state, max_precond_dim, merge_dims)

        if state["step"] > 0:
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
        merge_dims: bool,
    ) -> None:
        """Incrementally refresh the eigenbasis via one power iteration + QR.

        Equivalent to one step of subspace iteration on each Kronecker
        factor ``GG[idx]`` starting from the current ``Q[idx]``: estimate
        eigenvalues from the Rayleigh diagonal ``diag(Q^T GG Q)``, sort
        descending, do ``power = GG @ Q``, then QR-orthonormalize. Converges
        to the true eigenbasis over multiple refresh cycles, but each
        refresh is far cheaper than a full ``eigh`` — and avoids the
        CPU↔MPS bounce that ``eigh`` requires on Apple Silicon.

        ``gnd_m`` is a per-coordinate variance in the rotated basis; when
        we reorder Q's columns by ``sort_idx`` we must permute ``gnd_m``
        along the corresponding tensor dimension so it stays aligned with
        the new eigenvalue ordering. (SOAP applies the same permutation to
        its ``exp_avg_sq``.)
        """
        precond_list = state["GG"]
        orth_list = state["Q"]
        gnd_m = state["gnd_m"]

        # Apply merge_dims to gnd_m so the per-factor dimensions line up.
        orig_shape = gnd_m.shape
        permuted_shape = None
        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                permuted_shape = gnd_m.permute(0, 3, 1, 2).shape
            gnd_m_view = self._merge_dims(gnd_m, max_precond_dim)
        else:
            gnd_m_view = gnd_m

        new_Q = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0 or len(o) == 0:
                new_Q.append([])
                continue
            m_f = m.data.float()
            o_f = o.float()

            est_eig = torch.diag(o_f.T @ m_f @ o_f)
            sort_idx = torch.argsort(est_eig, descending=True)
            gnd_m_view = gnd_m_view.index_select(ind, sort_idx)
            o_f = o_f[:, sort_idx]
            power_iter = m_f @ o_f

            if _prefer_cpu(power_iter):
                Q_cpu, _ = torch.linalg.qr(power_iter.cpu())
                Q = Q_cpu.to(power_iter.device)
            else:
                Q, _ = torch.linalg.qr(power_iter)

            if m.data.dtype != torch.float:
                Q = Q.to(dtype=m.data.dtype)
            new_Q.append(Q)

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                gnd_m_view = gnd_m_view.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                gnd_m_view = gnd_m_view.reshape(orig_shape)

        state["gnd_m"] = gnd_m_view
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

        CCE: ``F.cross_entropy(y_hat, y, reduction='mean')``.
        """
        if self._loss_mode == "mse":
            return ((y_hat - y) ** 2).sum() / y_hat.shape[0]
        # cce, cce_hutchinson — both train against the same CCE main loss;
        # only the surrogate construction differs between them.
        return F.cross_entropy(y_hat, y, reduction="mean")

    def _build_surrogate(
        self,
        y_hat: torch.Tensor,
        aux_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return a scalar S whose gradient w.r.t. parameters is an unbiased
        Hutchinson-style factor of the GGN.

        All branches divide their summed surrogate by ``sqrt(K)`` so that
        ``E[(dS/dθ)²] = (1/K) · sum_aux L''·J²`` is an unbiased estimator
        of ``E_data[L''·J²]``, independent of ``K``.

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
        """
        K = aux_idx.shape[0]
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

        # MSE: known intrinsic L'' = 2, no double-backward needed.
        aux_y_hat = y_hat[aux_idx]
        eps_signs = _rademacher_like(aux_y_hat)
        return (self._SQRT_TWO * eps_signs * aux_y_hat).sum() * inv_sqrt_K

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(  # type: ignore[override]
        self,
        main_closure: Union[
            Callable[[], ClosureReturn],
            Sequence[Callable[[], ClosureReturn]],
        ],
        aux_closure: Callable[[], ClosureReturn],
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

        The closures run independent forward passes so each backward
        releases its computation graph immediately — there is no
        ``retain_graph`` and no holding of activations across passes.

        Returns:
            The scalar main loss (detached) for logging. When multiple
            micro-batch closures are passed, this is their size-weighted
            mean, i.e. the loss over the concatenated main batch.
        """
        if main_closure is None or aux_closure is None:
            raise RuntimeError(
                "Gnome.step requires (main_closure, aux_closure)."
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

        self._step_count += 1

        params, groups_for_params = [], []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)
                    groups_for_params.append(group)

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

        # AUX: forward + backward on the aux batch. Independent graph.
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
        aux_idx = torch.arange(y_hat_aux.shape[0], device=y_hat_aux.device)
        S = self._build_surrogate(y_hat_aux, aux_idx)
        g_aux = torch.autograd.grad(S, params, allow_unused=True)
        # aux graph is now freed
        del y_hat_aux, aux_result

        with torch.no_grad():
            for p, g, G_s, group in zip(params, g_accum, g_aux, groups_for_params):
                if g is None:
                    continue
                if G_s is None:
                    G_s = torch.zeros_like(g)
                self._param_step(p, g.detach(), G_s.detach(), group)

        return mean_loss

    def _param_step(
        self,
        p: torch.Tensor,
        g: torch.Tensor,
        G_s: torch.Tensor,
        group: dict,
    ) -> None:
        state = self.state[p]

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
                G_s, state,
                precondition_frequency=group["precondition_frequency"],
                shampoo_beta=shampoo_beta,
                max_precond_dim=group["max_precond_dim"],
                precondition_1d=group["precondition_1d"],
                merge_dims=group["merge_dims"],
            )
            self._update_preconditioner(
                G_s, state,
                max_precond_dim=group["max_precond_dim"],
                merge_dims=group["merge_dims"],
                precondition_1d=group["precondition_1d"],
            )
            return  # first step: build the basis, skip the update.

        state["step"] += 1
        beta1, beta2 = group["betas"]
        clip = group["clip"]
        warmup = group["warmup"]
        lr = group["lr"]
        if warmup > 0:
            lr = lr * max(min((self._step_count-1) / warmup, 1.0), .01)
        grad_m, gnd_m = state["grad_m"], state["gnd_m"]

        # Project loss gradient and surrogate into the GGN-derived eigenbasis.
        g_rot = self._project(
            g, state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )
        Gs_rot = self._project(
            G_s, state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )

        # GND estimate in the rotated basis: squared surrogate, EMA-aggregated.
        gnd_m.mul_(beta2).add_(Gs_rot.square(), alpha=(1.0 - beta2))

        # Gradient EMA in the rotated basis.
        grad_m.mul_(beta1).add_(g_rot, alpha=(1.0 - beta1))

        # Bias correction.
        bc1 = 1.0 - beta1 ** state["step"]
        bc2 = 1.0 - beta2 ** state["step"]
        grad_hat = grad_m / bc1
        gnd_hat = gnd_m / bc2

        # Adam/GNOME-style step in the rotated basis.
        update_rot = grad_hat / gnd_hat.add(group["eps"])
        if clip is not None:
            update_rot = update_rot.clamp(min=-clip, max=clip)

        # Rotate back into the parameter basis.
        update = self._project_back(
            update_rot, state,
            merge_dims=group["merge_dims"],
            max_precond_dim=group["max_precond_dim"],
        )
        if clip is not None:
            update = update.clamp(min=-clip, max=clip)

        p.add_(update, alpha=-lr)
        if group["weight_decay"] > 0.0:
            p.add_(p, alpha=-lr * group["weight_decay"])

        # Refresh Kronecker factors with the new surrogate; eigenbasis is
        # recomputed on schedule.
        self._update_preconditioner(
            G_s, state,
            max_precond_dim=group["max_precond_dim"],
            merge_dims=group["merge_dims"],
            precondition_1d=group["precondition_1d"],
        )
