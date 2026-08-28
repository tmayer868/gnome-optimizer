"""Standalone SOAP optimizer (https://arxiv.org/abs/2409.11321).

Original implementation by Nikhil Vyas et al.
Parts of the code are modifications of PyTorch's AdamW optimizer.
Parts of the code are modifications of code from GaLore:
  https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py

MPS note: torch.linalg.eigh and torch.linalg.qr are not implemented on MPS.
Both are run on CPU and moved back to the original device.
"""

import math
from itertools import chain
from typing import Tuple

import torch
import torch.optim as optim

from gnome.optimizer import MergeDims, _normalize_merge_dims


def _to_cpu_and_back(tensor, fn):
    """Run fn on tensor on CPU, then move result back to original device."""
    device = tensor.device
    result = fn(tensor.cpu())
    return result.to(device)


class SOAP(optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params: Iterable of parameters to optimize or dictionaries defining
            parameter groups.
        lr (float): Learning rate (default: 3e-3).
        betas (Tuple[float, float]): Adam beta parameters (default: (0.95, 0.95)).
        shampoo_beta (float): If >= 0, use this beta for the preconditioner EMA
            instead of betas[1] (default: -1).
        eps (float): Adam epsilon for numerical stability (default: 1e-8).
        weight_decay (float): Weight decay coefficient (default: 0.01).
        precondition_frequency (int): How often to update the preconditioner
            (default: 10).
        max_precond_dim (int): Maximum preconditioner dimension (default: 10000).
        merge_dims: Dimension grouping used before forming Kronecker factors.
            ``False`` keeps one factor per tensor axis. ``True`` or
            ``"greedy"`` uses SOAP's historical size-bounded greedy merging.
            An explicit partition such as ``((0,), (1,), (2, 3))`` produces
            ``[O][I][HW]`` factors for a convolution weight. A partition is
            applied only to tensors with the same number of axes.
        precondition_1d (bool): Whether to precondition 1D gradients
            (default: False).
        normalize_grads (bool): Whether to normalize gradients per layer
            (default: False).
        data_format (str): Data format for conv layers — "channels_first" or
            "channels_last" (default: "channels_first").
        correct_bias (bool): Whether to use bias correction in Adam
            (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float = -1,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        merge_dims: MergeDims = False,
        precondition_1d: bool = False,
        normalize_grads: bool = False,
        data_format: str = "channels_first",
        correct_bias: bool = True,
    ):
        merge_dims = _normalize_merge_dims(merge_dims)
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "merge_dims": merge_dims,
            "precondition_1d": precondition_1d,
            "normalize_grads": normalize_grads,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self._data_format = data_format

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

        # Explicit partitions target tensors of one particular rank. This lets
        # a convolution partition coexist with Linear and normalization
        # parameters in the same optimizer without reshaping those tensors.
        if spec is not True:
            partition_rank = sum(len(group) for group in spec)
            if tensor.dim() != partition_rank:
                return tensor, identity_meta

        assert self._data_format in ["channels_first", "channels_last"]
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
            axis_order = tuple(axis for group in spec for axis in group)
            if axis_order != tuple(range(tensor.dim())):
                tensor = tensor.permute(axis_order)
            merged_shape = tuple(
                math.prod(canonical_shape[axis] for axis in group)
                for group in spec
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
        """Undo grouping, including explicit axis and data-format permutations."""
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

    def merge_dims(
        self,
        grad: torch.Tensor,
        max_precond_dim: int,
        merge_dims: MergeDims = True,
    ) -> torch.Tensor:
        """Return the grouped view; ``True`` preserves historical behavior."""
        return self._reshape_for_preconditioner(
            grad, max_precond_dim, merge_dims
        )[0]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                if "Q" not in state:
                    self.init_preconditioner(
                        grad,
                        state,
                        precondition_frequency=group["precondition_frequency"],
                        precondition_1d=group["precondition_1d"],
                        shampoo_beta=(
                            group["shampoo_beta"]
                            if group["shampoo_beta"] >= 0
                            else group["betas"][1]
                        ),
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                    )
                    self.update_preconditioner(
                        grad, state,
                        max_precond_dim=group["max_precond_dim"],
                        merge_dims=group["merge_dims"],
                        precondition_1d=group["precondition_1d"],
                    )
                    continue  # skip first step so current grads aren't used in projection

                grad_projected = self.project(
                    grad, state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                norm_grad = self.project_back(
                    exp_avg / denom, state,
                    merge_dims=group["merge_dims"],
                    max_precond_dim=group["max_precond_dim"],
                )

                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad ** 2) ** 0.5)

                p.add_(norm_grad, alpha=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                self.update_preconditioner(
                    grad, state,
                    max_precond_dim=group["max_precond_dim"],
                    merge_dims=group["merge_dims"],
                    precondition_1d=group["precondition_1d"],
                )

        return loss

    def init_preconditioner(self, grad, state, precondition_frequency=10,
                            shampoo_beta=0.95, max_precond_dim=10000,
                            precondition_1d=False, merge_dims: MergeDims = False):
        """Initializes the preconditioner matrices (L and R in the paper)."""
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim, merge_dims)
            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims: MergeDims = False,
                max_precond_dim=10000):
        """Projects the gradient onto the eigenbases of the preconditioner."""
        grad, merge_metadata = self._reshape_for_preconditioner(
            grad, max_precond_dim, merge_dims
        )

        for mat in state["Q"]:
            if len(mat) > 0:
                # Q is stored in float32 (its eigen-decomposition precision);
                # align to grad's dtype — a no-op for float32, an upcast for
                # float64 — so the projection matches.
                grad = torch.tensordot(grad, mat.to(grad.dtype), dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        return self._restore_merged_dims(grad, merge_metadata)

    def project_back(self, grad, state, merge_dims: MergeDims = False,
                     max_precond_dim=10000):
        """Projects the gradient back to the original space."""
        grad, merge_metadata = self._reshape_for_preconditioner(
            grad, max_precond_dim, merge_dims
        )

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat.to(grad.dtype), dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        return self._restore_merged_dims(grad, merge_metadata)

    def update_preconditioner(self, grad, state,
                              max_precond_dim=10000,
                              merge_dims: MergeDims = False,
                              precondition_1d=False):
        """Updates preconditioner matrices and eigenbases."""
        if state["Q"] is not None:
            state["exp_avg"] = self.project_back(
                state["exp_avg"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )
        # GG is kept in float32 (its eigenbasis working precision); cast each
        # contribution to GG's dtype so higher-precision (e.g. float64) grads
        # accumulate. For float32 grads this .to() is a no-op — behaviour of
        # existing runs is unchanged.
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                gg = state["GG"][0]
                gg.lerp_(
                    (grad.unsqueeze(1) @ grad.unsqueeze(0)).to(gg.dtype),
                    1 - state["shampoo_beta"],
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim, merge_dims)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad, new_grad,
                            dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2,
                        )
                        gg = state["GG"][idx]
                        gg.lerp_(outer_product.to(gg.dtype), 1 - state["shampoo_beta"])
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad, grad,
                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                        )
                        gg = state["GG"][idx]
                        gg.lerp_(outer_product.to(gg.dtype), 1 - state["shampoo_beta"])

        if state["Q"] is None:
            state["Q"] = self.get_orthogonal_matrix(state["GG"])
        if state["step"] > 0 and state["step"] % state["precondition_frequency"] == 0:
            state["Q"] = self.get_orthogonal_matrix_QR(state, max_precond_dim, merge_dims)

        if state["step"] > 0:
            state["exp_avg"] = self.project(
                state["exp_avg"], state,
                merge_dims=merge_dims, max_precond_dim=max_precond_dim,
            )

    def get_orthogonal_matrix(self, mat):
        """Computes eigenbases via torch.linalg.eigh.

        Runs on CPU when the tensor is on MPS (eigh is not implemented for MPS).
        """
        final = []
        for m in mat:
            if len(m) == 0:
                final.append([])
                continue
            m_f = m.data.float()
            eye = torch.eye(m_f.shape[0], device=m_f.device)
            try:
                _, Q = torch.linalg.eigh(m_f + 1e-30 * eye)
            except Exception:
                # Fall back to CPU (e.g. MPS doesn't support eigh)
                m_cpu = m_f.cpu()
                eye_cpu = torch.eye(m_cpu.shape[0])
                _, Q = torch.linalg.eigh(m_cpu + 1e-30 * eye_cpu)
                Q = Q.to(m_f.device)
            Q = torch.flip(Q, [1])
            if m.data.dtype != torch.float:
                Q = Q.to(dtype=m.data.dtype)
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state, max_precond_dim=10000,
                                 merge_dims: MergeDims = False):
        """Computes eigenbases via one power iteration step + torch.linalg.qr.

        Runs qr on CPU when the tensor is on MPS (qr is not implemented for MPS).
        """
        precond_list = state["GG"]
        orth_list = state["Q"]

        exp_avg_sq, merge_metadata = self._reshape_for_preconditioner(
            state["exp_avg_sq"], max_precond_dim, merge_dims
        )

        final = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            if len(m) == 0:
                final.append([])
                continue
            m_f = m.data.float()
            o_f = o.data.float()

            est_eig = torch.diag(o_f.T @ m_f @ o_f)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o_f = o_f[:, sort_idx]
            power_iter = m_f @ o_f

            # qr is not implemented on MPS — run on CPU
            try:
                Q, _ = torch.linalg.qr(power_iter)
            except Exception:
                Q, _ = torch.linalg.qr(power_iter.cpu())
                Q = Q.to(power_iter.device)

            if m.data.dtype != torch.float:
                Q = Q.to(dtype=m.data.dtype)
            final.append(Q)

        state["exp_avg_sq"] = self._restore_merged_dims(
            exp_avg_sq, merge_metadata
        )
        return final
