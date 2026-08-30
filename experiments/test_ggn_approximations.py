"""Measure how well Gnome's separable bases diagonalize a Poisson PINN GGN.

This experiment uses the same manufactured 2D Poisson problem as
``experiments.pinns.poisson_pinn``:

    -Δu = 2π² sin(πx) sin(πy),  (x, y) in (0, 1)²,
      u = 0,                    (x, y) on the boundary.

The network is deliberately small enough that each weight tensor's full
Generalized Gauss--Newton (GGN) block can be materialized. On every step we
compute per-collocation-point Jacobians of both the PDE and boundary residuals.
For the equal-weight two-block loss

    loss = mean(r_pde²) + mean(r_bc²),

the exact layer GGN block is

    G = (2 / N_pde) J_pde^T J_pde + (2 / N_bc) J_bc^T J_bc.

We keep an EMA of that dense block. From the same Jacobians we also keep the
two factor EMAs used by Gnome for a matrix-shaped parameter W:

    L = sum_blocks (2 / N_block) * sum J J^T,
    R = sum_blocks (2 / N_block) * sum J^T J,

where each J is one scalar residual's Jacobian reshaped like W, with the same
per-block scaling as the dense GGN. These are the exact
conditional expectations of Gnome's one-probe factor updates, so this test
isolates the separable-basis approximation from Hutchinson sampling noise.

For each Linear weight, the script rotates the dense GGN EMA by
``Q_sep = kron(Q_L, Q_R)`` and reports the fraction of squared Frobenius energy
on the diagonal. One minus that fraction is the squared relative error of the
best diagonal GGN approximation in Gnome's basis.

Example:

    uv run -m experiments.test_ggn_approximations --steps 100
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacfwd, jacrev, vmap


PI = math.pi
SOURCE_COEFF = 2.0 * PI * PI


class TinyMLP(nn.Module):
    """Small enough that every per-layer dense GGN block is tractable."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
    ) -> None:
        super().__init__()
        widths = [input_dim, *hidden_dims, output_dim]
        if any(width <= 0 for width in widths):
            raise ValueError("all layer widths must be positive")

        layers: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layers.append(nn.Linear(in_features, out_features))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LayerEMA:
    """Dense GGN and exact Gnome-factor EMAs for one matrix parameter."""

    shape: tuple[int, int]
    ggn: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    @classmethod
    def zeros(cls, parameter: torch.Tensor) -> "LayerEMA":
        rows, cols = parameter.shape
        options = {"device": parameter.device, "dtype": parameter.dtype}
        return cls(
            shape=(rows, cols),
            ggn=torch.zeros(rows * cols, rows * cols, **options),
            left=torch.zeros(rows, rows, **options),
            right=torch.zeros(cols, cols, **options),
        )

    def update(self, jacobian_blocks: Sequence[torch.Tensor], beta: float) -> None:
        """Update from equally weighted residual-block Jacobians.

        Each tensor has shape ``(*residual_shape, rows, cols)``. A block with
        ``N`` scalar residuals is scaled by ``sqrt(2 / N)``, so concatenating
        the resulting matrix samples gives the exact GGN of a sum of block
        mean-squared errors.
        """
        rows, cols = self.shape
        samples = []
        for block in jacobian_blocks:
            matrix_samples = block.reshape(-1, rows, cols)
            scale = (2.0 / matrix_samples.shape[0]) ** 0.5
            samples.append(matrix_samples * scale)
        samples = torch.cat(samples, dim=0)
        flat = samples.reshape(samples.shape[0], -1)
        batch_ggn = flat.T @ flat
        batch_left = torch.einsum("nai,nbi->ab", samples, samples)
        batch_right = torch.einsum("nia,nib->ab", samples, samples)

        alpha = 1.0 - beta
        self.ggn.lerp_(batch_ggn, alpha)
        self.left.lerp_(batch_left, alpha)
        self.right.lerp_(batch_right, alpha)


@dataclass(frozen=True)
class DiagonalizationMetrics:
    diagonal_energy: float
    off_diagonal_ratio: float
    parameter_basis_energy: float
    left_trace_error: float
    right_trace_error: float


def source_term(points: torch.Tensor) -> torch.Tensor:
    """Manufactured Poisson source ``2π² sin(πx) sin(πy)``."""
    x, y = points.unbind(dim=-1)
    return SOURCE_COEFF * torch.sin(PI * x) * torch.sin(PI * y)


def sample_poisson_batch(
    n_pde: int,
    n_bc_per_edge: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform interior points and four equal-sized unit-square boundary sets."""
    interior = torch.rand(n_pde, 2, generator=generator, dtype=torch.float64)
    s = torch.rand(n_bc_per_edge, generator=generator, dtype=torch.float64)
    zeros = torch.zeros_like(s)
    ones = torch.ones_like(s)
    boundary = torch.cat(
        (
            torch.stack((zeros, s), dim=1),
            torch.stack((ones, s), dim=1),
            torch.stack((s, zeros), dim=1),
            torch.stack((s, ones), dim=1),
        ),
        dim=0,
    )
    return interior, boundary


def poisson_residuals(
    model: nn.Module,
    interior: torch.Tensor,
    boundary: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute differentiable PDE and zero-Dirichlet residual vectors."""
    interior = interior.detach().requires_grad_(True)
    u = model(interior).squeeze(-1)
    grad_u = torch.autograd.grad(u.sum(), interior, create_graph=True)[0]
    u_xx = torch.autograd.grad(
        grad_u[:, 0].sum(), interior, create_graph=True
    )[0][:, 0]
    u_yy = torch.autograd.grad(
        grad_u[:, 1].sum(), interior, create_graph=True
    )[0][:, 1]
    pde = u_xx + u_yy + source_term(interior)
    bc = model(boundary).squeeze(-1)
    return pde, bc


def poisson_residual_jacobians(
    model: nn.Module,
    interior: torch.Tensor,
    boundary: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return exact per-point parameter Jacobians of both residual blocks."""
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def solution_one(
        parameter_dict: dict[str, torch.Tensor],
        point: torch.Tensor,
    ) -> torch.Tensor:
        return functional_call(
            model,
            (parameter_dict, buffers),
            (point.unsqueeze(0),),
        ).squeeze()

    def pde_one(
        parameter_dict: dict[str, torch.Tensor],
        point: torch.Tensor,
    ) -> torch.Tensor:
        input_hessian = jacfwd(
            jacrev(solution_one, argnums=1), argnums=1
        )(parameter_dict, point)
        return input_hessian.diagonal().sum() + source_term(point)

    pde_jacobians = vmap(
        jacrev(pde_one, argnums=0), in_dims=(None, 0)
    )(params, interior)
    bc_jacobians = vmap(
        jacrev(solution_one, argnums=0), in_dims=(None, 0)
    )(params, boundary)
    return pde_jacobians, bc_jacobians


def partial_trace_factors(
    ggn: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the two partial traces of a row-major matrix-parameter GGN."""
    rows, cols = shape
    block = ggn.reshape(rows, cols, rows, cols)
    left = torch.einsum("abcb->ac", block)
    right = torch.einsum("abad->bd", block)
    return left, right


def diagonalization_metrics(state: LayerEMA) -> DiagonalizationMetrics:
    """Measure the GGN EMA in the eigenbases of its two partial traces."""
    ggn = 0.5 * (state.ggn + state.ggn.T)
    left = 0.5 * (state.left + state.left.T)
    right = 0.5 * (state.right + state.right.T)

    _, q_left = torch.linalg.eigh(left)
    _, q_right = torch.linalg.eigh(right)
    rows, cols = state.shape
    rotated = ggn.reshape(rows, cols, rows, cols)
    rotated = torch.tensordot(q_left.T, rotated, dims=([1], [0]))
    rotated = torch.tensordot(q_right.T, rotated, dims=([1], [1])).permute(1, 0, 2, 3)
    rotated = torch.tensordot(rotated, q_left, dims=([2], [0])).permute(0, 1, 3, 2)
    rotated = torch.tensordot(rotated, q_right, dims=([3], [0]))
    rotated = rotated.reshape(rows * cols, rows * cols)

    total_energy = rotated.square().sum().clamp_min(torch.finfo(ggn.dtype).tiny)
    diagonal_energy = rotated.diagonal().square().sum() / total_energy
    off_diagonal_ratio = (1.0 - diagonal_energy).clamp_min(0.0).sqrt()
    parameter_basis_energy = ggn.diagonal().square().sum() / total_energy

    # These should be at roundoff: factor EMA and dense-GGN EMA use the same
    # batches, scaling, and beta, and partial trace commutes with an EMA.
    left_from_ggn, right_from_ggn = partial_trace_factors(ggn, state.shape)
    tiny = torch.finfo(ggn.dtype).tiny
    left_error = (left - left_from_ggn).norm() / left_from_ggn.norm().clamp_min(tiny)
    right_error = (right - right_from_ggn).norm() / right_from_ggn.norm().clamp_min(tiny)

    return DiagonalizationMetrics(
        diagonal_energy=diagonal_energy.item(),
        off_diagonal_ratio=off_diagonal_ratio.item(),
        parameter_basis_energy=parameter_basis_energy.item(),
        left_trace_error=left_error.item(),
        right_trace_error=right_error.item(),
    )


def print_report(
    step: int,
    pde_loss: float,
    bc_loss: float,
    states: dict[str, LayerEMA],
) -> None:
    print(
        f"\nstep {step:4d} | loss {pde_loss + bc_loss:.6f} "
        f"(PDE {pde_loss:.6f}, BC {bc_loss:.6f})"
    )
    print(
        "layer          shape      params  diag energy  offdiag ||.||F  "
        "raw diag energy  trace check"
    )
    print("-" * 96)
    for name, state in states.items():
        metrics = diagonalization_metrics(state)
        trace_error = max(metrics.left_trace_error, metrics.right_trace_error)
        rows, cols = state.shape
        print(
            f"{name:<14} {rows:>2}x{cols:<7} {rows * cols:>6d}  "
            f"{100.0 * metrics.diagonal_energy:>10.2f}%  "
            f"{100.0 * metrics.off_diagonal_ratio:>12.2f}%  "
            f"{100.0 * metrics.parameter_basis_energy:>14.2f}%  "
            f"{trace_error:>10.2e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--n-pde",
        type=int,
        default=32,
        help="Interior Poisson collocation points per step.",
    )
    parser.add_argument(
        "--n-bc-per-edge",
        type=int,
        default=8,
        help="Boundary points per edge (four times this many in total).",
    )
    parser.add_argument("--ema-beta", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[32, 64, 32],
        metavar="WIDTH",
        help="Hidden-layer widths, for example: --hidden-dims 32 64 32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.n_pde <= 0 or args.n_bc_per_edge <= 0:
        raise ValueError("steps, n-pde, and n-bc-per-edge must be positive")
    if not 0.0 <= args.ema_beta < 1.0:
        raise ValueError("ema-beta must satisfy 0 <= beta < 1")
    if args.report_every <= 0:
        raise ValueError("report-every must be positive")
    if any(width <= 0 for width in args.hidden_dims):
        raise ValueError("hidden-dims must contain only positive widths")

    torch.manual_seed(args.seed)
    input_dim = 2
    output_dim = 1
    model = TinyMLP(input_dim, args.hidden_dims, output_dim).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    batch_generator = torch.Generator().manual_seed(args.seed + 1)

    states = {
        name: LayerEMA.zeros(parameter)
        for name, parameter in model.named_parameters()
        if parameter.ndim == 2
    }

    architecture = " -> ".join(
        str(width) for width in (input_dim, *args.hidden_dims, output_dim)
    )
    print(
        "Exact Poisson PINN GGN; Gnome factors are exact probe expectations\n"
        f"network={architecture}, PDE points={args.n_pde}, "
        f"BC points={4 * args.n_bc_per_edge}, "
        f"steps={args.steps}, EMA beta={args.ema_beta:.3f}"
    )
    print("Bias vectors are omitted because they do not have a two-sided separable basis.")

    for step in range(1, args.steps + 1):
        interior, boundary = sample_poisson_batch(
            args.n_pde,
            args.n_bc_per_edge,
            batch_generator,
        )

        pde_jacobians, bc_jacobians = poisson_residual_jacobians(
            model, interior, boundary
        )
        with torch.no_grad():
            for name, state in states.items():
                state.update(
                    (pde_jacobians[name], bc_jacobians[name]),
                    args.ema_beta,
                )

        optimizer.zero_grad(set_to_none=True)
        pde_residual, bc_residual = poisson_residuals(model, interior, boundary)
        pde_loss = F.mse_loss(pde_residual, torch.zeros_like(pde_residual))
        bc_loss = F.mse_loss(bc_residual, torch.zeros_like(bc_residual))
        loss = pde_loss + bc_loss
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.report_every == 0 or step == args.steps:
            print_report(step, pde_loss.item(), bc_loss.item(), states)

    print(
        "\nInterpretation: 'diag energy' is the fraction of ||GGN||_F^2 captured "
        "by a diagonal in Gnome's Q_L/Q_R basis; 'offdiag ||.||F' is the "
        "relative Frobenius error of the best such diagonal approximation."
    )


if __name__ == "__main__":
    main()
