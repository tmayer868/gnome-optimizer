"""Sampling helpers for time-dependent PINN residual blocks."""

import torch


def sample_aux_from_batch(batch, n_pde: int, n_ic: int, n_bc: int):
    """Subsample PDE/IC/BC points without replacement, preserving (t, x) pairs."""
    t_pde, x_pde, x_ic, t_bc = batch
    if not (1 <= n_pde <= len(t_pde) and 1 <= n_ic <= len(x_ic)
            and 1 <= n_bc <= len(t_bc)):
        raise ValueError("auxiliary sample counts must fit the main PDE/IC/BC batch")
    interior = torch.randperm(len(t_pde), device=t_pde.device)[:n_pde]
    initial = torch.randperm(len(x_ic), device=x_ic.device)[:n_ic]
    boundary = torch.randperm(len(t_bc), device=t_bc.device)[:n_bc]
    return t_pde[interior], x_pde[interior], x_ic[initial], t_bc[boundary]
