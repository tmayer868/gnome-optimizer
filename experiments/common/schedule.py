"""Learning-rate schedules for the baseline optimizers.

Protocol: on MSE tasks Gnome runs with a *fixed* learning rate — its update is
the diagonal Gauss-Newton step, whose magnitude vanishes on its own as the
residual shrinks, so no decay is needed. The baselines (SOAP, AdamW) normalize
by the gradient RMS and never self-anneal, so to give them their standard
strong treatment we wrap them in a linear-warmup + cosine-decay schedule.

The decay floor is a single knob (``min_frac``, exposed by experiment scripts
as ``--cosine-decay``): ``0.0`` decays all the way to zero, ``1.0`` is a flat
schedule (warmup then constant lr, i.e. decay disabled).
"""

from __future__ import annotations

import math

import torch


def cosine_with_warmup(
    step: int, warmup: int, total: int, min_frac: float = 0.0
) -> float:
    """LR multiplier in ``[min_frac, 1]``.

    Linear warmup ``0 → 1`` over ``warmup`` steps, then cosine ``1 → min_frac``
    over the rest of ``total``. ``min_frac=1.0`` disables decay (constant lr
    after warmup).
    """
    if warmup > 0 and step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * progress))


def baseline_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    total: int,
    min_frac: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Standard linear-warmup + cosine-decay ``LambdaLR`` for SOAP/AdamW."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, warmup, total, min_frac)
    )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """The current learning rate of the optimizer's first parameter group.

    For the scheduled baselines this reflects the cosine curve; for Gnome
    (fixed lr) it is the constant base lr.
    """
    return float(optimizer.param_groups[0]["lr"])
