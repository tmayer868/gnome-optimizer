"""Learning-rate schedules, shared by every optimizer.

Protocol: all optimizers — Gnome included — get the same linear-warmup +
cosine-decay schedule, so the comparison is over the update rule and not over
who was handed a schedule. Gnome owns no schedule of its own; ``group["lr"]``
is whatever this ``LambdaLR`` last wrote.

The decay floor is a single knob (``min_frac``, exposed by experiment scripts
as ``--cosine-decay``): ``0.0`` decays all the way to zero, ``1.0`` is a flat
schedule (warmup then constant lr, i.e. decay disabled). ``1.0`` is the setting
that recovers Gnome's fixed-lr behaviour, which on MSE is defensible on its own
terms — the Gauss-Newton step self-anneals as the residual shrinks — where the
gradient-RMS baselines never self-anneal and do want the decay.
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


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    total: int,
    min_frac: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Standard linear-warmup + cosine-decay ``LambdaLR``.

    Works for Gnome and for the SOAP/AdamW baselines alike — Gnome is a stock
    ``torch.optim.Optimizer`` whose step scales linearly in ``group["lr"]``.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, warmup, total, min_frac)
    )


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """The current learning rate of the optimizer's first parameter group.

    This is the lr actually applied: no optimizer here transforms ``lr``
    internally, so the scheduler's last write is what the update used.
    """
    return float(optimizer.param_groups[0]["lr"])
