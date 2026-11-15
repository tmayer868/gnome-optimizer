"""Precision transitions that preserve optimizer state and parameter bindings."""

import torch
from torch import nn


def switch_to_float64(model: nn.Module, opt: torch.optim.Optimizer) -> torch.device:
    """Promote a model and its accumulated optimizer state between steps."""
    old_params = list(model.parameters())
    device = old_params[0].device
    opt.zero_grad(set_to_none=True)
    if device.type == "mps":
        device = torch.device("cpu")
        # Transfer first: MPS cannot hold even a temporary float64 tensor.
        model.to(device)
    model.double()

    def convert(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device).to(
                dtype=torch.float64 if value.is_floating_point() else value.dtype
            )
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        return value

    # Module conversion can replace Parameters on a device change. Rebind
    # optimizer references as well as state keys, retaining the same optimizer.
    replacements = dict(zip(old_params, model.parameters()))
    for group in opt.param_groups:
        group["params"] = [replacements[p] for p in group["params"]]
    old_state = list(opt.state.items())
    opt.state.clear()
    for parameter, state in old_state:
        opt.state[replacements[parameter]] = {
            # AdamW's scalar step counter keeps its original dtype/device.
            key: value if key == "step" else convert(value)
            for key, value in state.items()
        }
    torch.set_default_dtype(torch.float64)
    return device
