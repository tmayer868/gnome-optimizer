"""First-order warm starts and Scimba's full-memory SS-Broyden refinement."""

import math
from importlib.metadata import version

from experiments.common import cosine_scheduler
from experiments.baselines.soap import SOAP


def warm_start_steps(args):
    return getattr(args, args.optimizer.split("+")[0] + "_steps")


def validate_ssbroyden(args):
    if not 0 <= warm_start_steps(args) < args.steps:
        name = args.optimizer.split("+")[0] + "_steps"
        raise ValueError(f"{name} must be in [0, steps) (steps is the total budget)")
    for name in ("lr", "ssbroyden_lr", "ssbroyden_tolerance_grad"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.diagnostics_every:
        raise ValueError("diagnostics_every is Gnome-only")
    for name in ("beta1", "beta2"):
        if not math.isfinite(getattr(args, name)) or not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name} must be finite and in [0, 1)")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")


def ssbroyden_class():
    try:
        from scimba_torch.optimizers.ssbroyden import SSBroyden
    except ImportError as error:
        raise ImportError("SS-Broyden hybrids require: uv sync --extra ssbroyden") from error
    return SSBroyden


def build_soap_phase(model, args, *, precondition_frequency=10):
    # Check the optional dependency before spending the SOAP training budget.
    ssbroyden_class()
    soap_cfg = dict(lr=args.lr, betas=(args.beta1, args.beta2),
                    shampoo_beta=args.beta2, eps=1e-8, weight_decay=args.weight_decay,
                    precondition_frequency=precondition_frequency, precondition_1d=True)
    opt = SOAP(model.parameters(), **soap_cfg)
    scheduler = cosine_scheduler(opt, args.warmup_steps, max(1, args.soap_steps),
                                 args.cosine_decay)
    n = sum(p.numel() for p in model.parameters())
    cfg = dict(**soap_cfg,
               soap_steps=args.soap_steps, ssbroyden_steps=args.steps - args.soap_steps,
               warmup=args.warmup_steps, cosine_decay_floor=args.cosine_decay,
               ssbroyden_lr=args.ssbroyden_lr,
               ssbroyden_tolerance_grad=args.ssbroyden_tolerance_grad,
               method="ssbroyden", scimba_version=version("scimba"),
               ssbroyden_sampling="fixed", ssbroyden_schedule="none",
               ssbroyden_weight_decay=0.0,
               ssbroyden_dtype="torch.float64",
               inverse_hessian_bytes=n * n * 8)
    return opt, cfg, scheduler

def build_ssbroyden_phase(model, args):
    opt = ssbroyden_class()(model.parameters(), lr=args.ssbroyden_lr,
                           tolerance_grad=args.ssbroyden_tolerance_grad,
                           method="ssbroyden")
    # Scimba 1.4 initializes Hk on CPU, even for CUDA parameters.
    first = next(model.parameters())
    state = opt.state[first]
    state["Hk"] = state["Hk"].to(device=first.device, dtype=first.dtype)
    return opt
