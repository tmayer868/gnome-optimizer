"""Common optimizer, embedding, and sampling options for PINN runners."""

import math
from functools import wraps

import torch
from torch.func import functional_call

from experiments.baselines import ENGDW
from experiments.common.embedding_layers import (
    ConcatEmbedding, PeriodicEmbedding, TrainableFourierEmbedding,
)
from experiments.baselines.ssbroyden import (
    build_soap_phase, build_ssbroyden_phase, validate_ssbroyden,
)


OPTIMIZERS = ("adamw", "soap", "gnome", "soap+ssbroyden", "engdw")


def preserve_default_dtype(train):
    """Keep a runner's precision choices local to its training invocation."""
    @wraps(train)
    def wrapped(args):
        previous = torch.get_default_dtype()
        requested = (getattr(args, "float64", False)
                     or getattr(args, "dtype", "float32") == "float64"
                     or getattr(args, "float64_after", None) == 0
                     or args.optimizer == "engdw")
        torch.set_default_dtype(torch.float64 if requested else torch.float32)
        try:
            return train(args)
        finally:
            torch.set_default_dtype(previous)
    return wrapped


def build_embedding(name, n_inputs, *, spatial_dims, length=1.0,
                    periodic=False, embed_dim=32, embed_scale=1.0, n_harmonics=1):
    """Use full spatial periods only for problems with periodic boundaries.

    Other problems use a period of twice the domain width, so opposite faces
    remain distinguishable. Time and coefficient inputs always pass through.
    """
    if name == "none":
        return ConcatEmbedding(n_inputs)
    kwargs = dict(n_harmonics=n_harmonics, periodic_dims=spatial_dims,
                  wavenumber=(2 if periodic else 1) * math.pi / length)
    if name == "periodic":
        return PeriodicEmbedding(n_inputs, **kwargs)
    if name == "fourier":
        return TrainableFourierEmbedding(
            n_inputs, embed_dim, embed_scale, include_input=True, **kwargs,
        )
    raise ValueError(f"unknown embedding: {name}")


def validate_options(args):
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.fix_after is not None and not 0 <= args.fix_after < args.steps:
        raise ValueError("fix_after must satisfy 0 <= fix_after < steps, or be None")
    if args.embed_dim < 2 or args.embed_dim % 2:
        raise ValueError("embed_dim must be a positive even integer")
    if not math.isfinite(args.embed_scale) or args.embed_scale <= 0:
        raise ValueError("embed_scale must be finite and positive")
    if args.diagnostics_every and args.optimizer != "gnome":
        raise ValueError("diagnostics_every is Gnome-only")
    if args.optimizer == "gnome" and hasattr(args, "aux_frac"):
        if not math.isfinite(args.aux_frac) or args.aux_frac <= 0:
            raise ValueError("aux_frac must be finite and positive")
        if (args.fixed_dataset or args.fix_after is not None) and args.aux_frac > 1:
            raise ValueError("fixed_dataset or fix_after requires aux_frac in (0, 1]")
    if args.optimizer == "soap+ssbroyden":
        validate_ssbroyden(args)
    if args.optimizer == "engdw":
        if args.engdw_damping <= 0:
            raise ValueError("engdw_damping must be positive")
        if not 0 <= args.engdw_momentum < 1:
            raise ValueError("engdw_momentum must be in [0, 1)")
        if args.engdw_chunk < 1:
            raise ValueError("engdw_chunk must be positive")
        if args.lr < 0:
            raise ValueError("lr must be nonnegative")


def sampling_settings(args):
    fixed = args.fixed_dataset or getattr(args, "engdw_fixed", False)
    after = 0 if fixed else args.fix_after
    if args.optimizer == "soap+ssbroyden":
        after = args.soap_steps if after is None else min(after, args.soap_steps)
    sampling = ("fixed" if after == 0 else
                "resample_then_fixed" if after is not None else "resample_each_step")
    return after, sampling


class BatchSampler:
    """Keep the complete training objective fixed once its freeze step arrives."""

    def __init__(self, args, draw):
        self.fixed_from_step, self.sampling = sampling_settings(args)
        self.draw = draw
        self.fixed_batch = None

    def batch(self, step):
        if self.fixed_batch is not None:
            return self.fixed_batch
        batch = self.draw()
        if self.fixed_from_step is not None and step >= self.fixed_from_step:
            self.fixed_batch = batch
        return batch

    def to(self, device, dtype):
        def convert(value):
            if torch.is_tensor(value):
                return value.to(device=device).to(dtype=dtype)
            if value is None:
                return None
            return tuple(convert(item) for item in value)
        if self.fixed_batch is not None:
            self.fixed_batch = convert(self.fixed_batch)


def subsample_block(block, n):
    if block is None:
        return None
    indices = torch.randperm(len(block[0]), device=block[0].device)[:n]
    return tuple(t[indices] for t in block)


def subsample_nested(batch, *counts):
    return tuple(subsample_block(block, n) for block, n in zip(batch, counts))


def subsample_txc(batch, n_pde, n_ic, n_bc):
    """Subsample the flat (t_pde, x_pde, x_ic, t_bc) layout."""
    pde = subsample_block(batch[:2], n_pde)
    ic = subsample_block(batch[2:3], n_ic)
    bc = subsample_block(batch[3:], n_bc)
    return (*pde, *ic, *bc)


def metadata(args):
    after, sampling = sampling_settings(args)
    return dict(embedding=args.embedding, embed_dim=args.embed_dim,
                embed_scale=args.embed_scale, fix_after=args.fix_after,
                fixed_dataset=args.fixed_dataset or getattr(args, "engdw_fixed", False),
                fixed_from_step=after, sampling=sampling)


def phase_metrics(args, step):
    if args.optimizer != "soap+ssbroyden":
        return {}
    return {"phase": "soap" if step < args.soap_steps else "ssbroyden"}


def refinement_step(opt, model, batch, residual_fn):
    def closure():
        opt.zero_grad()
        loss = residual_fn(model, batch).square().mean()
        loss.backward()
        return loss
    return opt.step(closure)


class _FunctionalModel:
    """Evaluate a runner's autograd residual with explicit network parameters."""

    def __init__(self, model, params):
        self.model, self.params = model, params

    def __call__(self, *coords):
        return functional_call(self.model, self.params, coords)

    def __getattr__(self, name):
        if name in self.params:
            return self.params[name]
        return getattr(self.model, name)


def make_autograd_engdw_functions(model, residual_fn, chunk_size=32):
    """Differentiate the exact training objective, including higher PDE derivatives.

    Batched reverse-mode chunks avoid materializing an N-by-N identity. This
    fallback supports residuals written with torch.autograd; runners with
    specialized per-point torch.func Jacobians can continue using those.
    """
    def residual(params, batch):
        r = residual_fn(_FunctionalModel(model, params), batch).reshape(-1, 1)
        return r / math.sqrt(r.numel())

    def jacobian(params, batch):
        active = {k: p.detach().requires_grad_(True) for k, p in params.items()}
        r = residual(active, batch).flatten()
        blocks = {k: [] for k in active}
        values = tuple(active.values())
        for start in range(0, r.numel(), chunk_size):
            count = min(chunk_size, r.numel() - start)
            seeds = r.new_zeros(count, r.numel())
            seeds[torch.arange(count, device=r.device), start + torch.arange(count, device=r.device)] = 1
            grads = torch.autograd.grad(
                r, values, grad_outputs=seeds, is_grads_batched=True,
                retain_graph=start + count < r.numel(), allow_unused=True,
            )
            for (key, parameter), grad in zip(active.items(), grads):
                blocks[key].append(grad if grad is not None else parameter.new_zeros(count, *parameter.shape))
        return {k: torch.cat(chunks) for k, chunks in blocks.items()}

    return residual, jacobian


def build_extra_optimizer(model, args, residual_fn, *, precondition_frequency=10):
    """Build a SOAP warm start or the common ENGD-W autograd path."""
    if args.optimizer == "soap+ssbroyden":
        return build_soap_phase(model, args, precondition_frequency=precondition_frequency)
    residual, jacobian = make_autograd_engdw_functions(model, residual_fn, args.engdw_chunk)
    opt = ENGDW(model, residual, jacobian_fn=jacobian, lr=args.lr,
                damping=args.engdw_damping, momentum=args.engdw_momentum,
                line_search=args.engdw_line_search, chunk_size=args.engdw_chunk)
    cfg = dict(lr=args.lr, damping=args.engdw_damping, momentum=args.engdw_momentum,
               line_search=args.engdw_line_search,
               ls_grid=opt.ls_grid if args.engdw_line_search else None,
               chunk_size=args.engdw_chunk, schedule="none", weight_decay=0.0,
               energy="0.5 * sum(block MSEs)", residual_scaling="1/sqrt(N_block)",
               jacobian="chunked batched autograd")
    return opt, cfg, None
