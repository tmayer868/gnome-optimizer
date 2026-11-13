"""1D reaction PINN: AdamW vs SOAP vs Gnome vs ENGD-W.

PDE:  u_t - rho*u*(1-u) = 0,    x in [0, 2*pi],  t in [0, 1]
IC:   u(0, x) = exp(-(x-pi)^2 / (2*(pi/4)^2))
BC:   u(t, 0) = u(t, 2*pi)      (periodic)

This is the pure logistic-reaction benchmark from Krishnapriyan et al.
(NeurIPS 2021). There is deliberately no spatial derivative: x labels a
continuum of independent logistic ODEs. The exact solution is

    u(t, x) = h(x)*exp(rho*t) / (1 - h(x) + h(x)*exp(rho*t)).

Despite that simplicity, a whole-space-time PINN trained with L-BFGS reports
relative L2 errors 9.79e-1 at rho=5 and 9.96e-1 at rho=10. The network settles
near the zero equilibrium, which makes the PDE residual small while ignoring
the narrow initial-condition manifold. Sequence-to-sequence training gets
roughly 7e-2 on the same rungs. This makes reaction a useful complement to
convection: failure cannot be blamed on transport, high spatial frequencies,
or noisy second derivatives. It is a nonlinear stiffness / temporal-credit
assignment problem and a clean test of whether curvature helps reconcile the
PDE and IC blocks.

The default model matches the paper's function class: raw (t, x), four hidden
layers of width 50 with tanh activations, a soft IC, and a soft C0 periodic BC.
By default this repo resamples collocation points everyƒ optimizer step, whereas the
paper's released implementation samples a fixed set, so the numbers are an
order-of-magnitude anchor rather than a bit-for-bit reproduction.

Set ``fixed_dataset=True`` in the config to sample training points once for
any optimizer. Gnome draws fresh auxiliary subsets of those points; validation
remains independent. ``engdw_fixed`` remains an ENGD-W-only compatibility alias.
Set ``fix_after=N`` to resample for N steps, then reuse PDE/IC/BC points.
``None`` preserves default sampling; ``fixed_dataset=True`` overrides the delay.
SS-Broyden hybrids freeze no later than their optimizer handoff.

``--embed periodic`` is an ablation that feeds [t, cos(x), sin(x)]. It makes
the BC exact and drops that residual block. It is not the published setup and
slightly changes the function class: the Gaussian agrees in value at the two
endpoints but its periodic extension is not C1 there. The PDE contains no
x-derivatives, so only C0 agreement is part of the canonical problem.

Usage:

    # Edit experiments/configs/pinns/reaction_pinn.py: optimizer='gnome', rho=5.0
    uv run -m experiments.pinns.reaction_pinn
    # Edit experiments/configs/pinns/reaction_pinn.py: optimizer='adamw', rho=5.0
    uv run -m experiments.pinns.reaction_pinn
    # Edit experiments/configs/pinns/reaction_pinn.py: optimizer='soap', rho=10.0
    uv run -m experiments.pinns.reaction_pinn
    # Edit experiments/configs/pinns/reaction_pinn.py: optimizer='engdw', rho=5.0, steps=1000
    # In CONFIG["optimizers"]['engdw']: engdw_line_search=True, engdw_damping=1e-06, engdw_chunk=128
    uv run -m experiments.pinns.reaction_pinn
    # Edit experiments/configs/pinns/reaction_pinn.py: optimizer='gnome', rho=5.0, arch='fused-modified', fuse_every=2, depth=6
    uv run -m experiments.pinns.reaction_pinn
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from experiments.baselines import ENGDW, SOAP
from experiments.common import (
    DIVERGED_EXIT,
    FusedMLP,
    FusedModifiedMLP,
    MLP,
    ModifiedMLP,
    RunLogger,
    cosine_scheduler,
    current_lr,
    diverged,
    pick_device,
    ConcatEmbedding,
    PeriodicEmbedding,
)
from experiments.common.ssbroyden import (
    build_gnome_phase, build_soap_phase, build_ssbroyden_phase,
    validate_ssbroyden, warm_start_steps,
)
from experiments.common.sampling import sample_aux_from_batch
from gnome import Gnome, JsonlDiagnostics, stack_residuals


EXPERIMENT = "reaction_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 2.0 * math.pi
GAUSSIAN_CENTER = math.pi
GAUSSIAN_SIGMA = math.pi / 4.0


# ========================= Exact solution =========================

def initial_condition(x: torch.Tensor) -> torch.Tensor:
    """Canonical Gaussian ``h(x)`` centered at pi with width pi/4."""
    z = (x - GAUSSIAN_CENTER) / GAUSSIAN_SIGMA
    return torch.exp(-0.5 * z.square())


def exact_solution(
    t: torch.Tensor, x: torch.Tensor, rho: float
) -> torch.Tensor:
    """Closed-form logistic solution initialized by ``h(x)``.

    Written through the logit to avoid ``exp(rho*t)`` overflow on stiffness
    sweeps. At the Gaussian peak h=1, ``log1p(-h)`` is ``-inf`` and sigmoid
    correctly returns the equilibrium value one.
    """
    h = initial_condition(x)
    logit_h = torch.log(h) - torch.log1p(-h)
    return torch.sigmoid(logit_h + rho * t)


# ========================= Models =========================


def build_embedding(name: str) -> nn.Module:
    if name == "none":
        return ConcatEmbedding(2)
    if name == "periodic":
        return PeriodicEmbedding(2, wavenumber=1.0)
    raise ValueError(f"unknown embedding: {name}")


def build_model(
    arch: str,
    embed: nn.Module,
    hidden: int,
    depth: int,
    fuse_every: int = 0,
) -> nn.Module:
    """Build ``(t, x) -> u`` with the shared experiment architectures."""
    if arch == "mlp":
        return MLP(embed, hidden=hidden, depth=depth)
    if arch == "modified":
        return ModifiedMLP(embed, hidden=hidden, depth=depth)
    if arch == "fused":
        return FusedMLP(
            embed, hidden=hidden, depth=depth, fuse_every=fuse_every
        )
    if arch == "fused-modified":
        return FusedModifiedMLP(
            embed, hidden=hidden, depth=depth, fuse_every=fuse_every
        )
    raise ValueError(f"unknown architecture: {arch}")


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor, rho: float
) -> torch.Tensor:
    """Reaction residual ``u_t - rho*u*(1-u)`` at ``(t, x)``."""
    t = t.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(
        u, t, torch.ones_like(u), create_graph=True
    )[0]
    return u_t - rho * u * (1.0 - u)


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Initial-condition residual ``u(0, x) - h(x)``."""
    return model(torch.zeros_like(x), x) - initial_condition(x)


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """C0 periodic boundary residual ``u(t, 0) - u(t, 2*pi)``."""
    x_l = torch.full_like(t, X_MIN)
    x_r = torch.full_like(t, X_MAX)
    return model(t, x_l) - model(t, x_r)


def residual_partition_sizes(batch, use_bc: bool = True) -> list[int]:
    """Element counts of the PDE, IC, and optional BC residual blocks."""
    t_pde, _x_pde, x_ic, t_bc = batch
    sizes = [t_pde.numel(), x_ic.numel()]
    if use_bc:
        sizes.append(t_bc.numel())
    return sizes


def stacked_residuals(
    model: nn.Module, batch, rho: float, use_bc: bool = True,
    weights=(1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Stack PDE, IC, and optional periodic-BC blocks with loss weights."""
    t_pde, x_pde, x_ic, t_bc = batch
    blocks = [
        pde_residual(model, t_pde, x_pde, rho),
        ic_residual(model, x_ic),
    ]
    if use_bc:
        blocks.append(bc_residual(model, t_bc))
    return stack_residuals(blocks, weights=weights[:len(blocks)])


def term_losses(
    model: nn.Module, batch, rho: float
) -> dict[str, float]:
    """Per-block MSEs for validation logging.

    BC is reported even with the periodic embedding, where it should remain
    at numerical round-off and is not included in the training loss.
    """
    t_pde, x_pde, x_ic, t_bc = batch
    return {
        "pde": pde_residual(model, t_pde, x_pde, rho).square().mean().item(),
        "ic": ic_residual(model, x_ic).square().mean().item(),
        "bc": bc_residual(model, t_bc).square().mean().item(),
    }


# ========================= Sampling and evaluation =========================

def sample_batch(
    n_pde: int, n_ic: int, n_bc: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent uniform samples for the PDE, IC, and BC blocks."""
    t_pde = torch.rand(n_pde, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x_pde = torch.rand(n_pde, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    x_ic = torch.rand(n_ic, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t_bc = torch.rand(n_bc, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    return t_pde, x_pde, x_ic, t_bc


def reaction_reference(
    rho: float, nt: int = 101, nx: int = 256
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic solution on the paper's 256-by-100-style evaluation grid."""
    t = torch.linspace(T_MIN, T_MAX, nt)
    # Exclude the duplicate periodic endpoint, following the released code.
    x = torch.linspace(X_MIN, X_MAX, nx + 1)[:-1]
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    return t, x, exact_solution(tt, xx, rho)


def eval_rel_l2(
    model: nn.Module,
    t_ref: torch.Tensor,
    x_ref: torch.Tensor,
    u_ref: torch.Tensor,
    device: torch.device,
    batch_size: int = 8192,
) -> float:
    """Discrete relative L2 error against an independent analytic grid."""
    nt, nx = u_ref.shape
    tt, xx = torch.meshgrid(t_ref, x_ref, indexing="ij")
    t_flat = tt.reshape(-1, 1).to(device)
    x_flat = xx.reshape(-1, 1).to(device)
    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, t_flat.shape[0], batch_size):
            preds.append(
                model(
                    t_flat[i:i + batch_size], x_flat[i:i + batch_size]
                ).cpu()
            )
    if was_training:
        model.train()
    u_pred = torch.cat(preds).reshape(nt, nx)
    return float(
        (u_pred - u_ref).square().sum().sqrt()
        / u_ref.square().sum().sqrt()
    )


# ========================= Optimizer factory =========================

def make_engdw_functions(
    model: nn.Module, rho: float, use_bc: bool = True, chunk_size: int | None = 32,
    weights=(1.0, 1.0, 1.0),
):
    """Return the normalized ENGD-W residual and its per-sample Jacobian.

    Its squared norm equals ``stacked_residuals(...).square().mean()``.
    ENGD-W uses half this value as its energy; damping therefore belongs to
    J.T @ J for this normalization, independently of total batch size.

    torch.func input differentiation allows jacrev to differentiate through
    u_t as well as the nonlinear reaction term. The PDE Jacobian is
    d_theta(u_t) - rho * (1 - 2*u) * d_theta(u).
    """
    def u_single(params, tx):
        return functional_call(
            model, params, (tx[0:1].reshape(1, 1), tx[1:2].reshape(1, 1))
        ).squeeze()

    du = jacrev(u_single, argnums=1)

    def pde_single(params, tx):
        u = u_single(params, tx)
        return du(params, tx)[0] - rho * u * (1.0 - u)

    def ic_single(params, x):
        tx = torch.cat((torch.zeros_like(x), x))
        return u_single(params, tx) - initial_condition(x).squeeze()

    def bc_single(params, t):
        left = torch.cat((t, torch.full_like(t, X_MIN)))
        right = torch.cat((t, torch.full_like(t, X_MAX)))
        return u_single(params, left) - u_single(params, right)

    def residual(params, batch):
        t_pde, x_pde, x_ic, t_bc = batch
        r_pde = vmap(pde_single, in_dims=(None, 0))(
            params, torch.cat((t_pde, x_pde), dim=1)
        )
        u_ic = functional_call(model, params, (torch.zeros_like(x_ic), x_ic))
        blocks = [r_pde, (u_ic - initial_condition(x_ic)).reshape(-1)]
        if use_bc:
            u_l = functional_call(model, params, (t_bc, torch.full_like(t_bc, X_MIN)))
            u_r = functional_call(model, params, (t_bc, torch.full_like(t_bc, X_MAX)))
            blocks.append((u_l - u_r).reshape(-1))
        return torch.cat([r * math.sqrt(w / r.numel()) for r, w in zip(blocks, weights)]).unsqueeze(1)

    def jacobian(params, batch):
        t_pde, x_pde, x_ic, t_bc = batch
        blocks = [(pde_single, torch.cat((t_pde, x_pde), dim=1)),
                  (ic_single, x_ic)]
        if use_bc:
            blocks.append((bc_single, t_bc))
        # Samples are independent. vmap(jacrev(one residual)) computes only
        # the necessary N parameter gradients, rather than differentiating
        # a batched forward N times with mostly-zero cotangents.
        jacobians = []
        for (fn, points), weight in zip(blocks, weights):
            block = vmap(jacrev(fn), in_dims=(None, 0), chunk_size=chunk_size)(
                params, points
            )
            jacobians.append({k: v * math.sqrt(weight / len(points)) for k, v in block.items()})
        return {k: torch.cat([block[k] for block in jacobians]) for k in params}

    return residual, jacobian


def build_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float,
    warmup: int,
    total_steps: int,
    cosine_decay: float,
    eps: float = 1e-6,
    beta1: float = 0.9,
    beta2: float = 0.99,
    trust_region: float = 1.0,
    basis_transport: str = "overlap",
):
    """Construct an optimizer and the shared warmup/cosine schedule."""
    if name == "gnome":
        cfg = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            shampoo_beta=beta2,
            eps=eps,
            precondition_frequency=10,
            basis_transport=basis_transport,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse",
            precondition_1d=True,
        )
        opt = Gnome(params, **cfg)
    elif name == "soap":
        cfg = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            shampoo_beta=beta2,
            eps=1e-8,
            precondition_frequency=10,
            precondition_1d=True,
        )
        opt = SOAP(params, **cfg)
    elif name == "adamw":
        cfg = dict(
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        opt = torch.optim.AdamW(params, **cfg)
    else:
        raise ValueError(f"unknown optimizer: {name}")

    scheduler = cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    cfg["warmup"] = warmup
    cfg["cosine_decay_floor"] = cosine_decay
    return opt, cfg, scheduler


# ========================= CLI and training =========================

def parse_args(argv=None) -> argparse.Namespace:
    from experiments.common.config import ConfigParser
    p = ConfigParser(__file__)
    args = p.parse_args(argv)
    if any(w < 0 for w in args.weights):
        p.error("weights must be non-negative")
    if min(args.n_pde, args.n_ic, args.n_bc, args.steps) < 1:
        p.error("sample counts and steps must be positive")
    if args.fix_after is not None and not 0 <= args.fix_after < args.steps:
        p.error("fix_after must satisfy 0 <= fix_after < steps, or be None")
    if (args.optimizer == "gnome" and (args.fixed_dataset or args.fix_after is not None)
            and not 0 < args.aux_frac <= 1):
        p.error("fixed_dataset or fix_after requires aux_frac in (0,1] for Gnome")
    if args.optimizer in ("soap+ssbroyden", "gnome+ssbroyden"):
        try:
            validate_ssbroyden(args)
        except ValueError as error:
            p.error(str(error))
    return args


def train(args: argparse.Namespace) -> str:
    hybrid = args.optimizer in ("soap+ssbroyden", "gnome+ssbroyden")
    warm_start = args.optimizer.split("+")[0]
    is_gnome = warm_start == "gnome"
    phase_steps = warm_start_steps(args) if hybrid else args.steps
    if hybrid:
        validate_ssbroyden(args)
    if args.engdw_fixed and args.optimizer != "engdw":
        raise SystemExit("--engdw-fixed requires --optimizer engdw")
    if args.rho < 0.0:
        raise SystemExit("--rho must be non-negative")
    if args.aux_frac <= 0.0:
        raise SystemExit("--aux-frac must be positive")
    if min(args.n_pde, args.n_ic, args.n_bc, args.steps) < 1:
        raise SystemExit("sample counts and --steps must be positive")
    if args.diagnostics_every > 0 and not is_gnome:
        raise SystemExit("--diagnostics-every is Gnome-only")
    fixed_dataset = args.fixed_dataset or args.engdw_fixed
    fixed_from_step = 0 if fixed_dataset else args.fix_after
    if hybrid:
        # Quasi-Newton curvature pairs require the same objective across steps.
        fixed_from_step = (phase_steps if fixed_from_step is None
                           else min(fixed_from_step, phase_steps))
    sampling = ("fixed" if fixed_from_step == 0 else
                "resample_then_fixed" if args.fix_after is not None else
                f"resample_{warm_start}_then_fixed" if hybrid else "resample_each_step")
    if args.optimizer == "engdw":
        if not math.isfinite(args.engdw_momentum) or not 0 <= args.engdw_momentum < 1:
            raise SystemExit("--engdw-momentum must be finite and in [0, 1)")
        if not math.isfinite(args.engdw_damping) or args.engdw_damping <= 0:
            raise SystemExit("--engdw-damping must be finite and positive")
        if args.engdw_chunk < 1:
            raise SystemExit("--engdw-chunk must be positive")
        if not math.isfinite(args.lr) or args.lr < 0:
            raise SystemExit("--lr must be finite and non-negative")

    device = pick_device()
    if args.float64 or hybrid or args.optimizer == "engdw":
        torch.set_default_dtype(torch.float64)
        if device.type == "mps":
            if not args.quiet:
                print(
                    f"[{EXPERIMENT}] float64: MPS has no double support; "
                    "using CPU.", flush=True,
                )
            device = torch.device("cpu")
    torch.manual_seed(args.seed)

    use_bc = args.embed != "periodic"
    model = build_model(
        args.arch,
        build_embedding(args.embed),
        args.hidden,
        args.depth,
        args.fuse_every,
    ).to(device)
    if args.optimizer == "engdw":
        residual_fn, jacobian_fn = make_engdw_functions(
            model, args.rho, use_bc, chunk_size=args.engdw_chunk, weights=args.weights,
        )
        opt = ENGDW(
            model, residual_fn, jacobian_fn=jacobian_fn,
            lr=args.lr, damping=args.engdw_damping,
            momentum=args.engdw_momentum,
            line_search=args.engdw_line_search, chunk_size=args.engdw_chunk,
        )
        opt_cfg = dict(
            lr=args.lr, damping=args.engdw_damping,
            momentum=args.engdw_momentum,
            line_search=args.engdw_line_search,
            ls_grid=opt.ls_grid if args.engdw_line_search else None,
            chunk_size=args.engdw_chunk, weight_decay=0.0, schedule="none",
            jacobian="vmap(jacrev(single_residual))",
            energy="0.5 * sum(block MSEs)", residual_scaling="1/sqrt(N_block)",
        )
        scheduler = None
    elif args.optimizer == "gnome+ssbroyden":
        opt, opt_cfg, scheduler = build_gnome_phase(model, args, build_optimizer)
    elif hybrid:
        opt, opt_cfg, scheduler = build_soap_phase(model, args)
    else:
        opt, opt_cfg, scheduler = build_optimizer(
            args.optimizer,
            model.parameters(),
            args.lr,
            args.weight_decay,
            warmup=args.warmup_steps,
            total_steps=args.steps,
            cosine_decay=args.cosine_decay,
            eps=args.eps,
            beta1=args.beta1,
            beta2=args.beta2,
            trust_region=args.trust_region,
            basis_transport=args.basis_transport,
        )

    n_pde_aux = max(1, int(args.n_pde * args.aux_frac)) if is_gnome else 0
    n_ic_aux = max(1, int(args.n_ic * args.aux_frac)) if is_gnome else 0
    n_bc_aux = max(1, int(args.n_bc * args.aux_frac)) if is_gnome else 0
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "partition_residuals": is_gnome and args.partition_residuals,
        "basis_transport": args.basis_transport if is_gnome else None,
        "steps": args.steps,
        "rho": args.rho,
        "arch": args.arch,
        "fuse_every": getattr(model, "chunks", None),
        "embed": args.embed,
        "blocks": ["pde", "ic", "bc"] if use_bc else ["pde", "ic"],
        "weights": list(args.weights),
        "hidden": args.hidden,
        "depth": args.depth,
        "n_params": n_params,
        "n_tensors": sum(1 for _ in model.parameters()),
        "n_pde": args.n_pde,
        "n_ic": args.n_ic,
        "n_bc": args.n_bc,
        "n_pde_aux": n_pde_aux,
        "n_ic_aux": n_ic_aux,
        "n_bc_aux": n_bc_aux,
        "fixed_dataset": fixed_dataset,
        "fix_after": args.fix_after,
        "fixed_from_step": fixed_from_step,
        "sampling": sampling,
        "aux_sampling": (
            "subsample_fixed_main_each_step" if fixed_from_step == 0 else
            "resample_then_subsample_fixed_main"
            if fixed_from_step is not None and fixed_from_step < phase_steps
            else "resample_each_step"
        ) if is_gnome else None,
        "diagnostics_every": args.diagnostics_every,
        "x_domain": (X_MIN, X_MAX),
        "t_domain": (T_MIN, T_MAX),
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }
    run = RunLogger(
        experiment=EXPERIMENT,
        optimizer=args.optimizer,
        seed=args.seed,
        hyperparameters=hyperparameters,
        runs_dir=args.runs_dir,
    )

    diag = None
    if args.diagnostics_every > 0:
        diag_params = (
            None if not args.diagnostics_params
            else [int(s) for s in args.diagnostics_params.split(",")]
        )
        diag_path = os.path.join(
            os.path.dirname(run.path) or ".", f"{run.run_id}.diag.jsonl"
        )
        diag = JsonlDiagnostics(diag_path, params=diag_params)

    if not args.quiet:
        blocks = "pde+ic+bc" if use_bc else "pde+ic (exact periodic BC)"
        print(
            f"[{EXPERIMENT}] {args.optimizer} | rho={args.rho:g} | "
            f"arch={args.arch} {args.depth}x{args.hidden} | "
            f"embed={args.embed} | params={n_params:,} | device={device} | "
            f"dtype={torch.get_default_dtype()}\n"
            f"  blocks={blocks} | N_pde={args.n_pde} | steps={args.steps} | "
            f"sampling={hyperparameters['sampling']}",
            flush=True,
        )

    t_ref, x_ref, u_ref = reaction_reference(args.rho)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}
    best_avg = best_rel_l2 = float("inf")
    training_seconds = 0.0
    fixed_batch = None

    for step in range(args.steps):
        step_start = time.perf_counter()
        quasi_newton = hybrid and step >= phase_steps
        if hybrid and step == phase_steps:
            opt = build_ssbroyden_phase(model, args)
            scheduler = None
            if not args.quiet:
                print(f"[{EXPERIMENT}] switching to SS-Broyden at step {step}", flush=True)
        if fixed_batch is None:
            main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
            if fixed_from_step is not None and step >= fixed_from_step:
                fixed_batch = main_batch
                run.log("sampling", step=step, sampling="fixed")
                if not args.quiet and step > 0:
                    print(f"[{EXPERIMENT}] freezing training points after {step} steps", flush=True)
        else:
            main_batch = fixed_batch
        if is_gnome and not quasi_newton:
            aux_batch = (sample_aux_from_batch(main_batch, n_pde_aux, n_ic_aux, n_bc_aux)
                         if fixed_batch is not None else sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device))

            def main_closure():
                r = stacked_residuals(model, main_batch, args.rho, use_bc, args.weights)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, args.rho, use_bc, args.weights)
                return r, torch.zeros_like(r)

            loss = opt.step(
                main_closure, aux_closure,
                surrogate_partition_sizes=(
                    residual_partition_sizes(aux_batch, use_bc) if args.partition_residuals else None
                ),
            )
        elif args.optimizer == "engdw":
            loss = opt.step(main_batch)
        elif quasi_newton:
            def closure():
                opt.zero_grad()
                r = stacked_residuals(model, main_batch, args.rho, use_bc, args.weights)
                value = r.square().mean()
                value.backward()
                return value

            loss = opt.step(closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, args.rho, use_bc, args.weights)
            loss = r.square().sum() / r.shape[0]
            loss.backward()
            opt.step()

        if (diag is not None and is_gnome and not quasi_newton
                and (step + 1) % args.diagnostics_every == 0):
            for record in opt.get_diagnostics(params=diag.params):
                diag(record)

        applied_lr = current_lr(opt)
        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        training_seconds += time.perf_counter() - step_start
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            if diag is not None:
                diag.close()
            print(
                f"[{EXPERIMENT}] diverged at step {step}; stopping.",
                flush=True,
            )
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val, lr=applied_lr,
                      training_seconds=training_seconds, **(
                          {"phase": "ssbroyden" if quasi_newton else warm_start} if hybrid else {}
                      ))
        window.append(loss_val)

        should_log = args.log_every and (step + 1) % args.log_every == 0
        is_last = step + 1 == args.steps
        if should_log or is_last or (hybrid and step + 1 == phase_steps):
            tl = term_losses(
                model,
                sample_batch(args.n_pde, args.n_ic, args.n_bc, device),
                args.rho,
            )
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_avg = sum(window) / len(window)
            last_terms, last_rel_l2 = tl, rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(
                step + 1,
                loss=last_avg,
                lr=current_lr(opt),
                rel_l2=rl2,
                training_seconds=training_seconds,
                **tl,
            )
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  pde={tl['pde']:.3e}  "
                    f"ic={tl['ic']:.3e}  bc={tl['bc']:.3e}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    path = run.finish(
        completed=True,
        final_avg_train=last_avg,
        best_avg_train=best_avg,
        final_rel_l2=last_rel_l2,
        best_rel_l2=best_rel_l2,
        training_seconds=training_seconds,
    )
    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics -> {diag.path}")
    print(f"[{EXPERIMENT}] saved -> {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print("  final " + "  ".join(f"{k}={v:.3e}" for k, v in last_terms.items()))
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
