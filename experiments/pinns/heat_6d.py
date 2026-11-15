"""Heat equation on the five-dimensional unit ball, with time in [0, 1].

The six network inputs are (t, x1, x2, x3, x4, x5). With r² = sum_i x_i²:

    u_t - (1/5) * sum_i u_{x_i x_i} = -(1/5) * r² * exp(r²/2 + t)
    u(0, x) = exp(r²/2)
    du/dn = exp(1/2 + t) on r = 1
    u*(t, x) = exp(r²/2 + t)

The outward unit normal on the sphere is x. PDE and IC points are uniform
in ball volume; BC points are uniform on its surface. Relative L2 uses an
independent fixed sample, uniform in time and ball volume, against the analytic
reference. No external reference data is needed.

Adapted from the Heat 5D runner: tanh MLP, equal PDE/IC/BC loss weights, optional
Gnome residual partitions, SOAP, AdamW, ENGD-W, and experiment-owned diagnostics.
Configure experiments/configs/pinns/heat_6d.py, then run:

    uv run --extra experiments -m experiments.pinns.heat_6d
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

from gnome import Gnome, JsonlDiagnostics, stack_residuals
from experiments.baselines import ENGDW, SOAP
from experiments.common import (
    DIVERGED_EXIT,
    diverged,
    RunLogger,
    cosine_scheduler,
    current_lr,
    pick_device,
)
from experiments.common import MLP as _SharedMLP, ConcatEmbedding


from experiments.common import pinn_options as pinn

EXPERIMENT = "heat_6d"

T_MIN, T_MAX = 0.0, 1.0
SPATIAL_DIMENSIONS = 5
DIFFUSIVITY = 1.0 / SPATIAL_DIMENSIONS


# ========================= Exact solution =========================

def radius_squared(*spatial: torch.Tensor) -> torch.Tensor:
    return sum(coordinate.square() for coordinate in spatial)


def initial_condition(*spatial: torch.Tensor) -> torch.Tensor:
    """Initial value exp(||x||²/2) in the spatial unit ball."""
    return torch.exp(0.5 * radius_squared(*spatial))


def exact_solution(t: torch.Tensor, *spatial: torch.Tensor) -> torch.Tensor:
    """Analytic solution exp(||x||²/2 + t)."""
    return torch.exp(0.5 * radius_squared(*spatial) + t)


def forcing(t: torch.Tensor, *spatial: torch.Tensor) -> torch.Tensor:
    """Source f = -(1/5) ||x||² exp(||x||²/2 + t)."""
    return -DIFFUSIVITY * radius_squared(*spatial) * exact_solution(t, *spatial)


def boundary_flux(t: torch.Tensor) -> torch.Tensor:
    """Prescribed outward normal derivative on ||x||=1."""
    return torch.exp(0.5 + t)


# ========================= Model =========================

class PINN(_SharedMLP):
    """Tanh MLP mapping (t, x1, x2, x3, x4, x5) to u; depth counts all linear layers."""

    def __init__(self, hidden: int = 64, depth: int = 5,
                 embedding: nn.Module | None = None):
        super().__init__(embedding if embedding is not None else ConcatEmbedding(1 + SPATIAL_DIMENSIONS), hidden=hidden, depth=depth)


# ========================= Residuals =========================

def pde_residual(model: nn.Module, t: torch.Tensor, *spatial: torch.Tensor) -> torch.Tensor:
    """u_t - (1/5) * Laplacian(u) - f, with parameter derivatives retained."""
    coords = tuple(value.clone().requires_grad_(True) for value in (t, *spatial))
    u = model(*coords)
    derivatives = autograd.grad(u, coords, torch.ones_like(u), create_graph=True)
    laplacian = torch.zeros_like(u)
    for derivative, coordinate in zip(derivatives[1:], coords[1:]):
        laplacian = laplacian + autograd.grad(
            derivative, coordinate, torch.ones_like(derivative), create_graph=True,
        )[0]
    return derivatives[0] - DIFFUSIVITY * laplacian - forcing(t, *spatial)


def ic_residual(model: nn.Module, *spatial: torch.Tensor) -> torch.Tensor:
    """Initial residual at t=0 against exp(||x||²/2)."""
    return model(torch.zeros_like(spatial[0]), *spatial) - initial_condition(*spatial)


def bc_residual(model: nn.Module, t: torch.Tensor, *spatial: torch.Tensor) -> torch.Tensor:
    """Outward Neumann residual grad_x(u) dot x - exp(1/2+t) on ||x||=1."""
    coords = tuple(value.clone().requires_grad_(True) for value in spatial)
    u = model(t, *coords)
    gradient = autograd.grad(u, coords, torch.ones_like(u), create_graph=True)
    normal_derivative = sum(derivative * coordinate
                            for derivative, coordinate in zip(gradient, coords))
    return normal_derivative - boundary_flux(t)


# ========================= Sampling =========================

def sample_sphere(n, device, dtype, generator=None):
    """Uniform surface samples from normalized isotropic Gaussian vectors."""
    points = torch.randn(n, SPATIAL_DIMENSIONS, device=device, dtype=dtype,
                         generator=generator)
    return points / points.norm(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)


def sample_ball(n, device, dtype, generator=None):
    """Uniform volume samples: isotropic direction and radius U**(1/5)."""
    directions = sample_sphere(n, device, dtype, generator)
    radius = torch.rand(n, 1, device=device, dtype=dtype,
                        generator=generator).pow(1.0 / SPATIAL_DIMENSIONS)
    return radius * directions


def sample_batch(n_pde: int, n_ic: int, n_bc: int, device: torch.device, dtype: torch.dtype):
    """Independent PDE/IC volume points and BC surface points; time is uniform."""
    kw = dict(device=device, dtype=dtype)
    pde_t = torch.rand(n_pde, 1, **kw) * (T_MAX - T_MIN) + T_MIN
    bc_t = torch.rand(n_bc, 1, **kw) * (T_MAX - T_MIN) + T_MIN
    pde_x = sample_ball(n_pde, device, dtype)
    ic_x = sample_ball(n_ic, device, dtype)
    bc_x = sample_sphere(n_bc, device, dtype)
    return ((pde_t, *pde_x.split(1, dim=1)), ic_x.split(1, dim=1),
            (bc_t, *bc_x.split(1, dim=1)))


def sample_aux_from_batch(batch, n_pde: int, n_ic: int, n_bc: int):
    """Fresh subsets of each fixed block, preserving all coordinate tuples."""
    subsets = []
    for coords, count in zip(batch, (n_pde, n_ic, n_bc)):
        if not 1 <= count <= len(coords[0]):
            raise ValueError("auxiliary sample counts must fit the fixed main batch")
        indices = torch.randperm(len(coords[0]), device=coords[0].device)[:count]
        subsets.append(tuple(coordinate[indices] for coordinate in coords))
    return tuple(subsets)


def residual_partition_sizes(batch) -> list[int]:
    """One scalar residual per point in each actual auxiliary block."""
    return [coords[0].numel() for coords in batch]


def stacked_residuals(model: nn.Module, batch) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights)."""
    pde, ic, bc = batch
    return stack_residuals([
        pde_residual(model, *pde),
        ic_residual(model, *ic),
        bc_residual(model, *bc),
    ])


def make_engdw_functions(model: nn.Module, chunk_size: int = 32):
    """Functional residual/Jacobian with ||r||² equal to sum of block MSEs."""
    def u_single(params, p):
        return functional_call(
            model, params,
            tuple(p[i:i + 1].reshape(1, 1) for i in range(1 + SPATIAL_DIMENSIONS)),
        ).squeeze()

    gradient = jacrev(u_single, argnums=1)
    hessian = jacrev(gradient, argnums=1)

    def pde_single(params, p):
        d = gradient(params, p)
        h = hessian(params, p)
        return d[0] - DIFFUSIVITY * h.diagonal()[1:].sum() - forcing(*p.unbind())

    def ic_single(params, p):
        return u_single(params, p) - initial_condition(*p[1:].unbind())

    def bc_single(params, p):
        return (gradient(params, p)[1:] * p[1:]).sum() - boundary_flux(p[0])

    def points(batch):
        pde, ic, bc = batch
        return (torch.cat(pde, 1),
                torch.cat((torch.zeros_like(ic[0]), *ic), 1), torch.cat(bc, 1))

    functions = (pde_single, ic_single, bc_single)

    def residual(params, batch):
        blocks = [vmap(fn, in_dims=(None, 0), chunk_size=chunk_size)(params, pts)
                  / math.sqrt(len(pts)) for fn, pts in zip(functions, points(batch))]
        return torch.cat(blocks).unsqueeze(1)

    def jacobian(params, batch):
        blocks = []
        for fn, pts in zip(functions, points(batch)):
            j = vmap(jacrev(fn), in_dims=(None, 0), chunk_size=chunk_size)(params, pts)
            blocks.append({k: v / math.sqrt(len(pts)) for k, v in j.items()})
        return {k: torch.cat([block[k] for block in blocks]) for k in params}

    return residual, jacobian


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-term MSE for diagnostic logging."""
    pde, ic, bc = batch
    return {
        "pde": pde_residual(model, *pde).pow(2).mean().item(),
        "ic": ic_residual(model, *ic).pow(2).mean().item(),
        "bc": bc_residual(model, *bc).pow(2).mean().item(),
    }


# ========================= Eval =========================

def make_eval_set(
    n: int, device: torch.device, dtype: torch.dtype, seed: int = 1234,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Fixed IID uniform time/ball-volume points, ordered (t, x1, x2, x3, x4, x5).

    A private CPU generator leaves training RNG states untouched. Draw in
    float64 before casting so every device and precision uses the same points
    up to rounding. The caller reuses this set for every evaluation.
    """
    if n < 1:
        raise ValueError("n must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.rand(n, 1, generator=generator, dtype=torch.float64)
    x = sample_ball(n, torch.device("cpu"), torch.float64, generator)
    points = torch.cat((t, x), dim=1).to(device=device, dtype=dtype)
    coords = points.split(1, dim=1)
    return coords, exact_solution(*coords)


def eval_rel_l2(
    model: nn.Module, coords, u_ref: torch.Tensor, batch_size: int = 65536,
) -> float:
    """``||u_pred - u*||_2 / ||u*||_2`` over the fixed uniform eval points.

    Queried in batches under ``no_grad`` so the eval set can be far larger than a
    training batch without OOM.
    """
    was_training = model.training
    model.eval()
    num = torch.zeros((), device=u_ref.device, dtype=u_ref.dtype)
    with torch.no_grad():
        for i in range(0, coords[0].shape[0], batch_size):
            sl = slice(i, i + batch_size)
            pred = model(*(coordinate[sl] for coordinate in coords))
            num += (pred - u_ref[sl]).pow(2).sum()
    if was_training:
        model.train()
    return float((num.sqrt() / u_ref.pow(2).sum().sqrt()))


# ========================= Optimizer factory =========================

def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    warmup: int, total_steps: int, cosine_decay: float, eps: float = 1e-6,
    beta1: float = 0.9, beta2: float = 0.99,
    trust_region: float = 1.0,
):
    """Construct the optimizer and its LR schedule.

    Returns ``(optimizer, config, scheduler)``. Every optimizer gets the same
    linear-warmup + cosine-decay schedule; ``cosine_decay`` is the final-lr
    fraction (0.0 -> decay to zero, 1.0 -> warmup then constant). On MSE, 1.0
    is the natural setting for Gnome -- its Gauss-Newton step self-anneals as
    the residual shrinks -- where the gradient-RMS baselines do want the decay.
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=10,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse", precondition_1d=True, norm_free=False,
        )
        opt = Gnome(params, **cfg)
    elif name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=10, precondition_1d=True,
        )
        opt = SOAP(params, **cfg)
    elif name == "adamw":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.9, 0.999), eps=1e-8,
        )
        opt = torch.optim.AdamW(params, **cfg)
    else:
        raise ValueError(f"unknown optimizer: {name}")

    scheduler = cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    cfg["warmup"] = warmup
    cfg["cosine_decay_floor"] = cosine_decay
    return opt, cfg, scheduler


# ========================= CLI / training =========================

def parse_args(argv=None) -> argparse.Namespace:
    from experiments.common.config import ConfigParser
    p = ConfigParser(__file__)
    args = p.parse_args(argv)
    for name in ("steps", "n_pde", "n_ic", "n_bc", "hidden", "n_eval"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.fix_after is not None and not 0 <= args.fix_after < args.steps:
        p.error("fix_after must satisfy 0 <= fix_after < steps, or be None")
    if args.depth < 2:
        p.error("depth must be at least 2")
    for name in ("log_every", "diagnostics_every", "warmup_steps"):
        if getattr(args, name) < 0:
            p.error(f"{name} must be nonnegative")
    if not 0 < args.aux_frac <= 1:
        p.error("aux_frac must be in (0, 1]")
    if not 0 <= args.cosine_decay <= 1:
        p.error("cosine_decay must be in [0, 1]")
    if args.diagnostics_every and args.optimizer != "gnome":
        p.error("diagnostics_every is Gnome-only")
    if args.engdw_fixed and args.optimizer != "engdw":
        p.error("engdw_fixed requires optimizer='engdw'")
    if args.optimizer == "engdw":
        if args.engdw_damping <= 0:
            p.error("engdw_damping must be positive")
        if not 0 <= args.engdw_momentum < 1:
            p.error("engdw_momentum must be in [0, 1)")
        if args.engdw_chunk < 1:
            p.error("engdw_chunk must be positive")
        if args.lr <= 0:
            p.error("lr must be positive")
    return args


@pinn.preserve_default_dtype
def train(args: argparse.Namespace) -> str:
    hybrid = args.optimizer == "soap+ssbroyden"
    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = torch.float64 if hybrid or args.optimizer == "engdw" else getattr(torch, args.dtype)
    dtype_name = str(dtype).removeprefix("torch.")
    if device.type == "mps" and dtype is torch.float64:
        device = torch.device("cpu")            # MPS has no float64
        if not args.quiet:
            print(f"[{EXPERIMENT}] float64 requested; MPS has no float64 "
                  f"— falling back to CPU.", flush=True)

    embedding = pinn.build_embedding(
        args.embedding, 1 + SPATIAL_DIMENSIONS, spatial_dims=tuple(range(1, 1 + SPATIAL_DIMENSIONS)), length=2.0,
        embed_dim=args.embed_dim, embed_scale=args.embed_scale,
    )
    model = PINN(hidden=args.hidden, depth=args.depth, embedding=embedding).to(device=device, dtype=dtype)
    if args.optimizer == "engdw":
        residual, jacobian = make_engdw_functions(model, args.engdw_chunk)
        opt = ENGDW(model, residual, jacobian_fn=jacobian, lr=args.lr,
                    damping=args.engdw_damping, momentum=args.engdw_momentum,
                    line_search=args.engdw_line_search, chunk_size=args.engdw_chunk)
        opt_cfg = dict(lr=args.lr, damping=args.engdw_damping, momentum=args.engdw_momentum,
                       line_search=args.engdw_line_search,
                       ls_grid=opt.ls_grid if args.engdw_line_search else None,
                       chunk_size=args.engdw_chunk, schedule="none", weight_decay=0.0,
                       energy="0.5 * sum(block MSEs)", residual_scaling="1/sqrt(N_block)",
                       jacobian="vmap(jacrev(single_residual))")
        scheduler = None
    elif hybrid:
        opt, opt_cfg, scheduler = pinn.build_soap_phase(model, args)
    else:
        opt, opt_cfg, scheduler = build_optimizer(
            args.optimizer, model.parameters(), args.lr, args.weight_decay,
            warmup=args.warmup_steps, total_steps=args.steps,
            cosine_decay=args.cosine_decay, eps=args.eps,
            beta1=args.beta1, beta2=args.beta2,
            trust_region=args.trust_region,
        )

    n_pde_aux = max(8, int(args.n_pde * args.aux_frac))
    n_ic_aux = max(2, int(args.n_ic * args.aux_frac))
    n_bc_aux = max(2, int(args.n_bc * args.aux_frac))
    if args.optimizer != "gnome":
        n_pde_aux = n_ic_aux = n_bc_aux = 0
    fixed_from_step, sampling = pinn.sampling_settings(args)
    sampling = ("fixed" if fixed_from_step == 0 else
                "resample_then_fixed" if fixed_from_step is not None else "resample_each_step")
    if fixed_from_step == 0:
        n_pde_aux = min(args.n_pde, n_pde_aux)
        n_ic_aux = min(args.n_ic, n_ic_aux)
        n_bc_aux = min(args.n_bc, n_bc_aux)
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        **pinn.metadata(args),
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
        "partition_residuals": args.optimizer == "gnome" and args.partition_residuals,
        "steps": args.steps,
        "dtype": dtype_name,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_pde": args.n_pde,
        "n_ic": args.n_ic,
        "n_bc": args.n_bc,
        "n_pde_aux": n_pde_aux,
        "n_ic_aux": n_ic_aux,
        "n_bc_aux": n_bc_aux,
        "n_params": n_params,
        "n_eval": args.n_eval,
        "eval_seed": args.eval_seed,
        "eval_sampling": "fixed_uniform_time_ball_volume",
        "benchmark": "5+1D forced heat on the unit ball",
        "spatial_domain": "5D unit ball",
        "boundary_condition": "Neumann",
        "boundary_sampling": "uniform_sphere",
        "t_domain": (T_MIN, T_MAX),
        "diffusivity": DIFFUSIVITY,
        "spatial_dimensions": SPATIAL_DIMENSIONS,
        "sampling": sampling,
        "fixed_dataset": args.fixed_dataset or args.engdw_fixed,
        "fix_after": args.fix_after,
        "fixed_from_step": fixed_from_step,
        "aux_sampling": (
            "subsample_fixed_main_each_step" if fixed_from_step == 0 else
            "resample_then_subsample_fixed_main" if fixed_from_step is not None else
            "resample_each_step"
        ) if args.optimizer == "gnome" else None,
        "device": str(device),
        **{f"opt.{k}": v for k, v in opt_cfg.items()},
    }
    run = RunLogger(
        experiment=EXPERIMENT,
        optimizer=args.optimizer,
        seed=args.seed,
        hyperparameters=hyperparameters,
        runs_dir=args.runs_dir,
    )

    # Optional optimizer-internals log. Kept in its own file rather than as
    # extra records in the run's JSONL: it is one record per *parameter* per
    # logged step, so it would outnumber the training records several times
    # over and slow load_run() down for everyone not looking at it.
    diag = None
    if args.diagnostics_every > 0:
        if args.optimizer != "gnome":
            raise SystemExit(
                f"--diagnostics-every is Gnome-only; --optimizer "
                f"{args.optimizer} exposes no diagnostics hook."
            )
        diag_params = (
            None if not args.diagnostics_params
            else [int(s) for s in args.diagnostics_params.split(",")]
        )
        diag_path = os.path.join(
            os.path.dirname(run.path) or ".", f"{run.run_id}.diag.jsonl"
        )
        diag = JsonlDiagnostics(diag_path, params=diag_params)

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | params={n_params:,} | "
            f"5+1D unit-ball heat | diffusivity={DIFFUSIVITY} | "
            f"dtype={dtype_name} | device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}",
            flush=True,
        )
    eval_coords, u_ref = make_eval_set(
        args.n_eval, device, dtype, seed=args.eval_seed)

    if diag is not None and not args.quiet:
        print(f"  diagnostics every {args.diagnostics_every} steps "
              f"→ {diag.path}", flush=True)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_pde = last_ic = last_bc = float("nan")
    best_avg = best_rel_l2 = float("inf")

    fixed_batch = None
    for step in range(args.steps):
        quasi_newton = hybrid and step >= args.soap_steps
        if hybrid and step == args.soap_steps:
            opt = pinn.build_ssbroyden_phase(model, args)
            scheduler = None
            run.log("optimizer", step=step, optimizer="ssbroyden")
        if fixed_batch is None:
            main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device, dtype)
            if fixed_from_step is not None and step >= fixed_from_step:
                fixed_batch = main_batch
                # Auxiliary minimum sizes may exceed small main blocks.
                n_pde_aux = min(args.n_pde, n_pde_aux)
                n_ic_aux = min(args.n_ic, n_ic_aux)
                n_bc_aux = min(args.n_bc, n_bc_aux)
                run.log("sampling", step=step, sampling="fixed",
                        n_pde_aux=n_pde_aux, n_ic_aux=n_ic_aux, n_bc_aux=n_bc_aux)
                if not args.quiet:
                    print(f"[{EXPERIMENT}] after {step} steps: training dataset fixed", flush=True)
        else:
            main_batch = fixed_batch
        if args.optimizer == "gnome":
            aux_batch = (sample_aux_from_batch(main_batch, n_pde_aux, n_ic_aux, n_bc_aux)
                         if fixed_batch is not None else
                         sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device, dtype))

            def main_closure():
                r = stacked_residuals(model, main_batch)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch)
                return r, torch.zeros_like(r)

            loss = opt.step(
                main_closure, aux_closure,
                surrogate_partition_sizes=(
                    residual_partition_sizes(aux_batch) if args.partition_residuals else None
                ),
            )
        elif args.optimizer == "engdw":
            loss = opt.step(main_batch)
        elif quasi_newton:
            loss = pinn.refinement_step(opt, model, main_batch, stacked_residuals)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch)
            # Match Gnome's internal MSE reduction: sum-of-squares / N.
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if (diag is not None and args.optimizer == "gnome"
                and (step + 1) % args.diagnostics_every == 0):
            for record in opt.get_diagnostics(params=diag.params):
                diag(record)

        applied_lr = current_lr(opt)
        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if diverged(loss_val):
            if diag is not None:
                diag.close()
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val, lr=applied_lr, **pinn.phase_metrics(args, step))
        window.append(loss_val)

        if (step + 1 == args.steps or (hybrid and step + 1 == args.soap_steps) or
                (args.log_every and (step + 1) % args.log_every == 0)):
            tl = term_losses(model, sample_batch(
                args.n_pde, args.n_ic, args.n_bc, device, dtype))
            rl2 = eval_rel_l2(model, eval_coords, u_ref)
            last_avg = sum(window) / len(window)
            last_pde, last_ic, last_bc = tl["pde"], tl["ic"], tl["bc"]
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        pde=tl["pde"], ic=tl["ic"], bc=tl["bc"], rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  ic={tl['ic']:.3e}  "
                    f"bc={tl['bc']:.3e}  rel_l2={rl2:.3e}  "
                    f"{ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )

    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final pde={last_pde:.3e}  ic={last_ic:.3e}  bc={last_bc:.3e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
