"""Heat 2D benchmark: AdamW vs SOAP vs Gnome vs ENGD-W.

Wang et al., arXiv:2502.00604v2, Table 2 calls this "Heat 2D" because the
network has two inputs (t, x). The underlying benchmark is Section 4.3 of
Muller & Zeinhofer (ICML 2023), not a two-spatial-dimensional heat equation:

    u_t = 0.25 u_xx,             (t, x) in [0, 1]^2
    u(0, x) = sin(pi x),         x in [0, 1]
    u(t, 0) = u(t, 1) = 0,       t in [0, 1]
    u_exact(t, x) = exp(-pi^2 t / 4) sin(pi x)

Sources:
https://arxiv.org/html/2502.00604v2#S5.T2
https://proceedings.mlr.press/v202/muller23b/muller23b.pdf (Section 4.3)

The training loop follows burgers2d_pinn: independent uniform PDE/IC/BC
samples each step, equal block MSE weights, Gnome auxiliary samples, shared
tanh MLP, optimizer-specific warmup/cosine schedules, and JSONL diagnostics.
The default [2, 256, 256, 256, 1] architecture is a Table 2 SOAP architecture.
Sampling counts, schedules and iteration budget are local experiment defaults;
this runner implements the benchmark, not the paper's full comparison protocol.
The analytic reference needs no download or numerical PDE solver.

ENGD-W uses float64, fixed damping, optional momentum and grid line search,
with no weight decay or LR schedule. Set optimizers['engdw']['engdw_fixed']
to reuse a sampled training batch. Validation remains independently sampled.
Start with n_pde=128, n_ic=32, n_bc=32: the dense sample-space solve scales
cubically with the total residual count. engdw_chunk chunks Jacobian evaluation,
not the full solve.

Usage (edit the mirrored CONFIG to select an optimizer or change settings)::

    uv run -m experiments.pinns.heat2d_pinn
    uv run -m experiments.pinns.heat2d_pinn --config path/to/config.py
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


EXPERIMENT = "heat2d_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 1.0
DIFFUSIVITY = 0.25


# ========================= Exact solution =========================

def exact_solution(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Analytic reference on the unit space-time square."""
    return torch.exp(-DIFFUSIVITY * math.pi**2 * t) * torch.sin(math.pi * x)


# ========================= Model =========================

class PINN(_SharedMLP):
    """Tanh MLP mapping (t, x) to u; depth counts all linear layers."""

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__(ConcatEmbedding(2), hidden=hidden, depth=depth)


# ========================= Residuals =========================

def pde_residual(model: nn.Module, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Heat residual u_t - 0.25 u_xx, differentiable in model parameters."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t - DIFFUSIVITY * u_xx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Initial value u(0, x) = sin(pi x)."""
    return model(torch.zeros_like(x), x) - torch.sin(math.pi * x)


def bc_residual(model: nn.Module, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Homogeneous Dirichlet residual at x=0 or x=1 (no terminal condition)."""
    return model(t, x)


# ========================= Sampling =========================

def sample_batch(n_pde: int, n_ic: int, n_bc: int,
                 device: torch.device, dtype: torch.dtype):
    """Uniform PDE/IC samples and n_bc points total over the two spatial ends."""
    kw = dict(device=device, dtype=dtype)
    t_pde = torch.rand(n_pde, 1, **kw)
    x_pde = torch.rand(n_pde, 1, **kw)
    x_ic = torch.rand(n_ic, 1, **kw)
    t_bc = torch.rand(n_bc, 1, **kw)
    x_bc = torch.randint(0, 2, (n_bc, 1), device=device).to(dtype)
    return (t_pde, x_pde), (x_ic,), (t_bc, x_bc)


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
    def u_single(params, tx):
        return functional_call(
            model, params, (tx[:1].reshape(1, 1), tx[1:].reshape(1, 1))
        ).squeeze()

    gradient = jacrev(u_single, argnums=1)
    hessian = jacrev(gradient, argnums=1)

    def pde_single(params, tx):
        return gradient(params, tx)[0] - DIFFUSIVITY * hessian(params, tx)[1, 1]

    def ic_single(params, tx):
        return u_single(params, tx) - torch.sin(math.pi * tx[1])

    def points(batch):
        pde, (x_ic,), bc = batch
        return (torch.cat(pde, 1),
                torch.cat((torch.zeros_like(x_ic), x_ic), 1), torch.cat(bc, 1))

    functions = (pde_single, ic_single, u_single)

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
    nt: int, nx: int, device: torch.device, dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Fixed nt by nx tensor-product grid and analytic reference."""
    kw = dict(device=device, dtype=dtype)
    t = torch.linspace(T_MIN, T_MAX, nt, **kw)
    x = torch.linspace(X_MIN, X_MAX, nx, **kw)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    coords = (tt.reshape(-1, 1), xx.reshape(-1, 1))
    return coords, exact_solution(*coords)


def eval_rel_l2(
    model: nn.Module, coords, u_ref: torch.Tensor, batch_size: int = 65536,
) -> float:
    """``||u_pred - u*||_2 / ||u*||_2`` over the fixed eval grid.

    Queried in batches under ``no_grad`` so the grid can be far larger than a
    training batch without OOM.
    """
    t, x = coords
    was_training = model.training
    model.eval()
    num = torch.zeros((), device=u_ref.device, dtype=u_ref.dtype)
    with torch.no_grad():
        for i in range(0, t.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            pred = model(t[sl], x[sl])
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
    for name in ("steps", "n_pde", "n_ic", "n_bc", "hidden"):
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    for name in ("depth", "n_eval_t", "n_eval_x"):
        if getattr(args, name) < 2:
            p.error(f"{name} must be at least 2")
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


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = torch.float64 if args.optimizer == "engdw" else getattr(torch, args.dtype)
    dtype_name = str(dtype).removeprefix("torch.")
    if device.type == "mps" and dtype is torch.float64:
        device = torch.device("cpu")            # MPS has no float64
        if not args.quiet:
            print(f"[{EXPERIMENT}] float64 requested; MPS has no float64 "
                  f"— falling back to CPU.", flush=True)

    model = PINN(hidden=args.hidden, depth=args.depth).to(device=device, dtype=dtype)
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
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
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
        "n_eval_t": args.n_eval_t,
        "n_eval_x": args.n_eval_x,
        "diffusivity": DIFFUSIVITY,
        "spatial_dimensions": 1,
        "sampling": "fixed" if args.engdw_fixed else "resample_each_step",
        "fixed_dataset": args.engdw_fixed,
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
            f"diffusivity={DIFFUSIVITY} | dtype={dtype_name} | device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}",
            flush=True,
        )
    eval_coords, u_ref = make_eval_set(
        args.n_eval_t, args.n_eval_x, device, dtype)

    if diag is not None and not args.quiet:
        print(f"  diagnostics every {args.diagnostics_every} steps "
              f"→ {diag.path}", flush=True)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_pde = last_ic = last_bc = float("nan")
    best_avg = best_rel_l2 = float("inf")

    fixed_batch = (sample_batch(args.n_pde, args.n_ic, args.n_bc, device, dtype)
                   if args.engdw_fixed else None)
    for step in range(args.steps):
        main_batch = fixed_batch if fixed_batch is not None else sample_batch(
            args.n_pde, args.n_ic, args.n_bc, device, dtype)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(
                n_pde_aux, n_ic_aux, n_bc_aux, device, dtype)

            def main_closure():
                r = stacked_residuals(model, main_batch)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        elif args.optimizer == "engdw":
            loss = opt.step(main_batch)
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
        run.log_train(step, loss=loss_val, lr=applied_lr)
        window.append(loss_val)

        if (step + 1 == args.steps or
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
