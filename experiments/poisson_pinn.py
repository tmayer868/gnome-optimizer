"""2D Poisson PINN: AdamW vs SOAP vs Gnome.

PDE:  -Δu = f(x, y),    (x, y) ∈ (0, 1)²
BC:   u = 0  on  ∂Ω    (Dirichlet)

Manufactured solution::

    u_exact(x, y) = sin(πx) sin(πy)
    f(x, y) = -Δu_exact = 2π² sin(πx) sin(πy)

This is the canonical "clean PINN" benchmark, deliberately picked here as a
sanity-check companion to Burgers:

* **Elliptic PDE, no time** — no propagation failure, no causal-training
  question, no IC vs PDE balance issue.
* **Unique solution** — the linear Poisson operator with Dirichlet zero BC
  has a single global minimizer, no multi-modality, no trivial-solution
  attractor.
* **Exact analytical reference** — no spectral solver to verify, no
  reference-accuracy floor to worry about.
* **Smooth low-frequency solution** — a small tanh MLP fits it, so any
  remaining differences in PINN rel_L2 are attributable to the optimizer's
  asymptotic behavior, not to architecture limitations.

Two-block residual: PDE (interior), BC (boundary). Stacked through
``gnome.stack_residuals`` so the multi-block MSE rides Gnome's single-MSE
surrogate as the per-block independent Rademacher GGN estimator.

All three optimizers share one plain tanh MLP so the only variable is the
optimizer. The baselines (SOAP, AdamW) get a linear-warmup + cosine-decay
schedule (``--cosine-decay`` sets the final-lr fraction; 1.0 disables it);
Gnome runs at a fixed lr — its Gauss-Newton step self-anneals as the residual
shrinks.

Usage:

    uv run -m experiments.poisson_pinn --optimizer gnome --seed 0
    uv run -m experiments.poisson_pinn --optimizer soap  --seed 0
    uv run -m experiments.poisson_pinn --optimizer adamw --seed 0
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.autograd as autograd
import torch.nn as nn

from gnome import Gnome, stack_residuals
from experiments.baselines import SOAP
from experiments.common import (
    RunLogger,
    baseline_cosine_scheduler,
    current_lr,
    pick_device,
)


EXPERIMENT = "poisson_pinn"

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
PI = math.pi
SOURCE_COEFF = 2.0 * PI * PI


# ========================= Model =========================

class PINN(nn.Module):
    """Maps ``(x, y) → u`` via a plain tanh MLP."""

    def __init__(self, hidden: int = 64, depth: int = 5):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


# ========================= Residuals =========================

def _source_term(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """RHS of the Poisson equation: ``f(x, y) = 2π² sin(πx) sin(πy)``."""
    return SOURCE_COEFF * torch.sin(PI * x) * torch.sin(PI * y)


def pde_residual(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Poisson PDE residual ``Δu + f`` (target zero for ``-Δu = f``).

    Sign convention: writing the residual as ``u_xx + u_yy + f`` makes the
    minimization target zero — equivalent to ``-Δu = f`` because the
    Rademacher / Hutchinson surrogate is sign-invariant.
    """
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = model(x, y)
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy + _source_term(x, y)


def bc_residual(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Dirichlet BC residual: ``u(boundary) - 0 = u(boundary)``."""
    return model(x, y)


# ========================= Sampling =========================

def sample_batch(
    n_pde: int, n_bc_per_edge: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform interior draws + ``n_bc_per_edge`` points on each of 4 edges."""
    x_pde = torch.rand(n_pde, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y_pde = torch.rand(n_pde, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN

    s = torch.rand(n_bc_per_edge, 1, device=device)
    # Four edges of the unit square.
    x_left  = torch.full_like(s, X_MIN);    y_left  = s * (Y_MAX - Y_MIN) + Y_MIN
    x_right = torch.full_like(s, X_MAX);    y_right = s * (Y_MAX - Y_MIN) + Y_MIN
    x_bot   = s * (X_MAX - X_MIN) + X_MIN;  y_bot   = torch.full_like(s, Y_MIN)
    x_top   = s * (X_MAX - X_MIN) + X_MIN;  y_top   = torch.full_like(s, Y_MAX)

    x_bc = torch.cat([x_left, x_right, x_bot, x_top], dim=0)
    y_bc = torch.cat([y_left, y_right, y_bot, y_top], dim=0)
    return x_pde, y_pde, x_bc, y_bc


def stacked_residuals(model: nn.Module, batch) -> torch.Tensor:
    """PDE + BC residuals stacked via ``stack_residuals`` (equal weights)."""
    x_pde, y_pde, x_bc, y_bc = batch
    return stack_residuals([
        pde_residual(model, x_pde, y_pde),
        bc_residual(model, x_bc, y_bc),
    ])


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-block MSE for diagnostic logging."""
    x_pde, y_pde, x_bc, y_bc = batch
    return {
        "pde": pde_residual(model, x_pde, y_pde).pow(2).mean().item(),
        "bc": bc_residual(model, x_bc, y_bc).pow(2).mean().item(),
    }


# ========================= Reference solution + eval =========================

def poisson_reference(
    nx: int = 128, ny: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytical reference ``u(x, y) = sin(π x) sin(π y)`` on a uniform grid.

    No solver, no caching — the closed-form expression is evaluated directly.
    Grid spans the full closed unit square including the Dirichlet boundary
    (where ``u = 0`` by construction of the IC).
    """
    x = torch.linspace(X_MIN, X_MAX, nx)
    y = torch.linspace(Y_MIN, Y_MAX, ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    u = torch.sin(PI * xx) * torch.sin(PI * yy)
    return x, y, u


def eval_rel_l2(
    model: nn.Module,
    x_ref: torch.Tensor, y_ref: torch.Tensor, u_ref: torch.Tensor,
    device: torch.device, batch_size: int = 8192,
) -> float:
    """Relative L2 against the analytical reference on its grid."""
    nx, ny = u_ref.shape
    xx, yy = torch.meshgrid(x_ref, y_ref, indexing="ij")
    x_flat = xx.reshape(-1, 1).to(device)
    y_flat = yy.reshape(-1, 1).to(device)
    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x_flat.shape[0], batch_size):
            preds.append(
                model(x_flat[i:i + batch_size], y_flat[i:i + batch_size]).cpu()
            )
    if was_training:
        model.train()
    u_pred = torch.cat(preds).reshape(nx, ny)
    num = (u_pred - u_ref).pow(2).sum().sqrt()
    den = u_ref.pow(2).sum().sqrt()
    return float(num / den)


# ========================= Optimizer factory =========================

def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    warmup: int, total_steps: int, cosine_decay: float,
):
    """Construct the optimizer and its LR schedule.

    Returns ``(optimizer, config, scheduler_or_None)``. Gnome runs at a fixed
    lr (its Gauss-Newton step self-anneals as the residual shrinks) so it gets
    no scheduler — only its own internal warmup. SOAP and AdamW get the
    standard linear-warmup + cosine-decay treatment; ``cosine_decay`` is the
    final-lr fraction (0.0 → decay to zero, 1.0 → decay disabled).
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.95, 0.99), shampoo_beta=0.99, eps=1e-4,
            precondition_frequency=50, aux_batch_size=10,
            clip=1.0, warmup=warmup,
            loss="mse", precondition_1d=True,
        )
        return Gnome(params, **cfg), cfg, None
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(0.95, 0.99), shampoo_beta=0.99, eps=1e-8,
            precondition_frequency=50, precondition_1d=True,
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

    scheduler = baseline_cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    cfg["warmup"] = warmup
    cfg["cosine_decay_floor"] = cosine_decay
    return opt, cfg, scheduler


# ========================= CLI / training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000,
                   help="Default 50k — Poisson converges much faster than "
                        "evolution PDEs; usually plateaus within 20-30k.")
    p.add_argument("--n-pde", type=int, default=2000)
    p.add_argument("--n-bc-per-edge", type=int, default=50,
                   help="Boundary points per edge (total BC sample is 4× this).")
    p.add_argument("--aux-frac", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=64, help="MLP width.")
    p.add_argument("--depth", type=int, default=5, help="MLP depth.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps. For the SOAP/AdamW baselines "
                        "this is the schedule warmup; for Gnome it is passed "
                        "as its internal `warmup=`.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    model = PINN(hidden=args.hidden, depth=args.depth).to(device)
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        warmup=args.warmup_steps, total_steps=args.steps,
        cosine_decay=args.cosine_decay,
    )

    n_pde_aux = max(1, int(args.n_pde * args.aux_frac))
    n_bc_aux_per_edge = max(1, int(args.n_bc_per_edge * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_params": n_params,
        "n_pde": args.n_pde,
        "n_bc_per_edge": args.n_bc_per_edge,
        "n_pde_aux": n_pde_aux,
        "n_bc_aux_per_edge": n_bc_aux_per_edge,
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

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | params={n_params:,} | "
            f"device={device}\n"
            f"  N_pde={args.n_pde} N_bc_per_edge={args.n_bc_per_edge} | "
            f"aux={n_pde_aux}/{n_bc_aux_per_edge} | steps={args.steps}",
            flush=True,
        )
    x_ref, y_ref, u_ref = poisson_reference()

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(args.n_pde, args.n_bc_per_edge, device)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_bc_aux_per_edge, device)

            def main_closure():
                r = stacked_residuals(model, main_batch)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch)
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            tl = term_losses(
                model, sample_batch(args.n_pde, args.n_bc_per_edge, device)
            )
            rl2 = eval_rel_l2(model, x_ref, y_ref, u_ref, device)
            last_avg = sum(window) / len(window)
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        pde=tl["pde"], bc=tl["bc"], rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  bc={tl['bc']:.3e}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
