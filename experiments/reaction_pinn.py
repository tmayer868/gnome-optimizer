"""1D reaction PINN: AdamW vs SOAP vs Gnome.

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
This repo resamples collocation points every optimizer step, whereas the
paper's released implementation samples a fixed set, so the numbers are an
order-of-magnitude anchor rather than a bit-for-bit reproduction.

``--embed periodic`` is an ablation that feeds [t, cos(x), sin(x)]. It makes
the BC exact and drops that residual block. It is not the published setup and
slightly changes the function class: the Gaussian agrees in value at the two
endpoints but its periodic extension is not C1 there. The PDE contains no
x-derivatives, so only C0 agreement is part of the canonical problem.

Usage:

    uv run -m experiments.reaction_pinn --optimizer gnome --rho 5
    uv run -m experiments.reaction_pinn --optimizer adamw --rho 5
    uv run -m experiments.reaction_pinn --optimizer soap --rho 10
    uv run -m experiments.reaction_pinn --optimizer gnome --rho 5 \
        --arch fused-modified --fuse-every 2 --depth 6
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.autograd as autograd
import torch.nn as nn

from experiments.baselines import SOAP
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
)
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

class RawEmbed(nn.Module):
    """Raw ``[t, x]`` inputs; periodicity is imposed as a soft block."""

    out_dim = 2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, x], dim=1)


class PeriodicEmbed(nn.Module):
    """``[t, cos(x), sin(x)]``; exactly 2*pi-periodic in x."""

    out_dim = 3

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, torch.cos(x), torch.sin(x)], dim=1)


def build_embedding(name: str) -> nn.Module:
    if name == "none":
        return RawEmbed()
    if name == "periodic":
        return PeriodicEmbed()
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


def stacked_residuals(
    model: nn.Module, batch, rho: float, use_bc: bool = True
) -> torch.Tensor:
    """Stack equal-weight PDE, IC, and optional periodic-BC blocks."""
    t_pde, x_pde, x_ic, t_bc = batch
    blocks = [
        pde_residual(model, t_pde, x_pde, rho),
        ic_residual(model, x_ic),
    ]
    if use_bc:
        blocks.append(bc_residual(model, t_bc))
    return stack_residuals(blocks)


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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--optimizer", required=True, choices=["gnome", "soap", "adamw"]
    )
    p.add_argument(
        "--rho", type=float, default=5.0,
        help="Reaction coefficient. Vanilla whole-domain PINNs are already "
             "near 100%% relative error at rho=5; rho=10 is the harder "
             "published rung.",
    )
    p.add_argument(
        "--arch",
        choices=["mlp", "modified", "fused", "fused-modified"],
        default="mlp",
        help="The plain MLP matches the published benchmark; other choices "
             "are architecture or tensor-grouping ablations.",
    )
    p.add_argument(
        "--fuse-every", type=int, default=0,
        help="Fused architectures only: hidden layers per parameter tensor. "
             "One is the per-layer control; zero fuses the whole stack.",
    )
    p.add_argument(
        "--embed", choices=["none", "periodic"], default="none",
        help="Raw inputs plus a soft BC (published setup), or an exact "
             "periodic embedding that drops the BC block.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=256)
    p.add_argument("--n-bc", type=int, default=256)
    p.add_argument(
        "--aux-frac", type=float, default=0.05,
        help="Fraction of each residual block used by Gnome's auxiliary "
             "curvature pass.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--trust-region", type=float, default=1.0,
        help="Gnome trust radius; zero disables the trust-region solve.",
    )
    p.add_argument(
        "--eps", type=float, default=1e-6,
        help="Gnome curvature damping. Baselines retain their own epsilon.",
    )
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument(
        "--hidden", type=int, default=50,
        help="Network width; 50 matches the published model.",
    )
    p.add_argument(
        "--depth", type=int, default=5,
        help="Linear-layer count for MLP/fused (four hidden layers at the "
             "default); gated hidden-layer count for modified variants.",
    )
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument(
        "--cosine-decay", type=float, default=0.0,
        help="Final learning-rate fraction after cosine decay.",
    )
    p.add_argument(
        "--float64", action="store_true",
        help="Use float64, falling back to CPU when the selected device is "
             "MPS.",
    )
    p.add_argument(
        "--diagnostics-every", type=int, default=0,
        help="Write Gnome internals every N steps; zero disables it.",
    )
    p.add_argument(
        "--diagnostics-params", type=str, default=None,
        help="Comma-separated parameter indices for Gnome diagnostics.",
    )
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    if args.rho < 0.0:
        raise SystemExit("--rho must be non-negative")
    if args.aux_frac <= 0.0:
        raise SystemExit("--aux-frac must be positive")

    device = pick_device()
    if args.float64:
        torch.set_default_dtype(torch.float64)
        if device.type == "mps":
            if not args.quiet:
                print(
                    f"[{EXPERIMENT}] --float64: MPS has no double support; "
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
    )

    n_pde_aux = max(1, int(args.n_pde * args.aux_frac))
    n_ic_aux = max(1, int(args.n_ic * args.aux_frac))
    n_bc_aux = max(1, int(args.n_bc * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "rho": args.rho,
        "arch": args.arch,
        "fuse_every": getattr(model, "chunks", None),
        "embed": args.embed,
        "blocks": ["pde", "ic", "bc"] if use_bc else ["pde", "ic"],
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
        opt.diagnostics = diag
        opt.diagnostics_every = args.diagnostics_every

    if not args.quiet:
        blocks = "pde+ic+bc" if use_bc else "pde+ic (exact periodic BC)"
        print(
            f"[{EXPERIMENT}] {args.optimizer} | rho={args.rho:g} | "
            f"arch={args.arch} {args.depth}x{args.hidden} | "
            f"embed={args.embed} | params={n_params:,} | device={device} | "
            f"dtype={torch.get_default_dtype()}\n"
            f"  blocks={blocks} | N_pde={args.n_pde} | steps={args.steps}",
            flush=True,
        )

    t_ref, x_ref, u_ref = reaction_reference(args.rho)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(
            args.n_pde, args.n_ic, args.n_bc, device
        )
        if args.optimizer == "gnome":
            aux_batch = sample_batch(
                n_pde_aux, n_ic_aux, n_bc_aux, device
            )

            def main_closure():
                r = stacked_residuals(model, main_batch, args.rho, use_bc)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, args.rho, use_bc)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, args.rho, use_bc)
            loss = r.square().sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            if diag is not None:
                diag.close()
            print(
                f"[{EXPERIMENT}] diverged at step {step}; stopping.",
                flush=True,
            )
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        should_log = args.log_every and (step + 1) % args.log_every == 0
        is_last = step + 1 == args.steps
        if should_log or is_last:
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
