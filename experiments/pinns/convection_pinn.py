"""1D convection PINN: AdamW vs SOAP vs Gnome vs ENGD-W — the canonical PINN failure.

PDE:  u_t + β·u_x = 0,    x ∈ [0, 2π],  t ∈ [0, 1]
IC:   u(0, x) = sin(x)
BC:   u(t, 0) = u(t, 2π)    (periodic)

Exact solution ``u(t, x) = sin(x - βt)`` — pure translation at speed β, no
diffusion, no shock, no stiffness in the usual sense. A first-order linear PDE
in one space dimension: the cheapest problem in this suite by a wide margin,
one first derivative in each of t and x per collocation point.

**Why it is here.** Krishnapriyan et al. (2021) made this the canonical PINN
failure mode: the PDE is trivial, but past ``β ≈ 20`` a standard tanh MLP
trained with Adam does not solve it at all — it converges happily to a wrong
answer. The failure is in the optimization, not the approximation: a network
of this size *can* represent ``sin(x - βt)`` to several digits, and the loss
landscape simply stops being navigable by first-order methods as β grows.
That makes it the sharpest possible test of an optimizer claim. Every other
benchmark here asks "how many digits"; this one asks "does it work at all",
and the standard recipe visibly answers no.

Published relative L2 for β = 30-40 (Rathore et al. 2024, *Challenges in
Training PINNs: A Loss Landscape Perspective*):

    Adam            5.96e-2
    L-BFGS          8.26e-3
    Adam + L-BFGS   4.19e-3
    NNCG            1.94e-3

**Do not quote our numbers against that table without matching the setup
first.** This file implements the standard Krishnapriyan problem — domain, IC,
periodic BC and analytic solution are all fixed by the paper — but the
architecture, collocation sampling and step budget follow *this repo's*
conventions (random uniform sampling, the shared ``MLP``/``ModifiedMLP``
trunks) rather than being copied from Rathore. Those choices move the number.
The table is here as an order-of-magnitude sanity anchor: "first-order methods
land around 1e-2, second-order methods around 1e-3". A run that reports 1e-1 is
failing the way the paper describes; a run that reports 1e-4 has beaten
everything in the table and deserves a hard look for a bug before it is
believed. See ``--beta`` for the one knob that matters most.

Two or three residual blocks depending on ``--embed``: PDE (collocation),
IC, and — unless the periodic embedding makes it exact — the periodic BC.
Stacked through ``gnome.stack_residuals`` so the multi-block MSE rides
Gnome's single-MSE surrogate as the per-block independent Rademacher GGN
estimator.

ENGD-W uses the shared dense Woodbury solver, float64 on CPU/CUDA, and
no weight decay or learning-rate schedule. Its residuals are normalized by
sqrt(block size), preserving the sum of block MSEs. Use small sample counts:
the dense sample Gram has n_pde + n_ic (+ n_bc) rows. ``--engdw-chunk`` limits
Jacobian differentiation workspace, not the full Jacobian or Gram storage.
``--engdw-fixed`` reuses training points; validation still samples fresh points.

Usage:

    uv run -m experiments.pinns.convection_pinn --optimizer gnome --beta 40
    uv run -m experiments.pinns.convection_pinn --optimizer adamw --beta 40
    uv run -m experiments.pinns.convection_pinn --optimizer engdw --beta 40 \\
        --n-pde 128 --n-ic 32 --n-bc 32 --engdw-line-search --steps 1000
    uv run -m experiments.pinns.convection_pinn --optimizer gnome --beta 40 \\
        --embed periodic --arch modified
    uv run -m experiments.pinns.convection_pinn --optimizer gnome \\
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
from torch.func import functional_call, jacrev, vmap

from gnome import Gnome, JsonlDiagnostics, stack_residuals
from experiments.baselines import ENGDW, SOAP
from experiments.common import (
    DIVERGED_EXIT,
    ConcatEmbedding,
    FusedMLP,
    FusedModifiedMLP,
    MLP,
    ModifiedMLP,
    diverged,
    RunLogger,
    cosine_scheduler,
    current_lr,
    pick_device,
    PeriodicEmbedding,
)


EXPERIMENT = "convection_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 2.0 * math.pi


# ========================= Models =========================


def build_embedding(embed: str) -> nn.Module:
    if embed == "none":
        return ConcatEmbedding(2)
    if embed == "periodic":
        return PeriodicEmbedding(
            2, wavenumber=2.0 * math.pi / (X_MAX - X_MIN)
        )
    raise ValueError(f"unknown embedding: {embed}")


def build_model(arch: str, embed: nn.Module, hidden: int, depth: int,
                fuse_every: int = 0) -> nn.Module:
    """``(t, x) → u``. The input embedding is whatever ``--embed`` selects.

    ``fused`` and ``fused-modified`` are the same function classes as ``mlp``
    and ``modified`` respectively — they differ only in how the weights are
    grouped into parameter tensors, which is what Gnome preconditions over.
    ``--fuse-every 1`` is the control for both.
    """
    if arch == "mlp":
        return MLP(embed, hidden=hidden, depth=depth)
    if arch == "modified":
        return ModifiedMLP(embed, hidden=hidden, depth=depth)
    if arch == "fused":
        return FusedMLP(embed, hidden=hidden, depth=depth,
                        fuse_every=fuse_every)
    if arch == "fused-modified":
        return FusedModifiedMLP(embed, hidden=hidden, depth=depth,
                                fuse_every=fuse_every)
    raise ValueError(f"unknown arch: {arch}")


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor, beta: float
) -> torch.Tensor:
    """Convection residual ``u_t + β·u_x`` at (t, x).

    First order in both variables, so one backward pass each — no
    ``u_xx``, which is why this benchmark is so much cheaper per step than
    the diffusion problems in this suite.
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return u_t + beta * u_x


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual: ``u(0, x) - sin(x)``."""
    t0 = torch.zeros_like(x)
    return model(t0, x) - torch.sin(x)


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Periodic BC residual ``u(t, 0) - u(t, 2π)``.

    C⁰ only: the PDE is first order, so matching ``u`` across the endpoints
    is the whole boundary condition — unlike the second-order problems here,
    which also need ``u_x`` to match. Only used with ``--embed none``; the
    periodic embedding makes this identically zero.
    """
    x_l = torch.full_like(t, X_MIN)
    x_r = torch.full_like(t, X_MAX)
    return model(t, x_l) - model(t, x_r)


# ========================= Sampling =========================

def sample_batch(
    n_pde: int, n_ic: int, n_bc: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent uniform draws for collocation / IC / BC point sets."""
    t_pde = torch.rand(n_pde, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x_pde = torch.rand(n_pde, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    x_ic = torch.rand(n_ic, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t_bc = torch.rand(n_bc, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    return t_pde, x_pde, x_ic, t_bc


def stacked_residuals(
    model: nn.Module, batch, beta: float, use_bc: bool = True
) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights)."""
    t_pde, x_pde, x_ic, t_bc = batch
    blocks = [
        pde_residual(model, t_pde, x_pde, beta),
        ic_residual(model, x_ic),
    ]
    if use_bc:
        blocks.append(bc_residual(model, t_bc))
    return stack_residuals(blocks)


def term_losses(model: nn.Module, batch, beta: float) -> dict[str, float]:
    """Per-block MSE for diagnostic logging.

    ``bc`` is always reported, even under ``--embed periodic`` where it is
    not part of the loss — there it is the check that the embedding is doing
    its job, and should sit at float32 round-off.
    """
    t_pde, x_pde, x_ic, t_bc = batch
    return {
        "pde": pde_residual(model, t_pde, x_pde, beta).pow(2).mean().item(),
        "ic": ic_residual(model, x_ic).pow(2).mean().item(),
        "bc": bc_residual(model, t_bc).pow(2).mean().item(),
    }


# ========================= Reference solution + eval =========================

def convection_reference(
    beta: float, nt: int = 101, nx: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic reference ``u(t, x) = sin(x - βt)`` on a uniform grid.

    No solver and no cached data file — the method of characteristics gives
    the solution in closed form, which is part of why this problem isolates
    the optimizer so cleanly. The x grid excludes the duplicate right
    endpoint (``2π ≡ 0``) so periodic points are not double-counted.
    """
    t = torch.linspace(T_MIN, T_MAX, nt)
    x = torch.linspace(X_MIN, X_MAX, nx + 1)[:-1]
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    u = torch.sin(xx - beta * tt)
    return t, x, u


def eval_rel_l2(
    model: nn.Module,
    t_ref: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor,
    device: torch.device, batch_size: int = 8192,
) -> float:
    """Relative L2 against the analytic reference on its grid."""
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
                model(t_flat[i:i + batch_size], x_flat[i:i + batch_size]).cpu()
            )
    if was_training:
        model.train()
    u_pred = torch.cat(preds).reshape(nt, nx)
    num = (u_pred - u_ref).pow(2).sum().sqrt()
    den = u_ref.pow(2).sum().sqrt()
    return float(num / den)


# ========================= Optimizer factory =========================

def make_engdw_functions(
    model: nn.Module, beta: float, use_bc: bool = True, chunk_size: int | None = 32,
):
    """Return the normalized ENGD-W residual and its per-sample Jacobian.

    Its squared norm equals ``stacked_residuals(...).square().mean()``.
    ENGD-W uses half this value as its energy; damping therefore belongs to
    J.T @ J for this normalization, independently of total batch size.

    torch.func input differentiation allows jacrev to differentiate through
    both u_t and u_x in the convection residual u_t + beta * u_x.
    """
    def u_single(params, tx):
        return functional_call(
            model, params, (tx[0:1].reshape(1, 1), tx[1:2].reshape(1, 1))
        ).squeeze()

    du = jacrev(u_single, argnums=1)

    def pde_single(params, tx):
        derivatives = du(params, tx)
        return derivatives[0] + beta * derivatives[1]

    def ic_single(params, x):
        tx = torch.cat((torch.zeros_like(x), x))
        return u_single(params, tx) - torch.sin(x).squeeze()

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
        blocks = [r_pde, (u_ic - torch.sin(x_ic)).reshape(-1)]
        if use_bc:
            u_l = functional_call(model, params, (t_bc, torch.full_like(t_bc, X_MIN)))
            u_r = functional_call(model, params, (t_bc, torch.full_like(t_bc, X_MAX)))
            blocks.append((u_l - u_r).reshape(-1))
        return torch.cat([r / math.sqrt(r.numel()) for r in blocks]).unsqueeze(1)

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
        for fn, points in blocks:
            block = vmap(jacrev(fn), in_dims=(None, 0), chunk_size=chunk_size)(
                params, points
            )
            jacobians.append({k: v / math.sqrt(len(points)) for k, v in block.items()})
        return {k: torch.cat([block[k] for block in jacobians]) for k in params}

    return residual, jacobian


def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    warmup: int, total_steps: int, cosine_decay: float, eps: float = 1e-6,
    beta1: float = 0.9, beta2: float = 0.99,
    trust_region: float = 1.0,
):
    """Construct the optimizer and its LR schedule.

    Returns ``(optimizer, config, scheduler)``. Every optimizer gets the same
    linear-warmup + cosine-decay schedule; ``cosine_decay`` is the final-lr
    fraction (0.0 -> decay to zero, 1.0 -> warmup then constant).
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=10,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse", precondition_1d=True,
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
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw", "engdw"])
    p.add_argument("--beta", type=float, default=40.0,
                   help="Convection speed. THE knob for this benchmark: the "
                        "PDE is trivial at every value, but first-order "
                        "optimizers start failing around 20 and are "
                        "comprehensively broken by 30-40. The published "
                        "comparison numbers in the module docstring are for "
                        "30-40. Sweep it to draw the failure curve — that "
                        "curve is the result, more than any single number.")
    p.add_argument("--arch",
                   choices=["mlp", "modified", "fused", "fused-modified"],
                   default="mlp",
                   help="Network. Defaults to the plain MLP, which is what "
                        "the published failure results use — switching to "
                        "'modified' changes the architecture as well as the "
                        "optimizer and muddies the comparison.")
    p.add_argument("--fuse-every", type=int, default=0,
                   help="--arch fused / fused-modified only: consecutive "
                        "hidden layers per parameter tensor. 1 = one tensor "
                        "per layer, the control every larger value is "
                        "compared against. 0 (default) fuses the whole stack.")
    p.add_argument("--embed", choices=["none", "periodic"], default="none",
                   help="Input embedding. 'none' (default) feeds raw [t, x] "
                        "and imposes periodicity softly, as a third residual "
                        "block — this is the published setup. 'periodic' "
                        "feeds [t, cos x, sin x], making periodicity exact "
                        "and DROPPING the BC block; note this also hands the "
                        "network a basis in which sin(x - beta t) is linear, "
                        "so it is not a like-for-like comparison.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50_000)
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=256)
    p.add_argument("--n-bc", type=int, default=256)
    p.add_argument("--aux-frac", type=float, default=0.05,
                   help="Aux batch fraction for Gnome's curvature surrogate.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome trust region: lambda is set to the smallest "
                        "value with ||m/(v+lambda)||_2 <= this * sqrt(P). "
                        "0 disables the solve.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature damping in m/(v+eps). Gnome only.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA for Gnome, SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument(
        "--engdw-damping", type=float, default=1e-6,
        help="ENGD-W fixed damping in JJ^T + lambda I; convection starting "
             "value, not a tuned result. No weight decay or LR schedule is used.",
    )
    p.add_argument(
        "--engdw-momentum", type=float, default=0.0,
        help="ENGD-W SPRING-style regularization-center momentum in [0, 1). "
             "Zero preserves the original solver; try 0.9, 0.95, or 0.99.",
    )
    p.add_argument(
        "--engdw-line-search", action="store_true",
        help="Choose the lowest same-batch loss on 13 log-spaced step sizes "
             "from 1e-3 to 1, overriding --lr (same grid as Poisson-5D).",
    )
    p.add_argument(
        "--engdw-fixed", action="store_true",
        help="ENGD-W only: sample PDE/IC/BC training points once and reuse "
             "them every step. Validation points remain independently sampled.",
    )
    p.add_argument(
        "--engdw-chunk", type=int, default=32,
        help="Samples per vectorized Jacobian chunk for ENGD-W; limits differentiation "
             "workspace, while the full Jacobian and sample Gram remain dense.",
    )
    p.add_argument("--hidden", type=int, default=50,
                   help="Network width. Default 50 follows Krishnapriyan.")
    p.add_argument("--depth", type=int, default=5,
                   help="Linear-layer count for --arch mlp/fused (4 hidden "
                        "layers of width 50 is the published net); gated "
                        "hidden-layer count for --arch modified.")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the cosine decay: 0.0 decays "
                        "to zero, 1.0 disables decay.")
    p.add_argument("--float64", action="store_true",
                   help="Run in double precision on CPU. ENGD-W automatically uses "
                        "float64 on CPU/CUDA without this flag. Forces device=cpu "
                        "(MPS has no float64) and sets the global default "
                        "dtype, so params, collocation samples and the "
                        "analytic reference are all double. Worth having "
                        "here because the PDE residual carries a factor of "
                        "beta: at beta=40 any error in u_x is amplified 40x "
                        "before it reaches the loss, so the residual hits "
                        "the float32 floor while u itself still looks clean. "
                        "Costs roughly an order of magnitude in wall time.")
    p.add_argument("--diagnostics-every", type=int, default=0,
                   help="Log Gnome's internal state every N steps to a "
                        "sibling {run_id}.diag.jsonl. 0 disables. Gnome only.")
    p.add_argument("--diagnostics-params", type=str, default=None,
                   help="Comma-separated parameter indices to log.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> str:
    if args.engdw_fixed and args.optimizer != "engdw":
        raise SystemExit("--engdw-fixed requires --optimizer engdw")
    if args.optimizer == "engdw":
        if not math.isfinite(args.engdw_momentum) or not 0 <= args.engdw_momentum < 1:
            raise SystemExit("--engdw-momentum must be finite and in [0, 1)")
        if not math.isfinite(args.engdw_damping) or args.engdw_damping <= 0:
            raise SystemExit("--engdw-damping must be finite and positive")
        if args.engdw_chunk < 1:
            raise SystemExit("--engdw-chunk must be positive")
        if not math.isfinite(args.lr) or args.lr < 0:
            raise SystemExit("--lr must be finite and non-negative")
        if min(args.n_pde, args.n_ic, args.n_bc, args.steps) < 1:
            raise SystemExit("sample counts and --steps must be positive")
        if args.diagnostics_every > 0:
            raise SystemExit("--diagnostics-every is Gnome-only")

    # float64 must come before any tensor/model construction so params,
    # samples and the reference grid are all double. Explicit --float64
    # retains the CPU behavior; ENGD-W also allows CUDA, falling back from MPS.
    if args.float64:
        torch.set_default_dtype(torch.float64)
        device = torch.device("cpu")
    elif args.optimizer == "engdw":
        torch.set_default_dtype(torch.float64)
        device = pick_device()
        if device.type == "mps":
            device = torch.device("cpu")
    else:
        device = pick_device()
    torch.manual_seed(args.seed)

    use_bc = args.embed != "periodic"
    model = build_model(
        args.arch, build_embedding(args.embed),
        args.hidden, args.depth, args.fuse_every,
    ).to(device)
    if args.optimizer == "engdw":
        residual_fn, jacobian_fn = make_engdw_functions(
            model, args.beta, use_bc, chunk_size=args.engdw_chunk,
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
    else:
        opt, opt_cfg, scheduler = build_optimizer(
            args.optimizer, model.parameters(), args.lr, args.weight_decay,
            warmup=args.warmup_steps, total_steps=args.steps,
            cosine_decay=args.cosine_decay, eps=args.eps,
            beta1=args.beta1, beta2=args.beta2,
            trust_region=args.trust_region,
        )

    n_pde_aux = max(1, int(args.n_pde * args.aux_frac))
    n_ic_aux = max(1, int(args.n_ic * args.aux_frac))
    n_bc_aux = max(1, int(args.n_bc * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "beta": args.beta,
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
        "sampling": "fixed" if args.engdw_fixed else "resample_each_step",
        "diagnostics_every": args.diagnostics_every,
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
            f"[{EXPERIMENT}] {args.optimizer} | beta={args.beta:g} | "
            f"arch={args.arch} {args.depth}x{args.hidden} | "
            f"embed={args.embed} | params={n_params:,} | device={device} | "
            f"dtype={torch.get_default_dtype()}\n"
            f"  blocks={blocks} | N_pde={args.n_pde} | steps={args.steps}",
            flush=True,
        )
    t_ref, x_ref, u_ref = convection_reference(args.beta)

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    best_avg = best_rel_l2 = float("inf")

    fixed_batch = None
    for step in range(args.steps):
        if fixed_batch is None:
            main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
            if args.engdw_fixed:
                fixed_batch = main_batch
        else:
            main_batch = fixed_batch
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device)

            def main_closure():
                r = stacked_residuals(model, main_batch, args.beta, use_bc)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, args.beta, use_bc)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        elif args.optimizer == "engdw":
            loss = opt.step(main_batch)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, args.beta, use_bc)
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            if diag is not None:
                diag.close()
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.",
                  flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            tl = term_losses(
                model,
                sample_batch(args.n_pde, args.n_ic, args.n_bc, device),
                args.beta,
            )
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_avg = sum(window) / len(window)
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
                    f"bc={tl['bc']:.3e}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )
    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
