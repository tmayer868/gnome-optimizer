"""2D Poisson PINN: AdamW vs SOAP vs Gnome vs ENGD-W.

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

Gnome, SOAP, and AdamW get the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then constant,
which suits Gnome on MSE since its step self-anneals as the residual shrinks).

ENGD-W uses the shared exact damped sample-space solve in float64, with a
fixed learning rate or same-batch grid line search and no schedule/weight decay.
Points are resampled each step unless ``--fixed-dataset`` is supplied.
This flag works for every optimizer. Gnome draws fresh auxiliary subsets from
the fixed main points each step. ``--engdw-fixed`` remains an ENGD-W-only
compatibility flag.

Usage:

    uv run -m experiments.pinns.poisson_pinn --optimizer gnome --seed 0
    uv run -m experiments.pinns.poisson_pinn --optimizer soap  --seed 0
    uv run -m experiments.pinns.poisson_pinn --optimizer adamw --seed 0
    uv run -m experiments.pinns.poisson_pinn --optimizer engdw --n-pde 128 \
        --n-bc-per-edge 16 --engdw-line-search --steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.func import functional_call, jacrev, vmap

from gnome import (
    Gnome,
    JsonlDiagnostics,
    format_records,
    measure_rho,
    stack_residuals,
)
from experiments.baselines import ENGDW, SOAP
from experiments.common import (
    DIVERGED_EXIT,
    ConcatEmbedding,
    FusedMLP,
    diverged,
    RunLogger,
    cosine_scheduler,
    current_lr,
    pick_device,
)
from experiments.common import MLP as _SharedMLP


EXPERIMENT = "poisson_pinn"

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
PI = math.pi
SOURCE_COEFF = 2.0 * PI * PI


# ========================= Model =========================

class PINN(_SharedMLP):
    """Maps ``(x, y) → u`` via a plain tanh MLP."""

    def __init__(self, hidden: int = 64, depth: int = 5):
        super().__init__(ConcatEmbedding(2), hidden=hidden, depth=depth)


def build_model(
    arch: str, hidden: int, depth: int, fuse_every: int = 0
) -> nn.Module:
    """``(x, y) -> u``. Both archs compute the same function class; they differ
    only in how the weights are grouped into parameter tensors, which is what
    Gnome preconditions over.

    * ``mlp``   — ``nn.Linear`` stack. Weight and bias are separate tensors, so
      each bias is preconditioned as its own 1-D factor (this experiment runs
      Gnome with ``precondition_1d=True``). The historical baseline: do not
      change it, existing runs are compared against it.
    * ``fused`` — ``FusedMLP``: ``[W | b]`` merged per layer, and the
      ``depth - 2`` hidden layers grouped ``fuse_every`` to a tensor.
      ``fuse_every=1`` is one tensor per layer and is the **control** for every
      larger setting — same function, same init, one variable. ``0`` fuses the
      whole stack.

    ``mlp`` is not the control for ``fused``: it differs in bias handling too.
    """
    if arch == "mlp":
        return PINN(hidden=hidden, depth=depth)
    if arch == "fused":
        return FusedMLP(
            ConcatEmbedding(2), hidden=hidden, depth=depth,
            fuse_every=fuse_every,
        )
    raise ValueError(f"unknown arch: {arch}")


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


def sample_aux_from_batch(batch, n_pde: int, n_bc_per_edge: int):
    """Uniform subsets without replacement from PDE points and each BC edge.

    Each edge has the same main/aux sample count, preserving the original
    equal-weight PDE/BC objective in expectation. Coordinate pairs stay intact.
    """
    x, y, xb, yb = batch
    edge_size = len(xb) // 4
    if len(xb) % 4 or not 1 <= n_pde <= len(x) or not 1 <= n_bc_per_edge <= edge_size:
        raise ValueError("auxiliary sample counts must fit the main PDE batch and each of its four edges")
    interior = torch.randperm(len(x), device=x.device)[:n_pde]
    boundary = torch.cat([
        edge*edge_size + torch.randperm(edge_size, device=xb.device)[:n_bc_per_edge]
        for edge in range(4)
    ])
    return x[interior], y[interior], xb[boundary], yb[boundary]


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

def make_engdw_functions(model: nn.Module, chunk_size: int = 32):
    """Normalized PDE/BC residuals and their per-point parameter Jacobian.

    Squared norm equals mean(PDE**2) + mean(BC**2), even for unequal block
    sizes. BC contains all four edges. Functional input Hessians preserve
    parameter differentiation through the Laplacian.
    """
    def u_single(params, xy):
        return functional_call(model, params, (xy[:1].reshape(1, 1), xy[1:].reshape(1, 1))).squeeze()

    hessian = jacrev(jacrev(u_single, argnums=1), argnums=1)

    def pde_single(params, xy):
        h = hessian(params, xy)
        return h[0, 0] + h[1, 1] + _source_term(xy[0], xy[1])

    def points(batch):
        x, y, xb, yb = batch
        return torch.cat((x, y), 1), torch.cat((xb, yb), 1)

    def residual(params, batch):
        blocks = [vmap(fn, in_dims=(None, 0), chunk_size=chunk_size)(params, pts)
                  / math.sqrt(len(pts))
                  for fn, pts in zip((pde_single, u_single), points(batch))]
        return torch.cat(blocks).unsqueeze(1)

    def jacobian(params, batch):
        blocks = []
        for fn, pts in zip((pde_single, u_single), points(batch)):
            j = vmap(jacrev(fn), in_dims=(None, 0), chunk_size=chunk_size)(params, pts)
            blocks.append({k: v / math.sqrt(len(pts)) for k, v in j.items()})
        return {k: torch.cat([b[k] for b in blocks]) for k in params}

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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000,
                   help="Default 50k — Poisson converges much faster than "
                        "evolution PDEs; usually plateaus within 20-30k.")
    p.add_argument("--n-pde", type=int, default=2000)
    p.add_argument("--n-bc-per-edge", type=int, default=50,
                   help="Boundary points per edge (total BC sample is 4× this).")
    p.add_argument("--fixed-dataset", action="store_true",
                   help="Reuse the same PDE/BC training points every step for any optimizer. "
                        "Gnome draws fresh auxiliary subsets of those points; validation still resamples.")
    p.add_argument("--aux-frac", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome per-coordinate update bound: lambda is set to "
                        "the smallest value with max|m̂/(v̂+lambda)| <= this, "
                        "so no coordinate moves more than lr*trust_region in "
                        "a step. Larger -> weaker bound -> longer steps. "
                        "0 disables it, falling back to plain m̂/(v̂+eps) "
                        "damping.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps): larger "
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--engdw-damping", type=float, default=1e-6,
                   help="ENGD-W fixed damping in J J^T + lambda I; no weight decay or schedule.")
    p.add_argument("--engdw-momentum", type=float, default=0.0,
                   help="ENGD-W regularization-center momentum in [0,1).")
    p.add_argument("--engdw-line-search", action="store_true",
                   help="ENGD-W same-batch grid line search over [1e-3,1], overriding --lr.")
    p.add_argument("--engdw-chunk", type=int, default=32,
                   help="Points per Jacobian chunk; the full Jacobian and sample Gram remain dense.")
    p.add_argument("--engdw-fixed", action="store_true",
                   help="ENGD-W-only compatibility flag for --fixed-dataset.")
    p.add_argument("--hidden", type=int, default=64, help="MLP width.")
    p.add_argument("--depth", type=int, default=5, help="MLP depth.")
    p.add_argument("--arch", type=str, default="mlp",
                   choices=["mlp", "fused"],
                   help="Weight grouping — same function class either way, "
                        "different parameter tensors for the optimizer to "
                        "precondition over. 'mlp' (default) is the nn.Linear "
                        "baseline; 'fused' merges [W|b] per layer and groups "
                        "hidden layers per --fuse-every.")
    p.add_argument("--fuse-every", type=int, default=0,
                   help="--arch fused only: consecutive hidden layers per "
                        "parameter tensor. 1 = one tensor per layer, which is "
                        "the control every larger value is compared against "
                        "(identical function and initialization, only the "
                        "grouping differs). 0 (default) fuses the whole stack. "
                        "A short final chunk is fine if it does not divide "
                        "depth-2. Not just a modelling knob: fewer, bigger "
                        "tensors cut Gnome's per-tensor per-step overhead, so "
                        "at modest width raising this is also a speedup (~8%% "
                        "for 1 -> 2 at hidden=64 depth=10). That reverses once "
                        "the eigenbasis refresh dominates, well above "
                        "hidden=64.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup for Gnome, SOAP, and AdamW; ENGD-W has no schedule.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Applies to Gnome, SOAP, and AdamW; ignored by ENGD-W.")
    p.add_argument("--float64", action="store_true",
                   help="Run in double precision on CPU. float32 floors "
                        "rel_L2 around 1e-5..1e-6; the ENGD high-accuracy "
                        "benchmark (Müller & Zeinhofer) reaches ~1e-7, which "
                        "is only reproducible in float64. Forces device=cpu "
                        "(MPS has no float64) and sets the global default "
                        "dtype to float64 so params, samples and the "
                        "reference are all double.")
    p.add_argument("--diagnostics-every", type=int, default=0,
                   help="Log Gnome's internal state — curvature spectrum, LM "
                        "damping, trust-region usage — every N steps to a "
                        "sibling runs/.../{run_id}.diag.jsonl. 0 (default) "
                        "disables it entirely. Gnome only: SOAP and AdamW "
                        "expose no such hook.")
    p.add_argument("--diagnostics-params", type=str, default=None,
                   help="Comma-separated parameter indices to log, e.g. "
                        "'0,4'. Default logs every parameter, which is one "
                        "record per tensor per logged step — narrow it to "
                        "keep the file readable.")
    p.add_argument("--measure-rho-every", type=int, default=0,
                   help="Every N steps, measure exactly how much curvature "
                        "the eigenbasis fails to diagonalize (rho), writing "
                        "runs/.../{run_id}.rho.jsonl. 0 (default) disables. "
                        "Costs one backward pass per sample (see "
                        "--measure-rho-samples), so keep N large. Gnome only.")
    p.add_argument("--measure-rho-samples", type=int, default=256,
                   help="Residual entries per rho measurement. This is the "
                        "cost knob: one backward pass each.")
    p.add_argument("--measure-rho-kron-floor", action="store_true",
                   help="Also compute the best-Kronecker-product error. "
                        "Builds a (P x P) matrix, so small layers only.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if min(args.steps, args.n_pde, args.n_bc_per_edge) < 1:
        p.error("steps and collocation counts must be positive")
    if args.engdw_fixed and args.optimizer != "engdw":
        p.error("--engdw-fixed requires --optimizer engdw")
    if args.optimizer == "gnome" and args.fixed_dataset and not 0 < args.aux_frac <= 1:
        p.error("--fixed-dataset requires --aux-frac in (0,1] for Gnome")
    if args.optimizer != "gnome" and (args.diagnostics_every > 0 or args.measure_rho_every > 0):
        p.error("diagnostics and rho measurements are Gnome-only")
    if args.optimizer == "engdw":
        if not math.isfinite(args.engdw_damping) or args.engdw_damping <= 0:
            p.error("--engdw-damping must be finite and positive")
        if not 0 <= args.engdw_momentum < 1 or not math.isfinite(args.engdw_momentum):
            p.error("--engdw-momentum must be finite and in [0,1)")
        if args.engdw_chunk < 1:
            p.error("--engdw-chunk must be positive")
        if not math.isfinite(args.lr) or args.lr < 0:
            p.error("--lr must be finite and nonnegative")
    return args


def train(args: argparse.Namespace) -> str:
    fixed_dataset = args.fixed_dataset or args.engdw_fixed
    # float64 must come before any tensor/model construction so params,
    # samples and the reference grid are all double. MPS has no float64
    # support, so double precision forces CPU.
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
    model = build_model(
        args.arch, args.hidden, args.depth, args.fuse_every
    ).to(device)
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

    n_pde_aux = max(1, int(args.n_pde * args.aux_frac)) if args.optimizer == "gnome" else 0
    n_bc_aux_per_edge = max(1, int(args.n_bc_per_edge * args.aux_frac)) if args.optimizer == "gnome" else 0
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "arch": args.arch,
        # Layers per fused tensor. Recorded as the *realized* chunk sizes, not
        # the raw flag: 0 and any value >= depth-2 both mean "one chunk", and
        # the sweep needs those runs to compare equal.
        "fuse_every": (
            getattr(model, "chunks", None) if args.arch == "fused" else None
        ),
        "hidden": args.hidden,
        "depth": args.depth,
        "n_params": n_params,
        # Parameter *tensor* count, not element count: the variable --arch
        # actually changes, and what Gnome's per-tensor preconditioning and
        # per-tensor trust region see.
        "n_tensors": sum(1 for _ in model.parameters()),
        "n_pde": args.n_pde,
        "n_bc_per_edge": args.n_bc_per_edge,
        "n_pde_aux": n_pde_aux,
        "n_bc_aux_per_edge": n_bc_aux_per_edge,
        "fixed_dataset": fixed_dataset,
        "sampling": "fixed" if fixed_dataset else "resample_each_step",
        "aux_sampling": ("subsample_fixed_main_each_step" if fixed_dataset else "resample_each_step") if args.optimizer == "gnome" else None,
        # Non-zero means a sibling {run_id}.diag.jsonl exists for this run.
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
        # Plain attributes, so this attaches to the already-built optimizer —
        # which is what lets the file be named after the run id.
        opt.diagnostics = diag
        opt.diagnostics_every = args.diagnostics_every

    # Exact rho measurement, in its own file. Unlike the diagnostics above this
    # is not free — it needs per-sample gradients, i.e. one backward pass per
    # sample — so it runs on its own much coarser cadence.
    rho_log = None
    if args.measure_rho_every > 0:
        if args.optimizer != "gnome":
            raise SystemExit(
                f"--measure-rho-every measures Gnome's eigenbasis; "
                f"--optimizer {args.optimizer} has none."
            )
        rho_log = open(os.path.join(
            os.path.dirname(run.path) or ".", f"{run.run_id}.rho.jsonl"
        ), "a")
    params_measured = [q for q in model.parameters() if q.requires_grad]

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | params={n_params:,} | "
            f"device={device} | dtype={torch.get_default_dtype()}\n"
            f"  N_pde={args.n_pde} N_bc_per_edge={args.n_bc_per_edge} | "
            f"aux={n_pde_aux}/{n_bc_aux_per_edge} | steps={args.steps} | "
            f"sampling={'fixed' if fixed_dataset else 'resample_each_step'}",
            flush=True,
        )
        if diag is not None:
            print(f"  diagnostics every {args.diagnostics_every} steps "
                  f"→ {diag.path}", flush=True)
    x_ref, y_ref, u_ref = poisson_reference()

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    best_avg = best_rel_l2 = float("inf")
    fixed_batch = None

    for step in range(args.steps):
        if fixed_batch is None:
            main_batch = sample_batch(args.n_pde, args.n_bc_per_edge, device)
            if fixed_dataset:
                fixed_batch = main_batch
        else:
            main_batch = fixed_batch
        if args.optimizer == "gnome":
            if fixed_dataset:
                aux_batch = sample_aux_from_batch(main_batch, n_pde_aux, n_bc_aux_per_edge)
            else:
                aux_batch = sample_batch(n_pde_aux, n_bc_aux_per_edge, device)

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
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if rho_log is not None and (step + 1) % args.measure_rho_every == 0:
            # A fresh batch, so the measurement is not tied to the points the
            # optimizer just stepped on. opt= adds the live-eigenbasis column.
            rb = sample_batch(args.n_pde, args.n_bc_per_edge, device)
            records = measure_rho(
                stacked_residuals(model, rb), params_measured, opt=opt,
                max_samples=args.measure_rho_samples,
                with_kron_floor=args.measure_rho_kron_floor,
            )
            for rec in records:
                rho_log.write(json.dumps({"step": step + 1, **rec}) + "\n")
            rho_log.flush()
            if not args.quiet:
                print(f"  [rho @ step {step + 1}]")
                print(format_records(records, prefix="    "), flush=True)

        applied_lr = current_lr(opt)
        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            if diag is not None:
                diag.close()
            if rho_log is not None:
                rho_log.close()
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val, lr=applied_lr)
        window.append(loss_val)

        if (args.log_every and (step + 1) % args.log_every == 0) or step + 1 == args.steps:
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
    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    if rho_log is not None:
        print(f"[{EXPERIMENT}] rho → {rho_log.name}")
        rho_log.close()
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
