"""Allen-Cahn PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + 5u³ - 5u - 0.0001·u_xx = 0,    x ∈ [-1, 1],  t ∈ [0, 1]
IC:   u(0, x) = x²·cos(πx)
BC:   u(t, -1) = u(t, 1),  u_x(t, -1) = u_x(t, 1)    (periodic)

The Allen-Cahn reaction-diffusion equation — a canonical stiff PINN
benchmark. The reaction term ``5u - 5u³`` is a double-well that drives the
field toward the metastable plateaus ``u = ±1``, and ``u ≡ ±1`` (and their
combinations) are near-zero-residual attractors; the tiny diffusion
``0.0001·u_xx`` only resolves the thin interfaces between plateaus. That
interface-translation structure gives the loss a canyon geometry — a few
stiff directions plus a large flat subspace — which is why Allen-Cahn is
the classic case where plain PINNs stall.

Two architectures, selectable via ``--arch``:

* ``mlp`` — a plain tanh MLP (as in the other torch PINN experiments).
* ``modified`` — the modified MLP of Wang, Teng & Perdikaris (2021): two
  input encoders ``u, v`` gate every hidden layer via
  ``h = tanh(W h); h = h·u + (1-h)·v``. This is the *architecture only* —
  none of the rest of the jaxpi pipeline (random weight factorization,
  Fourier features, causal weighting, grad-norm balancing) is ported.
  The multi-block loss is the same plain ``gnome.stack_residuals``
  pattern with equal block weights used throughout these experiments.

Three input embeddings, selectable via ``--embed`` (both architectures):

* ``none`` (default) feeds raw ``[t, x]`` and the periodic BC is imposed
  *softly*, as a third residual block matching ``u`` and ``u_x`` across
  the endpoints.
* ``periodic`` feeds ``[t, cos(πx), sin(πx)]``, which makes the network
  exactly period-2 in x — the domain width. Periodicity then holds to
  machine precision for ``u`` and *every* x-derivative, so the soft BC
  block is redundant and is **dropped** (two blocks instead of three).
  This is how jaxpi imposes exact periodic BCs for Allen-Cahn.
* ``fourier`` feeds ``[sin(zB), cos(zB)]`` with ``z = [t, x]`` the raw
  coordinates and ``B`` a *trainable* matrix initialized ``~ N(0, scale²)``,
  sized by ``--embed-dim`` and ``--embed-scale``. The frequencies are
  learned, so ``--embed-scale`` sets the starting bandwidth rather than the
  final one. Projecting raw ``x`` means the frequencies are not multiples
  of ``k``, so this is **not** exactly periodic and it keeps the soft BC
  block (three blocks, like ``none``). Both departures from jaxpi — which
  freezes ``B`` and applies it on top of the periodic features — are
  deliberate.

All optimizers share the chosen network so the only variable is the
optimizer. Every optimizer gets the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then
constant, which suits Gnome on MSE since its step self-anneals as the
residual shrinks).

Reference: jaxpi's ``allen_cahn.mat`` (auto-downloaded to
``experiments/reference_solutions/``), the Chebfun solution for this benchmark. A finer,
independently generated spectral reference can be selected with
``--reference highres`` and is generated and cached automatically on first use.

Usage:

    uv run -m experiments.pinns.allen_cahn_pinn --optimizer gnome --arch modified
    uv run -m experiments.pinns.allen_cahn_pinn --optimizer soap  --arch mlp
    uv run -m experiments.pinns.allen_cahn_pinn --optimizer gnome \
        --embed fourier --embed-dim 32 --embed-scale 10
    uv run -m experiments.pinns.allen_cahn_pinn --optimizer gnome \
        --diagnostics-every 500 --diagnostics-params 0
    uv run -m experiments.pinns.allen_cahn_pinn --optimizer adamw --arch modified
    uv run -m experiments.pinns.allen_cahn_pinn --optimizer gnome --embed periodic
"""

from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request

import torch
import torch.autograd as autograd
import torch.nn as nn

from gnome import Gnome, JsonlDiagnostics, stack_residuals
from experiments.baselines import SOAP
from experiments.common import (
    DIVERGED_EXIT,
    FusedMLP,
    FusedModifiedMLP,
    ModifiedMLP,
    diverged,
    RunLogger,
    cosine_scheduler,
    current_lr,
    pick_device,
    FusedLinear,
    ConcatEmbedding,
    PeriodicEmbedding,
    TrainableFourierEmbedding,
)
from experiments.common import MLP as _SharedMLP
from experiments.reference_solutions import (
    REFERENCE_SOLUTIONS_DIR,
    cached_reference_path,
)


EXPERIMENT = "allen_cahn_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0


# ========================= Models =========================


def build_embedding(embed: str, embed_dim: int = 256,
                    scale: float = 10.0) -> nn.Module:
    if embed == "none":
        return ConcatEmbedding(2)
    if embed == "periodic":
        return PeriodicEmbedding(
            2, wavenumber=2.0 * math.pi / (X_MAX - X_MIN)
        )
    if embed == "fourier":
        return TrainableFourierEmbedding(
            2, embed_dim, scale, include_input=True
        )
    raise ValueError(f"unknown embedding: {embed}")


class MLP(_SharedMLP):
    """Plain tanh MLP over an input embedding: ``(t, x) → u``.

    ``depth`` = number of Linear layers.
    """

    def __init__(self, embed: nn.Module, hidden: int = 256, depth: int = 4):
        super().__init__(embed, hidden=hidden, depth=depth)


def build_model(arch: str, embed: nn.Module, hidden: int, depth: int,
                fuse_every: int = 0) -> nn.Module:
    """``(t, x) → u``. The input embedding is whatever ``--embed`` selects.

    ``fused`` and ``fused-modified`` are the same function classes as ``mlp``
    and ``modified`` respectively — they differ only in how the weights are
    grouped into parameter tensors, which is what Gnome preconditions over.
    ``--fuse-every 1`` is the control for both: one tensor per layer,
    bit-identical initialization, one variable. Note the fusable run differs —
    ``depth - 2`` layers for ``fused``, ``depth - 1`` for ``fused-modified``,
    whose first gated layer consumes the embedding and cannot share a tensor.
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
    model: nn.Module, t: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Allen-Cahn PDE residual ``u_t + 5u³ - 5u - 0.0001·u_xx`` at (t, x)."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + 5.0 * u ** 3 - 5.0 * u - 0.0001 * u_xx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual: ``u(0, x) - x²·cos(πx)``."""
    t0 = torch.zeros_like(x)
    u0 = x ** 2 * torch.cos(math.pi * x)
    return model(t0, x) - u0


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Periodic BC residual: ``u(t, -1) - u(t, 1)`` and
    ``u_x(t, -1) - u_x(t, 1)``, stacked (C¹ periodicity).

    Only used with ``--embed none``; the periodic embedding makes this
    identically zero (see ``PeriodicEmbedding``).
    """
    x_l = torch.full_like(t, X_MIN, requires_grad=True)
    x_r = torch.full_like(t, X_MAX, requires_grad=True)
    u_l = model(t, x_l)
    u_r = model(t, x_r)
    u_l_x = autograd.grad(u_l, x_l, torch.ones_like(u_l), create_graph=True)[0]
    u_r_x = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
    return torch.cat([u_l - u_r, u_l_x - u_r_x], dim=0)


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
    model: nn.Module, batch, use_bc: bool = True
) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights).

    ``use_bc=False`` drops the soft periodicity block, which the periodic
    embedding satisfies exactly — computing it would just append a vector of
    zeros at the cost of two more backward passes per step.
    """
    t_pde, x_pde, x_ic, t_bc = batch
    blocks = [
        pde_residual(model, t_pde, x_pde),
        ic_residual(model, x_ic),
    ]
    if use_bc:
        blocks.append(bc_residual(model, t_bc))
    return stack_residuals(blocks)


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-term MSE for diagnostic logging.

    ``bc`` is always reported, even under ``--embed periodic`` where it is
    not part of the loss — there it is the check that the embedding is
    doing its job, and should sit at float32 round-off (~1e-14 as an MSE).
    This runs only every ``--log-every`` steps, so the two extra backward
    passes are free.
    """
    t_pde, x_pde, x_ic, t_bc = batch
    return {
        "pde": pde_residual(model, t_pde, x_pde).pow(2).mean().item(),
        "ic": ic_residual(model, x_ic).pow(2).mean().item(),
        "bc": bc_residual(model, t_bc).pow(2).mean().item(),
    }


# ========================= Reference solution + eval =========================

DEFAULT_REF_CACHE_DIR = str(REFERENCE_SOLUTIONS_DIR)
HIGHRES_REFERENCE_NAME = "allen_cahn_highres.mat"
REFERENCE_URL = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/allen_cahn/data/allen_cahn.mat"
)


def allen_cahn_reference(
    cache_path: str | None = None,
    *,
    reference: str = "jaxpi",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a jaxpi or independently generated Allen-Cahn reference.

    The jaxpi reference is downloaded on demand. The high-resolution spectral
    reference is generated and cached on demand. Returns
    ``(t, x, u)`` as CPU float64 tensors with shapes ``(nt,)``, ``(nx,)``,
    ``(nt, nx)`` and ``u[0]`` the initial condition.
    """
    import scipy.io

    if reference not in {"jaxpi", "highres"}:
        raise ValueError(f"unknown Allen-Cahn reference: {reference!r}")
    if cache_path is None:
        filename = (
            "allen_cahn.mat" if reference == "jaxpi" else HIGHRES_REFERENCE_NAME
        )
        cache_path = cached_reference_path(
            filename, [os.path.join("experiments/data", filename)]
        )
    if not os.path.isfile(cache_path):
        if reference == "highres":
            from experiments.pinns.generate_allen_cahn_reference import (
                generate_allen_cahn_reference,
            )
            print(
                f"[{EXPERIMENT}] high-resolution reference missing; "
                "generating it now...",
                flush=True,
            )
            generate_allen_cahn_reference(cache_path)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            print(f"[{EXPERIMENT}] downloading reference {REFERENCE_URL} ...",
                  flush=True)
            urllib.request.urlretrieve(REFERENCE_URL, cache_path)
    data = scipy.io.loadmat(cache_path)
    u = torch.as_tensor(data["usol"], dtype=torch.float64)
    t = torch.as_tensor(data["t"].flatten(), dtype=torch.float64)
    x = torch.as_tensor(data["x"].flatten(), dtype=torch.float64)
    return t, x, u


def eval_rel_l2(
    model: nn.Module,
    t_ref: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor,
    device: torch.device, batch_size: int = 8192,
) -> float:
    """Relative L2 error of the PINN prediction against ``u_ref`` on its grid."""
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
            precondition_frequency=20,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse", precondition_1d=True,
        )
        opt = Gnome(params, **cfg)
    elif name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=20, precondition_1d=True,
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw"])
    p.add_argument("--arch",
                   choices=["mlp", "modified", "fused", "fused-modified"],
                   default="modified",
                   help="Network: plain tanh MLP, the gated modified MLP "
                        "(Wang et al. 2021), or 'fused' — the plain MLP with "
                        "[W|b] merged per layer and hidden layers grouped "
                        "per --fuse-every. --hidden / --depth control all "
                        "three.")
    p.add_argument("--fuse-every", type=int, default=0,
                   help="--arch fused only: consecutive hidden layers per "
                        "parameter tensor. 1 = one tensor per layer, the "
                        "control every larger value is compared against "
                        "(identical function and initialization, only the "
                        "grouping differs). 0 (default) fuses the whole "
                        "stack. Note --depth 4 leaves only k=2 hidden layers "
                        "to group, so raise --depth to give this room.")
    p.add_argument("--embed", choices=["none", "periodic", "fourier"],
                   default="none",
                   help="Input embedding. 'none' feeds raw [t, x] and keeps "
                        "the soft periodic BC block. 'periodic' feeds "
                        "[t, cos(pi x), sin(pi x)], making the network "
                        "exactly period-2 in x — periodicity then holds for "
                        "every derivative and the BC block is DROPPED (two "
                        "blocks instead of three). 'fourier' feeds trainable "
                        "Fourier features [sin(zB), cos(zB)] over raw "
                        "z = [t, x]; not exactly periodic, so it KEEPS the "
                        "BC block. See --embed-dim / --embed-scale.")
    p.add_argument("--embed-dim", type=int, default=256,
                   help="Fourier feature output dim (even). --embed fourier.")
    p.add_argument("--embed-scale", type=float, default=10.0,
                   help="Init bandwidth of the Fourier frequencies: "
                        "B ~ N(0, scale^2). B is trained, so this is the "
                        "starting point, not a fixed bandwidth — but a high "
                        "init still gives a rough surface to start from. "
                        "Sweep it. --embed fourier.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000,
                   help="Allen-Cahn is stiff; may want more than the default.")
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=200)
    p.add_argument("--n-bc", type=int, default=200)
    p.add_argument("--aux-frac", type=float, default=0.25,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so keep small.")
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
                        "step. Allen-Cahn's canyon geometry may want it larger "
                        "than the 1e-6 default. Gnome only; SOAP/AdamW keep "
                        "eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.99,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.999,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) "
                        "for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=256, help="Network width.")
    p.add_argument("--depth", type=int, default=4,
                   help="Network depth: Linear-layer count for --arch mlp, "
                        "number of gated hidden layers for --arch modified.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps, applied to every optimizer.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--float64", action="store_true",
                   help="Run in double precision. Allen-Cahn's stiff canyon "
                        "geometry can push the residual below the float32 "
                        "floor (~1e-6 relative), where the Gauss-Newton "
                        "curvature estimate is reading round-off; float64 "
                        "moves that floor out of the way. MPS has no double "
                        "support, so this falls back to CPU there.")
    p.add_argument(
        "--reference",
        choices=["jaxpi", "highres"],
        default="jaxpi",
        help="Reference used for rel_l2. 'jaxpi' is the published 201x512 "
             "Chebfun table and remains the compatibility default. "
             "'highres' uses experiments/reference_solutions/"
             "allen_cahn_highres.mat and generates it automatically when "
             "missing.",
    )
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
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    # float64 must be set before the model is built. CUDA does doubles fine
    # (just slowly); MPS has no double support at all, so it forces CPU.
    device = pick_device()
    if args.float64:
        torch.set_default_dtype(torch.float64)
        if device.type == "mps":
            if not args.quiet:
                print(f"[{EXPERIMENT}] --float64: MPS has no double support "
                      f"— using CPU.", flush=True)
            device = torch.device("cpu")
    torch.manual_seed(args.seed)
    # Only 'periodic' satisfies the BC block exactly, so it is the only one
    # that drops it. 'fourier' projects raw (t, x) at frequencies that are not
    # multiples of k, so it needs the soft BC block just as 'none' does.
    use_bc = args.embed != "periodic"
    model = build_model(
        args.arch,
        build_embedding(args.embed, args.embed_dim, args.embed_scale),
        args.hidden, args.depth, args.fuse_every,
    ).to(device)
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
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "arch": args.arch,
        # Realized chunk sizes, not the raw flag: 0 and any value >= depth-2
        # both mean "one chunk", and the sweep needs those runs to compare
        # equal. None for the archs that have no such grouping.
        "fuse_every": getattr(model, "chunks", None),
        "embed": args.embed,
        "embed_dim": args.embed_dim if args.embed == "fourier" else None,
        "embed_scale": args.embed_scale if args.embed == "fourier" else None,
        "blocks": ["pde", "ic", "bc"] if use_bc else ["pde", "ic"],
        # Non-zero means a sibling {run_id}.diag.jsonl exists for this run.
        "diagnostics_every": args.diagnostics_every,
        "steps": args.steps,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_pde": args.n_pde,
        "n_ic": args.n_ic,
        "n_bc": args.n_bc,
        "n_pde_aux": n_pde_aux,
        "n_ic_aux": n_ic_aux,
        "n_bc_aux": n_bc_aux,
        "n_params": n_params,
        "x_domain": (X_MIN, X_MAX),
        "t_domain": (T_MIN, T_MAX),
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        "reference": args.reference,
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

    if not args.quiet:
        blocks = "pde+ic+bc" if use_bc else "pde+ic (exact periodic BC)"
        print(
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | embed={args.embed} | "
            f"params={n_params:,} | device={device} | "
            f"dtype={torch.get_default_dtype()}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps} | "
            f"blocks={blocks}",
            flush=True,
        )
        if diag is not None:
            print(f"  diagnostics every {args.diagnostics_every} steps "
                  f"→ {diag.path}", flush=True)
        print(f"  loading reference solution ({args.reference})...", flush=True)
    t_ref, x_ref, u_ref = allen_cahn_reference(reference=args.reference)
    # Evaluate in the run dtype. Both .mat files retain float64 on disk, so a
    # --float64 run now also gets a genuinely float64 reference tensor.
    dt = torch.get_default_dtype()
    t_ref, x_ref, u_ref = t_ref.to(dt), x_ref.to(dt), u_ref.to(dt)
    if not args.quiet:
        print(f"  reference grid={u_ref.shape[0]}x{u_ref.shape[1]}", flush=True)

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device)

            def main_closure():
                r = stacked_residuals(model, main_batch, use_bc)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, use_bc)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, use_bc)
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
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
                model, sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
            )
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_avg = sum(window) / len(window)
            last_terms, last_rel_l2 = tl, rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        rel_l2=rl2, **tl)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                terms = "  ".join(f"{k}={v:.3e}" for k, v in tl.items())
                print(
                    f"  step {step + 1:5d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  {terms}  "
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
    print("  final " + "  ".join(f"{k}={v:.3e}" for k, v in last_terms.items()))
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
