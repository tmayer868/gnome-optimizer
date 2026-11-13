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
  pattern with configurable block weights (equal by default).

Four input embeddings, selectable via ``--embed`` (all architectures):

* ``none`` (default) feeds raw ``[t, x]`` and the periodic BC is imposed
  *softly*, as a third residual block matching ``u`` and ``u_x`` across
  the endpoints.
* ``periodic`` feeds ``[t, cos(πx), sin(πx)]``, which makes the network
  exactly period-2 in x — the domain width. Periodicity then holds to
  machine precision for ``u`` and *every* x-derivative, so the soft BC
  block is redundant and is **dropped** (two blocks instead of three).
  This is how jaxpi imposes exact periodic BCs for Allen-Cahn.
* ``polynomial`` feeds ``[t, 1-x², x(1-x²)]``. Endpoint values match
  exactly, but endpoint spatial derivatives need not, so it keeps the
  soft BC block (three blocks).
* ``fourier`` first maps to ``z = [t, cos(πx), sin(πx)]``, then feeds
  ``[z, sin(zB), cos(zB)]`` to the network. ``B`` is a *trainable* matrix
  initialized ``~ N(0, scale²)``, sized by ``embed_dim`` and ``embed_scale``.
  Learned Fourier features retain exact period-2 spatial periodicity,
  including derivatives, so the soft BC block is **dropped**. ``embed_dim``
  counts the Fourier features; the three periodic input features are added.

All optimizers share the chosen network so the only variable is the
optimizer. Every optimizer gets the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then
constant, which suits Gnome on MSE since its step self-anneals as the
residual shrinks).

Points are resampled each step unless ``--fixed-dataset`` is supplied.
This reuses one uniformly sampled PDE/IC/BC batch for any optimizer; Gnome
draws fresh auxiliary subsets of that batch each step. Validation residual
points still resample.

Use ``--float64-after N`` to run N steps in float32, then continue in
float64 with the same optimizer state and LR schedule. On MPS the switch
also moves training to CPU. ``--float64`` (or ``--float64-after 0``) starts
in double precision immediately.

Reference: jaxpi's ``allen_cahn.mat`` (auto-downloaded to
``experiments/reference_solutions/``), the Chebfun solution for this benchmark. A finer,
independently generated spectral reference can be selected with
``--reference highres`` and is generated and cached automatically on first use.

Usage:

    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', arch='modified'
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='soap', arch='mlp'
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', embed='fourier', embed_dim=32, embed_scale=10.0
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', diagnostics_every=500, diagnostics_params='0'
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='adamw', arch='modified'
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', embed='periodic'
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', embed='polynomial'
    uv run -m experiments.pinns.allen_cahn_pinn
    # Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', float64_after=30000
    uv run -m experiments.pinns.allen_cahn_pinn
SOAP + SS-Broyden is available as ``optimizer='soap+ssbroyden'``. The
``soap_steps`` phase runs the usual SOAP updates; remaining total ``steps``
use Scimba's strong-Wolfe SS-Broyden. Sampling freezes and precision promotes
to float64 no later than the handoff; earlier ``fix_after`` and
``float64_after`` switches are retained.
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
from experiments.common.ssbroyden import (
    build_soap_phase, build_ssbroyden_phase, validate_ssbroyden,
)
from experiments.common.precision import switch_to_float64
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
    PolynomialEmbedding,
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
    if embed == "polynomial":
        return PolynomialEmbedding()
    if embed == "periodic":
        return PeriodicEmbedding(
            2, wavenumber=2.0 * math.pi / (X_MAX - X_MIN)
        )
    if embed == "fourier":
        return TrainableFourierEmbedding(
            2, embed_dim, scale, include_input=True,
            wavenumber=2.0 * math.pi / (X_MAX - X_MIN),
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

    Included in the loss except with ``periodic`` or ``fourier``, which
    make this zero up to roundoff. ``polynomial`` enforces matching values
    but still needs the derivative residual.
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


def sample_aux_from_batch(batch, n_pde: int, n_ic: int, n_bc: int):
    """Uniform subsets without replacement per block, preserving (t, x) pairs."""
    t_pde, x_pde, x_ic, t_bc = batch
    if not (1 <= n_pde <= len(t_pde) and 1 <= n_ic <= len(x_ic)
            and 1 <= n_bc <= len(t_bc)):
        raise ValueError("auxiliary sample counts must fit the main PDE/IC/BC batch")
    interior = torch.randperm(len(t_pde), device=t_pde.device)[:n_pde]
    initial = torch.randperm(len(x_ic), device=x_ic.device)[:n_ic]
    boundary = torch.randperm(len(t_bc), device=t_bc.device)[:n_bc]
    return t_pde[interior], x_pde[interior], x_ic[initial], t_bc[boundary]


def residual_partition_sizes(batch, use_bc: bool = True) -> list[int]:
    """Element counts of the PDE, IC, and optional BC residual blocks."""
    t_pde, _x_pde, x_ic, t_bc = batch
    sizes = [t_pde.numel(), x_ic.numel()]
    if use_bc:
        sizes.append(2 * t_bc.numel())
    return sizes


def stacked_residuals(
    model: nn.Module, batch, use_bc: bool = True, weights=(1.0, 1.0, 1.0)
) -> torch.Tensor:
    """Stack residuals with loss weights ordered as PDE, IC, periodic BC.

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
    return stack_residuals(blocks, weights=weights[:len(blocks)])


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-term MSE for diagnostic logging.

    ``bc`` is always reported, even for periodic/Fourier embeddings where it is
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
HIGHRES_REFERENCE_NAME = "allen_cahn_highres_8191_tight.mat"
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
    reference uses 8191 periodic points with rtol=2e-13 and atol=2e-15,
    and is generated and cached on demand. Returns
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
    """Relative L2 with model inference native and all error math in float64."""
    nt, nx = u_ref.shape
    tt, xx = torch.meshgrid(t_ref, x_ref, indexing="ij")
    model_dtype = next(
        (parameter.dtype for parameter in model.parameters()
         if parameter.is_floating_point()),
        torch.get_default_dtype(),
    )
    t_flat = tt.reshape(-1, 1).to(device=device, dtype=model_dtype)
    x_flat = xx.reshape(-1, 1).to(device=device, dtype=model_dtype)
    # Keep float64 conversion off accelerators such as MPS, which has no
    # float64 support. Transfer first, then promote using the CPU kernel.
    u_ref64 = u_ref.detach().cpu().to(dtype=torch.float64)
    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, t_flat.shape[0], batch_size):
            preds.append(
                model(t_flat[i:i + batch_size], x_flat[i:i + batch_size])
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )
    if was_training:
        model.train()
    u_pred = torch.cat(preds).reshape(nt, nx)
    num = (u_pred - u_ref64).pow(2).sum().sqrt()
    den = u_ref64.pow(2).sum().sqrt()
    return float(num / den)


# ========================= Optimizer factory =========================

def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    warmup: int, total_steps: int, cosine_decay: float, eps: float = 1e-6,
    beta1: float = 0.9, beta2: float = 0.99,
    trust_region: float = 1.0,
    basis_transport: str = "overlap",
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
            basis_transport=basis_transport,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse", precondition_1d=True,
        )
        opt = Gnome(params, **cfg)
    elif name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=1, precondition_1d=True,
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
    if any(w < 0 for w in args.weights):
        p.error("weights must be non-negative")
    if min(args.steps, args.n_pde, args.n_ic, args.n_bc) < 1:
        p.error("steps and collocation counts must be positive")
    if args.fix_after is not None and not 0 <= args.fix_after < args.steps:
        p.error("fix_after must satisfy 0 <= fix_after < steps, or be None")
    if args.optimizer == "gnome":
        if not math.isfinite(args.aux_frac) or args.aux_frac <= 0:
            p.error("--aux-frac must be finite and positive")
        if (args.fixed_dataset or args.fix_after is not None) and args.aux_frac > 1:
            p.error("fixed_dataset or fix_after requires aux_frac in (0,1] for Gnome")
    if args.float64_after is not None and not 0 <= args.float64_after < args.steps:
        p.error("--float64-after must satisfy 0 <= N < --steps")
    if args.optimizer == "soap+ssbroyden":
        try:
            validate_ssbroyden(args)
        except ValueError as error:
            p.error(str(error))
    return args



def train(args: argparse.Namespace) -> str:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(
        torch.float64 if (args.float64 or args.float64_after == 0 or
                          (args.optimizer == "soap+ssbroyden" and args.soap_steps == 0))
        else torch.float32
    )
    try:
        return _train(args)
    finally:
        torch.set_default_dtype(previous_dtype)


def _train(args: argparse.Namespace) -> str:
    hybrid = args.optimizer == "soap+ssbroyden"
    if hybrid:
        validate_ssbroyden(args)
    float64_from_step = 0 if torch.get_default_dtype() == torch.float64 else args.float64_after
    if hybrid:
        float64_from_step = (args.soap_steps if float64_from_step is None
                             else min(float64_from_step, args.soap_steps))
    device = pick_device()
    if torch.get_default_dtype() == torch.float64:
        if device.type == "mps":
            if not args.quiet:
                print(f"[{EXPERIMENT}] --float64: MPS has no double support "
                      f"— using CPU.", flush=True)
            device = torch.device("cpu")
    torch.manual_seed(args.seed)
    # Both periodic embeddings satisfy the BC block by construction, including
    # after training the Fourier projection.
    # 'polynomial' matches endpoint values but still needs derivative matching.
    use_bc = args.embed not in ("periodic", "fourier")
    model = build_model(
        args.arch,
        build_embedding(args.embed, args.embed_dim, args.embed_scale),
        args.hidden, args.depth, args.fuse_every,
    ).to(device)
    if hybrid:
        opt, opt_cfg, scheduler = build_soap_phase(model, args, precondition_frequency=1)
    else:
        opt, opt_cfg, scheduler = build_optimizer(
            args.optimizer, model.parameters(), args.lr, args.weight_decay,
            warmup=args.warmup_steps, total_steps=args.steps,
            cosine_decay=args.cosine_decay, eps=args.eps,
            beta1=args.beta1, beta2=args.beta2,
            trust_region=args.trust_region,
            basis_transport=args.basis_transport,
        )

    fixed_from_step = 0 if args.fixed_dataset else args.fix_after
    if hybrid:
        fixed_from_step = (args.soap_steps if fixed_from_step is None
                           else min(fixed_from_step, args.soap_steps))
    sampling = ("fixed" if fixed_from_step == 0 else
                "resample_then_fixed" if fixed_from_step is not None else "resample_each_step")
    n_pde_aux = max(8, int(args.n_pde * args.aux_frac))
    n_ic_aux = max(2, int(args.n_ic * args.aux_frac))
    n_bc_aux = max(2, int(args.n_bc * args.aux_frac))
    if fixed_from_step == 0:
        n_pde_aux = min(args.n_pde, n_pde_aux)
        n_ic_aux = min(args.n_ic, n_ic_aux)
        n_bc_aux = min(args.n_bc, n_bc_aux)
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "partition_residuals": args.optimizer == "gnome" and args.partition_residuals,
        "basis_transport": args.basis_transport if args.optimizer == "gnome" else None,
        "arch": args.arch,
        # Realized chunk sizes, not the raw flag: 0 and any value >= depth-2
        # both mean "one chunk", and the sweep needs those runs to compare
        # equal. None for the archs that have no such grouping.
        "fuse_every": getattr(model, "chunks", None),
        "embed": args.embed,
        "embed_dim": args.embed_dim if args.embed == "fourier" else None,
        "embed_scale": args.embed_scale if args.embed == "fourier" else None,
        "blocks": ["pde", "ic", "bc"] if use_bc else ["pde", "ic"],
        "weights": list(args.weights),
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
        "fixed_dataset": args.fixed_dataset,
        "fix_after": args.fix_after,
        "fixed_from_step": fixed_from_step,
        "sampling": sampling,
        "aux_sampling": (
            "subsample_fixed_main_each_step" if fixed_from_step == 0 else
            "resample_then_subsample_fixed_main" if fixed_from_step is not None else "resample_each_step"
        ) if args.optimizer == "gnome" else None,
        "n_params": n_params,
        "x_domain": (X_MIN, X_MAX),
        "t_domain": (T_MIN, T_MAX),
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        "float64_after": args.float64_after,
        "float64_from_step": float64_from_step,
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

    if not args.quiet:
        blocks = "pde+ic+bc" if use_bc else "pde+ic (exact periodic BC)"
        print(
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | embed={args.embed} | "
            f"params={n_params:,} | device={device} | "
            f"dtype={torch.get_default_dtype()}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps} | "
            f"blocks={blocks} | "
            f"sampling={sampling} | fix_after={fixed_from_step}",
            flush=True,
        )
        if diag is not None:
            print(f"  diagnostics every {args.diagnostics_every} steps "
                  f"→ {diag.path}", flush=True)
        print(f"  loading reference solution ({args.reference})...", flush=True)
    t_ref, x_ref, u_ref = allen_cahn_reference(reference=args.reference)
    # Retain the original float64 coordinates for the later precision switch;
    # eval_rel_l2 casts inference inputs to the model's current dtype/device.
    u_ref = u_ref.to(torch.float64)
    if not args.quiet:
        print(f"  reference grid={u_ref.shape[0]}x{u_ref.shape[1]}", flush=True)

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}
    best_avg = best_rel_l2 = float("inf")

    fixed_batch = None
    for step in range(args.steps):
        if step == float64_from_step and step > 0:
            previous_device = device
            device = switch_to_float64(model, opt)
            if fixed_batch is not None:
                # Move off MPS before promoting; preserve the sampled coordinates.
                fixed_batch = tuple(
                    tensor.to(device=device).to(dtype=torch.float64)
                    for tensor in fixed_batch
                )
            run.log("precision", step=step, from_dtype="torch.float32",
                    dtype="torch.float64", from_device=str(previous_device),
                    device=str(device), lr=current_lr(opt))
            if not args.quiet:
                print(f"[{EXPERIMENT}] after {step} steps: float32 → float64 "
                      f"| device={previous_device} → {device}", flush=True)
        quasi_newton = hybrid and step >= args.soap_steps
        if hybrid and step == args.soap_steps:
            opt = build_ssbroyden_phase(model, args)
            scheduler = None
            run.log("optimizer", step=step, optimizer="ssbroyden")
            if not args.quiet:
                print(f"[{EXPERIMENT}] switching to SS-Broyden at step {step}", flush=True)
        if fixed_batch is None:
            main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
            if fixed_from_step is not None and step >= fixed_from_step:
                fixed_batch = main_batch
                # Preserve the original auxiliary sampling before the switch;
                # subsets of the fixed dataset must fit its individual blocks.
                n_pde_aux = min(args.n_pde, n_pde_aux)
                n_ic_aux = min(args.n_ic, n_ic_aux)
                n_bc_aux = min(args.n_bc, n_bc_aux)
                run.log("sampling", step=step, sampling="fixed",
                        n_pde_aux=n_pde_aux, n_ic_aux=n_ic_aux, n_bc_aux=n_bc_aux)
                if not args.quiet and step > 0:
                    print(f"[{EXPERIMENT}] after {step} steps: training dataset fixed", flush=True)
        else:
            main_batch = fixed_batch
        if args.optimizer == "gnome":
            if fixed_batch is not None:
                aux_batch = sample_aux_from_batch(main_batch, n_pde_aux, n_ic_aux, n_bc_aux)
            else:
                aux_batch = sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device)

            def main_closure():
                r = stacked_residuals(model, main_batch, use_bc, args.weights)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, use_bc, args.weights)
                return r, torch.zeros_like(r)

            loss = opt.step(
                main_closure, aux_closure,
                surrogate_partition_sizes=(
                    residual_partition_sizes(aux_batch, use_bc) if args.partition_residuals else None
                ),
            )
        elif quasi_newton:
            def closure():
                opt.zero_grad()
                r = stacked_residuals(model, main_batch, use_bc, args.weights)
                value = r.square().mean()
                value.backward()
                return value

            loss = opt.step(closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, use_bc, args.weights)
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if (diag is not None and args.optimizer == "gnome"
                and (step + 1) % args.diagnostics_every == 0):
            for record in opt.get_diagnostics(params=diag.params):
                diag(record)

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
        run.log_train(step, loss=loss_val, **(
            {"phase": "ssbroyden" if quasi_newton else "soap"} if hybrid else {}
        ))
        window.append(loss_val)

        should_log = args.log_every and (step + 1) % args.log_every == 0
        at_handoff = hybrid and step + 1 == args.soap_steps
        if should_log or (hybrid and step + 1 == args.steps) or at_handoff:
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
        final_dtype=str(torch.get_default_dtype()), final_device=str(device),
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
