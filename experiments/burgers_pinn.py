"""Burgers PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + u·u_x - ν·u_xx = 0,    x ∈ [-1, 1],  t ∈ [0, 1]
IC:   u(0, x) = -sin(π x)
BC:   u(t, ±1) = 0    (Dirichlet)

The Raissi et al. (2019) canonical Burgers benchmark, with viscosity
``ν = 0.01 / π`` — small enough that the solution develops a sharp interior
shock around ``t ≈ 0.5``, making the cross-block (PDE/IC/BC) stiffness
diagnostic of the typical PINN pathology. PDE/IC/BC residuals are stacked
through ``gnome.stack_residuals`` so that the multi-block MSE
``L = mse(pde) + mse(ic) + mse(bc)`` rides Gnome's single-MSE surrogate
and the resulting probe is the per-block independent Rademacher GGN
estimator.

All three optimizers share one plain tanh MLP so the only variable is the
optimizer. The baselines (SOAP, AdamW) get a linear-warmup + cosine-decay
schedule (``--cosine-decay`` sets the final-lr fraction; 1.0 disables it);
Gnome runs at a fixed lr — its Gauss-Newton step self-anneals as the residual
shrinks.

Usage:

    uv run -m experiments.burgers_pinn --optimizer gnome --seed 0
    uv run -m experiments.burgers_pinn --optimizer soap  --seed 0
    uv run -m experiments.burgers_pinn --optimizer adamw --seed 0
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

from gnome import Gnome, stack_residuals
from experiments.baselines import SOAP
from experiments.common import (
    DIVERGED_EXIT,
    diverged,
    RunLogger,
    baseline_cosine_scheduler,
    current_lr,
    pick_device,
)


EXPERIMENT = "burgers_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0
NU = 0.01 / math.pi

# jaxpi (Wang et al. 2025, arXiv:2502.00604) Burgers reference — the canonical
# Raissi precomputed dataset their pirate-branch benchmark scores against. Used
# as a *second* eval reference alongside our own spectral solve: at their ~4e-5
# accuracy level the reference field and eval grid are first-order terms, so a
# rel_L2 is only directly comparable to their reported SOAP number on this exact
# file and grid.
JAXPI_REFERENCE_URL = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/burgers/data/burgers.mat"
)
JAXPI_REFERENCE_CACHE = "experiments/data/burgers.mat"


# ========================= Model =========================

class PINN(nn.Module):
    """Maps ``(t, x) → u`` via a plain tanh MLP."""

    def __init__(self, hidden: int = 20, depth: int = 9):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([t, x], dim=1))


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Burgers PDE residual ``u_t + u·u_x - ν·u_xx`` at (t, x)."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - NU * u_xx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual: ``u(0, x) + sin(π x)``."""
    t0 = torch.zeros_like(x)
    return model(t0, x) + torch.sin(math.pi * x)


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Dirichlet BC residual: ``u(t, ±1)`` stacked into one tensor."""
    x_l = torch.full_like(t, X_MIN)
    x_r = torch.full_like(t, X_MAX)
    return torch.cat([model(t, x_l), model(t, x_r)], dim=0)


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


def stacked_residuals(model: nn.Module, batch) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights)."""
    t_pde, x_pde, x_ic, t_bc = batch
    return stack_residuals([
        pde_residual(model, t_pde, x_pde),
        ic_residual(model, x_ic),
        bc_residual(model, t_bc),
    ])


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-term MSE for diagnostic logging."""
    t_pde, x_pde, x_ic, t_bc = batch
    return {
        "pde": pde_residual(model, t_pde, x_pde).pow(2).mean().item(),
        "ic": ic_residual(model, x_ic).pow(2).mean().item(),
        "bc": bc_residual(model, t_bc).pow(2).mean().item(),
    }


# ========================= Reference solution + eval =========================

DEFAULT_REF_CACHE_DIR = "experiments/data"


def burgers_reference(
    nx: int = 1024, nt: int = 101, nu: float = NU,
    cache_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fourier-spectral reference solution for Burgers.

    The IC ``u(0, x) = -sin(π x)`` is exactly periodic on ``[-1, 1)`` and
    Burgers preserves the resulting antisymmetry, so the Dirichlet BCs
    ``u(t, ±1) = 0`` are automatically enforced by a periodic Fourier
    basis. The solver discretizes the spatial domain with ``nx`` Fourier
    modes (``endpoint=False``), computes ``u_x`` and ``u_xx`` in spectral
    space, applies 2/3-rule dealiasing to the nonlinear ``u·u_x`` product
    to control aliasing of the shock front, and time-marches with RK4.

    The number of RK4 substeps per snapshot scales as ``nx²`` so the
    spectral parabolic CFL stays satisfied as resolution grows; at
    ``nx = 256`` that's 50 substeps, at ``nx = 1024`` it's 800 substeps.

    Reference accuracy as a function of ``nx`` (self-consistency vs the
    next-finer grid)::

        nx=256  vs nx=512  : rel_L2 ≈ 7e-3   ← under-resolves the shock
        nx=512  vs nx=1024 : rel_L2 ≈ 4e-4
        nx=1024 vs nx=2048 : rel_L2 ≈ 1e-6   ← converged

    The default ``nx = 1024`` therefore puts the reference's own accuracy
    well below any plausible PINN rel_L2 floor.

    Central finite differences are *not* used here — at ``ν = 0.01/π`` the
    cell Péclet number ``|u|·dx/ν`` is past the FD stability limit on
    moderate grids and the solution NaN-s through the shock. The spectral
    method has no such Péclet constraint.

    Returns ``(t_grid, x_grid, u_grid)`` with shapes ``(nt,)``, ``(nx,)``,
    ``(nt, nx)`` — CPU float32 tensors. Result is cached to disk under a
    filename that encodes ``nx`` so caches at different resolutions don't
    collide.
    """
    if cache_path is None:
        cache_path = os.path.join(DEFAULT_REF_CACHE_DIR, f"burgers_reference_nx{nx}.pt")
    import numpy as np

    if cache_path and os.path.isfile(cache_path):
        blob = torch.load(cache_path, weights_only=True)
        return blob["t"], blob["x"], blob["u"]

    L = X_MAX - X_MIN
    x = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    dx = L / nx
    k = 2.0 * np.pi * np.fft.fftfreq(nx, dx)
    k_max = np.abs(k).max()
    dealias = (np.abs(k) <= (2.0 / 3.0) * k_max).astype(np.float64)

    u = (-np.sin(np.pi * x)).astype(np.float64)

    def rhs(u_):
        u_hat = np.fft.fft(u_)
        u_x = np.fft.ifft(1j * k * u_hat).real
        u_xx = np.fft.ifft(-(k ** 2) * u_hat).real
        nonlin_hat = np.fft.fft(u_ * u_x) * dealias
        nonlin = np.fft.ifft(nonlin_hat).real
        return -nonlin + nu * u_xx

    # Spectral RK4 stability: dt < 2 / (ν · k_max²) ~ 2·dx²/(ν·π²). The substep
    # count therefore needs to grow ∝ nx² so larger references don't NaN through
    # the shock. The (nx/256)² scaling makes substeps=50 at the original nx=256
    # reference and bumps to 800 at nx=1024 (well inside the safe envelope).
    substeps_per_snap = max(50, int(50 * (nx / 256) ** 2))
    dt_sub = (T_MAX - T_MIN) / ((nt - 1) * substeps_per_snap)
    snapshots = [u.copy()]
    for _ in range(nt - 1):
        for _ in range(substeps_per_snap):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt_sub * k1)
            k3 = rhs(u + 0.5 * dt_sub * k2)
            k4 = rhs(u + dt_sub * k3)
            u = u + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        snapshots.append(u.copy())

    t_grid = np.linspace(T_MIN, T_MAX, nt).astype(np.float32)
    x_grid = x.astype(np.float32)
    u_grid = np.stack(snapshots).astype(np.float32)

    t = torch.from_numpy(t_grid)
    xt = torch.from_numpy(x_grid)
    ug = torch.from_numpy(u_grid)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"t": t, "x": xt, "u": ug}, cache_path)
    return t, xt, ug


def burgers_reference_jaxpi(
    cache_path: str = JAXPI_REFERENCE_CACHE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """jaxpi's Burgers reference (``burgers.mat``) for a head-to-head rel_L2.

    Loads the *exact* precomputed dataset Wang et al. (2025, arXiv:2502.00604)
    score against in their pirate-branch benchmark, so a rel_L2 measured here is
    directly comparable to their reported SOAP number — same reference field,
    same grid. Deliberately separate from ``burgers_reference`` (our own
    Fourier-spectral solve): at their 4e-5 accuracy level the reference dataset
    and eval grid are first-order terms, so the comparison is only valid on
    their own file. Verified contents: ``nu = 0.01/π`` and
    ``usol[0] == -sin(π x)`` to machine precision.

    File layout (Matlab v5): ``t (1, 201)``, ``x (1, 512)`` (both endpoints
    included), ``usol (201, 512)`` oriented ``(t, x)``. Downloaded on first use
    and cached to disk (gitignored like all reference data).

    Returns ``(t_grid, x_grid, u_grid)`` with shapes ``(201,)``, ``(512,)``,
    ``(201, 512)`` — CPU float32, matching ``burgers_reference`` so the same
    ``eval_rel_l2`` consumes either.
    """
    try:
        import scipy.io
    except ImportError as e:
        raise RuntimeError(
            "scipy is required to load jaxpi's .mat Burgers reference. "
            "Install with `uv pip install scipy`."
        ) from e

    if not os.path.isfile(cache_path):
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        print(
            f"  downloading jaxpi reference from {JAXPI_REFERENCE_URL} ...",
            flush=True,
        )
        try:
            urllib.request.urlretrieve(JAXPI_REFERENCE_URL, cache_path)
        except Exception as e:
            raise RuntimeError(
                f"failed to download jaxpi Burgers dataset: {e}\n"
                f"manually fetch {JAXPI_REFERENCE_URL!r} → {cache_path!r}."
            ) from e

    blob = scipy.io.loadmat(cache_path)
    t = torch.from_numpy(blob["t"].astype("float32").ravel())
    x = torch.from_numpy(blob["x"].astype("float32").ravel())
    u = torch.from_numpy(blob["usol"].astype("float32"))
    return t, x, u


def eval_rel_l2(
    model: nn.Module,
    t_ref: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor,
    device: torch.device, batch_size: int = 8192,
) -> float:
    """Relative L2 error of the PINN prediction against ``u_ref`` on its grid.

    Returns ``||u_pred - u_ref||_2 / ||u_ref||_2`` evaluated over every
    ``(t_i, x_j)`` on the reference grid. The model is queried in batches
    under ``torch.no_grad`` so the reference grid can be much larger than
    a training batch without OOM.
    """
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
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=10,
            clip=1.0, warmup=warmup,
            loss="mse", precondition_1d=True,
        )
        return Gnome(params, **cfg), cfg, None
    if name == "soap":
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
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--n-pde", type=int, default=2000)
    p.add_argument("--n-ic", type=int, default=100)
    p.add_argument("--n-bc", type=int, default=100)
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so this is not free — "
                        "keep small.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps): larger "
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=20, help="MLP width.")
    p.add_argument("--depth", type=int, default=9, help="MLP depth.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps. For the SOAP/AdamW baselines "
                        "this is the schedule warmup; for Gnome it is passed "
                        "as its internal `warmup=`.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--log-every", type=int, default=100,
                   help="Log a val entry (running train mean + per-term "
                        "diagnostics on a fresh probe batch) every N steps.")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--no-jaxpi-ref", action="store_true",
                   help="Skip the additional rel_l2 eval against jaxpi's "
                        "burgers.mat (Wang et al. 2025). On by default so a "
                        "head-to-head number against their reported SOAP "
                        "result is logged as `rel_l2_jaxpi` alongside our "
                        "spectral-reference `rel_l2`.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    model = PINN(hidden=args.hidden, depth=args.depth).to(device)
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        warmup=args.warmup_steps, total_steps=args.steps,
        cosine_decay=args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
    )

    n_pde_aux = max(8, int(args.n_pde * args.aux_frac))
    n_ic_aux = max(2, int(args.n_ic * args.aux_frac))
    n_bc_aux = max(2, int(args.n_bc * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
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
        "nu": NU,
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
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}",
            flush=True,
        )
        print("  loading reference solution...", flush=True)
    t_ref, x_ref, u_ref = burgers_reference()

    # Second reference: jaxpi's burgers.mat, scored on its own (201×512) grid
    # for a comparison that is directly valid against their reported SOAP number.
    t_ref_j = x_ref_j = u_ref_j = None
    if not args.no_jaxpi_ref:
        try:
            t_ref_j, x_ref_j, u_ref_j = burgers_reference_jaxpi()
        except Exception as e:
            print(
                f"  WARNING: jaxpi reference unavailable ({e}); "
                f"logging only our spectral rel_l2.",
                flush=True,
            )

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = last_rel_l2_j = float("nan")
    last_pde = last_ic = last_bc = float("nan")
    best_avg = best_rel_l2 = best_rel_l2_j = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device)

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
            # Match Gnome's internal MSE reduction: sum-of-squares / N.
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            tl = term_losses(
                model, sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
            )
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_avg = sum(window) / len(window)
            last_pde, last_ic, last_bc = tl["pde"], tl["ic"], tl["bc"]
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            val_kw = dict(
                loss=last_avg, lr=current_lr(opt),
                pde=tl["pde"], ic=tl["ic"], bc=tl["bc"], rel_l2=rl2,
            )
            if u_ref_j is not None:
                last_rel_l2_j = eval_rel_l2(model, t_ref_j, x_ref_j, u_ref_j, device)
                best_rel_l2_j = min(best_rel_l2_j, last_rel_l2_j)
                val_kw["rel_l2_jaxpi"] = last_rel_l2_j
            run.log_val(step + 1, **val_kw)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                jaxpi_str = (
                    f"  rel_l2_jaxpi={last_rel_l2_j:.3e}"
                    if u_ref_j is not None else ""
                )
                print(
                    f"  step {step + 1:5d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  ic={tl['ic']:.3e}  "
                    f"bc={tl['bc']:.3e}  rel_l2={rl2:.3e}{jaxpi_str}  "
                    f"{ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    summary = dict(
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )
    if u_ref_j is not None:
        summary["final_rel_l2_jaxpi"] = last_rel_l2_j
        summary["best_rel_l2_jaxpi"] = best_rel_l2_j
    path = run.finish(completed=True, **summary)

    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final pde={last_pde:.3e}  ic={last_ic:.3e}  bc={last_bc:.3e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    if u_ref_j is not None:
        print(f"  final rel_l2_jaxpi={last_rel_l2_j:.3e}  "
              f"best rel_l2_jaxpi={best_rel_l2_j:.3e}  "
              f"(vs Wang et al. SOAP 4.03e-5)")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
