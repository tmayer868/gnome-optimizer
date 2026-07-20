"""Korteweg-de Vries PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + η·u·u_x + μ²·u_xxx = 0,   η = 1,  μ = 0.022,  x ∈ [-1, 1],  t ∈ [0, 1]
IC:   u(0, x) = cos(πx)
BC:   u, u_x, u_xx  periodic in x    (u(t,-1)=u(t,1), etc.)

The Korteweg-de Vries equation — the canonical *dispersive* benchmark. The
third-order term ``μ²·u_xxx`` spreads energy across wavenumbers with no
dissipation to damp them, so the solution develops fine oscillatory
structure (soliton trains) that has to stay phase-coherent over the whole
domain. There is no smoothing to forgive errors and the derivative order is
one higher than the other 1-D benchmarks, so the residual is stiffer and
depends on ``u_xxx`` — a much harsher demand on the network's high-order
derivative representation. This is the benchmark with the most headroom in
the SOAP-PINN paper's single-window suite (their best is ~3.4e-4).

Two architectures, selectable via ``--arch``:

* ``mlp`` — a plain tanh MLP (as in the other torch PINN experiments).
* ``modified`` — the modified MLP of Wang, Teng & Perdikaris (2021): two
  input encoders ``u, v`` gate every hidden layer via
  ``h = tanh(W h); h = h·u + (1-h)·v``. This is the *architecture only* —
  none of the rest of the jaxpi pipeline (random weight factorization,
  Fourier features, causal weighting, grad-norm balancing) is ported. Both
  archs use a period-2 input embedding ``[t, cos(πx), sin(πx)]`` (matching
  the ``[-1, 1]`` domain), and the multi-block loss is the same plain
  ``gnome.stack_residuals`` pattern with equal block weights.

All optimizers share the chosen network so the only variable is the
optimizer. The baselines (SOAP, AdamW) get a linear-warmup + cosine-decay
schedule (``--cosine-decay`` sets the final-lr fraction; 1.0 disables it);
Gnome runs at a fixed lr — its Gauss-Newton step self-anneals as the
residual shrinks.

``--optimizer adamw+lbfgs`` is the classic PINN recipe (and the paper's
real first-order baseline — nobody runs Adam alone): the Adam phase runs
for ``--steps`` as usual, then an L-BFGS phase (``--lbfgs-steps``) refines
on a *fixed, full-batch* collocation set. L-BFGS builds a curvature history
from a sequence of (grad, step) pairs, which is only meaningful if the
objective is the same function each iteration — so, unlike the Adam phase,
its points are drawn once and held constant.

Reference: jaxpi's ``kdv.mat`` (auto-downloaded to ``experiments/data/``).

Usage:

    uv run -m experiments.kdv_pinn --optimizer gnome --arch modified
    uv run -m experiments.kdv_pinn --optimizer soap  --arch mlp
    uv run -m experiments.kdv_pinn --optimizer adamw+lbfgs --arch modified
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


EXPERIMENT = "kdv_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0
ETA = 1.0                 # u·u_x coefficient
MU_SQ = 0.022 ** 2        # u_xxx coefficient


# ========================= Models =========================

class MLP(nn.Module):
    """Plain tanh MLP: ``(t, x) → u``. ``depth`` = number of Linear layers.

    Input is the period-2 embedding ``[t, cos(πx), sin(πx)]`` (matches the
    ``[-1, 1]`` x-domain)."""

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([t,
                       x,
                       # torch.cos(math.pi * x),
                       # torch.sin(math.pi * x),
                       # torch.cos(math.pi * t),
                       # torch.sin(math.pi * t),
                       ], dim=1)
        return self.net(z)


class ModifiedMLP(nn.Module):
    """Modified MLP (Wang, Teng & Perdikaris 2021): ``(t, x) → u``.

    Two input encoders ``u, v`` gate every hidden layer:
    ``h = tanh(W_l h);  h = h·u + (1-h)·v``. ``depth`` = number of gated
    hidden layers. Architecture only — no random weight factorization,
    Fourier features, or causal weighting (those are jaxpi-pipeline pieces,
    deliberately not ported here). Same period-2 input embedding as ``MLP``.
    """

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        assert depth >= 1
        self.enc_u = nn.Linear(3, hidden)
        self.enc_v = nn.Linear(3, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(3 if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([t, torch.cos(math.pi * x), torch.sin(math.pi * x)], dim=1)
        u = torch.tanh(self.enc_u(z))
        v = torch.tanh(self.enc_v(z))
        h = z
        for layer in self.layers:
            h = torch.tanh(layer(h))
            h = h * u + (1.0 - h) * v
        return self.out(h)


def build_model(arch: str, hidden: int, depth: int) -> nn.Module:
    if arch == "mlp":
        return MLP(hidden=hidden, depth=depth)
    if arch == "modified":
        return ModifiedMLP(hidden=hidden, depth=depth)
    raise ValueError(f"unknown arch: {arch}")


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """KdV PDE residual ``u_t + η·u·u_x + μ²·u_xxx`` at (t, x)."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + ETA * u * u_x + MU_SQ * u_xxx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual: ``u(0, x) - cos(πx)``."""
    t0 = torch.zeros_like(x)
    u0 = torch.cos(math.pi * x)
    return model(t0, x) - u0


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Periodic BC residual through ``u_xx`` — KdV is third-order, so the
    well-posed periodic conditions match ``u``, ``u_x`` and ``u_xx`` across
    the ends. Returns ``[u_l-u_r, u_l_x-u_r_x, u_l_xx-u_r_xx]`` stacked."""
    x_l = torch.full_like(t, X_MIN, requires_grad=True)
    x_r = torch.full_like(t, X_MAX, requires_grad=True)
    u_l = model(t, x_l)
    u_r = model(t, x_r)
    u_l_x = autograd.grad(u_l, x_l, torch.ones_like(u_l), create_graph=True)[0]
    u_r_x = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
    u_l_xx = autograd.grad(u_l_x, x_l, torch.ones_like(u_l_x),
                           create_graph=True)[0]
    u_r_xx = autograd.grad(u_r_x, x_r, torch.ones_like(u_r_x),
                           create_graph=True)[0]
    return torch.cat([u_l - u_r, u_l_x - u_r_x, u_l_xx - u_r_xx], dim=0)


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
REFERENCE_URL = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/kdv/data/kdv.mat"
)


def kdv_reference(
    cache_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """jaxpi's KdV reference (``kdv.mat``).

    Auto-downloaded to ``experiments/data/``. Returns ``(t, x, u)`` with
    shapes ``(nt,)``, ``(nx,)``, ``(nt, nx)`` — CPU float32, ``u[0]`` the IC.
    """
    import scipy.io

    if cache_path is None:
        cache_path = os.path.join(DEFAULT_REF_CACHE_DIR, "kdv.mat")
    if not os.path.isfile(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        print(f"[{EXPERIMENT}] downloading reference {REFERENCE_URL} ...",
              flush=True)
        urllib.request.urlretrieve(REFERENCE_URL, cache_path)
    data = scipy.io.loadmat(cache_path)
    u = torch.as_tensor(data["usol"], dtype=torch.float32)
    t = torch.as_tensor(data["t"].flatten(), dtype=torch.float32)
    x = torch.as_tensor(data["x"].flatten(), dtype=torch.float32)
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
            precondition_frequency=2,
            clip=1.0, warmup=warmup,
            loss="mse", precondition_1d=True,
        )
        return Gnome(params, **cfg), cfg, None
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=2, precondition_1d=True,
        )
        opt = SOAP(params, **cfg)
    elif name in ("adamw", "adamw+lbfgs"):
        # adamw+lbfgs uses plain AdamW for phase 1; the L-BFGS refinement is
        # a separate phase appended after the main loop (see lbfgs_phase).
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


# ========================= L-BFGS refinement phase =========================

def lbfgs_phase(
    model: nn.Module, device: torch.device, args: argparse.Namespace,
    run: RunLogger, t_ref: torch.Tensor, x_ref: torch.Tensor,
    u_ref: torch.Tensor, start_step: int, best_rel_l2: float,
) -> dict:
    """Full-batch L-BFGS refinement, run after the Adam phase.

    L-BFGS approximates curvature from a history of (grad, step) pairs, which
    is only valid if the objective is a fixed function — so we draw ONE
    collocation set here and reuse it for every iteration (unlike the Adam
    phase, which resamples each step). ``torch.optim.LBFGS`` re-evaluates the
    closure several times per ``.step()`` for its strong-Wolfe line search;
    each outer step runs ``--lbfgs-max-iter`` inner iterations (must be >1 —
    the line search cannot recover from a cold identity Hessian in a single
    iteration and stalls). Returns the final/best metrics for the run summary.
    """
    fixed_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=args.lbfgs_lr, max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history, line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, tolerance_change=1e-12,
    )
    # Log at roughly the same iteration cadence as the Adam phase: one outer
    # step is lbfgs_max_iter inner iterations.
    log_every = max(1, args.log_every // args.lbfgs_max_iter)

    if not args.quiet:
        total_iters = args.lbfgs_steps * args.lbfgs_max_iter
        print(
            f"[{EXPERIMENT}] L-BFGS refinement: {args.lbfgs_steps} outer steps "
            f"x {args.lbfgs_max_iter} = {total_iters} iters on a fixed batch "
            f"(N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc}, "
            f"history={args.lbfgs_history}, lr={args.lbfgs_lr})",
            flush=True,
        )

    t_start = time.perf_counter()
    last_loss = last_rel_l2 = float("nan")
    last_pde = last_ic = last_bc = float("nan")

    for i in range(args.lbfgs_steps):
        def closure():
            opt.zero_grad()
            r = stacked_residuals(model, fixed_batch)
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            return loss

        loss = opt.step(closure)
        last_loss = float(loss.detach().item())
        step = start_step + i
        if diverged(last_loss):
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] L-BFGS diverged at step {step} — stopping.",
                  flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=last_loss)

        if (i + 1) % log_every == 0:
            tl = term_losses(model, fixed_batch)
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_pde, last_ic, last_bc = tl["pde"], tl["ic"], tl["bc"]
            last_rel_l2 = rl2
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_loss, lr=args.lbfgs_lr,
                        pde=tl["pde"], ic=tl["ic"], bc=tl["bc"], rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (i + 1) * 1000
                print(
                    f"  L-BFGS {i + 1:5d}/{args.lbfgs_steps}  "
                    f"loss={last_loss:.4e}  pde={tl['pde']:.3e}  "
                    f"ic={tl['ic']:.3e}  bc={tl['bc']:.3e}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )

    return {
        "last_avg": last_loss,
        "last_rel_l2": last_rel_l2,
        "best_rel_l2": best_rel_l2,
        "last_pde": last_pde, "last_ic": last_ic, "last_bc": last_bc,
    }


# ========================= CLI / training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw", "adamw+lbfgs"])
    p.add_argument("--arch", choices=["mlp", "modified"], default="modified",
                   help="Network: plain tanh MLP or the gated modified MLP "
                        "(Wang et al. 2021). --hidden / --depth control both.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000,
                   help="KdV is dispersive/stiff; may want more than default.")
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=200)
    p.add_argument("--n-bc", type=int, default=200)
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so keep small.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps): larger "
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep eps=1e-8.")
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
                   help="Linear LR warmup steps. For the SOAP/AdamW baselines "
                        "this is the schedule warmup; for Gnome it is passed "
                        "as its internal `warmup=`.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--lbfgs-steps", type=int, default=500,
                   help="L-BFGS OUTER steps after the Adam phase "
                        "(--optimizer adamw+lbfgs only). Each outer step runs "
                        "up to --lbfgs-max-iter inner iterations, so the total "
                        "L-BFGS budget is lbfgs_steps * lbfgs_max_iter. Runs "
                        "full-batch on a fixed collocation set.")
    p.add_argument("--lbfgs-max-iter", type=int, default=20,
                   help="Inner L-BFGS iterations per outer step. Must be >1: "
                        "the strong-Wolfe line search cannot recover from a "
                        "cold (identity) Hessian in a single iteration and "
                        "stalls. adamw+lbfgs only.")
    p.add_argument("--lbfgs-history", type=int, default=50,
                   help="L-BFGS history size (number of stored curvature "
                        "pairs). adamw+lbfgs only.")
    p.add_argument("--lbfgs-lr", type=float, default=1.0,
                   help="L-BFGS learning rate; with the strong-Wolfe line "
                        "search 1.0 is standard. adamw+lbfgs only.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    model = build_model(args.arch, args.hidden, args.depth).to(device)
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
        "arch": args.arch,
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
        "eta": ETA,
        "mu_sq": MU_SQ,
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
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | params={n_params:,} | "
            f"device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}",
            flush=True,
        )
        print("  loading / downloading reference solution...", flush=True)
    t_ref, x_ref, u_ref = kdv_reference()

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_pde = last_ic = last_bc = float("nan")
    best_avg = best_rel_l2 = float("inf")

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
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
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
            last_pde, last_ic, last_bc = tl["pde"], tl["ic"], tl["bc"]
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        pde=tl["pde"], ic=tl["ic"], bc=tl["bc"], rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:5d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  ic={tl['ic']:.3e}  "
                    f"bc={tl['bc']:.3e}  rel_l2={rl2:.3e}  "
                    f"{ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    if args.optimizer == "adamw+lbfgs" and args.lbfgs_steps > 0:
        res = lbfgs_phase(
            model, device, args, run, t_ref, x_ref, u_ref,
            start_step=args.steps, best_rel_l2=best_rel_l2,
        )
        last_avg = res["last_avg"]
        best_avg = min(best_avg, res["last_avg"])
        last_rel_l2 = res["last_rel_l2"]
        best_rel_l2 = res["best_rel_l2"]
        last_pde, last_ic, last_bc = (
            res["last_pde"], res["last_ic"], res["last_bc"])

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
