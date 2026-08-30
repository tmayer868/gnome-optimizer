"""(1+1)D inviscid Burgers PINN with Roe linearization: AdamW vs SOAP vs Gnome.

PDE:  u_t + u·u_x = 0,    x ∈ [-1, 1],  t ∈ [0, 1]
IC:   u(x, 0) = -sin(π x)
BC:   periodic — hard-enforced by the input map ``(t, x) → (t, cos πx, sin πx)``

Section 4.4.1 of Jnini et al., "Curvature-Aware Optimization for High-Accuracy
PINNs" (arXiv:2604.05230) — the LRPINN (linearized-Roe PINN) formulation. Their
Table 10, at 4,801 parameters and 5000 iterations:

    N_c      SSBroyden rel_L1 / rel_L2        NG rel_L1 / rel_L2         time
      625   (1.2±0.2)e-2  (7.8±1.9)e-2   (1.5±0.6)e-2  (2.6±1.8)e-1   20 / 54 s
     1250   (1.2±0.2)e-2  (9.2±2.5)e-2   (8.5±2.3)e-3  (9.1±3.1)e-2   21 / 73 s
     2500   (9.4±1.8)e-3  (8.2±2.0)e-2   (1.7±0.9)e-2  (1.8±0.8)e-1   22 / 113 s
     5000   (8.9±1.9)e-3  (7.9±1.3)e-2   (6.8±1.0)e-3  (6.8±0.4)e-2   26 / 121 s
    10000   (7.4±1.0)e-3  (7.8±1.4)e-2   (6.4±0.4)e-3  (7.1±0.5)e-2   31 / 193 s

**Why this rung.** Two reasons, both about the paper rather than the PDE.

First, Table 10 is the *only* table in the paper with error bars — five repeats
per setting, mean ± std, a fixed 5000-iteration budget for every case, and a
clean collocation sweep. It is the one result here that a reproduction can
actually be checked against.

Second, this is the one problem in the paper where the natural-gradient line
*loses*. NG is worse than SSBroyden on rel_L2 at every collocation count,
2-6x slower, and wildly variable (±1.8 on a 2.6e-1 mean). Gnome's curvature is
a GGN estimate like NG's, so the open question is whether it inherits that
weakness on a genuine shock. Testing where the related method fails is worth
more than another win.

**The Roe linearization.** A plain PINN residual ``u_t + u·u_x`` has no way to
pick out the entropy-admissible weak solution, so training is unstable once
characteristics cross. Following the paper, the advecting velocity is replaced
by the Roe average in regions where the flow is compressive enough to shock::

    Ω_s = {(x, t) : ∂u/∂x ≤ -δ}
    ũ   = (u(x+ε, t) + u(x-ε, t)) / 2   on Ω_s,     ũ = u(x, t)  elsewhere
    r   = u_t + ũ · u_x

``Ω_s`` is recomputed every step. Note the sign: a compressive (shock-forming)
region has strongly *negative* ``u_x``, which is why the test is one-sided —
expansions are left alone.

Loss is the paper's Eq. 50, two equally weighted blocks::

    L = mean(r²) + mean((u(x,0) - u*(x,0))²)

stacked through ``gnome.stack_residuals``. There is no BC block: the
``(cos πx, sin πx)`` input map makes the network exactly 2-periodic in ``x``,
so the periodic BCs hold identically. (Same trick, and the same single-
constraint caveat, as ``helmholtz_pinn``.)

**Reference solution: no WENO needed.** The paper compares against a
third-order WENO solve on 1000 spatial points, but for ``u_0 = -sin(πx)`` the
entropy solution is available in closed form via the Lax-Oleinik variational
formula::

    u(x, t) = (x - y*)/t,    y* = argmin_y [ (x-y)²/(2t) + U_0(y) ],
    U_0(y) = ∫_0^y u_0 = (cos(πy) - 1)/π

A shock forms at ``t = 1/π ≈ 0.3183`` and, since ``u_0`` is odd, sits
stationary at ``x = 0`` thereafter with ``u(0⁻) > 0 > u(0⁺)``.

**Do not use plain Newton on the implicit relation** ``u = -sin(π(x - ut))``
instead. Past shock formation that equation has three roots and Newton happily
converges to an inadmissible one — verified: at ``t = 0.5`` it returns
``u(-0.1) = -0.65`` where the entropy solution is ``+0.97``. Lax-Oleinik picks
the right branch by construction, which is why the basin is selected on a grid
before Newton polishes it.

**Two constants the paper does not give.** Both ``δ`` (the shock-detection
threshold) and ``ε`` (the offset for the left/right states) are described only
as "a given positive constant". Defaults here are ``--roe-delta 10.0`` and
``--roe-offset 0.01``, chosen so that δ sits well above the smooth-flow slope
(``|u_x| = π/(1-πt)`` at the origin before shock formation) and ε is under a
typical collocation spacing. Both are guesses — sweep them.

A third ambiguity: Eq. 48-49 read as an ordinary residual, so ``ũ`` is kept
differentiable by default. But "linearize" arguably means freezing it as a
coefficient, which is what makes the term linear in ``u`` — ``--roe-detach``
selects that reading. The two give different parameter gradients.

All three optimizers share one MLP so the only variable is the optimizer. The
optimizers all get the same linear-warmup + cosine-decay schedule (1.0 gives
warmup then constant, which suits Gnome's self-annealing Gauss-Newton step).

Usage::

    uv run -m experiments.pinns.inviscid_burgers_pinn --optimizer gnome --seed 0
    uv run -m experiments.pinns.inviscid_burgers_pinn --optimizer soap  --seed 0
    # the paper's collocation sweep:
    for n in 625 1250 2500 5000 10000; do
      uv run -m experiments.pinns.inviscid_burgers_pinn --optimizer gnome --n-pde $n
    done
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import torch.autograd as autograd
import torch.nn as nn

from gnome import Gnome, JsonlDiagnostics, stack_residuals
from experiments.baselines import SOAP
from experiments.common import (
    DIVERGED_EXIT,
    diverged,
    RunLogger,
    cosine_scheduler,
    current_lr,
    pick_device,
)
from experiments.common import (
    ConcatEmbedding,
    MLP as _SharedMLP,
    PeriodicEmbedding,
)


EXPERIMENT = "inviscid_burgers_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0
T_SHOCK = 1.0 / math.pi          # characteristics first cross here


# ========================= Exact entropy solution =========================

def u0(x: torch.Tensor) -> torch.Tensor:
    """Initial condition ``u(x, 0) = -sin(π x)``."""
    return -torch.sin(math.pi * x)


def exact_solution(
    x: torch.Tensor, t: float, ny: int = 20001, newton_iters: int = 60,
) -> torch.Tensor:
    """Entropy solution at time ``t`` via the Lax-Oleinik variational formula.

    Minimizes ``G(y) = (x-y)²/(2t) + U_0(y)`` with ``U_0(y) = (cos(πy) - 1)/π``,
    then returns ``u = (x - y*)/t``.

    Two stages, because the two things needed are different: the coarse grid
    scan selects the correct *basin* (this is what makes the result
    entropy-admissible rather than merely a root of the characteristic
    relation), and Newton on ``G'(y) = (y-x)/t - sin(πy) = 0`` then polishes it
    to machine precision. ``u_0`` is ``-sin(πy)`` for all real ``y``, which is
    exactly the periodic extension, so minimizing over an unbounded ``y`` is
    correct for the periodic problem.

    Characteristics satisfy ``|u| <= 1``, so with ``t <= 1`` the minimizer lies
    within 1 of ``x``; the scan window is padded well past that.
    """
    if t <= 0.0:
        return u0(x)
    x = x.reshape(-1)
    y = torch.linspace(x.min().item() - 1.5, x.max().item() + 1.5, ny,
                       dtype=x.dtype, device=x.device)
    U0 = (torch.cos(math.pi * y) - 1.0) / math.pi
    # Chunk over x: the scan is an (n_x, ny) outer difference.
    best = torch.empty_like(x)
    chunk = max(1, 2_000_000 // ny)
    for i in range(0, x.numel(), chunk):
        xi = x[i:i + chunk].unsqueeze(1)
        G = (xi - y.unsqueeze(0)) ** 2 / (2.0 * t) + U0.unsqueeze(0)
        best[i:i + chunk] = y[G.argmin(dim=1)]
    ystar = best
    for _ in range(newton_iters):                 # G'(y) = (y-x)/t - sin(pi y)
        g = (ystar - x) / t - torch.sin(math.pi * ystar)
        gp = 1.0 / t - math.pi * torch.cos(math.pi * ystar)
        ystar = ystar - g / gp.clamp(min=1e-12)
    u = (x - ystar) / t
    # Exactly on the shock, G has two equal minima and argmin picks one of them
    # arbitrarily — giving |u| ~ 1 at a point where the solution jumps through
    # zero. u_0 is odd, so the entropy solution is odd and u(0, t) = 0 for all
    # t; enforce that. Only bites for eval grids that contain x = 0 (the
    # default n_eval_x = 1000 straddles it), but a wrong value there is worth
    # about 1/n of the reported error.
    return torch.where(x.abs() < 1e-12, torch.zeros_like(u), u)


# ========================= Model =========================


class PINN(_SharedMLP):
    """Maps ``(t, x) → u`` through the periodic embedding then a tanh MLP.

    ``depth`` counts *total* linear layers, so the paper's "6 hidden layers,
    30 neurons" is ``--depth 7 --hidden 30`` → 4,801 parameters.
    """

    def __init__(self, hidden: int = 30, depth: int = 7,
                 embedding: str = "periodic"):
        if embedding == "periodic":
            embed: nn.Module = PeriodicEmbedding(2, wavenumber=math.pi)
        elif embedding == "raw":
            embed = ConcatEmbedding(2)
        else:
            raise ValueError(f"unknown embedding: {embedding}")
        super().__init__(embed, hidden=hidden, depth=depth)
        self.embedding = embedding


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor,
    roe_delta: float, roe_offset: float, roe_detach: bool,
) -> tuple[torch.Tensor, float]:
    """LRPINN residual ``u_t + ũ·u_x``, plus the fraction of points in ``Ω_s``.

    The shock mask is taken on ``u_x.detach()``: membership of ``Ω_s`` is a
    discrete set decision recomputed each step, not something to backpropagate
    through.

    Costs three forward passes (centre plus both offset states). The offset
    evaluations are done on the whole batch rather than only on ``Ω_s`` —
    ``torch.where`` needs both branches anyway, and masked indexing would churn
    every step as the mask moves.
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    shock = u_x.detach() <= -roe_delta          # compressive regions only
    roe = 0.5 * (model(t, x - roe_offset) + model(t, x + roe_offset))
    if roe_detach:
        roe = roe.detach()
    u_tilde = torch.where(shock, roe, u)
    return u_t + u_tilde * u_x, float(shock.to(u.dtype).mean())


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual ``u(x, 0) - u_0(x)``."""
    t0 = torch.zeros_like(x)
    return model(t0, x) - u0(x)


def residuals(
    model: nn.Module, batch, roe_delta: float, roe_offset: float,
    roe_detach: bool, ic_weight: float,
) -> tuple[torch.Tensor, float]:
    """Eq. 50: ``mean(r_pde²) + ic_weight·mean(r_ic²)`` as one stacked tensor."""
    (t, x), x_ic = batch
    r_pde, shock_frac = pde_residual(
        model, t, x, roe_delta, roe_offset, roe_detach)
    r_ic = ic_residual(model, x_ic)
    return stack_residuals([r_pde, r_ic], [1.0, ic_weight]), shock_frac


# ========================= Sampling =========================

def sample_batch(
    n_pde: int, n_ic: int, device: torch.device, dtype: torch.dtype
):
    """Uniform interior points on ``[-1,1]×[0,1]`` plus IC points on ``t=0``."""
    kw = dict(device=device, dtype=dtype)
    t = torch.rand(n_pde, 1, **kw) * (T_MAX - T_MIN) + T_MIN
    x = torch.rand(n_pde, 1, **kw) * (X_MAX - X_MIN) + X_MIN
    x_ic = torch.rand(n_ic, 1, **kw) * (X_MAX - X_MIN) + X_MIN
    return (t, x), x_ic


# ========================= Eval =========================

def make_eval_sets(
    nx: int, nt: int, device: torch.device, dtype: torch.dtype
):
    """Reference values at ``t = T_MAX`` and on a space-time grid.

    The ``t = T_MAX`` slice on ``nx`` points is the paper's likely protocol —
    Figure 13 plots ``t = 1`` and the caption describes a WENO reference on
    1000 spatial points. The space-time grid is logged alongside it because
    Table 10 never actually says which was used.
    """
    x = torch.linspace(X_MIN, X_MAX, nx, device=device, dtype=dtype)
    final = (x.reshape(-1, 1), exact_solution(x, T_MAX).reshape(-1, 1))

    ts = torch.linspace(T_MIN, T_MAX, nt, device=device, dtype=dtype)
    tt, xx, uu = [], [], []
    for tv in ts.tolist():
        tt.append(torch.full((nx, 1), tv, device=device, dtype=dtype))
        xx.append(x.reshape(-1, 1))
        uu.append(exact_solution(x, tv).reshape(-1, 1))
    grid = (torch.cat(tt), torch.cat(xx), torch.cat(uu))
    return final, grid


def eval_errors(model: nn.Module, final, grid) -> dict[str, float]:
    """Relative L1 and L2 at ``t = T_MAX`` and over the space-time grid."""
    was_training = model.training
    model.eval()
    out = {}
    with torch.no_grad():
        x_f, u_f = final
        e = model(torch.full_like(x_f, T_MAX), x_f) - u_f
        out["rel_l1"] = float(e.abs().sum() / u_f.abs().sum())
        out["rel_l2"] = float(e.pow(2).sum().sqrt() / u_f.pow(2).sum().sqrt())
        t_g, x_g, u_g = grid
        e = model(t_g, x_g) - u_g
        out["rel_l1_st"] = float(e.abs().sum() / u_g.abs().sum())
        out["rel_l2_st"] = float(e.pow(2).sum().sqrt() / u_g.pow(2).sum().sqrt())
    if was_training:
        model.train()
    return out


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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=5000,
                   help="Paper protocol: 5000 iterations for every case.")
    p.add_argument("--n-pde", type=int, default=2500,
                   help="Interior collocation points. Paper sweeps 625, 1250, "
                        "2500, 5000, 10000 (Table 10).")
    p.add_argument("--n-ic", type=int, default=256,
                   help="Initial-condition points at t=0. Not stated in the "
                        "paper for this rung.")
    p.add_argument("--ic-weight", type=float, default=1.0,
                   help="Weight on the IC block: mean(out^2) = mean(r_pde^2) + "
                        "ic_weight*mean(r_ic^2). Eq. 50 uses equal weights.")
    p.add_argument("--roe-delta", type=float, default=10.0,
                   help="Shock-detection threshold: Omega_s = {u_x <= -delta}. "
                        "Undocumented in the paper. Before shock formation the "
                        "slope at the origin is pi/(1-pi*t), so delta should "
                        "sit well above pi.")
    p.add_argument("--roe-offset", type=float, default=0.01,
                   help="Offset for the left/right states u(x -+ offset, t) in "
                        "the Roe average. Undocumented in the paper; should be "
                        "below a typical collocation spacing.")
    p.add_argument("--roe-detach", action="store_true",
                   help="Freeze the Roe average as a coefficient (the literal "
                        "'linearize' reading) instead of differentiating "
                        "through it. Changes the parameter gradients.")
    p.add_argument("--embedding", type=str, default="periodic",
                   choices=["periodic", "raw"],
                   help="periodic: (t, cos pi x, sin pi x), BCs exact, the "
                        "paper's setup. raw: (t, x) — periodicity is then not "
                        "enforced at all and there is no BC block, so expect "
                        "drift at the domain edges.")
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "residual eval, so this is not free — keep small.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome l2 trust radius: lambda is the smallest value "
                        ">= eps with ||m_hat/(v_hat+lambda)||_2 <= this * "
                        "sqrt(P), i.e. a bound on the RMS per-coordinate step. "
                        "0 disables it.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m_hat/(v_hat+eps). "
                        "Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=30, help="MLP width.")
    p.add_argument("--depth", type=int, default=7,
                   help="Total linear layers (paper: 6 hidden + output = 7, "
                        "giving 4801 params at --hidden 30).")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"],
                   help="float32 suffices here — the paper's accuracies are "
                        "~1e-2/1e-3, set by the shock, not by precision.")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero, 1.0 disables decay. Gnome never "
                        "decays regardless.")
    p.add_argument("--n-eval-x", type=int, default=1000,
                   help="Spatial points at t=1 (paper's WENO reference: 1000).")
    p.add_argument("--n-eval-t", type=int, default=51,
                   help="Time slices in the space-time eval grid.")
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
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = getattr(torch, args.dtype)
    if device.type == "mps" and dtype is torch.float64:
        device = torch.device("cpu")
        if not args.quiet:
            print(f"[{EXPERIMENT}] float64 requested; MPS has no float64 "
                  f"— falling back to CPU.", flush=True)

    model = PINN(hidden=args.hidden, depth=args.depth,
                 embedding=args.embedding).to(device=device, dtype=dtype)
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        warmup=args.warmup_steps, total_steps=args.steps,
        cosine_decay=args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
        trust_region=args.trust_region,
    )

    n_pde_aux = max(8, int(args.n_pde * args.aux_frac))
    n_ic_aux = max(2, int(args.n_ic * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
        "steps": args.steps,
        "dtype": args.dtype,
        "hidden": args.hidden,
        "depth": args.depth,
        "embedding": args.embedding,
        "n_params": n_params,
        "n_pde": args.n_pde,
        "n_ic": args.n_ic,
        "n_pde_aux": n_pde_aux,
        "n_ic_aux": n_ic_aux,
        "ic_weight": args.ic_weight,
        "roe_delta": args.roe_delta,
        "roe_offset": args.roe_offset,
        "roe_detach": args.roe_detach,
        "t_shock": T_SHOCK,
        "n_eval_x": args.n_eval_x,
        "n_eval_t": args.n_eval_t,
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
        # Plain attributes, so this attaches to the already-built optimizer —
        # which is what lets the file be named after the run id.
        opt.diagnostics = diag
        opt.diagnostics_every = args.diagnostics_every

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | params={n_params:,} | "
            f"embedding={args.embedding} | dtype={args.dtype} | "
            f"device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} | "
            f"aux={n_pde_aux}/{n_ic_aux} | steps={args.steps} | "
            f"roe: delta={args.roe_delta:g} offset={args.roe_offset:g} "
            f"detach={args.roe_detach}",
            flush=True,
        )
        print("  building Lax-Oleinik reference...", flush=True)
    final, grid = make_eval_sets(args.n_eval_x, args.n_eval_t, device, dtype)

    if diag is not None and not args.quiet:
        print(f"  diagnostics every {args.diagnostics_every} steps "
              f"→ {diag.path}", flush=True)
    t_start = time.perf_counter()
    window: list[float] = []
    last = {"rel_l1": float("nan"), "rel_l2": float("nan"),
            "rel_l1_st": float("nan"), "rel_l2_st": float("nan")}
    last_avg = float("nan")
    best_avg = float("inf")
    best = {k: float("inf") for k in last}
    shock_frac = 0.0

    for step in range(args.steps):
        main_batch = sample_batch(args.n_pde, args.n_ic, device, dtype)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_ic_aux, device, dtype)
            stats = {}

            def main_closure():
                r, sf = residuals(model, main_batch, args.roe_delta,
                                  args.roe_offset, args.roe_detach,
                                  args.ic_weight)
                stats["shock"] = sf
                return r, torch.zeros_like(r)

            def aux_closure():
                r, _ = residuals(model, aux_batch, args.roe_delta,
                                 args.roe_offset, args.roe_detach,
                                 args.ic_weight)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
            shock_frac = stats.get("shock", 0.0)
        else:
            opt.zero_grad()
            r, shock_frac = residuals(model, main_batch, args.roe_delta,
                                      args.roe_offset, args.roe_detach,
                                      args.ic_weight)
            # Match Gnome's internal MSE reduction: sum-of-squares / N.
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item())
        if diverged(loss_val):
            if diag is not None:
                diag.close()
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            last = eval_errors(model, final, grid)
            last_avg = sum(window) / len(window)
            best_avg = min(best_avg, last_avg)
            for k, v in last.items():
                best[k] = min(best[k], v)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        shock_frac=shock_frac, **last)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"rel_l1={last['rel_l1']:.3e}  "
                    f"rel_l2={last['rel_l2']:.3e}  "
                    f"shock={100 * shock_frac:.1f}%  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        **{f"final_{k}": v for k, v in last.items()},
        **{f"best_{k}": v for k, v in best.items()},
    )

    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  t=1     final rel_l1={last['rel_l1']:.3e}  "
          f"rel_l2={last['rel_l2']:.3e}   "
          f"(best {best['rel_l1']:.3e} / {best['rel_l2']:.3e})")
    print(f"  space-t final rel_l1={last['rel_l1_st']:.3e}  "
          f"rel_l2={last['rel_l2_st']:.3e}")
    print(f"  (Jnini et al. Table 10 @ N_c={args.n_pde}: see module docstring)")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
