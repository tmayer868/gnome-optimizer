"""2D Helmholtz PINN: AdamW vs SOAP vs Gnome.

PDE:  u_xx + u_yy + k²·u - q(x, y) = 0,    (x, y) ∈ [-1, 1]²
      q(x, y) = (k² - (a₁π)² - (a₂π)²)·sin(a₁πx)·sin(a₂πy)
BC:   periodic in both directions — enforced *exactly* by construction, see below

Exact solution::

    u(x, y) = sin(a₁πx)·sin(a₂πy)

Section 4.1 of Jnini et al., "Curvature-Aware Optimization for High-Accuracy
PINNs" (arXiv:2604.05230). Their Table 3, the reference rung ``a₁=1, a₂=4,
k=1`` — which is where SOAP loses by the widest margin anywhere in the paper:

    SOAP                      rel_L2 = 4.2e-4   rel_Linf = 5.5e-4   1038 s
    SSBroyden (Optax TR)      rel_L2 = 3.6e-9   rel_Linf = 4.8e-9    319 s
    NG                        rel_L2 = 4.2e-9   rel_Linf = 5.6e-9    251 s
    SPRING                    rel_L2 = 2.0e-9   rel_Linf = 2.4e-9    540 s

all at 2,971 parameters. SOAP is simultaneously ~100,000x less accurate and
4x slower than the natural-gradient line. That five-order gap is the point of
this experiment: Gnome is SOAP's machinery (rotate into the eigenbasis, EMA
the curvature, step there) with the curvature source swapped from gradient
outer products to a Hutchinson GGN estimate. This problem isolates exactly
that substitution, on the rung where the effect is largest.

**Toggling the embedding.** ``--embedding {fourier,raw}`` and
``--bc {auto,none,dirichlet}`` are independent, giving the full 2x2:

* ``fourier`` + ``none`` (default) — the paper's setup, single-block loss.
* ``raw`` + ``dirichlet`` (the other ``auto`` default) — a conventional
  two-block PINN. Periodicity is gone, so a BC block is *required*: without
  one, ``Δ + k²`` on a bounded domain admits any homogeneous solution and the
  problem is genuinely ill-posed. The script warns if you ask for this.
* ``fourier`` + ``dirichlet`` — the clean single-variable ablation. The
  embedding is unchanged and only the block count moves, so it separates "hard
  periodicity constraint" from "one block vs two".
* ``raw`` + ``none`` — ill-posed, warned about, kept only for completeness.

Note that adding the Dirichlet block does *not* rescue the weakly-observed
modes by much at ``k=1``. A constant offset ``c`` produces ``|r_pde| = k²c = c``
and ``|r_bc| = c``, so at ``bc_weight = 1`` the BC block merely doubles DC's
weight against the target mode's 166.8². Use ``--bc-weight`` to push harder.

**Single-block loss — no BC term (with the embedding on).** Inputs pass through a Fourier embedding
``[cos(πm·x), sin(πm·x), cos(πm·y), sin(πm·y)]`` for ``m = 1..M`` (the paper's
``cos(2πmx/Lₓ), sin(2πmx/Lₓ)`` with ``Lₓ = 2``). Every feature has period 2,
so the network is *exactly* 2-periodic in both directions and the periodic BCs
hold identically — there is nothing to penalize. The exact solution is
compatible: ``sin(a₁πx)`` has period ``2/a₁``, which divides 2 for integer
``a₁``.

This makes it the only single-block PINN in this repo. Every other one stacks
PDE/IC/BC through ``gnome.stack_residuals`` and so mixes "does the optimizer
handle block conflict" into the result. Here there is exactly one residual, so
what is measured is curvature quality and nothing else.

**The difficulty ladder.** The paper's four §4.1 architectures are recovered
exactly by two flags (parameter counts verified against Tables 3, 5 and 6)::

    (a₁,a₂)=(1,4)   k=1        --depth 5 --modes 1    2,971 params  (Table 3)
    (a₁,a₂)=(6,6)   k=1        --depth 7 --modes 1    4,831 params  (Table 5)
    (a₁,a₂)=(6,6)   k=10,100   --depth 7 --modes 10   5,911 params  (Table 5)
    (a₁,a₂)=(10,10) k=1        --depth 9 --modes 1    6,691 params  (Table 6)

**What this script does not reproduce.** The paper resamples collocation
points with RAD (residual-based adaptive sampling) every 500 epochs, and for
the ``k=100`` rung uses a progressive domain expansion curriculum
(``[-0.2,0.2]² → [-0.4,0.4]² → [-0.7,0.7]² → [-1,1]²``). Neither is
implemented here: we resample uniformly every step, which is this repo's
convention. The paper states NG converged reliably on the full domain without
curriculum at moderate frequencies, so the default rung needs neither. Expect
to need both if you push to ``k=100``.

Also note Table 4, a collocation sweep at otherwise fixed settings, which is a
free second axis to check against: 10,000 pts → rel_Linf 1.5e-8; 20,000 →
7.1e-8; 25,000 → 7.4e-9.

All three optimizers share one MLP so the only variable is the optimizer. The
optimizers all get the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then constant,
which suits Gnome since its step self-anneals as the residual shrinks).

Defaults to float64: the target here is 1e-9, which float32 cannot reach at
all. The network is tiny (2,971 params) so this costs little.

Usage::

    uv run -m experiments.pinns.helmholtz_pinn --optimizer gnome --seed 0
    uv run -m experiments.pinns.helmholtz_pinn --optimizer soap  --seed 0
    uv run -m experiments.pinns.helmholtz_pinn --optimizer adamw --seed 0
    # (6,6) k=10 rung:
    uv run -m experiments.pinns.helmholtz_pinn --optimizer gnome --a1 6 --a2 6 \
        --k 10 --depth 7 --modes 10
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


EXPERIMENT = "helmholtz_pinn"

X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
L_DOMAIN = X_MAX - X_MIN          # = 2, the Fourier embedding period


# ========================= Exact solution + forcing =========================

def exact_solution(
    x: torch.Tensor, y: torch.Tensor, a1: float, a2: float
) -> torch.Tensor:
    """``u* = sin(a1·π·x)·sin(a2·π·y)``."""
    return torch.sin(a1 * math.pi * x) * torch.sin(a2 * math.pi * y)


def forcing(
    x: torch.Tensor, y: torch.Tensor, a1: float, a2: float, k: float
) -> torch.Tensor:
    """``q = (k² - (a1π)² - (a2π)²)·u*``.

    Substituting ``u*`` into the operator gives ``u*_xx + u*_yy + k²u* =
    (k² - (a1π)² - (a2π)²)·u*``, so this ``q`` makes ``u*`` an exact solution
    by construction — no reference solve, no data file.
    """
    coeff = k ** 2 - (a1 * math.pi) ** 2 - (a2 * math.pi) ** 2
    return coeff * exact_solution(x, y, a1, a2)


# ========================= Model =========================


class PINN(_SharedMLP):
    """Maps ``(x, y) → u`` via an optional Fourier embedding then a tanh MLP.

    ``depth`` counts *total* linear layers, so the paper's "4 hidden layers,
    30 neurons" is ``--depth 5 --hidden 30``.

    ``embedding="raw"`` feeds ``(x, y)`` straight in. No input normalization is
    needed either way — the domain is already ``[-1, 1]²`` and the Fourier
    features are bounded by construction — so the two paths differ only in
    periodicity and frequency content, not in input scale.
    """

    def __init__(self, hidden: int = 30, depth: int = 5, modes: int = 1,
                 embedding: str = "fourier"):
        if embedding == "fourier":
            embed: nn.Module = PeriodicEmbedding(
                2,
                n_harmonics=modes,
                wavenumber=2.0 * math.pi / L_DOMAIN,
                periodic_dims=(0, 1),
            )
        elif embedding == "raw":
            embed = ConcatEmbedding(2)
        else:
            raise ValueError(f"unknown embedding: {embedding}")
        super().__init__(embed, hidden=hidden, depth=depth)
        self.embedding = embedding


# ========================= Residual =========================

def pde_residual(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    a1: float, a2: float, k: float,
) -> torch.Tensor:
    """Helmholtz residual ``u_xx + u_yy + k²u - q`` at ``(x, y)``.

    This is the *only* residual block — the Fourier embedding already
    satisfies the periodic BCs exactly, so there is no BC term to stack.
    """
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = model(x, y)
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy + (k ** 2) * u - forcing(x, y, a1, a2, k)


def bc_residual(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, a1: float, a2: float,
) -> torch.Tensor:
    """Dirichlet residual ``u - u*`` on points already lying on ∂Ω.

    For integer ``a1, a2`` the exact solution vanishes on ``∂Ω`` (``sin(a·π·(±1))
    = 0``), so this is just ``u`` there — but it is written as ``u - u*`` so
    non-integer frequencies still work.

    Only used when the Fourier embedding is off. With the embedding on,
    periodicity holds identically and there is nothing to penalize.
    """
    return model(x, y) - exact_solution(x, y, a1, a2)


# ========================= Sampling =========================

def sample_batch(
    n: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform interior collocation points on ``[-1, 1]²``.

    Resampled fresh every step (repo convention). The paper instead holds a
    fixed set and refreshes it with RAD every 500 epochs — see the module
    docstring.
    """
    kw = dict(device=device, dtype=dtype)
    x = torch.rand(n, 1, **kw) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n, 1, **kw) * (Y_MAX - Y_MIN) + Y_MIN
    return x, y


def sample_boundary(
    n: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniform draws on ``∂Ω``: pin one coordinate to a face at ±1."""
    kw = dict(device=device, dtype=dtype)
    x = torch.rand(n, 1, **kw) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n, 1, **kw) * (Y_MAX - Y_MIN) + Y_MIN
    pin_x = torch.randint(0, 2, (n, 1), device=device) == 0
    face = torch.where(
        torch.randint(0, 2, (n, 1), device=device) == 0,
        torch.tensor(X_MIN, **kw), torch.tensor(X_MAX, **kw),
    )
    x = torch.where(pin_x, face, x)
    y = torch.where(pin_x, y, face)
    return x, y


def residuals(
    model: nn.Module, batch, a1: float, a2: float, k: float,
    bc_weight: float | None,
) -> torch.Tensor:
    """PDE residual, optionally stacked with a Dirichlet BC block.

    ``bc_weight is None`` → single-block, PDE only (the Fourier-embedding
    case). Otherwise the two blocks are combined so that
    ``mean(out²) == mean(r_pde²) + bc_weight·mean(r_bc²)``.
    """
    (x, y), bnd = batch
    r_pde = pde_residual(model, x, y, a1, a2, k)
    if bc_weight is None:
        return r_pde
    return stack_residuals(
        [r_pde, bc_residual(model, *bnd, a1, a2)], [1.0, bc_weight])


# ========================= Eval =========================

def make_eval_set(
    ns: int, a1: float, a2: float, device: torch.device, dtype: torch.dtype
):
    """Tensor-product ``ns × ns`` grid on ``[-1, 1]²`` plus the exact ``u``.

    Endpoints included. Resolution has to beat the solution's own frequency:
    ``sin(a2·π·y)`` has period ``2/a2``, so ``ns`` should give many points per
    period at the highest ``a`` you run.
    """
    kw = dict(device=device, dtype=dtype)
    x = torch.linspace(X_MIN, X_MAX, ns, **kw)
    y = torch.linspace(Y_MIN, Y_MAX, ns, **kw)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coords = (xx.reshape(-1, 1), yy.reshape(-1, 1))
    return coords, exact_solution(*coords, a1, a2)


def eval_errors(
    model: nn.Module, coords, u_ref: torch.Tensor, batch_size: int = 65536,
) -> tuple[float, float]:
    """Return ``(rel_l2, rel_linf)`` against the exact solution on the grid.

    Both are reported because Table 3 reports both, and they separate cleanly
    here: rel_L2 is an average over a domain that is mostly easy, rel_Linf
    catches the antinodes where the oscillation is hardest to fit.
    """
    x, y = coords
    was_training = model.training
    model.eval()
    sq = torch.zeros((), device=u_ref.device, dtype=u_ref.dtype)
    mx = torch.zeros((), device=u_ref.device, dtype=u_ref.dtype)
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            err = (model(x[sl], y[sl]) - u_ref[sl]).abs()
            sq += err.pow(2).sum()
            mx = torch.maximum(mx, err.max())
    if was_training:
        model.train()
    rel_l2 = float(sq.sqrt() / u_ref.pow(2).sum().sqrt())
    rel_linf = float(mx / u_ref.abs().max())
    return rel_l2, rel_linf


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

    scheduler = cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    cfg["warmup"] = warmup
    cfg["cosine_decay_floor"] = cosine_decay
    return opt, cfg, scheduler


def lbfgs_phase(
    model: nn.Module, fixed_batch, eval_coords, u_ref: torch.Tensor,
    args: argparse.Namespace, run: RunLogger, bc_weight: float | None,
    start_step: int, best: dict[str, float],
) -> dict:
    """Full-batch L-BFGS refinement, run after the AdamW phase.

    This is the paper's Table 3 case 1 comparator — "BFGS (PyTorch)", which
    reached rel_L2 3.7e-7. Note that is ~100x *worse* than the ~1e-9 cluster;
    what gets to 1e-9 there is the **self-scaled** quasi-Newton variants
    (SSBFGS / SSBroyden) paired with a trust-region or Wolfe line search.
    Ordinary L-BFGS has no self-scaling, so 1e-7 is the number to beat here,
    not 1e-9.

    L-BFGS approximates curvature from a history of (grad, step) pairs, which
    is only valid if the objective is a fixed function — so we draw ONE
    collocation set here and reuse it for every iteration (unlike the AdamW
    phase, which resamples each step). ``torch.optim.LBFGS`` re-evaluates the
    closure several times per ``.step()`` for its strong-Wolfe line search;
    each outer step runs ``--lbfgs-max-iter`` inner iterations (must be >1 —
    the line search cannot recover from a cold identity Hessian in a single
    iteration and stalls).
    """
    opt = torch.optim.LBFGS(
        model.parameters(), lr=args.lbfgs_lr, max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history, line_search_fn="strong_wolfe",
        tolerance_grad=1e-16, tolerance_change=1e-16,
    )
    log_every = max(1, args.log_every // args.lbfgs_max_iter)

    if not args.quiet:
        total = args.lbfgs_steps * args.lbfgs_max_iter
        print(
            f"[{EXPERIMENT}] L-BFGS refinement: {args.lbfgs_steps} outer steps "
            f"x {args.lbfgs_max_iter} = {total} iters on a fixed batch "
            f"(N_pde={args.n_pde}, history={args.lbfgs_history}, "
            f"lr={args.lbfgs_lr})",
            flush=True,
        )

    t0 = time.perf_counter()
    last_loss = float("nan")
    last = {"rel_l2": float("nan"), "rel_linf": float("nan")}

    for i in range(args.lbfgs_steps):
        def closure():
            opt.zero_grad()
            r = residuals(model, fixed_batch, args.a1, args.a2, args.k,
                          bc_weight)
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
            rl2, rlinf = eval_errors(model, eval_coords, u_ref)
            last = {"rel_l2": rl2, "rel_linf": rlinf}
            best["rel_l2"] = min(best["rel_l2"], rl2)
            best["rel_linf"] = min(best["rel_linf"], rlinf)
            run.log_val(step + 1, loss=last_loss, lr=args.lbfgs_lr,
                        rel_l2=rl2, rel_linf=rlinf)
            if not args.quiet:
                ms_per = (time.perf_counter() - t0) / (i + 1) * 1000
                print(
                    f"  L-BFGS {i + 1:5d}/{args.lbfgs_steps}  "
                    f"loss={last_loss:.4e}  rel_l2={rl2:.3e}  "
                    f"rel_linf={rlinf:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )

    return {"last_avg": last_loss, "last": last, "best": best}


# ========================= CLI / training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw", "adamw+lbfgs"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--a1", type=float, default=1.0,
                   help="x-frequency of the exact solution sin(a1*pi*x).")
    p.add_argument("--a2", type=float, default=4.0,
                   help="y-frequency of the exact solution sin(a2*pi*y).")
    p.add_argument("--k", type=float, default=1.0,
                   help="Helmholtz wavenumber. Paper rungs: 1, 10, 100. "
                        "k=100 wants the domain-expansion curriculum, which "
                        "is not implemented here.")
    p.add_argument("--n-pde", type=int, default=10000,
                   help="Interior collocation points per step. Paper uses "
                        "15,000-25,000 with RAD refresh; Table 4 shows 10,000 "
                        "already reaches rel_Linf 1.5e-8, so that is the "
                        "default here.")
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch size for Gnome is max(8, int(n_pde * "
                        "aux_frac)). Each aux pass is a full second-order "
                        "residual eval, so this is not free — keep small.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--trust-region", type=float, default=1.0,
                   help="Gnome l2 trust radius: lambda is set to the smallest "
                        "value >= eps with ||m_hat/(v_hat+lambda)||_2 <= "
                        "this * sqrt(P), i.e. a bound on the RMS per-coordinate "
                        "step. Larger -> weaker bound -> longer steps. "
                        "0 disables it, falling back to plain m_hat/(v_hat+eps) "
                        "damping.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m_hat/(v_hat+eps): "
                        "larger -> more gradient-descent-like, smaller -> "
                        "fuller Newton step. Gnome only; SOAP/AdamW keep their "
                        "fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) "
                        "for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=30, help="MLP width.")
    p.add_argument("--depth", type=int, default=5,
                   help="Total linear layers (paper: 4 hidden + output = 5). "
                        "Ladder: 5 / 7 / 9 -> 2971 / 4831 / 6691 params at "
                        "--modes 1.")
    p.add_argument("--modes", type=int, default=1,
                   help="Fourier embedding modes M; input width is 4M. Paper "
                        "uses M=1 for low frequency and M=10 for the k=10/100 "
                        "rungs (--depth 7 --modes 10 -> 5911 params). Ignored "
                        "when --embedding raw.")
    p.add_argument("--embedding", type=str, default="fourier",
                   choices=["fourier", "raw"],
                   help="fourier: periodic input features, BCs hold exactly, "
                        "single-block PDE-only loss (the paper's setup). "
                        "raw: feed (x,y) directly — periodicity is then NOT "
                        "enforced, so a BC block is required (see --bc) or the "
                        "problem is ill-posed.")
    p.add_argument("--bc", type=str, default="auto",
                   choices=["auto", "none", "dirichlet"],
                   help="Boundary treatment. auto = none with --embedding "
                        "fourier, dirichlet with raw. Set explicitly to get "
                        "the off-diagonal combinations: `--embedding fourier "
                        "--bc dirichlet` adds a BC block while keeping the "
                        "embedding, which is the clean single-variable "
                        "ablation (embedding fixed, block count changed).")
    p.add_argument("--n-bc", type=int, default=1000,
                   help="Boundary points per step when the BC block is on.")
    p.add_argument("--bc-weight", type=float, default=1.0,
                   help="Weight on the BC block: mean(out^2) = mean(r_pde^2) "
                        "+ bc_weight*mean(r_bc^2). At k=1 a DC offset c gives "
                        "|r_pde| = k^2*c = c and |r_bc| = c, so bc_weight=1 "
                        "only doubles DC's visibility — raise it to actually "
                        "pin the constant mode.")
    p.add_argument("--dtype", type=str, default="float64",
                   choices=["float32", "float64"],
                   help="float64 by default: the target here is 1e-9, which "
                        "float32 cannot represent in the residual. MPS has no "
                        "float64 and falls back to CPU.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps, applied to every optimizer.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--n-eval", type=int, default=401,
                   help="Points per axis in the fixed eval grid (401 -> 160k "
                        "points). Raise it for high --a1/--a2.")
    p.add_argument("--lbfgs-steps", type=int, default=500,
                   help="Outer L-BFGS steps after the AdamW phase "
                        "(--optimizer adamw+lbfgs only). Each outer step runs "
                        "up to --lbfgs-max-iter inner iterations, so the total "
                        "budget is lbfgs_steps * lbfgs_max_iter. Runs on a "
                        "single fixed collocation batch. 0 skips the phase.")
    p.add_argument("--lbfgs-max-iter", type=int, default=20,
                   help="Inner iterations per L-BFGS outer step. Must be >1: "
                        "the strong-Wolfe line search cannot recover from a "
                        "cold identity Hessian in one iteration and stalls. "
                        "adamw+lbfgs only.")
    p.add_argument("--lbfgs-history", type=int, default=50,
                   help="L-BFGS history size (stored curvature pairs). "
                        "adamw+lbfgs only.")
    p.add_argument("--lbfgs-lr", type=float, default=1.0,
                   help="L-BFGS step scale; with a strong-Wolfe line search "
                        "1.0 is standard. adamw+lbfgs only.")
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
        device = torch.device("cpu")            # MPS has no float64
        if not args.quiet:
            print(f"[{EXPERIMENT}] float64 requested; MPS has no float64 "
                  f"— falling back to CPU.", flush=True)
    torch.set_default_dtype(dtype)              # periodic-frequency buffer dtype

    use_bc = args.bc == "dirichlet" or (
        args.bc == "auto" and args.embedding == "raw")
    bc_weight = args.bc_weight if use_bc else None
    if args.embedding == "raw" and not use_bc:
        print(f"[{EXPERIMENT}] WARNING: --embedding raw with no BC block. "
              f"Nothing enforces the boundary conditions, so Helmholtz on a "
              f"bounded domain is ill-posed here — any homogeneous solution "
              f"(Lv = 0) can be added freely. Results are not meaningful.",
              flush=True)

    model = PINN(hidden=args.hidden, depth=args.depth, modes=args.modes,
                 embedding=args.embedding).to(device=device, dtype=dtype)
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        warmup=args.warmup_steps, total_steps=args.steps,
        cosine_decay=args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
        trust_region=args.trust_region,
    )

    n_aux = max(8, int(args.n_pde * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
        "steps": args.steps,
        "dtype": args.dtype,
        "a1": args.a1,
        "a2": args.a2,
        "k": args.k,
        "hidden": args.hidden,
        "depth": args.depth,
        "modes": args.modes,
        "embedding": args.embedding,
        "bc": "dirichlet" if use_bc else "none",
        "bc_weight": bc_weight,
        "n_bc": args.n_bc if use_bc else 0,
        "n_pde": args.n_pde,
        "n_aux": n_aux,
        "n_params": n_params,
        "n_eval": args.n_eval,
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
            f"a=({args.a1:g},{args.a2:g}) k={args.k:g} | "
            f"embedding={args.embedding}"
            f"{f'(M={args.modes})' if args.embedding == 'fourier' else ''} | "
            f"bc={'dirichlet w=' + str(bc_weight) if use_bc else 'none'} | "
            f"dtype={args.dtype} | device={device}\n"
            f"  N_pde={args.n_pde}"
            f"{f' N_bc={args.n_bc}' if use_bc else ''} | "
            f"aux={n_aux} | steps={args.steps}",
            flush=True,
        )
    eval_coords, u_ref = make_eval_set(
        args.n_eval, args.a1, args.a2, device, dtype)

    if diag is not None and not args.quiet:
        print(f"  diagnostics every {args.diagnostics_every} steps "
              f"→ {diag.path}", flush=True)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = last_rel_linf = float("nan")
    best_avg = best_rel_l2 = best_rel_linf = float("inf")

    n_bc_aux = max(2, int(args.n_bc * args.aux_frac))

    def draw(n_int: int, n_bnd: int):
        return (sample_batch(n_int, device, dtype),
                sample_boundary(n_bnd, device, dtype) if use_bc else None)

    for step in range(args.steps):
        main_batch = draw(args.n_pde, args.n_bc)
        if args.optimizer == "gnome":
            aux_batch = draw(n_aux, n_bc_aux)

            def main_closure():
                r = residuals(model, main_batch, args.a1, args.a2, args.k,
                              bc_weight)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = residuals(model, aux_batch, args.a1, args.a2, args.k,
                              bc_weight)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = residuals(model, main_batch, args.a1, args.a2, args.k,
                          bc_weight)
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
            rl2, rlinf = eval_errors(model, eval_coords, u_ref)
            last_avg = sum(window) / len(window)
            last_rel_l2, last_rel_linf = rl2, rlinf
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            best_rel_linf = min(best_rel_linf, rlinf)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        rel_l2=rl2, rel_linf=rlinf)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"rel_l2={rl2:.3e}  rel_linf={rlinf:.3e}  "
                    f"{ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    if args.optimizer == "adamw+lbfgs" and args.lbfgs_steps > 0:
        best_d = {"rel_l2": best_rel_l2, "rel_linf": best_rel_linf}
        res = lbfgs_phase(
            model, draw(args.n_pde, args.n_bc), eval_coords, u_ref,
            args, run, bc_weight, args.steps, best_d,
        )
        last_avg = res["last_avg"]
        best_avg = min(best_avg, last_avg)
        last_rel_l2 = res["last"]["rel_l2"]
        last_rel_linf = res["last"]["rel_linf"]
        best_rel_l2 = res["best"]["rel_l2"]
        best_rel_linf = res["best"]["rel_linf"]

    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
        final_rel_linf=last_rel_linf, best_rel_linf=best_rel_linf,
    )

    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    print(f"  final rel_linf={last_rel_linf:.3e}  "
          f"best rel_linf={best_rel_linf:.3e}")
    if args.a1 == 1.0 and args.a2 == 4.0 and args.k == 1.0:
        print("  (Jnini et al. Table 3: SOAP 4.2e-4, NG 4.2e-9 rel_l2)")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
