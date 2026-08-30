"""(2+1)D viscous Burgers PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + f(u)_x + g(u)_y = ν·(u_xx + u_yy),   f(u) = g(u) = u²/2
      ⇔  u_t + u·u_x + u·u_y = ν·(u_xx + u_yy)
      (x, y) ∈ [0, 1]²,  t ∈ [0, 1],  ν = 0.004
IC/BC: taken from the exact solution (Dirichlet on all four spatial faces)

Exact solution::

    u(x, y, t) = 1 / (1 + exp[(x + y - t) / (2ν)])

Table 9 of Jnini et al., "Curvature-Aware Optimization for High-Accuracy
PINNs" (§4.3, the (2+1)D viscous Burgers rung). Their reported numbers on
this problem, for orientation:

    SOAP   (21001 iters)  rel_L2 = 1.7e-2      3881 params    414 s
    NG     (20000 iters)  rel_L2 = 5.8e-7      3881 params     75 s

**Why this problem is hard.** The exact solution is a travelling front along
``x + y = t`` whose e-folding thickness is ``2ν = 0.008``, while ``x + y - t``
ranges over ``[-1, 2]``. Measured on the default eval grid, only ~3% of the
domain has ``u`` anywhere between 0.02 and 0.98 — everywhere else the solution
is flat at 0 or 1. Uniform collocation therefore spends ~97% of its points
where there is nothing to learn, and that thin 3% is the whole benchmark. It
is very likely why their SOAP/NG gap here (4 orders) is so much wider than on
their other problems. Expect collocation density and ``--dtype float64`` to
matter more than the optimizer.

PDE/IC/BC residuals are stacked through ``gnome.stack_residuals`` with equal
block weights, so the multi-block MSE ``L = mse(pde) + mse(ic) + mse(bc)``
rides Gnome's single-MSE surrogate and the resulting probe is the per-block
independent Rademacher GGN estimator.

All three optimizers share one plain tanh MLP so the only variable is the
optimizer. Every optimizer gets the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then constant,
which suits Gnome on MSE since its step self-anneals as the residual shrinks).

**Reference architecture.** The paper gives "10 hidden layers, tanh" and a
parameter count of 3881, which pins the width exactly: with 3 inputs and 1
output, ``9w² + 14w + 1 = 3881`` ⟹ ``w = 20``. Hence the defaults
``--hidden 20 --depth 11`` (``depth`` counts *total* linear layers), which
reproduce 3881 parameters on the nose.

**What the paper does not specify**, and what this script assumes instead:

* *Collocation counts.* §4.3 says the training details are in Table 9, but
  Table 9 reports only optimizer/error/params/runtime — there is no ``N_c``
  column (unlike Table 10 for the inviscid case). Defaults here are copied
  from ``burgers_pinn`` (2000/100/100) and exposed as ``--n-pde/--n-ic/
  --n-bc``. This is the first knob to sweep.
* *Time domain.* Only ``t > 0`` is stated. The front sits at ``x + y = t``
  and so crosses the unit square over ``t ∈ [0, 2]``; Figure 12 shows
  ``t = 0.5``. We take ``t ∈ [0, 1]``, which spans the front's passage
  through the domain interior.
* *Loss weights.* Eq. 52 carries five ``λ`` coefficients, none of them given.
  We use equal block weights, which is this repo's default and Gnome's
  natural stacking.
* *Flux relaxation.* §4.3 claims it is used "throughout", per §4.4.2 — a
  second network for the flux plus an algebraic constraint ``F = u²``. But
  3881 parameters is exactly *one* net with a single output, and 2D would
  need flux nets for both ``f`` and ``g`` (~3× the count). We solve the
  plain residual form. The viscous front is smooth and resolved, so the
  entropy/relaxation machinery — which exists to handle true discontinuities
  in the *inviscid* case — should not be load-bearing here.

Usage::

    uv run -m experiments.pinns.burgers2d_pinn --optimizer gnome --seed 0
    uv run -m experiments.pinns.burgers2d_pinn --optimizer soap  --seed 0
    uv run -m experiments.pinns.burgers2d_pinn --optimizer adamw --seed 0
    # paper protocol (20k iters, float64):
    uv run -m experiments.pinns.burgers2d_pinn --optimizer gnome --dtype float64
"""

from __future__ import annotations

import argparse
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
from experiments.common import MLP as _SharedMLP, ConcatEmbedding


EXPERIMENT = "burgers2d_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
NU = 0.004


# ========================= Exact solution =========================

def exact_solution(
    t: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``u = 1/(1 + exp[(x+y-t)/(2ν)])``, written as a sigmoid.

    Algebraically ``1/(1+exp(s)) == sigmoid(-s)``, but the sigmoid form is
    the numerically safe one: ``(x+y-t)/(2ν)`` reaches ``+250`` at the far
    corner, which overflows ``exp`` in float32. ``torch.sigmoid`` saturates
    to 0/1 there instead of producing ``inf``.

    Verified against the PDE by autograd: the residual is ~7e-15 against
    terms of magnitude ~31, i.e. relative 2e-16.
    """
    return torch.sigmoid((t - x - y) / (2.0 * NU))


# ========================= Model =========================

class PINN(_SharedMLP):
    """Maps ``(t, x, y) → u`` via a plain tanh MLP.

    ``depth`` counts *total* linear layers, so ``depth = 11`` is the paper's
    "10 hidden layers" plus the output layer.
    """

    def __init__(self, hidden: int = 20, depth: int = 11):
        super().__init__(ConcatEmbedding(3), hidden=hidden, depth=depth)


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Residual ``u_t + u·u_x + u·u_y - ν·(u_xx + u_yy)`` at ``(t, x, y)``.

    Both fluxes are ``u²/2``, so both advective terms carry the *same* ``u``
    factor — this is a scalar conservation law, not a velocity field.
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = model(t, x, y)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_t + u * u_x + u * u_y - NU * (u_xx + u_yy)


def ic_residual(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """IC residual ``u(0, x, y) - u*(0, x, y)``."""
    t0 = torch.zeros_like(x)
    return model(t0, x, y) - exact_solution(t0, x, y)


def bc_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Dirichlet BC residual ``u - u*`` on points already lying on ∂Ω."""
    return model(t, x, y) - exact_solution(t, x, y)


# ========================= Sampling =========================

def sample_batch(
    n_pde: int, n_ic: int, n_bc: int,
    device: torch.device, dtype: torch.dtype,
):
    """Independent uniform draws for the collocation / IC / BC point sets.

    Boundary points pin one *spatial* coordinate to a face (0 or 1) and leave
    the other two free, so the four faces of the unit square are covered
    uniformly across all of ``t``.
    """
    kw = dict(device=device, dtype=dtype)

    def unif(n, lo, hi):
        return torch.rand(n, 1, **kw) * (hi - lo) + lo

    t_pde = unif(n_pde, T_MIN, T_MAX)
    x_pde = unif(n_pde, X_MIN, X_MAX)
    y_pde = unif(n_pde, Y_MIN, Y_MAX)

    x_ic = unif(n_ic, X_MIN, X_MAX)
    y_ic = unif(n_ic, Y_MIN, Y_MAX)

    t_bc = unif(n_bc, T_MIN, T_MAX)
    x_bc = unif(n_bc, X_MIN, X_MAX)
    y_bc = unif(n_bc, Y_MIN, Y_MAX)
    pin_x = torch.randint(0, 2, (n_bc, 1), device=device) == 0   # pin x or y
    face = torch.randint(0, 2, (n_bc, 1), device=device).to(dtype)
    x_bc = torch.where(pin_x, face, x_bc)
    y_bc = torch.where(pin_x, y_bc, face)

    return (t_pde, x_pde, y_pde), (x_ic, y_ic), (t_bc, x_bc, y_bc)


def stacked_residuals(model: nn.Module, batch) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights)."""
    pde, ic, bc = batch
    return stack_residuals([
        pde_residual(model, *pde),
        ic_residual(model, *ic),
        bc_residual(model, *bc),
    ])


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-term MSE for diagnostic logging."""
    pde, ic, bc = batch
    return {
        "pde": pde_residual(model, *pde).pow(2).mean().item(),
        "ic": ic_residual(model, *ic).pow(2).mean().item(),
        "bc": bc_residual(model, *bc).pow(2).mean().item(),
    }


# ========================= Eval =========================

def make_eval_set(
    nt: int, ns: int, device: torch.device, dtype: torch.dtype
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Tensor-product eval grid ``nt × ns × ns`` plus the exact ``u`` on it.

    A grid rather than a Monte-Carlo draw: the front is ``2ν = 0.008`` thick,
    so a random eval set of any affordable size lands only a handful of points
    inside it and the resulting rel_L2 is dominated by sampling noise. The
    default ``ns = 201`` gives spacing 0.005, just under the front thickness.
    """
    kw = dict(device=device, dtype=dtype)
    t = torch.linspace(T_MIN, T_MAX, nt, **kw)
    x = torch.linspace(X_MIN, X_MAX, ns, **kw)
    y = torch.linspace(Y_MIN, Y_MAX, ns, **kw)
    tt, xx, yy = torch.meshgrid(t, x, y, indexing="ij")
    coords = (tt.reshape(-1, 1), xx.reshape(-1, 1), yy.reshape(-1, 1))
    return coords, exact_solution(*coords)


def eval_rel_l2(
    model: nn.Module, coords, u_ref: torch.Tensor, batch_size: int = 65536,
) -> float:
    """``||u_pred - u*||_2 / ||u*||_2`` over the fixed eval grid.

    Queried in batches under ``no_grad`` so the grid can be far larger than a
    training batch without OOM.
    """
    t, x, y = coords
    was_training = model.training
    model.eval()
    num = torch.zeros((), device=u_ref.device, dtype=u_ref.dtype)
    with torch.no_grad():
        for i in range(0, t.shape[0], batch_size):
            sl = slice(i, i + batch_size)
            pred = model(t[sl], x[sl], y[sl])
            num += (pred - u_ref[sl]).pow(2).sum()
    if was_training:
        model.train()
    return float((num.sqrt() / u_ref.pow(2).sum().sqrt()))


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
    p.add_argument("--steps", type=int, default=20000,
                   help="Paper protocol is 20000 (NG / quasi-Newton) or 21001 "
                        "(SOAP); see Table 9.")
    p.add_argument("--n-pde", type=int, default=2000,
                   help="Interior collocation points per step. The paper does "
                        "not state this for the 2D viscous case, so the "
                        "default is inherited from burgers_pinn. The front is "
                        "only 2nu=0.008 thick, so this is the knob most likely "
                        "to set the accuracy floor — sweep it first.")
    p.add_argument("--n-ic", type=int, default=100,
                   help="Initial-condition points per step (t=0).")
    p.add_argument("--n-bc", type=int, default=100,
                   help="Boundary points per step, spread over all four "
                        "spatial faces of the unit square.")
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so this is not free — "
                        "keep small.")
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
    p.add_argument("--hidden", type=int, default=20,
                   help="MLP width. Default 20 reproduces the paper's 3881 "
                        "parameters with --depth 11.")
    p.add_argument("--depth", type=int, default=11,
                   help="Total linear layers (paper: 10 hidden + output).")
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"],
                   help="float64 is likely needed to approach the paper's "
                        "5.8e-7: second-order autodiff on a 0.008-thick front "
                        "hits a ~1e-5 floor in float32. MPS has no float64 and "
                        "falls back to CPU.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps, applied to every optimizer.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--n-eval-t", type=int, default=26,
                   help="Time slices in the fixed eval grid.")
    p.add_argument("--n-eval-x", type=int, default=201,
                   help="Points per spatial axis in the fixed eval grid "
                        "(201 -> spacing 0.005, just under the 0.008 front).")
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
    p.add_argument("--log-every", type=int, default=100,
                   help="Log a val entry (running train mean + per-term "
                        "diagnostics on a fresh probe batch) every N steps.")
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

    model = PINN(hidden=args.hidden, depth=args.depth).to(device=device, dtype=dtype)
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
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
        "steps": args.steps,
        "dtype": args.dtype,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_pde": args.n_pde,
        "n_ic": args.n_ic,
        "n_bc": args.n_bc,
        "n_pde_aux": n_pde_aux,
        "n_ic_aux": n_ic_aux,
        "n_bc_aux": n_bc_aux,
        "n_params": n_params,
        "n_eval_t": args.n_eval_t,
        "n_eval_x": args.n_eval_x,
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
            f"nu={NU} | dtype={args.dtype} | device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}",
            flush=True,
        )
    eval_coords, u_ref = make_eval_set(
        args.n_eval_t, args.n_eval_x, device, dtype)

    if diag is not None and not args.quiet:
        print(f"  diagnostics every {args.diagnostics_every} steps "
              f"→ {diag.path}", flush=True)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_pde = last_ic = last_bc = float("nan")
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(
            args.n_pde, args.n_ic, args.n_bc, device, dtype)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(
                n_pde_aux, n_ic_aux, n_bc_aux, device, dtype)

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
            if diag is not None:
                diag.close()
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            tl = term_losses(model, sample_batch(
                args.n_pde, args.n_ic, args.n_bc, device, dtype))
            rl2 = eval_rel_l2(model, eval_coords, u_ref)
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
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  ic={tl['ic']:.3e}  "
                    f"bc={tl['bc']:.3e}  rel_l2={rl2:.3e}  "
                    f"{ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )

    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final pde={last_pde:.3e}  ic={last_ic:.3e}  bc={last_bc:.3e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}  "
          f"(Jnini et al. Table 9: SOAP 1.7e-2, NG 5.8e-7)")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
