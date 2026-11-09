"""(2+1)D viscous Burgers PINN: AdamW vs SOAP vs Gnome vs ENGD-W.

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

All optimizers share one plain tanh MLP. AdamW, SOAP, and Gnome use a
linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then constant,
which suits Gnome on MSE since its step self-anneals as the residual shrinks).

ENGD-W uses float64, fixed damping, and no weight decay or LR schedule.
Its residual blocks are scaled by 1/sqrt(N_block). Optional momentum,
same-batch grid line search, and fixed collocation are configured in the
``optimizers['engdw']`` block. The dense solve uses all residual rows;
``engdw_chunk`` limits Jacobian evaluation chunks, not the solve size.

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

    # Edit experiments/configs/pinns/burgers2d_pinn.py: optimizer='gnome', seed=0
    uv run -m experiments.pinns.burgers2d_pinn
    # Edit experiments/configs/pinns/burgers2d_pinn.py: optimizer='soap', seed=0
    uv run -m experiments.pinns.burgers2d_pinn
    # Edit experiments/configs/pinns/burgers2d_pinn.py: optimizer='adamw', seed=0
    uv run -m experiments.pinns.burgers2d_pinn
    # ENGD-W: start with small batches for the dense solve.
    # Edit CONFIG: optimizer='engdw', n_pde=128, n_ic=32, n_bc=32, steps=1000
    uv run -m experiments.pinns.burgers2d_pinn
    # paper protocol (20k iters, float64):
    # Edit experiments/configs/pinns/burgers2d_pinn.py: optimizer='gnome', dtype='float64'
    uv run -m experiments.pinns.burgers2d_pinn
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


def residual_partition_sizes(batch) -> list[int]:
    """One scalar residual per point in each actual auxiliary block."""
    return [coords[0].numel() for coords in batch]


def stacked_residuals(model: nn.Module, batch) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights)."""
    pde, ic, bc = batch
    return stack_residuals([
        pde_residual(model, *pde),
        ic_residual(model, *ic),
        bc_residual(model, *bc),
    ])


def make_engdw_functions(model: nn.Module, chunk_size: int = 32):
    """Functional residual/Jacobian with ||r||² equal to sum of block MSEs.

    Parameter derivatives include both factors of u * (u_x + u_y) and both
    spatial second derivatives. IC and all four boundary faces use the same
    exact solution as the autograd training path.
    """
    def u_single(params, txy):
        return functional_call(
            model, params, tuple(txy[i:i + 1].reshape(1, 1) for i in range(3))
        ).squeeze()

    gradient = jacrev(u_single, argnums=1)
    hessian = jacrev(gradient, argnums=1)

    def pde_single(params, txy):
        du = gradient(params, txy)
        d2u = hessian(params, txy)
        return (du[0] + u_single(params, txy) * (du[1] + du[2])
                - NU * (d2u[1, 1] + d2u[2, 2]))

    def data_single(params, txy):
        return u_single(params, txy) - exact_solution(*txy.unbind())

    def points(batch):
        pde, (x_ic, y_ic), bc = batch
        return (torch.cat(pde, 1),
                torch.cat((torch.zeros_like(x_ic), x_ic, y_ic), 1),
                torch.cat(bc, 1))

    functions = (pde_single, data_single, data_single)

    def residual(params, batch):
        blocks = [vmap(fn, in_dims=(None, 0), chunk_size=chunk_size)(params, pts)
                  / math.sqrt(len(pts)) for fn, pts in zip(functions, points(batch))]
        return torch.cat(blocks).unsqueeze(1)

    def jacobian(params, batch):
        blocks = []
        for fn, pts in zip(functions, points(batch)):
            j = vmap(jacrev(fn), in_dims=(None, 0), chunk_size=chunk_size)(params, pts)
            blocks.append({k: v / math.sqrt(len(pts)) for k, v in j.items()})
        return {k: torch.cat([block[k] for block in blocks]) for k in params}

    return residual, jacobian


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

def parse_args(argv=None) -> argparse.Namespace:
    from experiments.common.config import ConfigParser
    p = ConfigParser(__file__)
    args = p.parse_args(argv)
    for name in ("steps", "n_pde", "n_ic", "n_bc", "hidden", "depth", "n_eval_t", "n_eval_x"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if args.engdw_fixed and args.optimizer != "engdw":
        p.error("engdw_fixed requires optimizer='engdw'")
    if args.diagnostics_every > 0 and args.optimizer != "gnome":
        p.error("diagnostics_every is Gnome-only")
    if args.optimizer == "engdw":
        if not math.isfinite(args.engdw_damping) or args.engdw_damping <= 0:
            p.error("engdw_damping must be finite and positive")
        if not 0 <= args.engdw_momentum < 1:
            p.error("engdw_momentum must be in [0, 1)")
        if args.engdw_chunk < 1:
            p.error("engdw_chunk must be positive")
        if not math.isfinite(args.lr) or args.lr < 0:
            p.error("lr must be finite and nonnegative")
    return args


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = torch.float64 if args.optimizer == "engdw" else getattr(torch, args.dtype)
    if device.type == "mps" and dtype is torch.float64:
        device = torch.device("cpu")            # MPS has no float64
        if not args.quiet:
            print(f"[{EXPERIMENT}] float64 requested; MPS has no float64 "
                  f"— falling back to CPU.", flush=True)

    model = PINN(hidden=args.hidden, depth=args.depth).to(device=device, dtype=dtype)
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

    n_pde_aux = max(8, int(args.n_pde * args.aux_frac)) if args.optimizer == "gnome" else 0
    n_ic_aux = max(2, int(args.n_ic * args.aux_frac)) if args.optimizer == "gnome" else 0
    n_bc_aux = max(2, int(args.n_bc * args.aux_frac)) if args.optimizer == "gnome" else 0
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
        "partition_residuals": args.optimizer == "gnome" and args.partition_residuals,
        "steps": args.steps,
        "dtype": str(dtype).removeprefix("torch."),
        "sampling": "fixed" if args.engdw_fixed else "resample_each_step",
        "fixed_dataset": args.engdw_fixed,
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

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | params={n_params:,} | "
            f"nu={NU} | dtype={dtype} | device={device}\n"
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

    fixed_batch = (sample_batch(args.n_pde, args.n_ic, args.n_bc, device, dtype)
                   if args.engdw_fixed else None)
    for step in range(args.steps):
        main_batch = fixed_batch if fixed_batch is not None else sample_batch(
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

            loss = opt.step(
                main_closure, aux_closure,
                surrogate_partition_sizes=(
                    residual_partition_sizes(aux_batch) if args.partition_residuals else None
                ),
            )
        elif args.optimizer == "engdw":
            loss = opt.step(main_batch)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch)
            # Match Gnome's internal MSE reduction: sum-of-squares / N.
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if (diag is not None and args.optimizer == "gnome"
                and (step + 1) % args.diagnostics_every == 0):
            for record in opt.get_diagnostics(params=diag.params):
                diag(record)

        applied_lr = current_lr(opt)
        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if diverged(loss_val):
            if diag is not None:
                diag.close()
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val, lr=applied_lr)
        window.append(loss_val)

        if (args.log_every and (step + 1) % args.log_every == 0) or step + 1 == args.steps:
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
