"""Lid-driven cavity PINN (steady Navier-Stokes): AdamW+L-BFGS vs SOAP vs Gnome.

PDE (steady, incompressible, velocity-pressure form) on ``Ω = (0,1)²``::

    ru:  u·u_x + v·u_y + p_x - ν·(u_xx + u_yy) = 0     (x-momentum)
    rv:  u·v_x + v·v_y + p_y - ν·(v_xx + v_yy) = 0     (y-momentum)
    rc:  u_x + v_y                             = 0     (continuity)

with ``ν = 1/Re``. Boundary conditions are no-slip on three walls and a
moving lid on top: ``v = 0`` everywhere on ``∂Ω``, ``u = 0`` on the left,
right and bottom walls, and ``u = lid(x)`` on ``y = 1``.

Why this benchmark. It is the one problem in the SOAP-PINN suite
(arXiv:2502.00604) that is **steady** — no time dimension, hence no causal
weighting and no time-marching — and it has the largest optimizer spread in
their table (Adam 3.24e-1 → SOAP 3.99e-2, an 8x gap). Two things make it
hard in a way that is specifically about *conditioning* rather than about
basin selection:

* **The incompressibility constraint.** ``u_x + v_y = 0`` is a hard
  algebraic constraint imposed as a soft penalty, so the loss has enormous
  curvature normal to the divergence-free manifold and very little along
  it. That is the classic ill-conditioned-saddle geometry.
* **Five competing blocks** (three residuals, two boundary sets) whose
  natural scales differ by orders of magnitude — pressure gradients against
  a dimensionless divergence against O(1) velocities.

Both are block/curvature-scaling problems with no temporal structure, which
makes this the cleanest venue in the suite for asking whether a true
Gauss-Newton basis buys anything that a schedule cannot.

Architecture is the modified MLP of Wang, Teng & Perdikaris (2021) — two
input encoders gating every hidden layer — with **three outputs** ``(u,v,p)``
from a shared trunk. That is the *architecture only*: none of the rest of
the jaxpi pipeline (random weight factorization, Fourier features, grad-norm
loss balancing, curriculum-aware weighting) is ported. The five residual
blocks go through ``gnome.stack_residuals`` with equal weights, exactly as in
the other torch PINN experiments here.

**Reynolds curriculum.** Re=5000 does not train from a cold start; the
standard remedy (and jaxpi's) is a warm-started ladder of steady solves at
increasing Re, each initialized from the previous one's weights. ``--re-stages``
sets the ladder and ``--stage-steps`` the per-stage budget. The optimizer state
carries across stages too, so each Re jump is a genuine objective shock — worth
watching, since a stale curvature EMA meeting a suddenly-changed objective is
the known failure mode for preconditioned methods.

**Pressure is determined only up to an additive constant** — nothing in the
loss pins its level, so the objective has an exact flat direction. It is
harmless for the metric (relative L2 is scored on ``u, v`` only) and benign
for the step (the gradient vanishes along it, so ``m̂/(v̂+eps)`` gives 0/eps),
but it is a real null mode of the GGN and the reason a curvature-trust floor
matters here.

**Lid profile.** ``--lid regularized`` (default) uses jaxpi's smoothed lid
``1 - cosh(50(x-0.5))/cosh(25)``, which is 1.0 except within ~5% of each top
corner where it rolls off to zero, removing the corner pressure singularity.
Their reference solutions, however, were generated with a *uniform* lid
(``u = 1`` along the whole top edge). So there is a train/reference mismatch
confined to two thin corner strips that no optimizer can fit away — a
candidate explanation for why the published errors floor around 4e-2.
``--lid uniform`` matches the reference exactly at the cost of reinstating the
singularity; the difference between the two is a direct measurement of how
much of the floor is the corner treatment.

Reference: jaxpi's ``ldc_Re{Re}.mat`` (256x256 grid, auto-downloaded to
``experiments/data/``). Relative L2 is reported on the stacked ``(u,v)``
field, with the per-component errors logged alongside.

All optimizers share the chosen network so the only variable is the
optimizer. SOAP and the AdamW phase get a linear-warmup + cosine-decay
schedule (``--cosine-decay`` sets the final-lr fraction; 1.0 disables it);
Gnome runs at a fixed lr — its Gauss-Newton step self-anneals as the residual
shrinks.

``--optimizer adamw+lbfgs`` is the classic PINN recipe (and the paper's real
first-order baseline): the AdamW phase runs the full curriculum, then an
L-BFGS phase (``--lbfgs-steps``) refines at the final Re on a *fixed,
full-batch* collocation set. L-BFGS builds curvature from a history of
(grad, step) pairs, which is only meaningful if the objective is the same
function each iteration — so its points are drawn once and held constant.

Usage::

    uv run -m experiments.ldc_pinn --optimizer gnome
    uv run -m experiments.ldc_pinn --optimizer soap
    uv run -m experiments.ldc_pinn --optimizer adamw+lbfgs

    # short smoke on the first rung only
    uv run -m experiments.ldc_pinn --optimizer gnome \\
        --re-stages 100 --stage-steps 2000 --log-every 200
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


EXPERIMENT = "ldc_pinn"

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0

# jaxpi's regularized lid: 1 - cosh(a(x-1/2))/cosh(a/2), a = 50.
LID_SHARPNESS = 50.0


# ========================= Models =========================

class MLP(nn.Module):
    """Plain tanh MLP: ``(x, y) → (u, v, p)``.

    ``depth`` = number of Linear layers. No input embedding: the cavity is a
    plain square with Dirichlet data, so there is no periodicity to encode.
    """

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=1))


class ModifiedMLP(nn.Module):
    """Modified MLP (Wang, Teng & Perdikaris 2021): ``(x, y) → (u, v, p)``.

    Two input encoders gate every hidden layer. The gate is written in the
    algebraically equivalent form ``h = v + h·(u - v)`` via one fused
    ``addcmul`` (rather than ``h·u + (1-h)·v``, three elementwise kernels and
    three autograd nodes), and the two encoders are fused into a single
    Linear producing ``2·hidden`` features. ``depth`` = gated-hidden-layer
    count.

    The three fields share one trunk and split only at the output layer, so
    ``u``, ``v`` and ``p`` are coupled in the representation the way they are
    coupled in the equations.

    Architecture only — no random weight factorization, Fourier features or
    grad-norm balancing (jaxpi-pipeline pieces, deliberately not ported).
    """

    def __init__(self, hidden: int = 256, depth: int = 4):
        super().__init__()
        assert depth >= 1
        # Fused u/v encoder: one matmul, chunked into the two gates.
        self.enc_uv = nn.Linear(2, 2 * hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(2 if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, 3)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x, y], dim=1)

        uv = torch.tanh(self.enc_uv(z))
        enc_a, enc_b = uv.chunk(2, dim=-1)
        w = enc_a - enc_b  # computed once; gate becomes enc_b + h*w

        h = z
        for layer in self.layers:
            h = torch.tanh(layer(h))
            h = torch.addcmul(enc_b, h, w)  # == h*enc_a + (1-h)*enc_b
        return self.out(h)


def build_model(arch: str, hidden: int, depth: int) -> nn.Module:
    if arch == "mlp":
        return MLP(hidden=hidden, depth=depth)
    if arch == "modified":
        return ModifiedMLP(hidden=hidden, depth=depth)
    raise ValueError(f"unknown arch: {arch}")


# ========================= Boundary data =========================

def lid_velocity(x: torch.Tensor, profile: str) -> torch.Tensor:
    """Horizontal velocity prescribed on the moving lid ``y = 1``.

    ``regularized`` is jaxpi's ``1 - cosh(50(x-1/2))/cosh(25)``: essentially 1
    across the lid but tapering to 0 at both top corners, which removes the
    pressure singularity where the moving lid meets a stationary wall.
    ``uniform`` is the textbook ``u = 1``, which matches the reference data
    exactly but reinstates the singularity.
    """
    if profile == "uniform":
        return torch.ones_like(x)
    if profile == "regularized":
        a = LID_SHARPNESS
        return 1.0 - torch.cosh(a * (x - 0.5)) / math.cosh(a * 0.5)
    raise ValueError(f"unknown lid profile: {profile}")


# ========================= Residuals =========================

def ns_residuals(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, nu: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Steady incompressible NS residuals ``(ru, rv, rc)`` at ``(x, y)``.

    One forward pass, then three first-order backward passes (one per output
    field, each returning both partials at once) and four second-order passes
    for the two Laplacians — only the diagonal of each Hessian is needed.
    """
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    out = model(x, y)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    ones = torch.ones_like(u)
    u_x, u_y = autograd.grad(u, (x, y), ones, create_graph=True)
    v_x, v_y = autograd.grad(v, (x, y), ones, create_graph=True)
    p_x, p_y = autograd.grad(p, (x, y), ones, create_graph=True)

    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    ru = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    rv = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    rc = u_x + v_y
    return ru, rv, rc


def bc_residuals(
    model: nn.Module, x_bc: torch.Tensor, y_bc: torch.Tensor,
    u_bc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dirichlet residuals on ``∂Ω``: ``(u - u_bc, v - 0)``.

    Kept as two blocks rather than one so the tangential (lid-driven) and
    normal (no-through-flow) constraints are weighted independently — they
    have genuinely different residual scales.
    """
    out = model(x_bc, y_bc)
    return out[:, 0:1] - u_bc, out[:, 1:2]


# ========================= Sampling =========================

def sample_interior(n: int, device: torch.device):
    x = torch.rand(n, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    return x, y


def sample_boundary(n_per_edge: int, device: torch.device, lid: str):
    """Uniform draws on each of the four walls.

    Returns ``(x, y, u_target)`` over the concatenated edges
    [bottom, top, left, right]. The lid profile supplies ``u_target`` on the
    top edge; every other wall is no-slip.
    """
    s = torch.rand(n_per_edge, 1, device=device)
    t = torch.rand(n_per_edge, 1, device=device)
    zeros = torch.zeros(n_per_edge, 1, device=device)
    ones = torch.ones(n_per_edge, 1, device=device)

    x_bot, y_bot = s, zeros
    x_top, y_top = t, ones
    x_lef, y_lef = zeros, s
    x_rig, y_rig = ones, t

    x = torch.cat([x_bot, x_top, x_lef, x_rig], dim=0)
    y = torch.cat([y_bot, y_top, y_lef, y_rig], dim=0)
    u_target = torch.cat([
        torch.zeros_like(x_bot),
        lid_velocity(x_top, lid),
        torch.zeros_like(x_lef),
        torch.zeros_like(x_rig),
    ], dim=0)
    return x, y, u_target


def sample_batch(n_pde: int, n_bc: int, device: torch.device, lid: str):
    x_pde, y_pde = sample_interior(n_pde, device)
    x_bc, y_bc, u_bc = sample_boundary(n_bc, device, lid)
    return x_pde, y_pde, x_bc, y_bc, u_bc


BLOCK_NAMES = ("ru", "rv", "rc", "u_bc", "v_bc")


def stacked_residuals(
    model: nn.Module, batch, nu: float,
    weights: list[float] | None = None,
) -> torch.Tensor:
    """The five blocks stacked via ``stack_residuals``.

    ``weights`` are the per-block λ_j in ``BLOCK_NAMES`` order, so the loss is
    ``Σ_j λ_j · mean(r_j²)`` — the same form as jaxpi's weighted sum, which
    makes their ``init_weights`` directly transferable. Defaults to equal
    weighting, as in the other torch PINN experiments here.
    """
    x_pde, y_pde, x_bc, y_bc, u_bc = batch
    ru, rv, rc = ns_residuals(model, x_pde, y_pde, nu)
    r_u_bc, r_v_bc = bc_residuals(model, x_bc, y_bc, u_bc)
    return stack_residuals([ru, rv, rc, r_u_bc, r_v_bc], weights)


def term_losses(model: nn.Module, batch, nu: float) -> dict[str, float]:
    """Per-block MSE for diagnostic logging."""
    x_pde, y_pde, x_bc, y_bc, u_bc = batch
    ru, rv, rc = ns_residuals(model, x_pde, y_pde, nu)
    r_u_bc, r_v_bc = bc_residuals(model, x_bc, y_bc, u_bc)
    return {
        "ru": ru.pow(2).mean().item(),
        "rv": rv.pow(2).mean().item(),
        "rc": rc.pow(2).mean().item(),
        "u_bc": r_u_bc.pow(2).mean().item(),
        "v_bc": r_v_bc.pow(2).mean().item(),
    }


# ========================= Reference solution + eval =========================

DEFAULT_REF_CACHE_DIR = "experiments/data"
REFERENCE_URL_FMT = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/ldc/data/ldc_Re{re}.mat"
)
AVAILABLE_RE = (100, 400, 1000, 1600, 3200, 5000)


def ldc_reference(re: int, cache_dir: str | None = None):
    """jaxpi's lid-driven-cavity reference at Reynolds number ``re``.

    Auto-downloaded to ``experiments/data/``. Returns ``(x, y, u, v)`` with
    shapes ``(nx,)``, ``(ny,)``, ``(nx, ny)``, ``(nx, ny)`` — the fields are
    indexed ``[x_index, y_index]`` (``meshgrid(..., indexing='ij')``).

    Note the reference was generated with a *uniform* lid; see ``--lid``.
    """
    import scipy.io

    if re not in AVAILABLE_RE:
        raise ValueError(
            f"no reference solution for Re={re}; available: {AVAILABLE_RE}"
        )
    cache_dir = cache_dir or DEFAULT_REF_CACHE_DIR
    cache_path = os.path.join(cache_dir, f"ldc_Re{re}.mat")
    if not os.path.isfile(cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        url = REFERENCE_URL_FMT.format(re=re)
        print(f"[{EXPERIMENT}] downloading reference {url} ...", flush=True)
        urllib.request.urlretrieve(url, cache_path)
    data = scipy.io.loadmat(cache_path)
    dt = torch.get_default_dtype()
    x = torch.as_tensor(data["x"].flatten()).to(dt)
    y = torch.as_tensor(data["y"].flatten()).to(dt)
    u = torch.as_tensor(data["u"]).to(dt)
    v = torch.as_tensor(data["v"]).to(dt)
    return x, y, u, v


def eval_rel_l2(
    model: nn.Module, ref, device: torch.device, batch_size: int = 8192,
) -> tuple[float, float, float]:
    """Relative L2 error against the reference on its own grid.

    Returns ``(rel_l2, rel_l2_u, rel_l2_v)`` — the headline number is the
    error of the stacked velocity field ``(u, v)``, which is the honest
    single-number summary when the two components have different magnitudes
    (``v`` is roughly half of ``u`` in a cavity, so averaging the two
    per-component ratios would overweight it).
    """
    x_ref, y_ref, u_ref, v_ref = ref
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

    out = torch.cat(preds)
    u_pred = out[:, 0].reshape(nx, ny)
    v_pred = out[:, 1].reshape(nx, ny)

    def _rel(pred, ref_field):
        return float((pred - ref_field).pow(2).sum().sqrt()
                     / ref_field.pow(2).sum().sqrt())

    num = ((u_pred - u_ref).pow(2).sum() + (v_pred - v_ref).pow(2).sum()).sqrt()
    den = (u_ref.pow(2).sum() + v_ref.pow(2).sum()).sqrt()
    return float(num / den), _rel(u_pred, u_ref), _rel(v_pred, v_ref)


# ========================= Optimizer factory =========================

def build_optimizer(
    name: str, params, lr: float, weight_decay: float,
    warmup: int, total_steps: int, cosine_decay: float, eps: float = 1e-6,
    beta1: float = 0.9, beta2: float = 0.99,
    trust_region: float = 1.0,
):
    """Construct the optimizer and its LR schedule.

    Returns ``(optimizer, config, scheduler_or_None)``. Gnome runs at a fixed
    lr (its Gauss-Newton step self-anneals as the residual shrinks) so it gets
    no scheduler — only its own internal warmup. SOAP and AdamW get the
    standard linear-warmup + cosine-decay treatment; ``cosine_decay`` is the
    final-lr fraction (0.0 → decay to zero, 1.0 → decay disabled). The
    schedule spans the whole Reynolds curriculum, not one stage.
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=20,
            warmup=warmup,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse", precondition_1d=True,
        )
        return Gnome(params, **cfg), cfg, None
    if name == "soap":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=1e-8,
            precondition_frequency=20, precondition_1d=True,
        )
        opt = SOAP(params, **cfg)
    elif name == "adamw+lbfgs":
        # Phase 1 is plain AdamW; the L-BFGS refinement is a separate phase
        # appended after the curriculum (see lbfgs_phase).
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
    run: RunLogger, ref, nu: float, re: int, start_step: int,
    best_rel_l2: float,
) -> dict:
    """Full-batch L-BFGS refinement at the final Reynolds number.

    L-BFGS approximates curvature from a history of (grad, step) pairs, which
    is only valid if the objective is a fixed function — so we draw ONE
    collocation set here and reuse it for every iteration (unlike the AdamW
    phase, which resamples each step). ``torch.optim.LBFGS`` re-evaluates the
    closure several times per ``.step()`` for its strong-Wolfe line search;
    each outer step runs ``--lbfgs-max-iter`` inner iterations (must be >1 —
    the line search cannot recover from a cold identity Hessian in a single
    iteration and stalls). Returns final/best metrics for the run summary.
    """
    fixed_batch = sample_batch(args.n_pde, args.n_bc, device, args.lid)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=args.lbfgs_lr, max_iter=args.lbfgs_max_iter,
        history_size=args.lbfgs_history, line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, tolerance_change=1e-12,
    )
    # Log at roughly the same iteration cadence as the AdamW phase: one outer
    # step is lbfgs_max_iter inner iterations.
    log_every = max(1, args.log_every // args.lbfgs_max_iter)

    if not args.quiet:
        total_iters = args.lbfgs_steps * args.lbfgs_max_iter
        print(
            f"[{EXPERIMENT}] L-BFGS refinement @ Re={re}: {args.lbfgs_steps} "
            f"outer steps x {args.lbfgs_max_iter} = {total_iters} iters on a "
            f"fixed batch (N_pde={args.n_pde} N_bc/edge={args.n_bc}, "
            f"history={args.lbfgs_history}, lr={args.lbfgs_lr})",
            flush=True,
        )

    t_start = time.perf_counter()
    last_loss = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}

    for i in range(args.lbfgs_steps):
        def closure():
            opt.zero_grad()
            r = stacked_residuals(model, fixed_batch, nu, args.block_weights)
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

        # Always evaluate on the last iteration, so a short phase (fewer outer
        # steps than the log cadence) still reports a final rel_l2.
        if (i + 1) % log_every == 0 or i == args.lbfgs_steps - 1:
            tl = term_losses(model, fixed_batch, nu)
            rl2, rl2_u, rl2_v = eval_rel_l2(model, ref, device)
            last_terms, last_rel_l2 = tl, rl2
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_loss, lr=args.lbfgs_lr,
                        re=re, nu=nu, rel_l2=rl2, rel_l2_u=rl2_u,
                        rel_l2_v=rl2_v, **tl)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (i + 1) * 1000
                print(
                    f"  L-BFGS {i + 1:5d}/{args.lbfgs_steps}  "
                    f"loss={last_loss:.4e}  "
                    f"ru={tl['ru']:.2e} rv={tl['rv']:.2e} rc={tl['rc']:.2e} "
                    f"u_bc={tl['u_bc']:.2e} v_bc={tl['v_bc']:.2e}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )

    return {
        "last_avg": last_loss,
        "last_rel_l2": last_rel_l2,
        "best_rel_l2": best_rel_l2,
        "last_terms": last_terms,
    }


# ========================= CLI / training =========================

def _int_list(s: str) -> list[int]:
    return [int(v) for v in s.replace(" ", "").split(",") if v]


def _float_list(s: str) -> list[float]:
    return [float(v) for v in s.replace(" ", "").split(",") if v]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw+lbfgs"])
    p.add_argument("--arch", choices=["mlp", "modified"], default="modified",
                   help="Network: plain tanh MLP or the gated modified MLP "
                        "(Wang et al. 2021). --hidden / --depth control both.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--re-stages", type=_int_list,
                   default=[100, 400, 1000, 3200, 5000],
                   help="Reynolds-number curriculum, comma-separated. Each "
                        "stage warm-starts from the previous one's weights "
                        "AND optimizer state. Every value needs a reference "
                        f"solution; available: {AVAILABLE_RE}.")
    p.add_argument("--stage-steps", type=_int_list, default=[20000],
                   help="Steps per curriculum stage, comma-separated. A single "
                        "value applies to every stage except the last, which "
                        "gets --final-stage-steps.")
    p.add_argument("--final-stage-steps", type=int, default=140_000,
                   help="Steps for the last curriculum stage when "
                        "--stage-steps is a single value. The final Re is "
                        "where the accuracy is actually earned.")
    p.add_argument("--lid", choices=["regularized", "uniform"],
                   default="regularized",
                   help="Lid velocity profile. 'regularized' is jaxpi's "
                        "1-cosh(50(x-1/2))/cosh(25), which removes the corner "
                        "singularity but does NOT match the reference data "
                        "(generated with a uniform lid) in two thin corner "
                        "strips. 'uniform' matches the reference exactly and "
                        "reinstates the singularity.")
    p.add_argument("--block-weights", type=_float_list, default=None,
                   help="Static per-block loss weights λ_j, comma-separated in "
                        f"{','.join(BLOCK_NAMES)} order, giving the loss "
                        "Σ_j λ_j·mean(r_j²). Default is equal weighting. "
                        "jaxpi's LDC init_weights are '1,1,10,100,100' — they "
                        "then adapt them with grad-norm balancing, which is "
                        "not ported, so passing that is the static "
                        "approximation of their starting point.")
    p.add_argument("--n-pde", type=int, default=4000,
                   help="Interior collocation points per step.")
    p.add_argument("--n-bc", type=int, default=256,
                   help="Boundary points per edge per step (4x this total).")
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "second-order residual eval, so keep small.")
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
                        "-> more gradient-descent-like, smaller -> fuller "
                        "Newton step. Gnome only; SOAP/AdamW keep eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
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
                   help="L-BFGS OUTER steps after the AdamW phase "
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


def resolve_stage_steps(args: argparse.Namespace) -> list[int]:
    """Expand ``--stage-steps`` against the curriculum length."""
    stages, steps = args.re_stages, args.stage_steps
    if not stages:
        raise ValueError("--re-stages must list at least one Reynolds number")
    if len(steps) == len(stages):
        return list(steps)
    if len(steps) == 1:
        if len(stages) == 1:
            return list(steps)
        return [steps[0]] * (len(stages) - 1) + [args.final_stage_steps]
    raise ValueError(
        f"--stage-steps has {len(steps)} entries but --re-stages has "
        f"{len(stages)}; pass one value per stage or a single value"
    )


def train(args: argparse.Namespace) -> str:
    device = pick_device()
    torch.manual_seed(args.seed)

    stage_steps = resolve_stage_steps(args)
    total_steps = sum(stage_steps)
    blk_w = args.block_weights
    if blk_w is not None and len(blk_w) != len(BLOCK_NAMES):
        raise ValueError(
            f"--block-weights needs {len(BLOCK_NAMES)} values in "
            f"{','.join(BLOCK_NAMES)} order, got {len(blk_w)}"
        )
    for re in args.re_stages:
        if re not in AVAILABLE_RE:
            raise ValueError(
                f"no reference solution for Re={re}; available: {AVAILABLE_RE}"
            )

    model = build_model(args.arch, args.hidden, args.depth).to(device)
    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        warmup=args.warmup_steps, total_steps=total_steps,
        cosine_decay=args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
        trust_region=args.trust_region,
    )

    n_pde_aux = max(8, int(args.n_pde * args.aux_frac))
    n_bc_aux = max(2, int(args.n_bc * args.aux_frac))
    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "arch": args.arch,
        "re_stages": list(args.re_stages),
        "stage_steps": stage_steps,
        "steps": total_steps,
        "lid": args.lid,
        "block_weights": blk_w or [1.0] * len(BLOCK_NAMES),
        "hidden": args.hidden,
        "depth": args.depth,
        "n_pde": args.n_pde,
        "n_bc": args.n_bc,
        "n_pde_aux": n_pde_aux,
        "n_bc_aux": n_bc_aux,
        "n_params": n_params,
        "x_domain": (X_MIN, X_MAX),
        "y_domain": (Y_MIN, Y_MAX),
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

    if not args.quiet:
        ladder = " -> ".join(
            f"Re{re}({n:,})" for re, n in zip(args.re_stages, stage_steps)
        )
        print(
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | params={n_params:,} | "
            f"device={device}\n"
            f"  N_pde={args.n_pde} N_bc/edge={args.n_bc} | "
            f"aux={n_pde_aux}/{n_bc_aux} | lid={args.lid}\n"
            f"  curriculum: {ladder} | total={total_steps:,} steps",
            flush=True,
        )
        print("  loading / downloading reference solutions...", flush=True)
    references = {re: ldc_reference(re) for re in args.re_stages}

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}
    best_avg = best_rel_l2 = float("inf")
    global_step = 0

    for stage, (re, n_steps) in enumerate(zip(args.re_stages, stage_steps)):
        nu = 1.0 / re
        ref = references[re]
        # Each stage's error is scored against that stage's own reference, so
        # the curve stays meaningful while the curriculum climbs.
        if not args.quiet:
            print(
                f"[{EXPERIMENT}] stage {stage + 1}/{len(args.re_stages)}: "
                f"Re={re} (nu={nu:.3e}), {n_steps:,} steps",
                flush=True,
            )

        for _ in range(n_steps):
            main_batch = sample_batch(args.n_pde, args.n_bc, device, args.lid)
            if args.optimizer == "gnome":
                aux_batch = sample_batch(n_pde_aux, n_bc_aux, device, args.lid)

                def main_closure():
                    r = stacked_residuals(model, main_batch, nu, blk_w)
                    return r, torch.zeros_like(r)

                def aux_closure():
                    r = stacked_residuals(model, aux_batch, nu, blk_w)
                    return r, torch.zeros_like(r)

                loss = opt.step(main_closure, aux_closure)
            else:
                opt.zero_grad()
                r = stacked_residuals(model, main_batch, nu, blk_w)
                loss = (r ** 2).sum() / r.shape[0]
                loss.backward()
                opt.step()

            if scheduler is not None:
                scheduler.step()

            loss_val = float(loss.detach().item())
            if diverged(loss_val):
                run.finish(completed=False, diverged=True,
                           diverged_step=global_step)
                print(f"[{EXPERIMENT}] diverged at step {global_step} "
                      f"(Re={re}) — stopping.", flush=True)
                raise SystemExit(DIVERGED_EXIT)
            run.log_train(global_step, loss=loss_val)
            window.append(loss_val)
            global_step += 1

            if args.log_every and global_step % args.log_every == 0:
                eval_batch = sample_batch(args.n_pde, args.n_bc, device,
                                          args.lid)
                tl = term_losses(model, eval_batch, nu)
                rl2, rl2_u, rl2_v = eval_rel_l2(model, ref, device)
                last_avg = sum(window) / len(window)
                last_terms, last_rel_l2 = tl, rl2
                best_avg = min(best_avg, last_avg)
                best_rel_l2 = min(best_rel_l2, rl2)
                run.log_val(global_step, loss=last_avg, lr=current_lr(opt),
                            re=re, nu=nu, rel_l2=rl2, rel_l2_u=rl2_u,
                            rel_l2_v=rl2_v, **tl)
                if not args.quiet:
                    ms_per = ((time.perf_counter() - t_start)
                              / global_step * 1000)
                    print(
                        f"  step {global_step:6d}/{total_steps}  Re={re:<5d} "
                        f"avg_train={last_avg:.4e}  "
                        f"ru={tl['ru']:.2e} rv={tl['rv']:.2e} "
                        f"rc={tl['rc']:.2e} u_bc={tl['u_bc']:.2e} "
                        f"v_bc={tl['v_bc']:.2e}  rel_l2={rl2:.3e}  "
                        f"{ms_per:.1f} ms/step",
                        flush=True,
                    )
                window.clear()

    if args.optimizer == "adamw+lbfgs" and args.lbfgs_steps > 0:
        final_re = args.re_stages[-1]
        res = lbfgs_phase(
            model, device, args, run, references[final_re],
            nu=1.0 / final_re, re=final_re, start_step=global_step,
            best_rel_l2=best_rel_l2,
        )
        last_avg = res["last_avg"]
        best_avg = min(best_avg, res["last_avg"])
        last_rel_l2 = res["last_rel_l2"]
        best_rel_l2 = res["best_rel_l2"]
        last_terms = res["last_terms"] or last_terms

    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    if last_terms:
        print("  final " + "  ".join(
            f"{k}={v:.3e}" for k, v in last_terms.items()))
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
