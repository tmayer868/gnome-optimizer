"""Navier-Stokes inverse problem: 2-D cylinder wake (Raissi et al., JCP 2019).

Inverse-problem PINN benchmark: given sparse `(u, v)` observations from a 2-D
cylinder-wake DNS at Re = 100, jointly recover the hidden pressure field and
two unknown PDE coefficients ``λ₁`` (advection) and ``λ₂`` (viscosity), under
the incompressible Navier-Stokes constraints

    continuity:  u_x + v_y = 0
    x-momentum:  u_t + λ₁(u·u_x + v·u_y) + p_x - λ₂(u_xx + u_yy) = 0
    y-momentum:  v_t + λ₁(u·v_x + v·v_y) + p_y - λ₂(v_xx + v_yy) = 0

True parameter values: ``λ₁ = 1``, ``λ₂ = 0.01`` (Re = 100). Both initialized
to zero — must be discovered.

Five-block residual stack — two data blocks (sparse ``u`` and ``v``
observations) plus three PDE blocks (continuity + two momentum equations).
This is the canonical multi-block-conflict test: data fits and physics
constraints almost always point the optimizer in different directions, so
the optimizer's curvature estimate has to balance them without manual
loss-weight tuning (all block weights default to 1).

Reference data is Raissi's ``cylinder_nektar_wake.mat`` from
``maziarraissi/PINNs`` on GitHub — DNS at Re=100, 5000 spatial points × 200
time snapshots in the wake region ``x ∈ [1, 8], y ∈ [-2, 2]``. Downloaded to
``experiments/data/`` on first run and cached.

Network outputs ``(u, v, p)`` directly (no streamfunction parameterization),
so continuity enters as an explicit residual block — giving Gnome the harder
4-conflicting-block test rather than the streamfunction's 3.

The baselines (SOAP, AdamW) get a linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 disables it); Gnome runs at
a fixed lr — its Gauss-Newton step self-anneals as the residual shrinks.

Usage:

    uv run -m experiments.navier_stokes_pinn --optimizer gnome --seed 0
    uv run -m experiments.navier_stokes_pinn --optimizer soap  --seed 0
    uv run -m experiments.navier_stokes_pinn --optimizer adamw --seed 0
"""

from __future__ import annotations

import argparse
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


EXPERIMENT = "navier_stokes_pinn"

# Domain matches Raissi's wake region. Time span is 200 snapshots × 0.1 dt.
T_MIN, T_MAX = 0.0, 19.9
X_MIN, X_MAX = 1.0, 8.0
Y_MIN, Y_MAX = -2.0, 2.0

# True coefficients at Re = 100. Initial guesses are zero — must be discovered.
TRUE_LAMBDA1 = 1.0
TRUE_LAMBDA2 = 0.01

REFERENCE_URL = (
    "https://raw.githubusercontent.com/maziarraissi/PINNs/master/"
    "main/Data/cylinder_nektar_wake.mat"
)
DEFAULT_REF_CACHE = "experiments/data/cylinder_nektar_wake.mat"


# ========================= Model =========================

class PINN(nn.Module):
    """Maps ``(t, x, y) → (u, v, p)`` via a tanh MLP plus two learnable scalars.

    ``self.lambda1`` and ``self.lambda2`` are ordinary ``nn.Parameter`` scalars
    so they participate in the optimizer step like any other weight. Initialized
    to zero so the discovery has to happen from a maximally uninformative prior.
    """

    def __init__(
        self,
        hidden: int = 128,
        depth: int = 8,
        lambda1_init: float = 0.0,
        lambda2_init: float = 0.0,
    ):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = [nn.Linear(3, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 3)]
        self.net = nn.Sequential(*layers)
        self.lambda1 = nn.Parameter(torch.tensor(float(lambda1_init)))
        self.lambda2 = nn.Parameter(torch.tensor(float(lambda2_init)))

    def forward(
        self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.net(torch.cat([t, x, y], dim=1))


# ========================= Residuals =========================

def ns_residual(
    model: PINN, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuity + x-momentum + y-momentum residuals at collocation points.

    Uses the combined-input ``autograd.grad`` form so first-order derivatives
    of ``u`` and ``v`` come from one backward pass each instead of three.
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    uvp = model(t, x, y)
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    u_t, u_x, u_y = autograd.grad(
        u, [t, x, y], torch.ones_like(u), create_graph=True
    )
    v_t, v_x, v_y = autograd.grad(
        v, [t, x, y], torch.ones_like(v), create_graph=True
    )
    p_x, p_y = autograd.grad(
        p, [x, y], torch.ones_like(p), create_graph=True
    )
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    l1 = model.lambda1
    l2 = model.lambda2
    continuity = u_x + v_y
    momentum_x = u_t + l1 * (u * u_x + v * u_y) + p_x - l2 * (u_xx + u_yy)
    momentum_y = v_t + l1 * (u * v_x + v * v_y) + p_y - l2 * (v_xx + v_yy)
    return continuity, momentum_x, momentum_y


def data_residuals(
    model: PINN,
    t_data: torch.Tensor, x_data: torch.Tensor, y_data: torch.Tensor,
    u_data: torch.Tensor, v_data: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse-observation residuals ``(u_pred - u_data, v_pred - v_data)``."""
    uvp = model(t_data, x_data, y_data)
    u_pred = uvp[:, 0:1]
    v_pred = uvp[:, 1:2]
    return u_pred - u_data, v_pred - v_data


def stacked_residuals(
    model: PINN,
    data_batch: tuple[torch.Tensor, ...],
    pde_batch: tuple[torch.Tensor, ...],
    weights: tuple[float, float, float, float, float],
) -> torch.Tensor:
    """Five-block stack: ``(u_data, v_data, continuity, momentum_x, momentum_y)``.

    Default equal weights — the whole point of the benchmark is to test
    whether the optimizer can balance these blocks without manual tuning.
    """
    t_d, x_d, y_d, u_d, v_d = data_batch
    t_p, x_p, y_p = pde_batch
    r_u, r_v = data_residuals(model, t_d, x_d, y_d, u_d, v_d)
    r_cont, r_mx, r_my = ns_residual(model, t_p, x_p, y_p)
    return stack_residuals(
        [r_u, r_v, r_cont, r_mx, r_my],
        weights=list(weights),
    )


def term_losses(
    model: PINN,
    data_batch: tuple[torch.Tensor, ...],
    pde_batch: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    """Per-block MSE plus parameter-discovery error, for diagnostic logging."""
    t_d, x_d, y_d, u_d, v_d = data_batch
    t_p, x_p, y_p = pde_batch
    r_u, r_v = data_residuals(model, t_d, x_d, y_d, u_d, v_d)
    r_cont, r_mx, r_my = ns_residual(model, t_p, x_p, y_p)
    return {
        "data_u": r_u.pow(2).mean().item(),
        "data_v": r_v.pow(2).mean().item(),
        "continuity": r_cont.pow(2).mean().item(),
        "momentum_x": r_mx.pow(2).mean().item(),
        "momentum_y": r_my.pow(2).mean().item(),
        "lambda1_err": abs(float(model.lambda1.item()) - TRUE_LAMBDA1),
        "lambda2_err": abs(float(model.lambda2.item()) - TRUE_LAMBDA2),
    }


# ========================= Reference data =========================

def download_reference(cache_path: str = DEFAULT_REF_CACHE) -> str:
    """Download Raissi's cylinder DNS data if not already cached on disk."""
    if os.path.isfile(cache_path):
        return cache_path
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"  downloading reference from {REFERENCE_URL} ...", flush=True)
    try:
        urllib.request.urlretrieve(REFERENCE_URL, cache_path)
    except Exception as e:
        raise RuntimeError(
            f"failed to download Raissi cylinder dataset: {e}\n"
            f"manually fetch {REFERENCE_URL!r} → {cache_path!r}."
        ) from e
    return cache_path


def load_reference(cache_path: str = DEFAULT_REF_CACHE) -> dict[str, torch.Tensor]:
    """Load ``cylinder_nektar_wake.mat`` into tensors.

    Raissi's file layout::

        U_star : (N, 2, T)   velocity (u, v) at N spatial points × T times
        p_star : (N, T)      pressure
        t      : (T, 1)      time stamps
        X_star : (N, 2)      (x, y) coordinates of the N points

    Returns a dict with ``t (T,), x (N,), y (N,), u (T, N), v (T, N), p (T, N)``
    all float32.
    """
    try:
        import scipy.io
    except ImportError as e:
        raise RuntimeError(
            "scipy is required to load Raissi's .mat reference. "
            "Install with `uv pip install scipy`."
        ) from e

    path = download_reference(cache_path)
    blob = scipy.io.loadmat(path)
    U_star = blob["U_star"]
    p_star = blob["p_star"]
    t_star = blob["t"].flatten()
    X_star = blob["X_star"]

    x = X_star[:, 0].astype("float32")
    y = X_star[:, 1].astype("float32")
    u = U_star[:, 0, :].T.astype("float32")  # (T, N)
    v = U_star[:, 1, :].T.astype("float32")
    p = p_star.T.astype("float32")
    t = t_star.astype("float32")

    return {
        "t": torch.from_numpy(t),
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "u": torch.from_numpy(u),
        "v": torch.from_numpy(v),
        "p": torch.from_numpy(p),
    }


def sparse_data_pool(
    ref: dict[str, torch.Tensor],
    n_data: int,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Pick a fixed sparse set of ``n_data`` observation tuples from the DNS.

    Sampling uniformly across the full ``T × N`` spacetime grid gives sparse
    coverage in both dimensions, matching Raissi's protocol. The pool is
    fixed once at training start — these are the network's only ``(u, v)``
    observations for the entire run.
    """
    t_grid = ref["t"]
    x_grid = ref["x"]
    y_grid = ref["y"]
    u_grid = ref["u"]
    v_grid = ref["v"]
    T = t_grid.shape[0]
    N = x_grid.shape[0]

    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(T * N, generator=gen)[:n_data]
    ti = idx // N
    si = idx % N

    t = t_grid[ti].unsqueeze(1)
    x = x_grid[si].unsqueeze(1)
    y = y_grid[si].unsqueeze(1)
    u = u_grid[ti, si].unsqueeze(1)
    v = v_grid[ti, si].unsqueeze(1)

    return tuple(z.to(device) for z in (t, x, y, u, v))


def sample_pde_batch(
    n_pde: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh uniform collocation points over the wake region each step."""
    t = torch.rand(n_pde, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    x = torch.rand(n_pde, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n_pde, 1, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    return t, x, y


def eval_rel_l2(
    model: PINN,
    ref: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int = 4096,
) -> dict[str, float]:
    """Relative L2 of inferred ``(u, v, p)`` against the full DNS grid.

    Pressure is defined up to a constant in incompressible NS, so we subtract
    the mean from both predicted and reference pressure before computing the
    L2 ratio — otherwise the constant offset dominates.
    """
    t_grid = ref["t"]
    x_grid = ref["x"]
    y_grid = ref["y"]
    u_ref = ref["u"]
    v_ref = ref["v"]
    p_ref = ref["p"]
    T = t_grid.shape[0]
    N = x_grid.shape[0]

    tt = t_grid.unsqueeze(1).expand(T, N).reshape(-1, 1)
    xx = x_grid.unsqueeze(0).expand(T, N).reshape(-1, 1)
    yy = y_grid.unsqueeze(0).expand(T, N).reshape(-1, 1)

    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, tt.shape[0], batch_size):
            preds.append(
                model(
                    tt[i:i + batch_size].to(device),
                    xx[i:i + batch_size].to(device),
                    yy[i:i + batch_size].to(device),
                ).cpu()
            )
    if was_training:
        model.train()
    uvp = torch.cat(preds).reshape(T, N, 3)
    u_p = uvp[..., 0]
    v_p = uvp[..., 1]
    p_p = uvp[..., 2] - uvp[..., 2].mean()
    p_r = p_ref - p_ref.mean()

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt())

    return {
        "rel_l2_u": rel(u_p, u_ref),
        "rel_l2_v": rel(v_p, v_ref),
        "rel_l2_p": rel(p_p, p_r),
    }


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
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--n-data", type=int, default=5000,
                   help="Number of sparse (u, v) observation points "
                        "(fixed at training start).")
    p.add_argument("--n-pde", type=int, default=10000,
                   help="Fresh collocation points per step.")
    p.add_argument("--aux-frac", type=float, default=0.03)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps): larger "
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) "
                        "for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps. For the SOAP/AdamW baselines "
                        "this is the schedule warmup; for Gnome it is passed "
                        "as its internal `warmup=`.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--w-data-u", type=float, default=1.0)
    p.add_argument("--w-data-v", type=float, default=1.0)
    p.add_argument("--w-continuity", type=float, default=1.0)
    p.add_argument("--w-momentum-x", type=float, default=1.0)
    p.add_argument("--w-momentum-y", type=float, default=1.0)
    p.add_argument("--lambda1-init", type=float, default=0.0)
    p.add_argument("--lambda2-init", type=float, default=0.0)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()

    if not args.quiet:
        print(f"[{EXPERIMENT}] loading reference DNS...", flush=True)
    ref = load_reference()
    data_pool = sparse_data_pool(ref, args.n_data, device, seed=args.seed)

    model = PINN(
        hidden=args.hidden, depth=args.depth,
        lambda1_init=args.lambda1_init, lambda2_init=args.lambda2_init,
    ).to(device)

    opt, opt_cfg, scheduler = build_optimizer(
        args.optimizer, model.parameters(), args.lr, args.weight_decay,
        warmup=args.warmup_steps, total_steps=args.steps,
        cosine_decay=args.cosine_decay, eps=args.eps,
        beta1=args.beta1, beta2=args.beta2,
    )

    n_pde_aux = max(8, int(args.n_pde * args.aux_frac))
    n_data_aux = max(8, int(args.n_data * args.aux_frac))
    block_weights = (
        args.w_data_u, args.w_data_v,
        args.w_continuity, args.w_momentum_x, args.w_momentum_y,
    )

    n_params = sum(p.numel() for p in model.parameters())
    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_params": n_params,
        "n_data": args.n_data,
        "n_pde": args.n_pde,
        "n_pde_aux": n_pde_aux,
        "n_data_aux": n_data_aux,
        "w_data_u": args.w_data_u,
        "w_data_v": args.w_data_v,
        "w_continuity": args.w_continuity,
        "w_momentum_x": args.w_momentum_x,
        "w_momentum_y": args.w_momentum_y,
        "lambda1_init": args.lambda1_init,
        "lambda2_init": args.lambda2_init,
        "true_lambda1": TRUE_LAMBDA1,
        "true_lambda2": TRUE_LAMBDA2,
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
            f"  N_data={args.n_data} N_pde={args.n_pde} | "
            f"aux={n_pde_aux}/{n_data_aux} | steps={args.steps}",
            flush=True,
        )

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = float("nan")
    best_avg = float("inf")
    last_rl2 = {"rel_l2_u": float("nan"), "rel_l2_v": float("nan"),
                "rel_l2_p": float("nan")}
    best_rl2_u = float("inf")
    last_l1 = float(model.lambda1.item())
    last_l2 = float(model.lambda2.item())

    for step in range(args.steps):
        pde_batch = sample_pde_batch(args.n_pde, device)

        if args.optimizer == "gnome":
            aux_pde = sample_pde_batch(n_pde_aux, device)
            aux_idx = torch.randperm(args.n_data, device=device)[:n_data_aux]
            aux_data = tuple(d[aux_idx] for d in data_pool)

            def main_closure():
                r = stacked_residuals(model, data_pool, pde_batch, block_weights)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_data, aux_pde, block_weights)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, data_pool, pde_batch, block_weights)
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
                model, data_pool, sample_pde_batch(args.n_pde, device)
            )
            rl2 = eval_rel_l2(model, ref, device)
            last_l1 = float(model.lambda1.item())
            last_l2 = float(model.lambda2.item())
            last_avg = sum(window) / len(window)
            last_rl2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rl2_u = min(best_rl2_u, rl2["rel_l2_u"])
            run.log_val(
                step + 1,
                loss=last_avg, lr=current_lr(opt),
                lambda1=last_l1, lambda2=last_l2,
                **tl, **rl2,
            )
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"du={tl['data_u']:.2e} dv={tl['data_v']:.2e} "
                    f"c={tl['continuity']:.2e} "
                    f"mx={tl['momentum_x']:.2e} my={tl['momentum_y']:.2e}  "
                    f"λ1={last_l1:.4f} λ2={last_l2:.5f}  "
                    f"rl2_u={rl2['rel_l2_u']:.2e} rl2_p={rl2['rel_l2_p']:.2e}  "
                    f"{ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2_u=last_rl2["rel_l2_u"], best_rel_l2_u=best_rl2_u,
        final_rel_l2_v=last_rl2["rel_l2_v"], final_rel_l2_p=last_rl2["rel_l2_p"],
        final_lambda1=last_l1, final_lambda2=last_l2,
        lambda1_err=abs(last_l1 - TRUE_LAMBDA1),
        lambda2_err=abs(last_l2 - TRUE_LAMBDA2),
    )
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final rel_l2: u={last_rl2['rel_l2_u']:.3e}  "
          f"v={last_rl2['rel_l2_v']:.3e}  p={last_rl2['rel_l2_p']:.3e}")
    print(f"  final λ1={last_l1:.4f} (true {TRUE_LAMBDA1})  "
          f"λ2={last_l2:.5f} (true {TRUE_LAMBDA2})")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
