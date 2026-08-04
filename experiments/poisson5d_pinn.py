"""5-D Poisson PINN: AdamW vs SOAP vs Gnome.

Apples-to-apples with the 5-D benchmark of "Improving Energy Natural Gradient
Descent through Woodbury, Momentum, and Randomization" (arXiv:2505.12149,
Appendix A.2), restructured into this repo's experiment format.

PDE:  -Δu = f(x),    x ∈ (0, 1)^5
BC:   u = u*  on  ∂Ω    (Dirichlet)

Manufactured solution::

    u*(x)   = Σ_i cos(π x_i)
    Δu*     = -π² u*        ⇒   -Δu* = π² u*
    f(x)    = π² Σ_i cos(π x_i)   (= π² u*, known in closed form)

**cos, not sin.** ``∫₀¹ cos(π x) dx = 0``, so ``u*`` has zero mean on the unit
cube and a constant predictor scores ``rel_l2 = 1.0``. A sine sum has mean
``2d/π``, so a constant predictor there scores ~0.05 — a degenerate metric.
The zero-mean cosine target keeps ``rel_l2`` honest.

Why 5-D (not 100-D): 5-D is the rung of the paper's ladder with a full baseline
set, it's cheap (D = 10,065 params with the reference architecture), and it's
where the single-precision floor for second-order PDE autodiff actually starts
to bite — so it's the right first rung.

**Exact Laplacian, no Hutchinson.** With a manufactured solution the PDE
residual is the only training signal, so a stochastic trace estimate would put
a variance noise-floor under the residual that no optimizer can escape. At
d=5 the exact Laplacian (``Σ_i ∂²u/∂x_i²`` via forward-over-reverse autograd)
is cheap, so we compute it exactly. This is the key difference from the
higher-dimensional variants.

Two-block residual: PDE (interior), BC (boundary). Stacked through
``gnome.stack_residuals`` so the multi-block MSE rides Gnome's single-MSE
surrogate as the per-block independent Rademacher GGN estimator. ``--bc-volume-
weight`` switches the BC block from equal weighting (the paper's Methods-section
residual, and Gnome's natural stacking) to the ``|∂Ω| = 2d`` weighting of the
paper's Background-section loss.

All three optimizers share one plain tanh MLP so the only variable is the
optimizer. The baselines (SOAP, AdamW) get a linear-warmup + cosine-decay
schedule (``--cosine-decay`` sets the final-lr fraction; 1.0 disables it);
Gnome runs at a fixed lr — its Gauss-Newton step self-anneals as the residual
shrinks.

Precision defaults to the device's native float32; ``--dtype float64`` pushes
below the ~1e-5 second-order-autodiff floor (MPS has no float64, so it falls
back to CPU — use CUDA for real float64 runs).

``--optimizer engdw`` adds the paper's exact Energy Natural Gradient with the
Woodbury identity as a reference method (the ~1e-6 natural-gradient line). It
uses the paper's ``1/√Nⱼ`` residual scaling so their tuned damping transfers,
and is forced to float64 on CUDA/CPU. Note it is a **GPU/reference-only path**:
the exact Jacobian is built with a naive ``jacrev`` (~O(N²) time, O(N·D)
memory), so at the paper's N=3500 it is a Thunder/CUDA job, not a laptop one.

Usage::

    uv run -m experiments.poisson5d_pinn --optimizer gnome --seed 0 --lr 1e-2
    uv run -m experiments.poisson5d_pinn --optimizer soap  --seed 0
    uv run -m experiments.poisson5d_pinn --optimizer adamw --seed 0
    # reference natural-gradient line (GPU/float64):
    uv run -m experiments.poisson5d_pinn --optimizer engdw --lr 5.2289e-2
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.func import functional_call, jacfwd, jacrev, vmap

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


EXPERIMENT = "poisson5d_pinn"

PI = math.pi


# ========================= Model =========================

class PINN(nn.Module):
    """Maps ``x ∈ ℝ^d → u`` via a plain tanh MLP with Xavier init.

    Uniform-width MLP following the repo convention (same as ``poisson_pinn``):
    ``hidden`` is the layer width and ``depth`` is the total number of linear
    layers (input + hidden + output). The paper's A.2 net was a tapered
    ``5→64→64→48→48→1``; here we use a plain uniform net instead, since the
    exact architecture isn't important for the optimizer comparison.
    """

    def __init__(self, d_in: int, hidden: int = 64, depth: int = 5):
        super().__init__()
        assert depth >= 2
        layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:                      # Xavier suits tanh
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ========================= Exact solution + forcing =========================

def u_star(x: torch.Tensor) -> torch.Tensor:
    """Manufactured solution ``u*(x) = Σ_i cos(π x_i)`` → ``(B, 1)`` (zero mean)."""
    return torch.cos(PI * x).sum(dim=1, keepdim=True)


def forcing(x: torch.Tensor) -> torch.Tensor:
    """RHS ``f = -Δu* = π² Σ_i cos(π x_i) = π² u*``."""
    return (PI ** 2) * u_star(x)


# ========================= Laplacian (exact) =========================

def laplacian_exact(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Exact ``Δu = Σ_i ∂²u/∂x_i²`` via forward-over-reverse autograd.

    ``d`` second-derivative passes; cheap at d=5. ``create_graph=True`` keeps
    the Laplacian differentiable back to θ so it flows through the residual
    into Gnome's surrogate (and ordinary backprop).
    """
    u = model(x)
    g = autograd.grad(u.sum(), x, create_graph=True)[0]          # (B, d)
    lap = torch.zeros_like(u)
    for i in range(x.shape[1]):
        gi = autograd.grad(g[:, i].sum(), x, create_graph=True)[0][:, i]
        lap = lap + gi.unsqueeze(1)
    return lap                                                   # (B, 1)


# ========================= Sampling =========================

def sample_interior(
    n: int, d: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform interior collocation ``x ∈ (0, 1)^d`` (needs grad for Δu)."""
    return torch.rand(n, d, device=device, dtype=dtype, requires_grad=True)


def sample_boundary(
    n: int, d: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Uniform boundary draws: pin one random coordinate of each point to 0/1."""
    x = torch.rand(n, d, device=device, dtype=dtype)
    k = torch.randint(0, d, (n,), device=device)                # coordinate pinned
    face = torch.randint(0, 2, (n,), device=device).to(x.dtype)  # face 0 or 1
    x[torch.arange(n, device=device), k] = face
    return x                                                    # no grad needed


# ========================= Residuals =========================

def pde_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Poisson PDE residual ``-Δu - f`` (target zero for ``-Δu = f``)."""
    return -laplacian_exact(model, x) - forcing(x)


def bc_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Dirichlet BC residual ``u(boundary) - u*(boundary)``."""
    return model(x) - u_star(x)


def stacked_residuals(
    model: nn.Module, n_int: int, n_bc: int, d: int,
    device: torch.device, dtype: torch.dtype, bc_volume_weight: bool = False,
) -> torch.Tensor:
    """PDE + BC residuals stacked via ``stack_residuals``.

    Equal block weights by default (paper's Methods residual); with
    ``bc_volume_weight`` the BC block gets the ``|∂Ω| = 2d`` factor of the
    paper's Background-section loss.
    """
    r_int = pde_residual(model, sample_interior(n_int, d, device, dtype))
    r_bc = bc_residual(model, sample_boundary(n_bc, d, device, dtype))
    weights = [1.0, float(2 * d)] if bc_volume_weight else None
    return stack_residuals([r_int, r_bc], weights)


def term_losses(
    model: nn.Module, n_int: int, n_bc: int, d: int,
    device: torch.device, dtype: torch.dtype,
) -> dict[str, float]:
    """Per-block MSE for diagnostic logging (fresh batch)."""
    r_int = pde_residual(model, sample_interior(n_int, d, device, dtype))
    r_bc = bc_residual(model, sample_boundary(n_bc, d, device, dtype))
    return {
        "pde": r_int.pow(2).mean().item(),
        "bc": r_bc.pow(2).mean().item(),
    }


# ========================= Reference eval =========================

def make_eval_set(
    d: int, device: torch.device, dtype: torch.dtype,
    n: int = 30000, seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed, seeded held-out set of ``n`` interior points and ``u*`` on them.

    Derivative-free (compares ``u_θ`` to ``u*`` directly), so the rel_l2 signal
    carries no autodiff noise. Held fixed across the whole run so the metric is
    a clean convergence curve rather than a resampled estimate.
    """
    g = torch.Generator().manual_seed(seed)                     # CPU gen (device-agnostic)
    x = torch.rand(n, d, dtype=dtype, generator=g).to(device)
    return x, u_star(x)


@torch.no_grad()
def eval_rel_l2(
    model: nn.Module, x_eval: torch.Tensor, u_eval: torch.Tensor,
) -> float:
    """Relative L2 against ``u*`` on the fixed held-out set."""
    was_training = model.training
    model.eval()
    pred = model(x_eval)
    rel = torch.linalg.norm(pred - u_eval) / torch.linalg.norm(u_eval)
    if was_training:
        model.train()
    return float(rel)


# ========================= ENGD-W (Energy Natural Gradient, Woodbury) =========

def make_functional_residual(model: nn.Module, d: int, bc_volume_weight: bool):
    """Build a *pure-functional* stacked residual ``r(params, x_int, x_bc)``.

    ENGD-W needs the Jacobian ``∂r/∂θ`` via ``torch.func.jacrev``, which requires
    the residual to be a pure function of a parameter *dict* (``functional_call``),
    with the Laplacian itself expressed through ``torch.func`` (``jacfwd∘jacrev``)
    — you cannot nest ``torch.autograd.grad`` inside a ``torch.func`` transform,
    so this path does *not* reuse ``laplacian_exact`` / ``stacked_residuals``.

    Scaling follows the paper's **Methods** residual: block ``j`` is divided by
    ``√Nⱼ`` so ``‖r‖² = MSE_int + MSE_bc`` (and ``0.5‖r‖²`` is the ENGD energy).
    This is *not* ``stack_residuals``' ``√(N/Nⱼ)`` weighting — the two differ by
    ``√N``, which would rescale the effective damping by ``N`` and break the
    paper's tuned ``λ``. ``--bc-volume-weight`` multiplies the BC block by
    ``√(2d)`` (paper's Background loss).
    """
    def u_single(params, x):                # x: (d,) -> scalar
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze()

    def lap_single(params, x):              # Σ_i ∂²u/∂x_i² at one point
        H = jacfwd(jacrev(u_single, argnums=1), argnums=1)(params, x)   # (d, d)
        return torch.diagonal(H).sum()

    v_u = vmap(u_single, in_dims=(None, 0))
    v_lap = vmap(lap_single, in_dims=(None, 0))

    def residual(params, x_int, x_bc):
        n_int, n_bc = x_int.shape[0], x_bc.shape[0]
        r_int = (-v_lap(params, x_int).unsqueeze(1) - forcing(x_int)) / math.sqrt(n_int)
        r_bc = (v_u(params, x_bc).unsqueeze(1) - u_star(x_bc)) / math.sqrt(n_bc)
        if bc_volume_weight:
            r_bc = r_bc * math.sqrt(2 * d)
        return torch.cat([r_int, r_bc], dim=0)                          # (N, 1)

    return residual


class ENGDW:
    """Energy Natural Gradient Descent via the Woodbury / push-through identity.

        δ = (JᵀJ + λI)⁻¹ Jᵀr  ==  Jᵀ(JJᵀ + λI)⁻¹ r,     θ ← θ − η·δ

    The right form solves an ``N×N`` system instead of ``D×D`` — that identity
    *is* ENGD-W. Cost ``O(N²D)`` to form the Gram, ``O(N³)`` to factor; the
    binding constraint is batch ``N``, not model size ``D``. float64/CPU only
    (the Gram is ill-conditioned; the tuned damping is float64-scale).

    ``residual_fn(params, *args) → (N, 1)`` with energy ``L = 0.5‖r‖²``.
    """

    def __init__(self, model, residual_fn, lr=5.2289e-2, damping=6.804474e-8,
                 line_search=False, ls_grid=None, chunk_size=None):
        self.model = model
        self.residual_fn = residual_fn
        self.lr = lr
        self.lam = damping
        self.line_search = line_search
        self.ls_grid = ls_grid or torch.logspace(-3, 0, 13).tolist()
        self.chunk_size = chunk_size
        self.params = dict(model.named_parameters())
        self.shapes = [(k, v.shape, v.numel()) for k, v in self.params.items()]
        # Duck-type shim so common.current_lr(opt) works; updated per step to the
        # actual η applied (meaningful when line search is on).
        self.param_groups = [{"lr": lr}]

    def _unflat(self, flat):
        out, i = {}, 0
        for k, shape, n in self.shapes:
            out[k] = flat[i:i + n].view(shape)
            i += n
        return out

    def _loss_at(self, eta, delta, args):
        d = self._unflat(delta)
        cand = {k: (v - eta * d[k]).detach() for k, v in self.params.items()}
        r = self.residual_fn(cand, *args).squeeze(-1)   # x-derivs still needed
        return (r ** 2).sum().item()

    def step(self, *args):
        def r_vec(p):
            return self.residual_fn(p, *args).squeeze(-1)            # (N,)

        params = {k: v.detach() for k, v in self.params.items()}
        r = r_vec(params).detach()
        N = r.shape[0]

        Jd = jacrev(r_vec, chunk_size=self.chunk_size)(params)       # dict (N, *shape)
        J = torch.cat([v.reshape(N, -1) for v in Jd.values()], dim=1)  # (N, D)

        G = J @ J.T
        G = 0.5 * (G + G.T)                    # symmetrize before factoring
        G.diagonal().add_(self.lam)

        try:                                    # G is SPD; Cholesky is the right call
            L = torch.linalg.cholesky(G)
            z = torch.cholesky_solve(r.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:        # damping too small for this dtype
            z = torch.linalg.lstsq(G, r.unsqueeze(1)).solution.squeeze(1)

        delta = J.T @ z                                              # (D,)

        if self.line_search:
            eta = min((self._loss_at(e, delta, args), e) for e in self.ls_grid)[1]
        else:
            eta = self.lr
        self.param_groups[0]["lr"] = eta

        upd = self._unflat(eta * delta)
        with torch.no_grad():
            for k, v in self.params.items():
                v -= upd[k]

        return float((r ** 2).sum())          # PINN loss = MSE_int + MSE_bc


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
    final-lr fraction (0.0 → decay to zero, 1.0 → decay disabled).
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=2,
            warmup=warmup,
            trust_radius=(trust_region if trust_region > 0 else None),
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
                   choices=["gnome", "soap", "adamw", "engdw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                   help="Compute precision. float64 pushes below the ~1e-5 "
                        "single-precision floor of second-order PDE autodiff, "
                        "but MPS has no float64 so it falls back to CPU (slow) "
                        "— use CUDA for real float64 runs. Note: Gnome computes "
                        "its preconditioner eigenbasis internally in float32 "
                        "regardless; only the Laplacian/gradient/param path is "
                        "true float64.")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--d", type=int, default=5, help="Problem dimension.")
    p.add_argument("--hidden", type=int, default=64, help="MLP width.")
    p.add_argument("--depth", type=int, default=5,
                   help="Total number of linear layers (input + hidden + output).")
    p.add_argument("--n-int", type=int, default=3000,
                   help="Interior collocation points per step (paper: 3000).")
    p.add_argument("--n-bc", type=int, default=500,
                   help="Boundary collocation points per step (paper: 500).")
    p.add_argument("--aux-n-int", type=int, default=180,
                   help="Interior points for Gnome's GGN aux batch (disjoint, "
                        "small). Gnome only.")
    p.add_argument("--aux-n-bc", type=int, default=30,
                   help="Boundary points for Gnome's GGN aux batch. Gnome only.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Base learning rate. Gnome typically wants a larger lr "
                        "here (e.g. 1e-2) than the SOAP/AdamW baselines.")
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
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps. For the SOAP/AdamW baselines "
                        "this is the schedule warmup; for Gnome it is passed "
                        "as its internal `warmup=`.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--bc-volume-weight", action="store_true",
                   help="Use |∂Ω|=2d weighting on the BC block (paper's "
                        "Background loss) instead of equal block weights "
                        "(paper's Methods residual). See header note.")
    p.add_argument("--engdw-damping", type=float, default=6.804474e-8,
                   help="ENGD-W Tikhonov damping λ in (JJᵀ+λI). Default is the "
                        "paper's Fig 8 fixed-lr 5D value; pair with --lr "
                        "5.2289e-2. engdw only.")
    p.add_argument("--engdw-line-search", action="store_true",
                   help="ENGD-W: use the paper's inherited line search instead "
                        "of a fixed --lr (then pair with damping 3.173212e-12, "
                        "the Fig 7 value). engdw only.")
    p.add_argument("--engdw-chunk", type=int, default=None,
                   help="jacrev chunk_size for the ENGD-W Jacobian (memory vs "
                        "speed; lower = less memory). None = no chunking. engdw only.")
    p.add_argument("--n-eval", type=int, default=30000,
                   help="Size of the fixed held-out set for rel_l2 (paper: 30k).")
    p.add_argument("--eval-seed", type=int, default=1234,
                   help="Seed for the fixed held-out eval set.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = pick_device()
    bcw = args.bc_volume_weight
    if args.optimizer == "engdw":
        # ENGD-W needs float64 (the Gram is ill-conditioned and the tuned
        # damping is float64-scale) and can't use MPS (no float64; torch.func +
        # cholesky unreliable). CUDA is its real home — the naive jacobian is
        # ~O(N²) per step, so it's a GPU job at the paper's N=3500. Falls back
        # to CPU (slow) when there's no GPU.
        target = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if dtype is not torch.float64 or device != target:
            if not args.quiet:
                print(f"[{EXPERIMENT}] engdw forces float64 on {target.type} "
                      f"(MPS unsupported); overriding --dtype/device.", flush=True)
            dtype, device = torch.float64, target
            args.dtype = "float64"                  # reflect in banner + hyperparams
    elif device.type == "mps" and dtype is torch.float64:
        device = torch.device("cpu")               # MPS has no float64
        if not args.quiet:
            print(f"[{EXPERIMENT}] float64 requested; MPS has no float64 "
                  f"— falling back to CPU.", flush=True)
    model = PINN(args.d, hidden=args.hidden, depth=args.depth).to(device=device, dtype=dtype)

    if args.optimizer == "engdw":
        residual_fn = make_functional_residual(model, args.d, bcw)
        opt = ENGDW(model, residual_fn, lr=args.lr, damping=args.engdw_damping,
                    line_search=args.engdw_line_search, chunk_size=args.engdw_chunk)
        opt_cfg = dict(lr=args.lr, damping=args.engdw_damping,
                       line_search=args.engdw_line_search,
                       loss="energy", residual_scaling="paper_methods")
        scheduler = None
    else:
        opt, opt_cfg, scheduler = build_optimizer(
            args.optimizer, model.parameters(), args.lr, args.weight_decay,
            warmup=args.warmup_steps, total_steps=args.steps,
            cosine_decay=args.cosine_decay, eps=args.eps,
            beta1=args.beta1, beta2=args.beta2,
        trust_region=args.trust_region,
        )

    n_params = sum(p.numel() for p in model.parameters())

    hyperparameters = {
        "optimizer": args.optimizer,
        "steps": args.steps,
        "dtype": args.dtype,
        "d": args.d,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_params": n_params,
        "n_int": args.n_int,
        "n_bc": args.n_bc,
        "aux_n_int": args.aux_n_int,
        "aux_n_bc": args.aux_n_bc,
        "bc_volume_weight": bcw,
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

    if not args.quiet:
        print(
            f"[{EXPERIMENT}] {args.optimizer} | d={args.d} | params={n_params:,} | "
            f"dtype={args.dtype} | device={device}\n"
            f"  N_int={args.n_int} N_bc={args.n_bc} | "
            f"aux={args.aux_n_int}/{args.aux_n_bc} | "
            f"bc_volume_weight={bcw} | steps={args.steps}",
            flush=True,
        )
    x_eval, u_eval = make_eval_set(
        args.d, device, dtype, args.n_eval, args.eval_seed)

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        if args.optimizer == "gnome":
            def main_closure():
                r = stacked_residuals(
                    model, args.n_int, args.n_bc, args.d, device, dtype, bcw)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(
                    model, args.aux_n_int, args.aux_n_bc, args.d, device, dtype, bcw)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        elif args.optimizer == "engdw":
            # Plain (no-grad) points: the functional Laplacian differentiates
            # w.r.t. x via torch.func, so x must not be an autograd leaf.
            x_int = torch.rand(args.n_int, args.d, device=device, dtype=dtype)
            x_bc = sample_boundary(args.n_bc, args.d, device, dtype)
            loss = opt.step(x_int, x_bc)                 # returns a float
        else:
            opt.zero_grad()
            r = stacked_residuals(
                model, args.n_int, args.n_bc, args.d, device, dtype, bcw)
            loss = (r ** 2).sum() / r.shape[0]
            loss.backward()
            opt.step()

        if scheduler is not None:
            scheduler.step()

        loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.", flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            tl = term_losses(model, args.n_int, args.n_bc, args.d, device, dtype)
            rl2 = eval_rel_l2(model, x_eval, u_eval)
            last_avg = sum(window) / len(window)
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        pde=tl["pde"], bc=tl["bc"], rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  bc={tl['bc']:.3e}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
