"""Kuramoto-Sivashinsky PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + u·u_x + u_xx + u_xxxx = 0,    x ∈ [0, 32π],  t ∈ [0, 30]
IC:   u(0, x) = cos(x/16)·(1 + sin(x/16))
BC:   u(t, 0) = u(t, 32π),  u_x(t, 0) = u_x(t, 32π)    (periodic)

The canonical chaotic-regime Kuramoto-Sivashinsky benchmark: the large-L
periodic domain + Kassam-Trefethen IC produces sustained spatiotemporal
chaos. The 4th-order linear part of the PDE (``u_xx + u_xxxx``) is what
makes KS distinctive — small-k modes are amplified (k² > k⁴), large-k
damped (k⁴ ≫ k²), and the nonlinear ``u·u_x`` term ties them together.

Compared to Burgers, the PINN residual costs roughly 2× more (four
``autograd.grad`` calls through the network input instead of two), but
the headline difficulty is the same family of PDE/IC/BC residual
stiffness — handled here by the same ``gnome.stack_residuals`` pattern
with equal block weights (no causal training, no grad-norm weighting).
Periodicity is enforced as soft constraints on ``u`` and ``u_x`` at the
endpoints.

Reference: ETDRK4 spectral integrator (Kassam & Trefethen 2005). RK4
would need a CFL of ``O(dx⁴ / 1)`` because of the 4th-order linear
operator; ETDRK4 handles the linear part exactly per step and lets us
take ``dt ≈ 0.025`` even at the chaotic-regime length scales.

All three optimizers share one plain tanh MLP so the only variable is the
optimizer. The baselines (SOAP, AdamW) get a linear-warmup + cosine-decay
schedule (``--cosine-decay`` sets the final-lr fraction; 1.0 disables it);
Gnome runs at a fixed lr — its Gauss-Newton step self-anneals as the residual
shrinks.

Usage:

    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer gnome --seed 0
    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer soap  --seed 0
    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer adamw --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
import time

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


EXPERIMENT = "kuramoto_sivashinsky_pinn"

T_MIN, T_MAX = 0.0, 30.0
X_MIN, X_MAX = 0.0, 32.0 * math.pi
L_DOMAIN = X_MAX - X_MIN


# ========================= Model =========================

class PINN(nn.Module):
    """Maps ``(t, x) → u`` via a plain tanh MLP."""

    def __init__(self, hidden: int = 128, depth: int = 6):
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
    """KS PDE residual ``u_t + u·u_x + u_xx + u_xxxx`` at (t, x).

    Four sequential autograd passes through the spatial input —
    ``create_graph=True`` on every one so the surrogate / loss backward
    can differentiate through the entire higher-order chain.
    """
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    u_xxxx = autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]
    return u_t + u * u_x + u_xx + u_xxxx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual: ``u(0, x) - cos(x/16)(1 + sin(x/16))``."""
    t0 = torch.zeros_like(x)
    z = x / 16.0
    u0 = torch.cos(z) * (1.0 + torch.sin(z))
    return model(t0, x) - u0


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Periodic BC residual: ``u(t, X_MIN) - u(t, X_MAX)`` and
    ``u_x(t, X_MIN) - u_x(t, X_MAX)``, stacked.

    For a 4th-order spatial PDE we'd ideally enforce periodicity up to
    the third derivative, but C¹ periodicity is the standard PINN
    treatment for KS and gives the network enough freedom for the
    higher derivatives to match interior values.
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


def ks_reference(
    nx: int = 256, nt: int = 151, dt: float = 0.025,
    cache_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ETDRK4 Fourier-spectral reference solution for KS.

    Following Kassam & Trefethen (2005, SIAM J. Sci. Comput.): the
    linear operator ``c(k) = k² - k⁴`` (diagonal in Fourier space) is
    integrated exactly per step, and the nonlinear term
    ``N(u) = -½ ∂_x (u²)`` is treated via the ETDRK4 quadrature on a
    contour of M=16 roots of unity around each ``h·c(k)`` — the standard
    trick that keeps the φ-function evaluations well-conditioned
    uniformly in k.

    The default ``nx=256`` covers the L=32π domain with ``k_max=4`` —
    well past the inertial range cut-off for this IC — and ``dt=0.025``
    is the Kassam-Trefethen reference step size for this regime. Returns
    ``(t_grid, x_grid, u_grid)`` with shapes ``(nt,)``, ``(nx,)``,
    ``(nt, nx)`` — CPU float32, snapshots at ``t = i·(T_MAX/(nt-1))``.
    """
    if cache_path is None:
        cache_path = os.path.join(
            DEFAULT_REF_CACHE_DIR, f"ks_reference_nx{nx}_nt{nt}.pt"
        )
    if cache_path and os.path.isfile(cache_path):
        blob = torch.load(cache_path, weights_only=True)
        return blob["t"], blob["x"], blob["u"]

    import numpy as np

    x = X_MIN + L_DOMAIN * np.arange(nx) / nx                    # x ∈ [0, L)
    # Fourier wavenumbers in physical units (2π/L · integer).
    k = 2.0 * np.pi * np.fft.fftfreq(nx, L_DOMAIN / nx)
    c = k ** 2 - k ** 4                                          # linear part
    # 2/3-rule dealiasing for the (u²)_x nonlinear product.
    dealias = (np.abs(k) <= (2.0 / 3.0) * np.abs(k).max()).astype(np.float64)

    # ETDRK4 coefficients via contour integration.
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)      # roots of unity
    LR = dt * c[:, None] + r[None, :]
    Q = dt * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1).real
    f1 = dt * np.mean(
        (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1
    ).real
    f2 = dt * np.mean(
        (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1
    ).real
    f3 = dt * np.mean(
        (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1
    ).real
    E = np.exp(dt * c)
    E2 = np.exp(dt * c / 2.0)

    # Initial condition.
    u = np.cos(x / 16.0) * (1.0 + np.sin(x / 16.0))
    v = np.fft.fft(u)

    g = -0.5j * k * dealias        # N(v) = g · fft(real(ifft(v))²)

    def nonlin(v_):
        return g * np.fft.fft(np.real(np.fft.ifft(v_)) ** 2)

    # Snapshot schedule: save every `stride` integration steps so the
    # output grid has exactly `nt` time points spanning [T_MIN, T_MAX].
    total_steps = int(round((T_MAX - T_MIN) / dt))
    if total_steps < nt - 1:
        raise ValueError(
            f"dt={dt} too large: only {total_steps} steps to cover "
            f"[{T_MIN}, {T_MAX}] but need at least {nt - 1} for nt={nt}"
        )
    stride = total_steps // (nt - 1)
    snapshots_v = [v.copy()]
    snapshots_t = [T_MIN]
    for n in range(1, total_steps + 1):
        Nv = nonlin(v)
        a = E2 * v + Q * Nv
        Na = nonlin(a)
        b = E2 * v + Q * Na
        Nb = nonlin(b)
        c_step = E2 * a + Q * (2.0 * Nb - Nv)
        Nc = nonlin(c_step)
        v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3
        if n % stride == 0 and len(snapshots_v) < nt:
            snapshots_v.append(v.copy())
            snapshots_t.append(T_MIN + n * dt)

    t_grid = np.asarray(snapshots_t, dtype=np.float32)
    x_grid = x.astype(np.float32)
    u_grid = np.stack(
        [np.real(np.fft.ifft(vk)) for vk in snapshots_v]
    ).astype(np.float32)

    t = torch.from_numpy(t_grid)
    xt = torch.from_numpy(x_grid)
    ug = torch.from_numpy(u_grid)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"t": t, "x": xt, "u": ug}, cache_path)
    return t, xt, ug


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
    beta1: float = 0.9, beta2: float = 0.95,
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
            precondition_frequency=10, aux_batch_size=10,
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
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=200)
    p.add_argument("--n-bc", type=int, default=200)
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so this is not free — "
                        "keep small. KS's 4th-order PDE makes the aux pass "
                        "~2x more expensive than Burgers per point.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps): larger "
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.95,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=128,
                   help="MLP width. KS chaotic regime needs more capacity than "
                        "Burgers; 128 is a reasonable default.")
    p.add_argument("--depth", type=int, default=6, help="MLP depth.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps. For the SOAP/AdamW baselines "
                        "this is the schedule warmup; for Gnome it is passed "
                        "as its internal `warmup=`.")
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero (standard treatment), 1.0 disables "
                        "decay. Gnome (MSE) never decays regardless.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
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
        "x_domain": (X_MIN, X_MAX),
        "t_domain": (T_MIN, T_MAX),
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
        print("  loading / building reference solution...", flush=True)
    t_ref, x_ref, u_ref = ks_reference()

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
