"""Kuramoto-Sivashinsky PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + u·u_x + u_xx + u_xxxx = 0,    x ∈ [0, 32π],  t ∈ [0, 30]
IC:   u(0, x) = cos(x/16)·(1 + sin(x/16))
BC:   u(t, 0) = u(t, 32π),  u_x(t, 0) = u_x(t, 32π)    (periodic)

The large-L (L = 32π) periodic domain and Kassam-Trefethen IC are the
canonical Kuramoto-Sivashinsky setup, but we integrate only to t = 30 —
which stays inside the *laminar pre-chaotic transient*. From this smooth
IC the solution stays spatially coherent and its energy decays through
t ≈ 30; the coherent structure then destabilizes and breaks up into
sustained spatiotemporal chaos near t ≈ 40. We deliberately stop short of
that breakup: once the flow is chaotic, sensitive dependence makes a
pointwise rel-L2 against a reference meaningless (it would measure the
chaos, not the optimizer). What remains on [0, 30] is the difficulty we
want — the stiffness of the 4th-order linear operator ``u_xx + u_xxxx``:
small-k modes are amplified (k² > k⁴), large-k damped (k⁴ ≫ k²), and the
nonlinear ``u·u_x`` term ties them together.

Compared to Burgers, the PINN residual costs roughly 2× more (four
``autograd.grad`` calls through the network input instead of two), but
the headline difficulty is the same family of PDE/IC/BC residual
stiffness — handled here by the same ``gnome.stack_residuals`` pattern
with equal block weights (no causal training, no grad-norm weighting).

Reference: ETDRK4 spectral integrator (Kassam & Trefethen 2005). RK4
would need a CFL of ``O(dx⁴ / 1)`` because of the 4th-order linear
operator; ETDRK4 handles the linear part exactly per step and lets us
take ``dt ≈ 0.025`` even on the large-L domain.

Two architectures, selectable via ``--arch``:

* ``mlp`` — a plain tanh MLP (the original setup here).
* ``modified`` — the modified MLP of Wang, Teng & Perdikaris (2021): two
  input encoders gate every hidden layer via
  ``h = tanh(W h); h = h·u + (1-h)·v``. This is the *architecture only* —
  none of the rest of the jaxpi pipeline (random weight factorization,
  Fourier features, causal weighting, grad-norm balancing) is ported.

``--embed`` chooses the input representation, and it changes the block
structure:

* ``none`` (default) feeds raw ``[t, x]``, and periodicity is enforced as
  *soft* constraints on ``u`` and ``u_x`` at the endpoints — a third
  residual block. For a 4th-order spatial PDE we would ideally match
  derivatives up to third order, but C¹ periodicity is the standard PINN
  treatment for KS and leaves the network enough freedom for the higher
  derivatives to agree with interior values.
* ``periodic`` feeds ``[t, cos(2πx/L), sin(2πx/L)]``, which makes the
  network **exactly** period-L in x. Periodicity then holds to machine
  precision for every derivative, so the BC block is redundant and is
  dropped — the loss becomes two blocks (PDE, IC) instead of three. This
  matches how the Allen-Cahn and KdV experiments here handle their
  periodic domains, and how jaxpi imposes exact periodic BCs.

All optimizers share the chosen network so the only variable is the
optimizer. Every optimizer gets the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then constant,
which suits Gnome on MSE since its step self-anneals as the residual shrinks).

``--optimizer adamw+lbfgs`` is the classic PINN recipe (and the SOAP-PINN
paper's real first-order baseline — nobody runs Adam alone): the AdamW phase
runs for ``--steps`` as usual, then an L-BFGS phase (``--lbfgs-steps``)
refines on a *fixed, full-batch* collocation set. L-BFGS builds a curvature
history from a sequence of (grad, step) pairs, which is only meaningful if
the objective is the same function each iteration — so, unlike the AdamW
phase, its points are drawn once and held constant.

Usage:

    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer gnome --seed 0
    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer soap  --seed 0
    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer adamw+lbfgs
    uv run -m experiments.kuramoto_sivashinsky_pinn --optimizer gnome \\
        --arch modified --embed periodic
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
    cosine_scheduler,
    current_lr,
    pick_device,
)


EXPERIMENT = "kuramoto_sivashinsky_pinn"

T_MIN, T_MAX = 0.0, 30.0
X_MIN, X_MAX = 0.0, 32.0 * math.pi
L_DOMAIN = X_MAX - X_MIN


# ========================= Model =========================

class RawEmbed(nn.Module):
    """``[t, x]`` — the raw coordinates. Periodicity must be imposed softly."""
    out_dim = 2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, x], dim=1)


class PeriodicEmbed(nn.Module):
    """``[t, cos(2πx/L), sin(2πx/L)]`` — exactly period-L in x.

    Any function of these features repeats with period ``L = X_MAX - X_MIN``
    by construction, so ``u`` and *all* of its x-derivatives match at the
    endpoints to machine precision. That makes the soft BC block redundant.
    """
    out_dim = 3

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        k = 2.0 * math.pi / L_DOMAIN
        return torch.cat([t, torch.cos(k * x), torch.sin(k * x)], dim=1)


def build_embedding(embed: str) -> nn.Module:
    if embed == "none":
        return RawEmbed()
    if embed == "periodic":
        return PeriodicEmbed()
    raise ValueError(f"unknown embedding: {embed}")


class MLP(nn.Module):
    """Plain tanh MLP over an input embedding: ``(t, x) → u``.

    ``depth`` = number of Linear layers.
    """

    def __init__(self, embed: nn.Module, hidden: int = 128, depth: int = 6):
        super().__init__()
        assert depth >= 2
        self.embed = embed
        d = embed.out_dim
        layers: list[nn.Module] = [nn.Linear(d, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(t, x))


class ModifiedMLP(nn.Module):
    """Modified MLP (Wang, Teng & Perdikaris 2021) over an input embedding.

    Two encoders gate every hidden layer. The gate is written in the
    algebraically equivalent form ``h = v + h·(u - v)`` via one fused
    ``addcmul`` (rather than ``h·u + (1-h)·v``, three elementwise kernels and
    three autograd nodes), and the two encoders are fused into a single
    Linear producing ``2·hidden`` features. ``depth`` = gated-hidden-layer
    count.

    Architecture only — no random weight factorization, Fourier features or
    causal weighting (jaxpi-pipeline pieces, deliberately not ported).
    """

    def __init__(self, embed: nn.Module, hidden: int = 128, depth: int = 6):
        super().__init__()
        assert depth >= 1
        self.embed = embed
        d = embed.out_dim
        # Fused u/v encoder: one matmul, chunked into the two gates.
        self.enc_uv = nn.Linear(d, 2 * hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(d if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(t, x)

        uv = torch.tanh(self.enc_uv(z))
        enc_a, enc_b = uv.chunk(2, dim=-1)
        w = enc_a - enc_b  # computed once; gate becomes enc_b + h*w

        h = z
        for layer in self.layers:
            h = torch.tanh(layer(h))
            h = torch.addcmul(enc_b, h, w)  # == h*enc_a + (1-h)*enc_b
        return self.out(h)


def build_model(arch: str, embed: nn.Module, hidden: int, depth: int
                ) -> nn.Module:
    if arch == "mlp":
        return MLP(embed, hidden=hidden, depth=depth)
    if arch == "modified":
        return ModifiedMLP(embed, hidden=hidden, depth=depth)
    raise ValueError(f"unknown arch: {arch}")


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

    Only used with ``--embed none``; the periodic embedding makes this
    identically zero (see ``PeriodicEmbed``).
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


def stacked_residuals(
    model: nn.Module, batch, use_bc: bool = True
) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights).

    ``use_bc=False`` drops the soft periodicity block, which the periodic
    embedding satisfies exactly — computing it would just append a vector of
    zeros at the cost of two more backward passes per step.
    """
    t_pde, x_pde, x_ic, t_bc = batch
    blocks = [
        pde_residual(model, t_pde, x_pde),
        ic_residual(model, x_ic),
    ]
    if use_bc:
        blocks.append(bc_residual(model, t_bc))
    return stack_residuals(blocks)


def term_losses(model: nn.Module, batch, use_bc: bool = True
                ) -> dict[str, float]:
    """Per-term MSE for diagnostic logging."""
    t_pde, x_pde, x_ic, t_bc = batch
    terms = {
        "pde": pde_residual(model, t_pde, x_pde).pow(2).mean().item(),
        "ic": ic_residual(model, x_ic).pow(2).mean().item(),
    }
    if use_bc:
        terms["bc"] = bc_residual(model, t_bc).pow(2).mean().item()
    return terms


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
            loss="mse", precondition_1d=True,
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


# ========================= L-BFGS refinement phase =========================

def lbfgs_phase(
    model: nn.Module, device: torch.device, args: argparse.Namespace,
    run: RunLogger, t_ref: torch.Tensor, x_ref: torch.Tensor,
    u_ref: torch.Tensor, use_bc: bool, start_step: int, best_rel_l2: float,
) -> dict:
    """Full-batch L-BFGS refinement, run after the AdamW phase.

    L-BFGS approximates curvature from a history of (grad, step) pairs, which
    is only valid if the objective is a fixed function — so we draw ONE
    collocation set here and reuse it for every iteration (unlike the AdamW
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
    # Log at roughly the same iteration cadence as the AdamW phase: one outer
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
    last_terms: dict[str, float] = {}

    for i in range(args.lbfgs_steps):
        def closure():
            opt.zero_grad()
            r = stacked_residuals(model, fixed_batch, use_bc)
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
            tl = term_losses(model, fixed_batch, use_bc)
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_terms, last_rel_l2 = tl, rl2
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_loss, lr=args.lbfgs_lr,
                        rel_l2=rl2, **tl)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (i + 1) * 1000
                terms = "  ".join(f"{k}={v:.3e}" for k, v in tl.items())
                print(
                    f"  L-BFGS {i + 1:5d}/{args.lbfgs_steps}  "
                    f"loss={last_loss:.4e}  {terms}  "
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw", "adamw+lbfgs"])
    p.add_argument("--arch", choices=["mlp", "modified"], default="mlp",
                   help="Network: plain tanh MLP or the gated modified MLP "
                        "(Wang et al. 2021). --hidden / --depth control both. "
                        "Defaults to mlp — the original setup for this "
                        "experiment, so existing runs stay comparable.")
    p.add_argument("--embed", choices=["none", "periodic"], default="none",
                   help="Input embedding. 'none' feeds raw [t, x] and keeps "
                        "the soft periodic BC block. 'periodic' feeds "
                        "[t, cos(2pi x/L), sin(2pi x/L)], making the network "
                        "exactly period-L in x — periodicity then holds for "
                        "every derivative and the BC block is DROPPED (two "
                        "blocks instead of three).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=200)
    p.add_argument("--n-bc", type=int, default=200)
    p.add_argument("--aux-frac", type=float, default=0.10,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so this is not free — "
                        "keep small. KS's 4th-order PDE makes the aux pass "
                        "~2x more expensive than Burgers per point.")
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
                        "-> more gradient-descent-like, smaller -> fuller Newton "
                        "step. Gnome only; SOAP/AdamW keep their fixed eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9,
                   help="First-moment (momentum) EMA for Gnome and SOAP.")
    p.add_argument("--beta2", type=float, default=0.99,
                   help="Second-moment / preconditioner EMA (also shampoo_beta) for Gnome and SOAP.")
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=128,
                   help="Network width. The stiff KS operator needs more "
                        "capacity than Burgers; 128 is a reasonable default.")
    p.add_argument("--depth", type=int, default=6,
                   help="Network depth: Linear-layer count for --arch mlp, "
                        "number of gated hidden layers for --arch modified.")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="Linear LR warmup steps, applied to every optimizer.")
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


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    # The periodic embedding satisfies the BC block exactly, so it is dropped.
    use_bc = args.embed == "none"
    model = build_model(
        args.arch, build_embedding(args.embed), args.hidden, args.depth
    ).to(device)
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
        "optimizer": args.optimizer,
        "arch": args.arch,
        "embed": args.embed,
        "blocks": ["pde", "ic", "bc"] if use_bc else ["pde", "ic"],
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
        blocks = "pde+ic+bc" if use_bc else "pde+ic (exact periodic BC)"
        print(
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | embed={args.embed} | "
            f"params={n_params:,} | device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps} | "
            f"blocks={blocks}",
            flush=True,
        )
        print("  loading / building reference solution...", flush=True)
    t_ref, x_ref, u_ref = ks_reference()

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_terms: dict[str, float] = {}
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device)

            def main_closure():
                r = stacked_residuals(model, main_batch, use_bc)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, use_bc)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, use_bc)
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
                model, sample_batch(args.n_pde, args.n_ic, args.n_bc, device),
                use_bc,
            )
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_avg = sum(window) / len(window)
            last_terms, last_rel_l2 = tl, rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        rel_l2=rl2, **tl)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                terms = "  ".join(f"{k}={v:.3e}" for k, v in tl.items())
                print(
                    f"  step {step + 1:5d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  {terms}  "
                    f"rel_l2={rl2:.3e}  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    if args.optimizer == "adamw+lbfgs" and args.lbfgs_steps > 0:
        res = lbfgs_phase(
            model, device, args, run, t_ref, x_ref, u_ref, use_bc,
            start_step=args.steps, best_rel_l2=best_rel_l2,
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
