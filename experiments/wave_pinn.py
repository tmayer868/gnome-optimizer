"""Wave-equation PINN: AdamW vs SOAP vs Gnome.

PDE:  u_tt - c²·u_xx = 0,    c = 2,    x ∈ [0, 1],  t ∈ [0, 1]
ICs:  u(0, x) = sin(πx) + 0.5·sin(2cπx),   u_t(0, x) = 0
BCs:  u(t, 0) = u(t, 1) = 0    (Dirichlet)

The second-order linear wave equation (jaxpi's wave benchmark, c=2). Two
things make it its own kind of hard: (1) the residual has an exact
d'Alembert null space — any ``f(x−ct) + g(x+ct)`` satisfies the PDE — so
the ICs/BCs carry all the anchoring; and (2) it is non-dissipative, so
phase errors never decay and must be held coherent over the whole time
domain. The reference solution
``u = sin(πx)cos(cπt) + 0.5·sin(2cπx)cos(4cπt)`` is a sum of pure sine
modes that vanish at x = 0, 1 (Dirichlet) and carries content up to the
``sin(4πx)cos(8πt)`` mode — which is why wave benefits from a spectral
(Fourier) input embedding more than the other 1-D benchmarks.

Architecture options (all share the optimizer, so it's the only variable):

* ``--arch {mlp, modified}`` — plain tanh MLP, or the gated modified MLP of
  Wang, Teng & Perdikaris (2021). ``--hidden`` / ``--depth`` size both.
* ``--embed {none, periodic, fourier}`` — the input embedding:
    - ``none``: raw ``(t, x)``.
    - ``periodic``: ``[t, cos(πx), sin(πx)]`` (period-2 in x). NOTE this
      does NOT enforce wave's *Dirichlet* BC (cos(πx)≠0 at the boundaries);
      it only helps spectrally. It's the embedding that makes sense for the
      *periodic* AC problem.
    - ``fourier``: fixed random Fourier features (Tancik et al. 2020) of
      ``(t, x)`` — ``[sin(zB), cos(zB)]`` with ``B ~ N(0, scale²)``. The
      spectral aid the wave solution's high modes want; ``--fourier-scale``
      is the key knob (jaxpi uses 10 for wave). ``B`` is fixed, not trained.
* ``--hard-bc`` — enforce the Dirichlet BC by construction via the output
  transform ``u = sin(πx)·N(t, x)`` (vanishes at x = 0, 1). When set, the
  soft ``bc`` loss block is dropped (it would be identically zero). This is
  the wave analog of AC's exact-periodicity embedding, and pairs naturally
  with ``--embed fourier``.

Baselines (SOAP, AdamW) get linear-warmup + cosine-decay (``--cosine-decay``
= final-lr fraction; 1.0 disables); Gnome runs at a fixed lr — its
Gauss-Newton step self-anneals. Reference: analytic (no download).

Usage:

    uv run -m experiments.wave_pinn --optimizer gnome --arch modified \\
        --embed fourier --fourier-scale 10 --hard-bc
    uv run -m experiments.wave_pinn --optimizer soap  --arch mlp --embed none
"""

from __future__ import annotations

import argparse
import math
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


EXPERIMENT = "wave_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = 0.0, 1.0
C_SPEED = 2.0
A_COEFF = 0.5


# ========================= Input embeddings =========================

class NoEmbed(nn.Module):
    """Raw ``(t, x)``."""
    out_dim = 2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, x], dim=1)


class PeriodicEmbed(nn.Module):
    """``[t, cos(πx), sin(πx), cos(πt), sin(πt)]`` — period-2 in x (spectral aid on wave; it
    does not enforce the Dirichlet BC)."""
    n_freq = 5
    out_dim = 2 + 4 * (n_freq - 1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        in_features = [t, x]
        in_features += [torch.sin(n * math.pi * x) for n in range(1, self.n_freq)]
        in_features += [torch.cos(n * math.pi * x) * x * (1.0 - x) for n in range(1, self.n_freq)]
        in_features += [torch.cos(n * math.pi * t) * t * (1.0 - t) for n in range(1, self.n_freq)]
        in_features += [torch.sin(n * math.pi * t) for n in range(1, self.n_freq)]
        return torch.cat(in_features, dim=1)


class FourierEmbed(nn.Module):
    """Random Fourier features (Tancik et al. 2020): embeds ``(t, x)`` as
    ``[sin(zB), cos(zB), t, x]``. Total output dimension matches ``embed_dim``.
    """

    def __init__(self, embed_dim: int = 128, scale: float = 2.0):
        super().__init__()
        assert embed_dim % 2 == 0, "fourier embed_dim must be even"
        self.out_dim = embed_dim

        # Calculate the size needed for the projection
        # We subtract 2 because t and x (2 features) are concatenated at the end
        proj_dim = (embed_dim - 2) // 2

        # 1. Correct logic: Pass string name first, do NOT assign the function output to a variable
        B_tensor = torch.randn(2, proj_dim) * scale
        self.register_buffer('B', B_tensor)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Assuming t and x have shape (batch_size, 1)
        p = torch.cat([t, x], dim=1) @ self.B  # self.B is now safely available
        return torch.cat([torch.sin(p), torch.cos(p), t, x], dim=1)

def build_embedding(embed: str, fourier_dim: int, fourier_scale: float
                    ) -> nn.Module:
    if embed == "none":
        return NoEmbed()
    if embed == "periodic":
        return PeriodicEmbed()
    if embed == "fourier":
        return FourierEmbed(fourier_dim, fourier_scale)
    raise ValueError(f"unknown embed: {embed}")


# ========================= Models =========================

class MLP(nn.Module):
    """Plain tanh MLP over an input embedding. ``depth`` = Linear-layer count.

    With ``hard_bc``, the output is transformed to ``sin(πx)·N(t,x)`` so it
    vanishes at x = 0, 1 (exact Dirichlet).
    """

    def __init__(self, embed: nn.Module, hidden: int = 256, depth: int = 4,
                 hard_bc: bool = False):
        super().__init__()
        assert depth >= 2
        self.embed = embed
        self.hard_bc = hard_bc
        layers: list[nn.Module] = [nn.Linear(embed.out_dim, hidden), nn.Tanh()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.net(self.embed(t, x))
        if self.hard_bc:
            out = torch.sin(math.pi * x) * out
        return out


class ModifiedMLP(nn.Module):
    """Modified MLP (Wang, Teng & Perdikaris 2021) over an input embedding.

    Two encoders ``u, v`` gate every hidden layer:
    ``h = tanh(W_l h);  h = h·u + (1-h)·v``. ``depth`` = gated-hidden-layer
    count. ``hard_bc`` applies the ``sin(πx)·N`` Dirichlet transform.
    """

    def __init__(self, embed: nn.Module, hidden: int = 256, depth: int = 4,
                 hard_bc: bool = False):
        super().__init__()
        assert depth >= 1
        self.embed = embed
        self.hard_bc = hard_bc
        d = embed.out_dim
        self.enc_u = nn.Linear(d, hidden)
        self.enc_v = nn.Linear(d, hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(d if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(t, x)
        u = torch.tanh(self.enc_u(z))
        v = torch.tanh(self.enc_v(z))
        h = z
        for layer in self.layers:
            h = torch.tanh(layer(h))
            h = h * u + (1.0 - h) * v
        out = self.out(h)
        if self.hard_bc:
            out = torch.sin(math.pi * x) * out
        return out


def build_model(arch: str, embed: nn.Module, hidden: int, depth: int,
                hard_bc: bool) -> nn.Module:
    if arch == "mlp":
        return MLP(embed, hidden, depth, hard_bc)
    if arch == "modified":
        return ModifiedMLP(embed, hidden, depth, hard_bc)
    raise ValueError(f"unknown arch: {arch}")


# ========================= Exact solution =========================

def u_exact(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Analytic wave solution
    ``sin(πx)cos(cπt) + A·sin(2cπx)cos(4cπt)``."""
    return (
        torch.sin(math.pi * x) * torch.cos(C_SPEED * math.pi * t)
        + A_COEFF * torch.sin(2 * C_SPEED * math.pi * x)
        * torch.cos(4 * C_SPEED * math.pi * t)
    )


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module, t: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Wave PDE residual ``u_tt - c²·u_xx`` at (t, x)."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_tt = autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_tt - C_SPEED ** 2 * u_xx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Displacement IC residual: ``u(0, x) - [sin(πx) + A·sin(2cπx)]``."""
    t0 = torch.zeros_like(x)
    u0 = torch.sin(math.pi * x) + A_COEFF * torch.sin(2 * C_SPEED * math.pi * x)
    return model(t0, x) - u0


def ict_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Velocity IC residual: ``u_t(0, x) - 0``."""
    t0 = torch.zeros_like(x).requires_grad_(True)
    u = model(t0, x)
    u_t = autograd.grad(u, t0, torch.ones_like(u), create_graph=True)[0]
    return u_t


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Dirichlet BC residual: ``u(t, 0)`` and ``u(t, 1)`` (both target 0),
    stacked. Identically zero when the model uses the hard-BC transform."""
    x_l = torch.full_like(t, X_MIN)
    x_r = torch.full_like(t, X_MAX)
    return torch.cat([model(t, x_l), model(t, x_r)], dim=0)


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
    model: nn.Module, batch, hard_bc: bool
) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (equal weights):
    pde + displacement IC + velocity IC (+ Dirichlet BC unless ``hard_bc``,
    which enforces it by construction)."""
    t_pde, x_pde, x_ic, t_bc = batch
    blocks = [
        pde_residual(model, t_pde, x_pde),
        ic_residual(model, x_ic),
        ict_residual(model, x_ic),
    ]
    if not hard_bc:
        blocks.append(bc_residual(model, t_bc))
    return stack_residuals(blocks)


def term_losses(model: nn.Module, batch) -> dict[str, float]:
    """Per-term MSE for diagnostic logging (bc logged always; ~0 under
    hard-BC, a sanity check on the transform)."""
    t_pde, x_pde, x_ic, t_bc = batch
    return {
        "pde": pde_residual(model, t_pde, x_pde).pow(2).mean().item(),
        "ic": ic_residual(model, x_ic).pow(2).mean().item(),
        "ict": ict_residual(model, x_ic).pow(2).mean().item(),
        "bc": bc_residual(model, t_bc).pow(2).mean().item(),
    }


# ========================= Reference + eval =========================

def wave_reference(
    nt: int = 200, nx: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic reference on a uniform grid. Returns ``(t, x, u)`` with
    shapes ``(nt,)``, ``(nx,)``, ``(nt, nx)``."""
    t = torch.linspace(T_MIN, T_MAX, nt)
    x = torch.linspace(X_MIN, X_MAX, nx)
    tt, xx = torch.meshgrid(t, x, indexing="ij")
    u = u_exact(tt, xx)
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

    Gnome runs at a fixed lr (self-annealing GGN step) so it gets no
    scheduler — only its internal warmup. SOAP and AdamW get the standard
    linear-warmup + cosine-decay; ``cosine_decay`` is the final-lr fraction
    (0.0 → decay to zero, 1.0 → decay disabled).
    """
    if name == "gnome":
        cfg = dict(
            lr=lr, weight_decay=weight_decay,
            betas=(beta1, beta2), shampoo_beta=beta2, eps=eps,
            precondition_frequency=20,
            clip=1.0, warmup=warmup,
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
    p.add_argument("--arch", choices=["mlp", "modified"], default="modified",
                   help="Network: plain tanh MLP or the gated modified MLP.")
    p.add_argument("--embed", choices=["none", "periodic", "fourier"],
                   default="fourier",
                   help="Input embedding. 'fourier' = fixed random Fourier "
                        "features (the spectral aid wave wants).")
    p.add_argument("--fourier-dim", type=int, default=128,
                   help="Fourier embedding output dim (even). --embed fourier.")
    p.add_argument("--fourier-scale", type=float, default=8.0,
                   help="Fourier frequency spread (B ~ N(0, scale²)). THE knob "
                        "— jaxpi uses ~10 for wave; sweep it. --embed fourier.")
    p.add_argument("--hard-bc", action="store_true",
                   help="Enforce Dirichlet u=0 by construction via "
                        "u = sin(πx)·N(t,x); drops the soft bc loss block.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--n-pde", type=int, default=4000)
    p.add_argument("--n-ic", type=int, default=200,
                   help="IC points (shared by the displacement and velocity "
                        "IC blocks).")
    p.add_argument("--n-bc", type=int, default=200)
    p.add_argument("--aux-frac", type=float, default=0.03,
                   help="Aux batch sizes for Gnome are max(K_min, int(N * "
                        "aux_frac)) per block. Each aux pass is a full "
                        "higher-order residual eval, so keep small.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon in m̂/(v̂+eps). Gnome "
                        "only; SOAP/AdamW keep eps=1e-8.")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=256, help="Network width.")
    p.add_argument("--depth", type=int, default=4,
                   help="Network depth: Linear-layer count for --arch mlp, "
                        "number of gated hidden layers for --arch modified.")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: 0.0 "
                        "decays to zero, 1.0 disables. Gnome never decays.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    torch.manual_seed(args.seed)
    device = pick_device()
    embed = build_embedding(args.embed, args.fourier_dim, args.fourier_scale)
    model = build_model(
        args.arch, embed, args.hidden, args.depth, args.hard_bc
    ).to(device)
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
        "embed": args.embed,
        "fourier_dim": args.fourier_dim if args.embed == "fourier" else None,
        "fourier_scale": args.fourier_scale if args.embed == "fourier" else None,
        "hard_bc": args.hard_bc,
        "steps": args.steps,
        "hidden": args.hidden,
        "depth": args.depth,
        "c_speed": C_SPEED,
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
        embed_desc = args.embed
        if args.embed == "fourier":
            embed_desc += f"(dim={args.fourier_dim}, scale={args.fourier_scale})"
        print(
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | embed={embed_desc} | "
            f"hard_bc={args.hard_bc} | params={n_params:,} | device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}",
            flush=True,
        )
    t_ref, x_ref, u_ref = wave_reference()

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_pde = last_ic = last_ict = last_bc = float("nan")
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        main_batch = sample_batch(args.n_pde, args.n_ic, args.n_bc, device)
        if args.optimizer == "gnome":
            aux_batch = sample_batch(n_pde_aux, n_ic_aux, n_bc_aux, device)

            def main_closure():
                r = stacked_residuals(model, main_batch, args.hard_bc)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, args.hard_bc)
                return r, torch.zeros_like(r)

            loss = opt.step(main_closure, aux_closure)
        else:
            opt.zero_grad()
            r = stacked_residuals(model, main_batch, args.hard_bc)
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
            last_pde, last_ic = tl["pde"], tl["ic"]
            last_ict, last_bc = tl["ict"], tl["bc"]
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            run.log_val(step + 1, loss=last_avg, lr=current_lr(opt),
                        pde=tl["pde"], ic=tl["ic"], ict=tl["ict"],
                        bc=tl["bc"], rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:5d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={tl['pde']:.3e}  ic={tl['ic']:.3e}  "
                    f"ict={tl['ict']:.3e}  bc={tl['bc']:.3e}  "
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
    print(f"  final pde={last_pde:.3e}  ic={last_ic:.3e}  "
          f"ict={last_ict:.3e}  bc={last_bc:.3e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
