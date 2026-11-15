"""Korteweg-de Vries PINN: AdamW vs SOAP vs Gnome.

PDE:  u_t + η·u·u_x + μ²·u_xxx = 0,   η = 1,  x ∈ [-1, 1],  t ∈ [0, 1]
IC:   u(0, x) = cos(πx)
BC:   u, u_x, u_xx  periodic in x    (u(t,-1)=u(t,1), etc.)

The Korteweg-de Vries equation — the canonical *dispersive* benchmark. The
third-order term ``μ²·u_xxx`` spreads energy across wavenumbers with no
dissipation to damp them, so the solution develops fine oscillatory
structure (soliton trains) that has to stay phase-coherent over the whole
domain. There is no smoothing to forgive errors and the derivative order is
one higher than the other 1-D benchmarks, so the residual is stiffer and
depends on ``u_xxx`` — a much harsher demand on the network's high-order
derivative representation. This is the benchmark with the most headroom in
the SOAP-PINN paper's single-window suite (their best is ~3.4e-4).

Two architectures, selectable via ``--arch``:

* ``mlp`` — a plain tanh MLP (as in the other torch PINN experiments).
* ``modified`` — the modified MLP of Wang, Teng & Perdikaris (2021): two
  input encoders ``u, v`` gate every hidden layer via
  ``h = tanh(W h); h = h·u + (1-h)·v``. This is the *architecture only* —
  none of the rest of the jaxpi pipeline (random weight factorization,
  Fourier features, causal weighting, grad-norm balancing) is ported.

Both architectures accept ``embedding periodic|none``. The default ``periodic``
uses ``[t, cos(πx), sin(πx)]``, enforcing period-2 spatial periodicity on
``[-1, 1]``. ``none`` feeds raw ``[t, x]``. Both retain the PDE, IC, and BC
blocks with configurable weights (equal by default) via ``gnome.stack_residuals``; the BC block enforces
matching ``u``, ``u_x``, and ``u_xx`` for raw inputs and is redundant up to
roundoff with the periodic embedding.

All optimizers share the chosen network so the only variable is the
optimizer. Every optimizer gets the same linear-warmup + cosine-decay schedule
(``--cosine-decay`` sets the final-lr fraction; 1.0 gives warmup then
constant, which suits Gnome on MSE since its step self-anneals as the
residual shrinks).

The optimizer choices are AdamW, SOAP, Gnome, SOAP + SS-Broyden, and
ENGD-W. The hybrid freezes one complete batch at the handoff and refines the
same weighted objective with a strong-Wolfe line search.

``--mu-sq`` controls the dispersive coefficient and defaults to ``0.022²``,
the JAX-PI benchmark value. That default uses the common static ``kdv.mat``
reference, downloading it from JAX-PI when missing and generating an equivalent
ETDRK4 solution only when offline. Other coefficients get a matching Fourier/ETDRK4
reference cached by solver parameters under ``experiments/reference_solutions``.

Usage:

    # Edit experiments/configs/pinns/kdv_pinn.py: optimizer='gnome', arch='modified'
    uv run -m experiments.pinns.kdv_pinn
    # Edit experiments/configs/pinns/kdv_pinn.py: optimizer='soap', arch='mlp'
    uv run -m experiments.pinns.kdv_pinn
    # Edit experiments/configs/pinns/kdv_pinn.py: optimizer='gnome', embedding='none'
    uv run -m experiments.pinns.kdv_pinn
    # Edit experiments/configs/pinns/kdv_pinn.py: optimizer='soap+ssbroyden', arch='modified'
    uv run -m experiments.pinns.kdv_pinn
"""

from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from gnome import Gnome, JsonlDiagnostics, stack_residuals
from experiments.baselines import SOAP
from experiments.common import (
    ConcatEmbedding,
    DIVERGED_EXIT,
    ModifiedMLP,
    diverged,
    RunLogger,
    cosine_scheduler,
    current_lr,
    pick_device,
    PeriodicEmbedding,
)
from experiments.reference_solutions import cached_reference_path, reference_path
from experiments.common.precision import switch_to_float64 as _switch_optimizer_to_float64


from experiments.common import pinn_options as pinn

EXPERIMENT = "kdv_pinn"

T_MIN, T_MAX = 0.0, 1.0
X_MIN, X_MAX = -1.0, 1.0
ETA = 1.0                 # u·u_x coefficient
DEFAULT_MU_SQ = 0.022 ** 2
DEFAULT_REFERENCE_MODES = 511
DEFAULT_REFERENCE_TIME_STEPS = 200
DEFAULT_REFERENCE_SUBSTEPS = 20
DEFAULT_REFERENCE_PATH = reference_path("kdv.mat")
LEGACY_REFERENCE_PATHS = ("experiments/data/kdv.mat", "kdv.mat")
# Raissi's ``main/Data/KdV.mat`` uses mu_sq=0.0025. JAX-PI's pirate branch
# contains the coefficient-matched mu_sq=0.022**2 reference used here.
KDV_URL = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/kdv/data/kdv.mat"
)


# ========================= Models =========================
def build_embedding(embed: str, embed_dim: int = 32, embed_scale: float = 1.0) -> nn.Module:
    return pinn.build_embedding(embed, 2, spatial_dims=(1,), length=X_MAX - X_MIN,
                                periodic=True, embed_dim=embed_dim, embed_scale=embed_scale)


class MLP(nn.Module):
    """Plain tanh MLP: ``(t, x) → u``. ``depth`` = number of Linear layers.

    Input is the period-2 embedding ``[t, cos(πx), sin(πx)]`` by default,
    or raw ``[t, x]`` with ``embed="none"``."""

    def __init__(self, hidden: int = 256, depth: int = 4,
                 embed: str = "periodic", embed_dim: int = 32, embed_scale: float = 1.0):
        super().__init__()
        assert depth >= 2
        self.embed = build_embedding(embed, embed_dim, embed_scale)
        layers: list[nn.Module] = [nn.Linear(self.embed.out_dim, hidden), nn.Tanh()]
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)


    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.embed(t, x))

def build_model(arch: str, hidden: int, depth: int,
                embed: str = "periodic", embed_dim: int = 32, embed_scale: float = 1.0) -> nn.Module:
    """``(t, x) → u`` with periodic features or raw coordinates."""
    if arch == "mlp":
        return MLP(hidden=hidden, depth=depth, embed=embed, embed_dim=embed_dim, embed_scale=embed_scale)
    if arch == "modified":
        return ModifiedMLP(
            build_embedding(embed, embed_dim, embed_scale),
            hidden=hidden,
            depth=depth,
        )
    raise ValueError(f"unknown arch: {arch}")


# ========================= Residuals =========================

def pde_residual(
    model: nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    mu_sq: float = DEFAULT_MU_SQ,
) -> torch.Tensor:
    """KdV PDE residual ``u_t + η·u·u_x + μ²·u_xxx`` at (t, x)."""
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    u = model(t, x)
    u_t = autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]
    return u_t + ETA * u * u_x + mu_sq * u_xxx


def ic_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """IC residual: ``u(0, x) - cos(πx)``."""
    t0 = torch.zeros_like(x)
    u0 = torch.cos(math.pi * x)
    return model(t0, x) - u0


def bc_residual(model: nn.Module, t: torch.Tensor) -> torch.Tensor:
    """Periodic BC residual through ``u_xx`` — KdV is third-order, so the
    well-posed periodic conditions match ``u``, ``u_x`` and ``u_xx`` across
    the ends. Returns ``[u_l-u_r, u_l_x-u_r_x, u_l_xx-u_r_xx]`` stacked."""
    x_l = torch.full_like(t, X_MIN, requires_grad=True)
    x_r = torch.full_like(t, X_MAX, requires_grad=True)
    u_l = model(t, x_l)
    u_r = model(t, x_r)
    u_l_x = autograd.grad(u_l, x_l, torch.ones_like(u_l), create_graph=True)[0]
    u_r_x = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
    u_l_xx = autograd.grad(u_l_x, x_l, torch.ones_like(u_l_x),
                           create_graph=True)[0]
    u_r_xx = autograd.grad(u_r_x, x_r, torch.ones_like(u_r_x),
                           create_graph=True)[0]
    return torch.cat([u_l - u_r, u_l_x - u_r_x, u_l_xx - u_r_xx], dim=0)


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


def residual_partition_sizes(batch) -> list[int]:
    """Scalar residual counts for PDE, IC, and the combined u/u_x/u_xx BC."""
    t_pde, _x_pde, x_ic, t_bc = batch
    return [t_pde.numel(), x_ic.numel(), 3 * t_bc.numel()]


def stacked_residuals(
    model: nn.Module,
    batch,
    mu_sq: float = DEFAULT_MU_SQ,
    weights=(1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Per-block residuals stacked via ``stack_residuals`` (PDE, IC, BC weights)."""
    t_pde, x_pde, x_ic, t_bc = batch
    return stack_residuals([
        pde_residual(model, t_pde, x_pde, mu_sq),
        ic_residual(model, x_ic),
        bc_residual(model, t_bc),
    ], weights=weights)


def term_losses(
    model: nn.Module,
    batch,
    mu_sq: float = DEFAULT_MU_SQ,
) -> dict[str, float]:
    """Per-term MSE for diagnostic logging."""
    t_pde, x_pde, x_ic, t_bc = batch
    return {
        "pde": pde_residual(model, t_pde, x_pde, mu_sq).pow(2).mean().item(),
        "ic": ic_residual(model, x_ic).pow(2).mean().item(),
        "bc": bc_residual(model, t_bc).pow(2).mean().item(),
    }


# ========================= Reference solution + eval =========================

def load_kdv_reference(
    path: str = DEFAULT_REFERENCE_PATH,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load ``kdv.mat``, generating the default reference when it is absent."""
    try:
        import scipy.io
    except ImportError as e:
        raise RuntimeError(
            "scipy is required to load kdv.mat. Install the experiment "
            "dependencies with `uv sync --extra experiments`."
        ) from e

    if path == DEFAULT_REFERENCE_PATH:
        path = cached_reference_path("kdv.mat", LEGACY_REFERENCE_PATHS)
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        download_path = f"{path}.download"
        print(
            f"[{EXPERIMENT}] kdv.mat missing; downloading {KDV_URL} ...",
            flush=True,
        )
        try:
            urllib.request.urlretrieve(KDV_URL, download_path)
            os.replace(download_path, path)
        except Exception as error:
            if os.path.isfile(download_path):
                os.remove(download_path)
            print(
                f"[{EXPERIMENT}] KdV download failed ({error}); generating "
                "the equivalent ETDRK4 reference instead...",
                flush=True,
            )
            t_generated, x_generated, u_generated = kdv_reference(
                DEFAULT_MU_SQ,
                modes=DEFAULT_REFERENCE_MODES,
                time_steps=DEFAULT_REFERENCE_TIME_STEPS,
                substeps_per_output=DEFAULT_REFERENCE_SUBSTEPS,
                use_cache=False,
            )
            temporary_path = f"{path}.tmp"
            scipy.io.savemat(
                temporary_path,
                {
                    "t": t_generated.numpy()[None, :],
                    "x": x_generated.numpy()[None, :],
                    "usol": u_generated.numpy(),
                    "method": "Fourier ETDRK4",
                    "mu_sq": np.float64(DEFAULT_MU_SQ),
                    "modes": np.int64(DEFAULT_REFERENCE_MODES),
                    "substeps_per_output": np.int64(
                        DEFAULT_REFERENCE_SUBSTEPS
                    ),
                },
                appendmat=False,
                do_compression=True,
            )
            os.replace(temporary_path, path)

    data = scipy.io.loadmat(path)
    required = {"t", "x", "usol"}
    missing = required.difference(data)
    if missing:
        raise ValueError(
            f"invalid KdV reference {path!r}: missing {sorted(missing)}"
        )

    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
    solution = np.asarray(data["usol"], dtype=np.float64)
    if solution.shape != (t.size, x.size):
        raise ValueError(
            f"invalid KdV reference {path!r}: usol has shape "
            f"{solution.shape}, expected {(t.size, x.size)}"
        )
    if not (
        np.isfinite(t).all()
        and np.isfinite(x).all()
        and np.isfinite(solution).all()
    ):
        raise ValueError(f"invalid KdV reference {path!r}: non-finite values")
    if not (
        np.all(np.diff(t) > 0.0)
        and np.all(np.diff(x) > 0.0)
        and np.isclose(t[0], T_MIN)
        and np.isclose(t[-1], T_MAX)
        and np.isclose(x[0], X_MIN)
        and np.isclose(x[-1], X_MAX)
    ):
        raise ValueError(
            f"invalid KdV reference {path!r}: grid does not match the "
            "configured time and space domains"
        )

    return (
        torch.from_numpy(t.copy()),
        torch.from_numpy(x.copy()),
        torch.from_numpy(solution.copy()),
    )


def kdv_reference_cache_path(
    mu_sq: float,
    modes: int,
    time_steps: int,
    substeps_per_output: int,
) -> str:
    """Return the parameter-specific cache path for a generated KdV solve."""
    filename = (
        f"kdv_reference_mu{mu_sq:.12g}_modes{modes}_nt{time_steps}"
        f"_substeps{substeps_per_output}.pt"
    )
    return cached_reference_path(
        filename, [os.path.join("experiments/data", filename)]
    )


def kdv_reference(
    mu_sq: float = DEFAULT_MU_SQ,
    *,
    modes: int = DEFAULT_REFERENCE_MODES,
    time_steps: int = DEFAULT_REFERENCE_TIME_STEPS,
    substeps_per_output: int = DEFAULT_REFERENCE_SUBSTEPS,
    cache_path: str | None = None,
    use_cache: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a periodic Fourier reference with fourth-order ETDRK4.

    The stiff dispersive term is advanced exactly in Fourier space, while the
    nonlinear advection term uses a two-thirds-dealiased pseudospectral
    evaluation.  This makes the solve both fast and insensitive to the large
    imaginary eigenvalues of ``mu_sq * d³/dx³``.  At the defaults the solve
    takes about 0.3 seconds on a laptop and matches JAX-PI's static ``kdv.mat``
    for ``mu_sq=0.022²`` to roughly 1e-8 relative L2.

    ``modes`` is the number of unique points on ``[-1, 1)`` and must be odd so
    there is no ambiguous Nyquist mode.  The returned spatial grid duplicates
    the periodic endpoint at ``x=1``, preserving the old reference layout:
    ``t.shape == (time_steps + 1,)``, ``x.shape == (modes + 1,)``, and
    ``u.shape == (time_steps + 1, modes + 1)``.
    """
    if not math.isfinite(mu_sq) or mu_sq <= 0.0:
        raise ValueError(f"mu_sq must be finite and positive; got {mu_sq}")
    if modes < 3 or modes % 2 == 0:
        raise ValueError(f"modes must be an odd integer >= 3; got {modes}")
    if time_steps < 1:
        raise ValueError(f"time_steps must be >= 1; got {time_steps}")
    if substeps_per_output < 1:
        raise ValueError(
            "substeps_per_output must be >= 1; "
            f"got {substeps_per_output}"
        )
    if use_cache:
        if cache_path is None:
            cache_path = kdv_reference_cache_path(
                mu_sq, modes, time_steps, substeps_per_output
            )
        if os.path.isfile(cache_path):
            blob = torch.load(cache_path, weights_only=True)
            return blob["t"], blob["x"], blob["u"]
    else:
        cache_path = None

    length = X_MAX - X_MIN
    t = np.linspace(T_MIN, T_MAX, time_steps + 1, dtype=np.float64)
    x_periodic = X_MIN + length * np.arange(modes, dtype=np.float64) / modes
    wave_number = 2.0 * np.pi * np.fft.fftfreq(modes, d=length / modes)

    # u_t = -eta/2 * d_x(u²) - mu_sq * u_xxx. Since the Fourier symbol
    # of d_xxx is -i*k³, the exactly integrated linear symbol is +i*mu²*k³.
    linear = 1j * mu_sq * wave_number**3
    step_size = (t[1] - t[0]) / substeps_per_output
    z = step_size * linear
    E = np.exp(z)
    E2 = np.exp(z / 2.0)

    # Contour-integral ETDRK4 coefficients.  Evaluating the phi-function
    # combinations on a small complex contour avoids cancellation near z=0;
    # a full circle is used because this KdV linear operator is imaginary.
    contour_points = 64
    roots = np.exp(
        2j * np.pi * (np.arange(contour_points) + 0.5) / contour_points
    )
    LR = z[:, None] + roots[None, :]
    exp_LR = np.exp(LR)
    Q = step_size * np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1)
    f1 = step_size * np.mean(
        (-4.0 - LR + exp_LR * (4.0 - 3.0 * LR + LR**2)) / LR**3,
        axis=1,
    )
    f2 = step_size * np.mean(
        (2.0 + LR + exp_LR * (-2.0 + LR)) / LR**3,
        axis=1,
    )
    f3 = step_size * np.mean(
        (-4.0 - 3.0 * LR - LR**2 + exp_LR * (4.0 - LR)) / LR**3,
        axis=1,
    )

    mode_number = np.fft.fftfreq(modes) * modes
    dealias = np.abs(mode_number) <= modes // 3

    def nonlinear(state_hat: np.ndarray) -> np.ndarray:
        filtered = state_hat * dealias
        state = np.fft.ifft(filtered).real
        result = (
            -0.5j * ETA * wave_number * np.fft.fft(state * state)
        )
        return result * dealias

    state_hat = np.fft.fft(np.cos(np.pi * x_periodic))
    solution = np.empty((time_steps + 1, modes), dtype=np.float64)
    solution[0] = np.fft.ifft(state_hat).real
    for output_index in range(1, time_steps + 1):
        for _ in range(substeps_per_output):
            Nv = nonlinear(state_hat)
            a = E2 * state_hat + Q * Nv
            Na = nonlinear(a)
            b = E2 * state_hat + Q * Na
            Nb = nonlinear(b)
            c = E2 * a + Q * (2.0 * Nb - Nv)
            Nc = nonlinear(c)
            state_hat = (
                E * state_hat
                + f1 * Nv
                + 2.0 * f2 * (Na + Nb)
                + f3 * Nc
            )
        solution[output_index] = np.fft.ifft(state_hat).real

    x_closed = np.linspace(X_MIN, X_MAX, modes + 1, dtype=np.float64)
    solution_closed = np.concatenate([solution, solution[:, :1]], axis=1)
    result = (
        torch.from_numpy(t),
        torch.from_numpy(x_closed),
        torch.from_numpy(solution_closed),
    )
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        torch.save({"t": result[0], "x": result[1], "u": result[2]}, cache_path)
    return result


def eval_rel_l2(
    model: nn.Module,
    t_ref: torch.Tensor, x_ref: torch.Tensor, u_ref: torch.Tensor,
    device: torch.device, batch_size: int = 8192,
) -> float:
    """Relative L2 with model inference native and all error math in float64."""
    nt, nx = u_ref.shape
    tt, xx = torch.meshgrid(t_ref, x_ref, indexing="ij")
    model_dtype = next(
        (parameter.dtype for parameter in model.parameters()
         if parameter.is_floating_point()),
        torch.get_default_dtype(),
    )
    t_flat = tt.reshape(-1, 1).to(device=device, dtype=model_dtype)
    x_flat = xx.reshape(-1, 1).to(device=device, dtype=model_dtype)
    # Keep float64 conversion off accelerators such as MPS, which has no
    # float64 support. Transfer first, then promote using the CPU kernel.
    u_ref64 = u_ref.detach().cpu().to(dtype=torch.float64)
    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, t_flat.shape[0], batch_size):
            preds.append(
                model(t_flat[i:i + batch_size], x_flat[i:i + batch_size])
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )
    if was_training:
        model.train()
    u_pred = torch.cat(preds).reshape(nt, nx)
    num = (u_pred - u_ref64).pow(2).sum().sqrt()
    den = u_ref64.pow(2).sum().sqrt()
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
            precondition_frequency=20,
            trust_radius=(trust_region if trust_region > 0 else None),
            loss="mse", precondition_1d=True,
        )
        opt = Gnome(params, **cfg)
    elif name == "soap":
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

    scheduler = cosine_scheduler(opt, warmup, total_steps, cosine_decay)
    cfg["warmup"] = warmup
    cfg["cosine_decay_floor"] = cosine_decay
    return opt, cfg, scheduler


# ========================= CLI / training =========================

def parse_args(argv=None) -> argparse.Namespace:
    from experiments.common.config import ConfigParser
    p = ConfigParser(__file__)
    args = p.parse_args(argv)
    if args.float64_after is not None and not 0 <= args.float64_after < args.steps:
        p.error("float64_after must satisfy 0 <= N < steps")
    if any(w < 0 for w in args.weights):
        p.error("weights must be non-negative")
    return args


def switch_to_float64(model, opt):
    """Promote the model and preserve the active optimizer and its schedule."""
    return _switch_optimizer_to_float64(model, opt)


@pinn.preserve_default_dtype
def train(args: argparse.Namespace) -> str:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(
        torch.float64 if args.float64 or args.float64_after == 0 or args.optimizer in ("engdw", "soap+ssbroyden") else torch.float32
    )
    try:
        return _train(args)
    finally:
        torch.set_default_dtype(previous_dtype)


def _train(args: argparse.Namespace) -> str:
    hybrid = args.optimizer == "soap+ssbroyden"
    if not math.isfinite(args.mu_sq) or args.mu_sq <= 0.0:
        raise ValueError(f"--mu-sq must be finite and positive; got {args.mu_sq}")
    use_mat_reference = args.mu_sq == DEFAULT_MU_SQ
    if (not use_mat_reference
            and (args.reference_modes < 3 or args.reference_modes % 2 == 0)):
        raise ValueError(
            "--reference-modes must be an odd integer >= 3; "
            f"got {args.reference_modes}"
        )
    if not use_mat_reference and args.reference_substeps < 1:
        raise ValueError(
            "--reference-substeps must be >= 1; "
            f"got {args.reference_substeps}"
        )

    # Choose the device after resolving precision; MPS has no float64 support.
    if torch.get_default_dtype() == torch.float64:
        device = torch.device("cpu")
    else:
        device = pick_device()
    torch.manual_seed(args.seed)
    model = build_model(args.arch, args.hidden, args.depth, args.embedding, args.embed_dim, args.embed_scale).to(device)
    residual_fn = lambda model, batch: stacked_residuals(model, batch, args.mu_sq, args.weights)
    if args.optimizer in ("soap+ssbroyden", "engdw"):
        opt, opt_cfg, scheduler = pinn.build_extra_optimizer(
            model, args, residual_fn, precondition_frequency=20)
    else:
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
    reference_cache_path = (
        cached_reference_path("kdv.mat", LEGACY_REFERENCE_PATHS)
        if use_mat_reference
        else kdv_reference_cache_path(
            args.mu_sq,
            args.reference_modes,
            DEFAULT_REFERENCE_TIME_STEPS,
            args.reference_substeps,
        )
    )
    reference_was_cached = os.path.isfile(reference_cache_path)

    hyperparameters = {
        **pinn.metadata(args),
        # Non-zero means a sibling {run_id}.diag.jsonl exists.
        "diagnostics_every": args.diagnostics_every,
        "optimizer": args.optimizer,
        "weights": list(args.weights),
        "partition_residuals": args.optimizer == "gnome" and args.partition_residuals,
        "arch": args.arch,
        "embed": args.embedding,
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
        "eta": ETA,
        "mu_sq": args.mu_sq,
        "reference_method": (
            "kdv.mat" if use_mat_reference else "Fourier ETDRK4"
        ),
        "reference_path": reference_cache_path,
        "reference_modes": (
            DEFAULT_REFERENCE_MODES
            if use_mat_reference
            else args.reference_modes
        ),
        "reference_time_steps": DEFAULT_REFERENCE_TIME_STEPS,
        "reference_substeps_per_output": (
            None if use_mat_reference else args.reference_substeps
        ),
        "device": str(device),
        "dtype": str(torch.get_default_dtype()),
        "float64_after": args.float64_after,
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
            f"[{EXPERIMENT}] {args.optimizer} | arch={args.arch} "
            f"{args.depth}x{args.hidden} | embed={args.embedding} | params={n_params:,} | "
            f"device={device}\n"
            f"  N_pde={args.n_pde} N_ic={args.n_ic} N_bc={args.n_bc} | "
            f"aux={n_pde_aux}/{n_ic_aux}/{n_bc_aux} | steps={args.steps}\n"
            f"  mu_sq={args.mu_sq:.9g}",
            flush=True,
        )
        if use_mat_reference:
            print(
                f"  loading reference solution from {reference_cache_path}...",
                flush=True,
            )
        elif reference_was_cached:
            print(
                f"  loading cached Fourier reference from "
                f"{reference_cache_path}...",
                flush=True,
            )
        else:
            print(
                "  generating matching Fourier reference solution...",
                flush=True,
            )
    reference_started = time.perf_counter()
    if use_mat_reference:
        t_ref, x_ref, u_ref = load_kdv_reference()
    else:
        t_ref, x_ref, u_ref = kdv_reference(
            args.mu_sq,
            modes=args.reference_modes,
            substeps_per_output=args.reference_substeps,
            cache_path=reference_cache_path,
        )
    if not args.quiet:
        print(
            f"  reference {u_ref.shape[0]}x{u_ref.shape[1]} "
            f"{'loaded' if reference_was_cached else 'generated'} in "
            f"{time.perf_counter() - reference_started:.2f}s",
            flush=True,
        )
    # Keep the original reference precision for the float64 finishing stage.
    # Evaluation casts input coordinates to the current model dtype.

    if diag is not None and not args.quiet:
        print(f"  diagnostics every {args.diagnostics_every} steps "
              f"→ {diag.path}", flush=True)
    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    last_pde = last_ic = last_bc = float("nan")
    best_avg = best_rel_l2 = float("inf")

    sampler = pinn.BatchSampler(args, lambda: sample_batch(args.n_pde, args.n_ic, args.n_bc, device))
    for step in range(args.steps):
        quasi_newton = hybrid and step >= args.soap_steps
        if hybrid and step == args.soap_steps:
            opt = pinn.build_ssbroyden_phase(model, args)
            scheduler = None
            run.log("optimizer", step=step, optimizer="ssbroyden")
        if step == args.float64_after and step > 0 and next(model.parameters()).dtype != torch.float64:
            previous_device = device
            device = switch_to_float64(model, opt)
            sampler.to(device, torch.float64)
            run.log("precision", step=step, from_dtype="torch.float32",
                    dtype="torch.float64", from_device=str(previous_device),
                    device=str(device),
                    lr=current_lr(opt))
            if not args.quiet:
                print(f"[{EXPERIMENT}] after {step} steps: float32 → float64 "
                      f"| device={previous_device} → {device}", flush=True)
        main_batch = sampler.batch(step)
        if args.optimizer == "gnome":
            aux_batch = (pinn.subsample_txc(main_batch, n_pde_aux, n_ic_aux, n_bc_aux)
                         if sampler.fixed_batch is not None else sample_batch(
                             n_pde_aux, n_ic_aux, n_bc_aux, device))
            # if step % 1000 == 0:
            #     args.weights[0] = min(1.0, 1.2 * args.weights[0])
            def main_closure():
                r = stacked_residuals(model, main_batch, args.mu_sq, args.weights)
                return r, torch.zeros_like(r)

            def aux_closure():
                r = stacked_residuals(model, aux_batch, args.mu_sq, args.weights)
                return r, torch.zeros_like(r)

            loss = opt.step(
                main_closure, aux_closure,
                surrogate_partition_sizes=(
                    residual_partition_sizes(aux_batch) if args.partition_residuals else None
                ),
            )

        elif args.optimizer == "engdw":
            loss = opt.step(main_batch)
        elif quasi_newton:
            loss = pinn.refinement_step(opt, model, main_batch, residual_fn)
        else:
            opt.zero_grad()
            loss = residual_fn(model, main_batch).square().mean()
            loss.backward()
            opt.step()

        if (diag is not None and args.optimizer == "gnome"
                and (step + 1) % args.diagnostics_every == 0):
            for record in opt.get_diagnostics(params=diag.params):
                diag(record)

        if scheduler is not None:
            scheduler.step()

        if loss is not None:
            loss_val = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
        if diverged(loss_val):
            if diag is not None:
                diag.close()
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.",
                  flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val, lr=current_lr(opt), **pinn.phase_metrics(args, step))
        window.append(loss_val)

        if ((args.log_every and (step + 1) % args.log_every == 0)
                or step + 1 == args.steps or (hybrid and step + 1 == args.soap_steps)):
            tl = term_losses(
                model,
                sample_batch(args.n_pde, args.n_ic, args.n_bc, device),
                args.mu_sq,
            )
            rl2 = eval_rel_l2(model, t_ref, x_ref, u_ref, device)
            last_avg = sum(window) / len(window)
            last_pde, last_ic, last_bc = tl["pde"], tl["ic"], tl["bc"]
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            cur_lr = current_lr(opt)
            run.log_val(step + 1, loss=last_avg, lr=cur_lr,
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


    if diag is not None:
        diag.close()
        print(f"[{EXPERIMENT}] diagnostics → {diag.path}")
    path = run.finish(
        completed=True,
        final_dtype=str(next(model.parameters()).dtype),
        final_device=str(device),
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
