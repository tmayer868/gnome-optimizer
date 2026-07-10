"""2D Poisson PINN in JAX: AdamW vs SOAP vs Gnome (gnome_jax).

The JAX twin of ``experiments/poisson_pinn.py`` — same PDE, model,
sampling, optimizer configs, and logging schema — used to validate the
``gnome_jax`` port (jax_port_plan.md Phase 6). Logged under its own
experiment name (``poisson_pinn_jax``) so JAX runs never mix with the
PyTorch runs in ``runs/poisson_pinn/``.

PDE:  -Δu = f(x, y),    (x, y) ∈ (0, 1)²
BC:   u = 0  on  ∂Ω    (Dirichlet)

Manufactured solution::

    u_exact(x, y) = sin(πx) sin(πy)
    f(x, y) = -Δu_exact = 2π² sin(πx) sin(πy)

The whole training step — point sampling, the two Gnome closures with
their nested ``jax.grad`` PDE derivatives, and the optimizer update —
runs under a single ``jax.jit`` (the port's acceptance criterion).

Baselines: the vendored SOAP_JAX (``experiments/baselines/soap_jax.py``)
and ``optax.adamw``, both with the standard linear-warmup + cosine-decay
schedule (``--cosine-decay`` = final-lr fraction; 1.0 disables). Gnome
runs at a fixed lr — its Gauss-Newton step self-anneals.

Usage:

    uv run -m experiments.poisson_pinn_jax --optimizer gnome --seed 0
    uv run -m experiments.poisson_pinn_jax --optimizer soap  --seed 0
    uv run -m experiments.poisson_pinn_jax --optimizer adamw --seed 0
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp
import optax

import gnome_jax
from experiments.baselines.soap_jax import soap
from experiments.common import DIVERGED_EXIT, RunLogger, diverged


EXPERIMENT = "poisson_pinn_jax"

X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
PI = math.pi
SOURCE_COEFF = 2.0 * PI * PI


# ========================= Model =========================

def init_mlp(key: jax.Array, hidden: int, depth: int) -> list:
    """Plain tanh MLP ``(x, y) → u`` as a list of ``(W, b)`` layers.

    Init matches ``torch.nn.Linear``'s default: both W and b uniform in
    ``±1/sqrt(fan_in)``.
    """
    assert depth >= 2
    sizes = [2] + [hidden] * (depth - 1) + [1]
    layers = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        key, kw, kb = jax.random.split(key, 3)
        bound = 1.0 / math.sqrt(fan_in)
        layers.append((
            jax.random.uniform(kw, (fan_in, fan_out), minval=-bound, maxval=bound),
            jax.random.uniform(kb, (fan_out,), minval=-bound, maxval=bound),
        ))
    return layers


def forward(params: list, xy: jax.Array) -> jax.Array:
    """Batched forward: ``xy`` of shape (N, 2) → u of shape (N, 1)."""
    h = xy
    for w, b in params[:-1]:
        h = jnp.tanh(h @ w + b)
    w, b = params[-1]
    return h @ w + b


def _u_scalar(params: list, x: jax.Array, y: jax.Array) -> jax.Array:
    """Pointwise u(x, y) for the derivative operators."""
    return forward(params, jnp.stack([x, y])[None, :])[0, 0]


# ========================= Residuals =========================

def _source_term(x: jax.Array, y: jax.Array) -> jax.Array:
    """RHS of the Poisson equation: ``f(x, y) = 2π² sin(πx) sin(πy)``."""
    return SOURCE_COEFF * jnp.sin(PI * x) * jnp.sin(PI * y)


def pde_residual(params: list, x: jax.Array, y: jax.Array) -> jax.Array:
    """Poisson PDE residual ``Δu + f`` (target zero for ``-Δu = f``).

    Sign convention matches the PyTorch experiment: the Rademacher
    surrogate is sign-invariant, so ``u_xx + u_yy + f`` targets zero.
    """
    u_x = jax.grad(_u_scalar, argnums=1)
    u_xx = jax.grad(lambda p, a, b: u_x(p, a, b), argnums=1)
    u_y = jax.grad(_u_scalar, argnums=2)
    u_yy = jax.grad(lambda p, a, b: u_y(p, a, b), argnums=2)
    lap = jax.vmap(lambda a, b: u_xx(params, a, b) + u_yy(params, a, b))
    return lap(x, y) + _source_term(x, y)


def bc_residual(params: list, x: jax.Array, y: jax.Array) -> jax.Array:
    """Dirichlet BC residual: ``u(boundary) - 0 = u(boundary)``."""
    return forward(params, jnp.stack([x, y], axis=1))[:, 0]


# ========================= Sampling =========================

def sample_batch(key: jax.Array, n_pde: int, n_bc_per_edge: int):
    """Uniform interior draws + ``n_bc_per_edge`` points on each of 4 edges."""
    k_pde, k_bc = jax.random.split(key)
    xy = jax.random.uniform(k_pde, (n_pde, 2))
    x_pde = xy[:, 0] * (X_MAX - X_MIN) + X_MIN
    y_pde = xy[:, 1] * (Y_MAX - Y_MIN) + Y_MIN

    s = jax.random.uniform(k_bc, (n_bc_per_edge,))
    x_bc = jnp.concatenate([
        jnp.full_like(s, X_MIN),              # left
        jnp.full_like(s, X_MAX),              # right
        s * (X_MAX - X_MIN) + X_MIN,          # bottom
        s * (X_MAX - X_MIN) + X_MIN,          # top
    ])
    y_bc = jnp.concatenate([
        s * (Y_MAX - Y_MIN) + Y_MIN,          # left
        s * (Y_MAX - Y_MIN) + Y_MIN,          # right
        jnp.full_like(s, Y_MIN),              # bottom
        jnp.full_like(s, Y_MAX),              # top
    ])
    return x_pde, y_pde, x_bc, y_bc


def stacked_residuals(params: list, batch) -> jax.Array:
    """PDE + BC residuals stacked via ``stack_residuals`` (equal weights)."""
    x_pde, y_pde, x_bc, y_bc = batch
    return gnome_jax.stack_residuals([
        pde_residual(params, x_pde, y_pde),
        bc_residual(params, x_bc, y_bc),
    ])


# ========================= Reference + eval =========================

def poisson_reference(nx: int = 128, ny: int = 128):
    """Analytical reference ``u = sin(πx) sin(πy)`` on a uniform grid."""
    x = jnp.linspace(X_MIN, X_MAX, nx)
    y = jnp.linspace(Y_MIN, Y_MAX, ny)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    xy = jnp.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    u = jnp.sin(PI * xx) * jnp.sin(PI * yy)
    return xy, u


@jax.jit
def _predict_grid(params: list, xy: jax.Array) -> jax.Array:
    return forward(params, xy)[:, 0]


def eval_rel_l2(params: list, xy_ref: jax.Array, u_ref: jax.Array) -> float:
    u_pred = _predict_grid(params, xy_ref).reshape(u_ref.shape)
    num = jnp.sqrt(jnp.sum(jnp.square(u_pred - u_ref)))
    den = jnp.sqrt(jnp.sum(jnp.square(u_ref)))
    return float(num / den)


@jax.jit
def term_losses(params: list, batch):
    """Per-block MSE for diagnostic logging."""
    x_pde, y_pde, x_bc, y_bc = batch
    return (
        jnp.mean(jnp.square(pde_residual(params, x_pde, y_pde))),
        jnp.mean(jnp.square(bc_residual(params, x_bc, y_bc))),
    )


# ========================= CLI / training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adamw"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--n-pde", type=int, default=2000)
    p.add_argument("--n-bc-per-edge", type=int, default=50)
    p.add_argument("--aux-frac", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature-damping epsilon (Gnome only; "
                        "SOAP/AdamW keep their fixed eps=1e-8).")
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=1e-8)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--cosine-decay", type=float, default=0.0,
                   help="Final-LR fraction for the baseline cosine decay: "
                        "0.0 decays to zero, 1.0 disables. Gnome never "
                        "decays regardless.")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    key = jax.random.PRNGKey(args.seed)
    key, k_model = jax.random.split(key)
    params = init_mlp(k_model, args.hidden, args.depth)

    n_pde_aux = max(1, int(args.n_pde * args.aux_frac))
    n_bc_aux_per_edge = max(1, int(args.n_bc_per_edge * args.aux_frac))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    schedule = None
    if args.optimizer == "gnome":
        opt_cfg = dict(
            lr=args.lr, weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2), shampoo_beta=args.beta2,
            eps=args.eps, precondition_frequency=10,
            clip=1.0, warmup=args.warmup_steps,
            loss="mse", precondition_1d=True,
        )
        opt = gnome_jax.gnome(
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=10,
            clip=1.0,
            warmup=args.warmup_steps,
            precondition_1d=True,
        )
        opt_state = opt.init(params)

        @jax.jit
        def train_step(params, opt_state, key):
            key, k_main, k_aux = jax.random.split(key, 3)
            main_batch = sample_batch(k_main, args.n_pde, args.n_bc_per_edge)
            aux_batch = sample_batch(k_aux, n_pde_aux, n_bc_aux_per_edge)

            def main_fn(p):
                r = stacked_residuals(p, main_batch)
                return r, jnp.zeros_like(r)

            def aux_fn(p):
                r = stacked_residuals(p, aux_batch)
                return r, jnp.zeros_like(r)

            return opt.step(params, opt_state, key, main_fn, aux_fn)

    else:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr,
            warmup_steps=args.warmup_steps, decay_steps=args.steps,
            end_value=args.lr * args.cosine_decay,
        )
        if args.optimizer == "soap":
            opt_cfg = dict(
                lr=args.lr, weight_decay=args.weight_decay,
                betas=(args.beta1, args.beta2), shampoo_beta=args.beta2,
                eps=1e-8, precondition_frequency=10, precondition_1d=True,
            )
            opt = soap(
                learning_rate=schedule,
                b1=args.beta1, b2=args.beta2, shampoo_beta=args.beta2,
                eps=1e-8, weight_decay=args.weight_decay,
                precondition_frequency=10, precondition_1d=True,
            )
        else:  # adamw
            opt_cfg = dict(
                lr=args.lr, weight_decay=args.weight_decay,
                betas=(0.9, 0.999), eps=1e-8,
            )
            opt = optax.adamw(
                learning_rate=schedule,
                b1=0.9, b2=0.999, eps=1e-8,
                weight_decay=args.weight_decay,
            )
        opt_cfg["warmup"] = args.warmup_steps
        opt_cfg["cosine_decay_floor"] = args.cosine_decay
        opt_state = opt.init(params)

        @jax.jit
        def train_step(params, opt_state, key):
            key, k_main = jax.random.split(key)
            batch = sample_batch(k_main, args.n_pde, args.n_bc_per_edge)

            def loss_fn(p):
                r = stacked_residuals(p, batch)
                return jnp.sum(jnp.square(r)) / r.shape[0]

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state_new = opt.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, loss, opt_state_new, key

    hyperparameters = {
        "optimizer": args.optimizer,
        "framework": "jax",
        "jax_version": jax.__version__,
        "steps": args.steps,
        "hidden": args.hidden,
        "depth": args.depth,
        "n_params": n_params,
        "n_pde": args.n_pde,
        "n_bc_per_edge": args.n_bc_per_edge,
        "n_pde_aux": n_pde_aux,
        "n_bc_aux_per_edge": n_bc_aux_per_edge,
        "device": str(jax.devices()[0]),
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
            f"device={jax.devices()[0]}\n"
            f"  N_pde={args.n_pde} N_bc_per_edge={args.n_bc_per_edge} | "
            f"aux={n_pde_aux}/{n_bc_aux_per_edge} | steps={args.steps}",
            flush=True,
        )
    xy_ref, u_ref = poisson_reference()

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        params, loss, opt_state, key = train_step(params, opt_state, key)

        loss_val = float(loss)
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.",
                  flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if args.log_every and (step + 1) % args.log_every == 0:
            key, k_diag = jax.random.split(key)
            pde_mse, bc_mse = term_losses(
                params, sample_batch(k_diag, args.n_pde, args.n_bc_per_edge)
            )
            rl2 = eval_rel_l2(params, xy_ref, u_ref)
            last_avg = sum(window) / len(window)
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            lr_now = (
                args.lr if schedule is None else float(schedule(step + 1))
            )
            run.log_val(step + 1, loss=last_avg, lr=lr_now,
                        pde=float(pde_mse), bc=float(bc_mse), rel_l2=rl2)
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  "
                    f"pde={float(pde_mse):.3e}  bc={float(bc_mse):.3e}  "
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
    train(parse_args())


if __name__ == "__main__":
    main()
