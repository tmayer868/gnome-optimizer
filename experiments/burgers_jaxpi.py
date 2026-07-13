"""Burgers under the jaxpi protocol: Adam vs SOAP vs Gnome (gnome_jax).

Replicates the training setup of Wang et al.'s jaxpi Burgers benchmark
(``examples/burgers``, ``pirate`` branch, ``configs/sota.py``) — the
codebase behind "Gradient Alignment in Physics-Informed Neural Networks"
(arXiv:2502.00604) — with Gnome added as a third optimizer. PirateNet is
deliberately out of scope for now; this uses their plain-arch protocol.

Protocol parity with their sota.py config:

* **Architecture** — ModifiedMlp 4x256 with RWF (mean 1.0, stddev 0.1), no
  Fourier features, no periodic embedding: see
  ``experiments/common/pinn_arch_jax.py`` (pure-JAX reimplementation of
  the published methods — jaxpi's own code is under a no-redistribution
  Penn license and is NOT copied).
* **Losses** — ics (u(0,x) from burgers.mat), bcs (u at both spatial
  boundaries over the time grid), res (collocation residual
  ``u_t + u·u_x − (0.01/π)·u_xx``) with **causal weighting** (sorted-t
  chunks, ``w = exp(−tol · M @ l)``) and **grad-norm loss balancing**
  (running average, momentum 0.9, updated every 1000 steps).
* **Baselines** — Adam and SOAP (the vendored SOAP_JAX package — the same
  implementation jaxpi itself imports), with their schedule: linear warmup
  (5000) joined to exponential decay (rate 0.9 per 1000 steps).
  ``--schedule-free`` additionally wraps the baseline in
  ``optax.contrib.schedule_free`` + global-norm clip 1.0, matching their
  ``pirate_soap.py`` optimizer treatment. SOAP uses
  ``precondition_frequency=2`` exactly as jaxpi hardcodes it.
* **Gnome** — fixed lr (no schedule; its Gauss-Newton step self-anneals),
  the weighted multi-block loss mapped exactly onto ``stack_residuals``:
  ics block (λ=w_ics), two bc blocks (λ=w_bcs each — their bcs loss is a
  sum of two means), and the causal res block with each point scaled by
  ``sqrt(w_chunk)`` (λ=w_res), so ``mean(stacked²)`` equals their weighted
  loss identically. Grad-norm weight updates use the same machinery as the
  baselines.

Data: jaxpi's ``burgers.mat`` (auto-downloaded to ``experiments/data/``,
same file ``burgers_pinn.py`` uses for its ``rel_l2_jaxpi`` metric).
Headline metric: ``rel_l2`` on their full 201x512 space-time grid. Final
params are pickled next to the run's JSONL for post-hoc evaluation.

Usage:

    uv run -m experiments.burgers_jaxpi --optimizer gnome --seed 0
    uv run -m experiments.burgers_jaxpi --optimizer soap  --seed 0
    uv run -m experiments.burgers_jaxpi --optimizer adam  --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import urllib.request

import jax

# On Ampere+ GPUs JAX defaults f32 matmuls to TF32 (~3 decimal digits),
# which degrades second-derivative PDE residuals and the curvature
# machinery. Force true float32; no-op on CPU.
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import optax
from jax import grad, vmap
from jax.tree_util import tree_leaves, tree_map

import gnome_jax
from experiments.baselines.soap_jax import soap
from experiments.common import DIVERGED_EXIT, RunLogger, diverged
from experiments.common.pinn_arch_jax import make_modified_mlp


EXPERIMENT = "burgers_jaxpi"

DATA_URL = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/burgers/data/burgers.mat"
)
DATA_CACHE = "experiments/data/burgers.mat"

NU_COEFF = 0.01 / math.pi  # viscosity in their r_net


# ========================= Data =========================

def load_dataset():
    """jaxpi's burgers.mat: usol (nt, nx), t (nt,), x (nx,)."""
    import scipy.io

    if not os.path.exists(DATA_CACHE):
        os.makedirs(os.path.dirname(DATA_CACHE), exist_ok=True)
        print(f"[{EXPERIMENT}] downloading {DATA_URL} ...", flush=True)
        urllib.request.urlretrieve(DATA_URL, DATA_CACHE)
    data = scipy.io.loadmat(DATA_CACHE)
    u_ref = jnp.asarray(data["usol"])
    t_star = jnp.asarray(data["t"].flatten())
    x_star = jnp.asarray(data["x"].flatten())
    return u_ref, t_star, x_star


def save_params(params, jsonl_path: str) -> str:
    """Pickle the (device_get) params pytree next to the run's JSONL."""
    out = os.path.splitext(jsonl_path)[0] + ".params.pkl"
    with open(out, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    return out


# ========================= Problem definition =========================

class BurgersProblem:
    """Holds the grids and implements jaxpi's losses / weighting / eval."""

    def __init__(self, u_ref, t_star, x_star, causal_tol, num_chunks,
                 apply_fn):
        self.u_ref = u_ref
        self.t_star = t_star
        self.x_star = x_star
        self.t0 = t_star[0]
        self.u0 = u_ref[0, :]
        self.tol = causal_tol
        self.num_chunks = num_chunks
        # Causal accumulation matrix: chunk i is discounted by the summed
        # residual mass of all earlier-time chunks.
        self.M = jnp.triu(jnp.ones((num_chunks, num_chunks)), k=1).T
        self.dom = jnp.array(
            [[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]]
        )
        self._apply = apply_fn

    # ----- network + residuals -----

    def u_net(self, params, t, x):
        return self._apply(params, jnp.stack([t, x]))[0]

    def r_net(self, params, t, x):
        """Burgers residual, matching jaxpi: u_t + u·u_x − (0.01/π)·u_xx."""
        u = self.u_net(params, t, x)
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_x = grad(self.u_net, argnums=2)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return u_t + u * u_x - NU_COEFF * u_xx

    # ----- residual blocks -----

    def ics_residual(self, params, x_idx):
        u_pred = vmap(self.u_net, (None, None, 0))(
            params, self.t0, self.x_star[x_idx]
        )
        return self.u0[x_idx] - u_pred

    def bc_residuals(self, params, t_idx):
        bc1 = vmap(self.u_net, (None, 0, None))(
            params, self.t_star[t_idx], self.x_star[0]
        )
        bc2 = vmap(self.u_net, (None, 0, None))(
            params, self.t_star[t_idx], self.x_star[-1]
        )
        return bc1, bc2

    def causal_res(self, params, batch):
        """Sorted-t residuals scaled per point by sqrt(w_chunk), so that
        mean(out²) == their causal res loss mean(l·w)."""
        t_sorted = batch[:, 0].sort()
        r = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
        rc = r.reshape(self.num_chunks, -1)
        l = jnp.mean(rc**2, axis=1)
        w = jax.lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return (rc * jnp.sqrt(w)[:, None]).reshape(-1)

    # ----- jaxpi's loss dict (used by grad-norm weighting + baselines) -----

    def losses(self, params, batch):
        ics_r = self.ics_residual(params, jnp.arange(self.x_star.shape[0]))
        bc1, bc2 = self.bc_residuals(params, jnp.arange(self.t_star.shape[0]))
        res_scaled = self.causal_res(params, batch)
        return {
            "ics": jnp.mean(ics_r**2),
            "bcs": jnp.mean(bc1**2) + jnp.mean(bc2**2),
            "res": jnp.mean(res_scaled**2),
        }

    def weighted_loss(self, params, weights, batch):
        losses = self.losses(params, batch)
        return sum(weights[k] * losses[k] for k in losses)

    def compute_grad_norm_weights(self, params, batch):
        """jaxpi's grad_norm scheme: w_k = mean_norm / (norm_k + eps·mean)."""
        norms = {}
        for k in ("ics", "bcs", "res"):
            g = grad(lambda p: self.losses(p, batch)[k])(params)
            flat = jnp.concatenate([x.reshape(-1) for x in tree_leaves(g)])
            norms[k] = jnp.linalg.norm(flat)
        mean_norm = jnp.mean(jnp.stack(list(norms.values())))
        return {
            k: mean_norm / (n + 1e-5 * mean_norm) for k, n in norms.items()
        }

    # ----- Gnome's stacked view of the same weighted loss -----

    def stacked_residuals(self, params, weights, batch, x_idx, t_idx):
        ics_r = self.ics_residual(params, x_idx)
        bc1, bc2 = self.bc_residuals(params, t_idx)
        res_scaled = self.causal_res(params, batch)
        return gnome_jax.stack_residuals(
            [ics_r, bc1, bc2, res_scaled],
            [weights["ics"], weights["bcs"], weights["bcs"], weights["res"]],
        )

    # ----- eval -----

    def rel_l2(self, params):
        u_pred = vmap(
            vmap(self.u_net, (None, None, 0)), (None, 0, None)
        )(params, self.t_star, self.x_star)
        return jnp.linalg.norm(u_pred - self.u_ref) / jnp.linalg.norm(
            self.u_ref
        )


def sample_res_batch(key, dom, n):
    """Uniform collocation draws over the (t, x) domain, like jaxpi's
    UniformSampler."""
    return jax.random.uniform(
        key, (n, 2), minval=dom[:, 0], maxval=dom[:, 1]
    )


# ========================= CLI / training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adam"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100000,
                   help="jaxpi sota budget.")
    p.add_argument("--batch-size", type=int, default=8192,
                   help="Res collocation batch (jaxpi sota: 8192). Must be "
                        "divisible by --num-chunks.")
    p.add_argument("--aux-batch-size", type=int, default=256,
                   help="Gnome aux res batch. Must be divisible by "
                        "--num-chunks.")
    p.add_argument("--aux-ics", type=int, default=64,
                   help="Fresh random ics-grid points resampled per aux "
                        "step for Gnome's surrogate.")
    p.add_argument("--aux-bcs", type=int, default=32,
                   help="Fresh random bc t-grid points resampled per aux "
                        "step for Gnome's surrogate.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999,
                   help="Baseline beta2 (jaxpi sota). Gnome uses "
                        "--gnome-beta2.")
    p.add_argument("--gnome-beta2", type=float, default=0.99,
                   help="Gnome second-moment / shampoo EMA.")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature damping (baselines keep 1e-8).")
    p.add_argument("--warmup-steps", type=int, default=5000,
                   help="Baseline linear warmup (jaxpi sota). Gnome uses "
                        "--gnome-warmup.")
    p.add_argument("--gnome-warmup", type=int, default=200)
    p.add_argument("--decay-rate", type=float, default=0.9)
    p.add_argument("--decay-steps", type=int, default=1000)
    p.add_argument("--schedule-free", action="store_true",
                   help="Wrap the baseline in optax.contrib.schedule_free + "
                        "global-norm clip 1.0 (jaxpi pirate_soap treatment).")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--rwf-mean", type=float, default=1.0)
    p.add_argument("--rwf-stddev", type=float, default=0.1)
    p.add_argument("--causal-tol", type=float, default=1.0)
    p.add_argument("--num-chunks", type=int, default=32)
    p.add_argument("--weight-update-every", type=int, default=1000)
    p.add_argument("--weight-momentum", type=float, default=0.9)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def train(args: argparse.Namespace) -> str:
    if args.batch_size % args.num_chunks:
        raise SystemExit("--batch-size must be divisible by --num-chunks")
    if args.aux_batch_size % args.num_chunks:
        raise SystemExit("--aux-batch-size must be divisible by --num-chunks")

    u_ref, t_star, x_star = load_dataset()

    init_fn, apply_fn = make_modified_mlp(
        in_dim=2, hidden=args.hidden, out_dim=1,
        num_layers=args.num_layers,
        rwf_mean=args.rwf_mean, rwf_stddev=args.rwf_stddev,
    )
    prob = BurgersProblem(
        u_ref, t_star, x_star, args.causal_tol, args.num_chunks, apply_fn
    )

    key = jax.random.PRNGKey(args.seed)
    key, k_model = jax.random.split(key)
    params = init_fn(k_model)
    n_params = sum(p.size for p in tree_leaves(params))

    weights = {"ics": jnp.asarray(1.0), "bcs": jnp.asarray(1.0),
               "res": jnp.asarray(1.0)}

    # ----- optimizers -----
    schedule = None
    if args.optimizer == "gnome":
        opt_cfg = dict(
            lr=args.lr, weight_decay=0.0,
            betas=(args.beta1, args.gnome_beta2),
            shampoo_beta=args.gnome_beta2, eps=args.eps,
            precondition_frequency=10, clip=1.0, warmup=args.gnome_warmup,
            loss="mse", precondition_1d=True,
        )
        opt = gnome_jax.gnome(
            lr=args.lr,
            betas=(args.beta1, args.gnome_beta2),
            shampoo_beta=args.gnome_beta2,
            eps=args.eps,
            weight_decay=0.0,
            precondition_frequency=10,
            clip=1.0,
            warmup=args.gnome_warmup,
            precondition_1d=True,
        )
        opt_state = opt.init(params)

        x_all = jnp.arange(x_star.shape[0])
        t_all = jnp.arange(t_star.shape[0])

        @jax.jit
        def train_step(params, opt_state, key, weights):
            key, k_main, k_aux, k_ic, k_bc = jax.random.split(key, 5)
            batch = sample_res_batch(k_main, prob.dom, args.batch_size)
            aux_batch = sample_res_batch(k_aux, prob.dom,
                                         args.aux_batch_size)
            # Fresh random ics/bc grid subsets every step (the reference is
            # gridded, so indices rather than coordinates), not a fixed
            # stride — coverage accumulates through the curvature EMA.
            x_aux = jax.random.permutation(k_ic, x_star.shape[0])[
                :args.aux_ics]
            t_aux = jax.random.permutation(k_bc, t_star.shape[0])[
                :args.aux_bcs]

            def main_fn(p):
                r = prob.stacked_residuals(p, weights, batch, x_all, t_all)
                return r, jnp.zeros_like(r)

            def aux_fn(p):
                r = prob.stacked_residuals(p, weights, aux_batch,
                                           x_aux, t_aux)
                return r, jnp.zeros_like(r)

            new_params, loss, new_state, key = opt.step(
                params, opt_state, key, main_fn, aux_fn
            )
            return new_params, loss, new_state, key

    else:
        # jaxpi's baseline schedule: linear warmup joined to exponential
        # decay (their _create_optimizer).
        exp_decay = optax.exponential_decay(
            init_value=args.lr,
            transition_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=False,
        )
        if args.warmup_steps > 0:
            warmup = optax.linear_schedule(
                init_value=0.0, end_value=args.lr,
                transition_steps=args.warmup_steps,
            )
            schedule = optax.join_schedules(
                [warmup, exp_decay], [args.warmup_steps]
            )
        else:
            schedule = exp_decay

        if args.optimizer == "soap":
            opt_cfg = dict(
                lr=args.lr, weight_decay=0.0,
                betas=(args.beta1, args.beta2), eps=1e-8,
                precondition_frequency=2,  # jaxpi hardcodes 2 for SOAP
                precondition_1d=False,
            )
            tx = soap(
                learning_rate=schedule,
                b1=args.beta1, b2=args.beta2,
                weight_decay=0.0, precondition_frequency=2,
            )
        else:  # adam
            opt_cfg = dict(
                lr=args.lr, weight_decay=0.0,
                betas=(args.beta1, args.beta2), eps=1e-8,
            )
            tx = optax.adam(
                learning_rate=schedule,
                b1=args.beta1, b2=args.beta2, eps=1e-8,
            )
        if args.schedule_free:
            tx = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.contrib.schedule_free(tx, schedule, b1=args.beta1),
            )
        opt_cfg["warmup"] = args.warmup_steps
        opt_cfg["decay_rate"] = args.decay_rate
        opt_cfg["decay_steps"] = args.decay_steps
        opt_cfg["schedule_free"] = args.schedule_free
        opt_state = tx.init(params)

        @jax.jit
        def train_step(params, opt_state, key, weights):
            key, k_main = jax.random.split(key)
            batch = sample_res_batch(k_main, prob.dom, args.batch_size)
            loss, grads = jax.value_and_grad(prob.weighted_loss)(
                params, weights, batch
            )
            updates, new_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, loss, new_state, key

    update_weights = jax.jit(
        lambda params, batch: prob.compute_grad_norm_weights(params, batch)
    )
    eval_rel_l2 = jax.jit(prob.rel_l2)

    hyperparameters = {
        "optimizer": args.optimizer,
        "framework": "jax",
        "jax_version": jax.__version__,
        "protocol": "jaxpi sota.py (pirate branch)",
        "steps": args.steps,
        "arch": "modified_mlp",
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "rwf_mean": args.rwf_mean,
        "rwf_stddev": args.rwf_stddev,
        "n_params": n_params,
        "batch_size": args.batch_size,
        "aux_batch_size": args.aux_batch_size,
        "aux_ics": args.aux_ics,
        "aux_bcs": args.aux_bcs,
        "causal_tol": args.causal_tol,
        "num_chunks": args.num_chunks,
        "weighting": "grad_norm",
        "weight_update_every": args.weight_update_every,
        "weight_momentum": args.weight_momentum,
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
            f"[{EXPERIMENT}] {args.optimizer} | modified_mlp "
            f"{args.num_layers}x{args.hidden} (RWF) | params={n_params:,} | "
            f"device={jax.devices()[0]}\n"
            f"  batch={args.batch_size} aux={args.aux_batch_size} | "
            f"causal(tol={args.causal_tol}, chunks={args.num_chunks}) + "
            f"grad_norm | steps={args.steps}"
            + (" | schedule_free" if args.schedule_free else ""),
            flush=True,
        )

    t_start = time.perf_counter()
    window: list[float] = []
    last_avg = last_rel_l2 = float("nan")
    best_avg = best_rel_l2 = float("inf")

    for step in range(args.steps):
        params, loss, opt_state, key = train_step(
            params, opt_state, key, weights
        )

        loss_val = float(loss)
        if diverged(loss_val):
            run.finish(completed=False, diverged=True, diverged_step=step)
            print(f"[{EXPERIMENT}] diverged at step {step} — stopping.",
                  flush=True)
            raise SystemExit(DIVERGED_EXIT)
        run.log_train(step, loss=loss_val)
        window.append(loss_val)

        if step % args.weight_update_every == 0:
            key, k_w = jax.random.split(key)
            wbatch = sample_res_batch(k_w, prob.dom, args.batch_size)
            new_w = update_weights(params, wbatch)
            m = args.weight_momentum
            weights = tree_map(
                lambda old, new: old * m + (1 - m) * new, weights, new_w
            )

        if args.log_every and (step + 1) % args.log_every == 0:
            rl2 = float(eval_rel_l2(params))
            last_avg = sum(window) / len(window)
            last_rel_l2 = rl2
            best_avg = min(best_avg, last_avg)
            best_rel_l2 = min(best_rel_l2, rl2)
            lr_now = (
                args.lr if schedule is None else float(schedule(step + 1))
            )
            run.log_val(
                step + 1, loss=last_avg, lr=lr_now, rel_l2=rl2,
                w_ics=float(weights["ics"]), w_bcs=float(weights["bcs"]),
                w_res=float(weights["res"]),
            )
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  rel_l2={rl2:.3e}  "
                    f"w=({float(weights['ics']):.2f},"
                    f"{float(weights['bcs']):.2f},"
                    f"{float(weights['res']):.2f})  {ms_per:.1f} ms/step",
                    flush=True,
                )
            window.clear()

    path = run.finish(
        completed=True,
        final_avg_train=last_avg, best_avg_train=best_avg,
        final_rel_l2=last_rel_l2, best_rel_l2=best_rel_l2,
    )
    params_path = save_params(params, path)
    print(f"[{EXPERIMENT}] saved → {path}")
    print(f"  params → {params_path}")
    print(f"  final avg_train={last_avg:.4e}  best={best_avg:.4e}")
    print(f"  final rel_l2={last_rel_l2:.3e}  best rel_l2={best_rel_l2:.3e}")
    return path


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
