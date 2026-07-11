"""Allen-Cahn under the jaxpi protocol: Adam vs SOAP vs Gnome (gnome_jax).

Replicates the training setup of Wang et al.'s jaxpi Allen-Cahn benchmark
(``examples/allen_cahn``, ``pirate`` branch, ``configs/default.py`` — the
plain-arch protocol; PirateNet is out of scope for now) from the paper
"Gradient Alignment in Physics-Informed Neural Networks"
(arXiv:2502.00604), with Gnome added as a third optimizer.

PDE:  u_t − 0.0001·u_xx + 5u³ − 5u = 0,   (t, x) ∈ [0,1] × [−1,1]
IC:   u(0, x) = x² cos(πx);   periodic BCs (enforced exactly by the
periodic input embedding — there is no bcs loss block).

Protocol parity with their default.py config:

* **Architecture** — ModifiedMlp 4x256 with RWF (1.0/0.1), periodic
  embedding (period π on the x axis) and trainable Random Fourier features
  (scale 2.0, dim 256): ``experiments/common/pinn_arch_jax.py``, a pure-JAX
  reimplementation of the published methods (jaxpi's code is under a
  no-redistribution license and is not copied).
* **Losses** — ics + causal-weighted res (tol 1.0, 32 chunks), balanced by
  their **NTK weighting**: per-block mean of the per-point NTK diagonal
  ``||∇_θ f||²`` (res chunk-averaged and multiplied by the causal
  weights), ``w_k = mean_ntk / (ntk_k + 1e-5·mean_ntk)``, running average
  momentum 0.9, updated every 1000 steps.
* **Baselines** — Adam and SOAP (vendored SOAP_JAX, the same package jaxpi
  imports; ``precondition_frequency=2`` as jaxpi hardcodes), with their
  schedule: linear warmup (5000) joined to exponential decay (0.9 per
  5000). ``--schedule-free`` applies their pirate-config optimizer
  treatment.
* **Gnome** — fixed lr, the weighted two-block loss mapped exactly onto
  ``stack_residuals``: ics block (λ=w_ics) + causal res block with each
  point scaled by ``sqrt(w_chunk)`` (λ=w_res), so ``mean(stacked²)``
  equals their weighted loss identically.

Their budget is 300k steps (AC is stiff). Data: jaxpi's
``allen_cahn.mat`` (auto-downloaded to ``experiments/data/``). Headline
metric: ``rel_l2`` on their full space-time grid. Final params are pickled
next to the run's JSONL.

Usage:

    uv run -m experiments.allen_cahn_jaxpi --optimizer gnome --seed 0
    uv run -m experiments.allen_cahn_jaxpi --optimizer soap  --seed 0
    uv run -m experiments.allen_cahn_jaxpi --optimizer adam  --seed 0
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import urllib.request

import jax
import jax.numpy as jnp
import optax
from jax import grad, vmap
from jax.tree_util import tree_leaves, tree_map

import gnome_jax
from experiments.baselines.soap_jax import soap
from experiments.common import DIVERGED_EXIT, RunLogger, diverged
from experiments.common.pinn_arch_jax import make_modified_mlp


EXPERIMENT = "allen_cahn_jaxpi"

DATA_URL = (
    "https://raw.githubusercontent.com/PredictiveIntelligenceLab/jaxpi/"
    "pirate/examples/allen_cahn/data/allen_cahn.mat"
)
DATA_CACHE = "experiments/data/allen_cahn.mat"


# ========================= Data =========================

def load_dataset():
    """jaxpi's allen_cahn.mat: usol (nt, nx), t (nt,), x (nx,)."""
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


def _segment_size(n: int, target: int = 256) -> int:
    """Largest divisor of n that is <= target (for memory-chunked maps)."""
    for s in range(min(target, n), 0, -1):
        if n % s == 0:
            return s
    return 1


# ========================= Problem definition =========================

class AllenCahnProblem:
    """Grids + jaxpi's losses / NTK weighting / eval for Allen-Cahn."""

    def __init__(self, u_ref, t_star, x_star, causal_tol, num_chunks,
                 apply_fn):
        self.u_ref = u_ref
        self.t_star = t_star
        self.x_star = x_star
        self.t0 = t_star[0]
        self.u0 = u_ref[0, :]
        self.tol = causal_tol
        self.num_chunks = num_chunks
        self.M = jnp.triu(jnp.ones((num_chunks, num_chunks)), k=1).T
        self.dom = jnp.array(
            [[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]]
        )
        self._apply = apply_fn

    # ----- network + residuals -----

    def u_net(self, params, t, x):
        return self._apply(params, jnp.stack([t, x]))[0]

    def r_net(self, params, t, x):
        """AC residual, matching jaxpi: u_t + 5u³ − 5u − 0.0001·u_xx."""
        u = self.u_net(params, t, x)
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return u_t + 5.0 * u**3 - 5.0 * u - 0.0001 * u_xx

    # ----- residual blocks -----

    def ics_residual(self, params, x_idx):
        u_pred = vmap(self.u_net, (None, None, 0))(
            params, self.t0, self.x_star[x_idx]
        )
        return self.u0[x_idx] - u_pred

    def _causal_l_w(self, params, batch):
        t_sorted = batch[:, 0].sort()
        r = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
        rc = r.reshape(self.num_chunks, -1)
        l = jnp.mean(rc**2, axis=1)
        w = jax.lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return rc, l, w

    def causal_res(self, params, batch):
        """Sorted-t residuals scaled per point by sqrt(w_chunk):
        mean(out²) == their causal res loss mean(l·w)."""
        rc, _, w = self._causal_l_w(params, batch)
        return (rc * jnp.sqrt(w)[:, None]).reshape(-1)

    # ----- jaxpi's loss dict -----

    def losses(self, params, batch):
        ics_r = self.ics_residual(params, jnp.arange(self.x_star.shape[0]))
        res_scaled = self.causal_res(params, batch)
        return {
            "ics": jnp.mean(ics_r**2),
            "res": jnp.mean(res_scaled**2),
        }

    def weighted_loss(self, params, weights, batch):
        losses = self.losses(params, batch)
        return sum(weights[k] * losses[k] for k in losses)

    # ----- NTK weighting (their scheme="ntk") -----

    def _pointwise_ntk(self, fn, params, ts, xs):
        """Per-point NTK diagonal ||∇_θ fn(params, t, x)||², computed in
        memory-bounded segments via lax.map (the full per-point jacobian
        over ~400k params would not fit at batch 8192)."""
        n = ts.shape[0]
        seg = _segment_size(n)

        def per_point(t, x):
            g = grad(fn, argnums=0)(params, t, x)
            return sum(jnp.vdot(v, v).real for v in tree_leaves(g))

        def one_segment(pair):
            t_s, x_s = pair
            return vmap(per_point)(t_s, x_s)

        out = jax.lax.map(
            one_segment, (ts.reshape(-1, seg), xs.reshape(-1, seg))
        )
        return out.reshape(n)

    def compute_ntk_weights(self, params, batch):
        """jaxpi's NTK balancing: per-block mean NTK diagonal, with the
        res block chunk-averaged and multiplied by the causal weights;
        w_k = mean_ntk / (ntk_k + 1e-5·mean_ntk)."""
        ics_ntk = self._pointwise_ntk(
            self.u_net, params,
            jnp.full_like(self.x_star, self.t0), self.x_star,
        )

        t_sorted = batch[:, 0].sort()
        res_ntk = self._pointwise_ntk(
            self.r_net, params, t_sorted, batch[:, 1]
        )
        res_ntk = res_ntk.reshape(self.num_chunks, -1).mean(axis=1)
        _, _, causal_w = self._causal_l_w(params, batch)
        res_ntk = res_ntk * causal_w

        ntk = {"ics": jnp.mean(ics_ntk), "res": jnp.mean(res_ntk)}
        mean_ntk = jnp.mean(jnp.stack(list(ntk.values())))
        return {
            k: mean_ntk / (v + 1e-5 * mean_ntk) for k, v in ntk.items()
        }

    # ----- Gnome's stacked view of the same weighted loss -----

    def stacked_residuals(self, params, weights, batch, x_idx):
        ics_r = self.ics_residual(params, x_idx)
        res_scaled = self.causal_res(params, batch)
        return gnome_jax.stack_residuals(
            [ics_r, res_scaled],
            [weights["ics"], weights["res"]],
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
    return jax.random.uniform(
        key, (n, 2), minval=dom[:, 0], maxval=dom[:, 1]
    )


# ========================= CLI / training =========================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", required=True,
                   choices=["gnome", "soap", "adam"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300000,
                   help="jaxpi's Allen-Cahn budget.")
    p.add_argument("--batch-size", type=int, default=8192,
                   help="Res collocation batch (jaxpi: 8192). Must be "
                        "divisible by --num-chunks.")
    p.add_argument("--aux-batch-size", type=int, default=256,
                   help="Gnome aux res batch. Must be divisible by "
                        "--num-chunks.")
    p.add_argument("--aux-stride", type=int, default=8,
                   help="Stride subsampling the ics grid for Gnome's aux "
                        "closure.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999,
                   help="Baseline beta2 (jaxpi). Gnome uses --gnome-beta2.")
    p.add_argument("--gnome-beta2", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Gnome curvature damping (baselines keep 1e-8).")
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--gnome-warmup", type=int, default=200)
    p.add_argument("--decay-rate", type=float, default=0.9)
    p.add_argument("--decay-steps", type=int, default=5000,
                   help="jaxpi's AC config decays 0.9 per 5000 (Burgers "
                        "used 1000).")
    p.add_argument("--schedule-free", action="store_true")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--rwf-mean", type=float, default=1.0)
    p.add_argument("--rwf-stddev", type=float, default=0.1)
    p.add_argument("--fourier-scale", type=float, default=2.0)
    p.add_argument("--fourier-dim", type=int, default=256)
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
        period=(math.pi,), period_axes=(1,),
        fourier_scale=args.fourier_scale, fourier_dim=args.fourier_dim,
    )
    prob = AllenCahnProblem(
        u_ref, t_star, x_star, args.causal_tol, args.num_chunks, apply_fn
    )

    key = jax.random.PRNGKey(args.seed)
    key, k_model = jax.random.split(key)
    params = init_fn(k_model)
    n_params = sum(p.size for p in tree_leaves(params))

    weights = {"ics": jnp.asarray(1.0), "res": jnp.asarray(1.0)}

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

        x_aux = jnp.arange(0, x_star.shape[0], args.aux_stride)
        x_all = jnp.arange(x_star.shape[0])

        @jax.jit
        def train_step(params, opt_state, key, weights):
            key, k_main, k_aux = jax.random.split(key, 3)
            batch = sample_res_batch(k_main, prob.dom, args.batch_size)
            aux_batch = sample_res_batch(k_aux, prob.dom,
                                         args.aux_batch_size)

            def main_fn(p):
                r = prob.stacked_residuals(p, weights, batch, x_all)
                return r, jnp.zeros_like(r)

            def aux_fn(p):
                r = prob.stacked_residuals(p, weights, aux_batch, x_aux)
                return r, jnp.zeros_like(r)

            new_params, loss, new_state, key = opt.step(
                params, opt_state, key, main_fn, aux_fn
            )
            return new_params, loss, new_state, key

    else:
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
                precondition_frequency=2,
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
        lambda params, batch: prob.compute_ntk_weights(params, batch)
    )
    eval_rel_l2 = jax.jit(prob.rel_l2)

    hyperparameters = {
        "optimizer": args.optimizer,
        "framework": "jax",
        "jax_version": jax.__version__,
        "protocol": "jaxpi allen_cahn default.py (pirate branch)",
        "steps": args.steps,
        "arch": "modified_mlp",
        "hidden": args.hidden,
        "num_layers": args.num_layers,
        "rwf_mean": args.rwf_mean,
        "rwf_stddev": args.rwf_stddev,
        "fourier_scale": args.fourier_scale,
        "fourier_dim": args.fourier_dim,
        "period": "pi (x axis)",
        "n_params": n_params,
        "batch_size": args.batch_size,
        "aux_batch_size": args.aux_batch_size,
        "aux_stride": args.aux_stride,
        "causal_tol": args.causal_tol,
        "num_chunks": args.num_chunks,
        "weighting": "ntk",
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
            f"{args.num_layers}x{args.hidden} (RWF+RFF+periodic) | "
            f"params={n_params:,} | device={jax.devices()[0]}\n"
            f"  batch={args.batch_size} aux={args.aux_batch_size} | "
            f"causal(tol={args.causal_tol}, chunks={args.num_chunks}) + "
            f"ntk weighting | steps={args.steps}"
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
                w_ics=float(weights["ics"]), w_res=float(weights["res"]),
            )
            if not args.quiet:
                ms_per = (time.perf_counter() - t_start) / (step + 1) * 1000
                print(
                    f"  step {step + 1:6d}/{args.steps}  "
                    f"avg_train={last_avg:.4e}  rel_l2={rl2:.3e}  "
                    f"w=({float(weights['ics']):.2f},"
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
