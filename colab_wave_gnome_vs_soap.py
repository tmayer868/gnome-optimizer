"""Gnome vs SOAP on the 1D Wave equation (jaxpi protocol) — self-contained
Colab script.

Paste into a single Colab cell (GPU runtime recommended) and run. Gnome
(JAX port) is inlined, SOAP is pip-installed from the same SOAP_JAX
package jaxpi uses, and the reference solution is analytic — no download.

PDE:  u_tt = c²·u_xx,  c = 2,  (t, x) ∈ [0,1] × [0,1]
ICs:  u(0,x) = sin(πx) + 0.5·sin(2cπx),  u_t(0,x) = 0
BCs:  u(t,0) = u(t,1) = 0  (Dirichlet, enforced as a loss block)
Reference: u = sin(πx)cos(cπt) + 0.5·sin(2cπx)cos(4cπt) on a 200x128 grid
(the jaxpi wave example's analytic solution; NOTE the paper's Table 5
says c=4 but the repo's dataset uses c=2 — the repo solution defines the
problem we replicate).

Protocol = jaxpi wave sota.py / paper appendix G.4: ModifiedMlp 4x256 +
RWF + trainable Random Fourier features (scale 10 — both sources agree;
wave's high frequencies need it), NO periodic embedding. Four SOFT loss
blocks {u0, u_t0, bcs, res} (paper-faithful) with causal weighting +
grad-norm balancing. The residual is normalized by (4cπ)² so per-chunk
causal losses are O(1) and causal tol = 1.0 (paper Table 6) behaves as
designed — the repo's 1e-2 was only a scale-compensation for the
un-normalized O(630) residual; this is a pure reparametrization (the
zero set is unchanged), shared by both optimizers. Gnome needs eps=1e-3
here, not the 1e-6 that worked on Poisson/Burgers — wave's curvature
estimate is unreliable in many directions and eps is the LM trust floor
(see the eps note in the repo). SOAP β1=0.99 with their warmup +
exponential-decay schedule; Gnome at a FIXED lr. Chunks 16 (paper) vs 32
(repo); steps 1e5 (paper) vs 2e5 (repo).

Paper reference: Table 1 (PirateNet-assisted): Adam 5.15e-5 |
Adam+L-BFGS 5.08e-5 | Kron 8.62e-6 | Muon 9.34e-6 | SOAP 8.05e-6 (best).
Figure 4 ModifiedMLP bars: Adam ~9.5e-5, SOAP ~2.7e-5.
"""

# ============================== CONFIG ==============================

STEPS = 100_000         # paper Table 6 (repo sota: 200_000)
BATCH_SIZE = 8192       # res collocation points; must divide NUM_CHUNKS
SEED = 0
LOG_EVERY = 100
OPTIMIZERS = ("gnome", "soap")   # run order

# "highest" = true float32 everywhere. "default" = TF32 model matmuls on
# Ampere+ GPUs; optimizer internals stay exact either way.
MATMUL_PRECISION = "highest"

# --- Gnome (fixed lr, no schedule) ---
GNOME_LR = 1e-2                  # gnome likes the largest stable lr —
                                 # crank it (with more warmup) once a
                                 # baseline run survives
GNOME_BETAS = (0.9, 0.99)      # β1 0.99 = ~10x noise averaging
# EXPERIMENT: relative (condition-number) damping — the denominator floor
# is eps·λ_max (per param tensor's largest curvature) instead of a fixed
# absolute eps. So eps is now a scale-free ratio: directions flatter than
# eps·λ_max get GD-fallback, stiffer ones get Newton. The old absolute
# 1e-3 does NOT transfer (it was ~1e-7 of λ_max); this is a fresh knob.
GNOME_EPS = 1e-6                 # relative damping ratio (1/eps ≈ condition cap)
GNOME_CLIP = 1.0                 # trust-region clip
GNOME_CLIP_MODE = "param"        # "both" | "rotated" | "param";
                                 # "param" won the Burgers 3-way ablation
GNOME_WARMUP = 2_000              # internal linear lr warmup (steps)
GNOME_PRECOND_FREQ = 10          # eigenbasis refresh interval
GNOME_AUX_BATCH = 256            # aux (surrogate) res points per step
GNOME_AUX_ICS = 64               # fresh IC x-points resampled per aux step
GNOME_AUX_BCS = 64               # fresh BC t-points resampled per aux step

# --- SOAP (paper protocol) ---
SOAP_LR = 1e-2
# Paper appendix G.4: β1=0.99, β2=0.999 for SOAP (their ablation optimum).
SOAP_BETAS = (0.99, 0.999)
SOAP_WARMUP = 5_000              # linear warmup steps
SOAP_DECAY_RATE = 0.9            # exponential decay ...
SOAP_DECAY_STEPS = 2_000         # ... per this many steps
SOAP_PRECOND_FREQ = 2            # jaxpi hardcodes 2

# --- Problem / pipeline ---
C_SPEED = 2.0                    # wave speed (repo train.py; paper says 4)
A_COEFF = 0.5                    # second-mode amplitude in the IC
HIDDEN = 256
NUM_LAYERS = 4
RWF_MEAN, RWF_STDDEV = 1.0, 0.1
FOURIER_SCALE, FOURIER_DIM = 10.0, 256  # scale 10: paper + repo agree
CAUSAL_TOL = 1.0                 # paper Table 6; valid because r_net is
                                 # normalized by (4cπ)² so per-chunk causal
                                 # losses are O(1) (see RES_SCALE)
NUM_CHUNKS = 16                  # paper Table 6 (repo sota: 32)
WEIGHT_UPDATE_EVERY = 1_000      # grad-norm refresh (paper+repo agree on
WEIGHT_MOMENTUM = 0.9            # the grad_norm scheme for wave)

# ====================================================================

import math
import os
import subprocess
import sys
import time

try:
    from soap_jax import soap
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "git+https://github.com/haydn-jones/SOAP_JAX.git",
    ])
    from soap_jax import soap

import jax

if MATMUL_PRECISION != "default":
    jax.config.update("jax_default_matmul_precision", MATMUL_PRECISION)

import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import grad, vmap
from jax.tree_util import tree_leaves, tree_map

print("devices:", jax.devices())


# ====================================================================
# Gnome (JAX port) — inlined from the gnome-optimizer repo (gnome_jax).
# ====================================================================

from itertools import chain
from typing import NamedTuple

_QR_DTYPE = jnp.float32
_PRECISION = jax.lax.Precision.HIGHEST
_SQRT_TWO = math.sqrt(2.0)


@jtu.register_pytree_node_class
class Preconditioner:
    __slots__ = ("matrices",)

    def __init__(self, matrices):
        self.matrices = tuple(matrices)

    def tree_flatten(self):
        return (self.matrices, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    def map(self, fn):
        return Preconditioner(fn(m) for m in self.matrices)


def _is_precond(v):
    return isinstance(v, Preconditioner)


class GnomeState(NamedTuple):
    count: jax.Array
    exp_avg: object
    exp_avg_sq: object
    GG: object
    Q: object


def _init_conditioner(p, max_dim, precond_1d):
    if p.ndim == 1:
        if not precond_1d or p.shape[0] > max_dim:
            return Preconditioner([None])
        return Preconditioner([jnp.zeros((p.shape[0],) * 2, dtype=_QR_DTYPE)])
    return Preconditioner([
        jnp.zeros((s, s), dtype=_QR_DTYPE) if s <= max_dim else None
        for s in p.shape
    ])


def _lerp(a, b, w):
    return a + w * (b - a)


def _update_precond(g, GG, beta):
    if g.ndim == 1:
        if GG.matrices[0] is None:
            return GG
        outer = jnp.matmul(g[:, None], g[None, :], precision=_PRECISION)
        return Preconditioner(
            [_lerp(GG.matrices[0], outer.astype(GG.matrices[0].dtype),
                   1 - beta)]
        )
    new = []
    for idx, gg in enumerate(GG.matrices):
        if gg is None:
            new.append(None)
            continue
        outer = jnp.tensordot(
            g, g,
            axes=[[*chain(range(idx), range(idx + 1, g.ndim))]] * 2,
            precision=_PRECISION,
        )
        new.append(_lerp(gg, outer.astype(gg.dtype), 1 - beta))
    return Preconditioner(new)


def _project(g, Q):
    for mat in Q.matrices:
        if mat is not None:
            g = jnp.tensordot(g, mat.astype(g.dtype), axes=((0,), (0,)),
                              precision=_PRECISION)
        else:
            g = jnp.moveaxis(g, 0, -1)
    return g


def _project_back(g, Q):
    for mat in Q.matrices:
        if mat is not None:
            g = jnp.tensordot(g, mat.astype(g.dtype), axes=((0,), (1,)),
                              precision=_PRECISION)
        else:
            g = jnp.moveaxis(g, 0, -1)
    return g


def _orthogonal_matrix(gg):
    if gg is None:
        return None
    m = gg.astype(_QR_DTYPE)
    n = m.shape[0]
    scale = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(m))), 1.0)
    eye = jnp.eye(n, dtype=_QR_DTYPE)
    _, evecs = jnp.linalg.eigh(m + (1e-6 * scale) * eye)
    q = jnp.flip(evecs, axis=1)
    return jnp.where(jnp.all(jnp.isfinite(q)), q, eye)


def _qr_refresh(GG, Q, exp_avg_sq):
    new_Q = []
    for ind, (m, o) in enumerate(zip(GG.matrices, Q.matrices)):
        if m is None or o is None:
            new_Q.append(None)
            continue
        m_f = m.astype(_QR_DTYPE)
        o_f = o.astype(_QR_DTYPE)
        est = jnp.diag(jnp.matmul(
            jnp.matmul(o_f.T, m_f, precision=_PRECISION), o_f,
            precision=_PRECISION))
        idx = jnp.argsort(est, descending=True)
        exp_avg_sq = jnp.take(exp_avg_sq, idx, axis=ind)
        o_f = o_f[:, idx]
        q_new, _ = jnp.linalg.qr(jnp.matmul(m_f, o_f, precision=_PRECISION))
        new_Q.append(q_new)
    return Preconditioner(new_Q), exp_avg_sq


def compute_main_loss(y_hat, y):
    return ((y_hat - y) ** 2).sum() / y_hat.shape[0]


def build_surrogate_mse(y_hat_aux, key):
    K = y_hat_aux.shape[0]
    signs = jax.random.rademacher(key, y_hat_aux.shape, dtype=y_hat_aux.dtype)
    signs = jax.lax.stop_gradient(signs)
    return (_SQRT_TWO * signs * y_hat_aux).sum() / math.sqrt(K)


def stack_residuals(residuals, weights=None):
    if weights is None:
        weights = [1.0] * len(residuals)
    flats = [r.reshape(-1) for r in residuals]
    sizes = [f.size for f in flats]
    n_total = sum(sizes)
    return jnp.concatenate([
        f * jnp.sqrt(w * (n_total / s))
        for f, w, s in zip(flats, weights, sizes)
    ])


def gnome(lr=1e-3, betas=(0.9, 0.999), shampoo_beta=0.95, eps=1e-4,
          weight_decay=0.01, precondition_frequency=10,
          max_precond_dim=10000, clip=1.0, warmup=200,
          precondition_1d=False, clip_mode="both"):
    beta1, beta2 = betas
    gg_beta = shampoo_beta if shampoo_beta >= 0 else beta2

    def init_fn(params):
        zeros = jtu.tree_map(jnp.zeros_like, params)
        cond = lambda: jtu.tree_map(
            lambda p: _init_conditioner(p, max_precond_dim, precondition_1d),
            params)
        return GnomeState(count=jnp.zeros([], jnp.int32), exp_avg=zeros,
                          exp_avg_sq=zeros, GG=cond(), Q=cond())

    def update_fn(g_main, g_aux, state, params):
        count_inc = state.count + 1
        state = state._replace(count=count_inc)

        def init_step():
            new_GG = jtu.tree_map(
                lambda g, gg: _update_precond(g, gg, gg_beta),
                g_aux, state.GG, is_leaf=_is_precond)
            new_Q = jtu.tree_map(
                lambda gg: gg.map(_orthogonal_matrix),
                new_GG, is_leaf=_is_precond)
            return params, state._replace(GG=new_GG, Q=new_Q)

        def update_step():
            eff = (count_inc - 1).astype(jnp.float32)
            g_rot = jtu.tree_map(lambda g, q: _project(g, q),
                                 g_main, state.Q, is_leaf=_is_precond)
            gs_rot = jtu.tree_map(lambda g, q: _project(g, q),
                                  g_aux, state.Q, is_leaf=_is_precond)
            exp_avg = jtu.tree_map(
                lambda m, g: beta1 * m + (1 - beta1) * g,
                state.exp_avg, g_rot)
            exp_avg_sq = jtu.tree_map(
                lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g),
                state.exp_avg_sq, gs_rot)
            bc1 = 1.0 - beta1**eff
            bc2 = 1.0 - beta2**eff

            def newton(m, v, q):
                # Relative (Levenberg-Marquardt) damping: floor the curvature
                # denominator at eps·λ_max (this param's largest bias-corrected
                # curvature), so eps is a scale-free condition-number cap
                # rather than an absolute number that tracks residual scaling.
                vhat = v / bc2
                eff_eps = eps * jnp.max(vhat) + 1e-30  # +tiny: guard all-zero
                upd = (m / bc1) / (vhat + eff_eps)
                if clip is not None and clip_mode in ("both", "rotated"):
                    upd = jnp.clip(upd, -clip, clip)
                upd = _project_back(upd, q)
                if clip is not None and clip_mode in ("both", "param"):
                    upd = jnp.clip(upd, -clip, clip)
                return upd

            updates = jtu.tree_map(newton, exp_avg, exp_avg_sq, state.Q,
                                   is_leaf=_is_precond)
            if warmup > 0:
                lr_eff = lr * jnp.minimum(eff / warmup, 1.0)
            else:
                lr_eff = jnp.asarray(lr, jnp.float32)
            if weight_decay > 0.0:
                apply_u = lambda p, u: p - lr_eff * u - lr_eff * weight_decay * p
            else:
                apply_u = lambda p, u: p - lr_eff * u
            new_params = jtu.tree_map(apply_u, params, updates)

            new_GG = jtu.tree_map(
                lambda g, gg: _update_precond(g, gg, gg_beta),
                g_aux, state.GG, is_leaf=_is_precond)

            def refresh():
                q_v = jtu.tree_map(
                    lambda gg, q, v: _qr_refresh(gg, q, v),
                    new_GG, state.Q, exp_avg_sq, is_leaf=_is_precond)
                new_Q = jtu.tree_map(lambda _, x: x[0], g_main, q_v)
                new_v = jtu.tree_map(lambda _, x: x[1], g_main, q_v)
                new_m = jtu.tree_map(
                    lambda m, oq, nq: _project(_project_back(m, oq), nq),
                    exp_avg, state.Q, new_Q, is_leaf=_is_precond)
                return new_Q, new_v, new_m

            def keep():
                return state.Q, exp_avg_sq, exp_avg

            new_Q, v_out, m_out = jax.lax.cond(
                (count_inc - 1) % precondition_frequency == 0, refresh, keep)
            return new_params, GnomeState(count=count_inc, exp_avg=m_out,
                                          exp_avg_sq=v_out, GG=new_GG,
                                          Q=new_Q)

        return jax.lax.cond(count_inc == 1, init_step, update_step)

    def step_fn(params, state, key, main_fn, aux_fn):
        key_aux, key_next = jax.random.split(key)

        def main_loss(p):
            y_hat, y = main_fn(p)
            return compute_main_loss(y_hat, jax.lax.stop_gradient(y))

        loss, g_main = jax.value_and_grad(main_loss)(params)

        def surrogate(p):
            y_hat_aux, _ = aux_fn(p)
            return build_surrogate_mse(y_hat_aux, key_aux)

        g_aux = jax.grad(surrogate)(params)
        new_params, new_state = update_fn(g_main, g_aux, state, params)
        return new_params, loss, new_state, key_next

    class Opt(NamedTuple):
        init: object
        update: object
        step: object

    return Opt(init=init_fn, update=update_fn, step=step_fn)


# ====================================================================
# Architecture: ModifiedMlp + RWF + RFF, no periodic embedding (pure JAX,
# reimplemented from the published methods; jaxpi's code is not copied).
# ====================================================================

def _rwf_dense_init(key, fan_in, fan_out, mean, stddev):
    kw, kg = jax.random.split(key)
    w = jax.nn.initializers.glorot_normal()(kw, (fan_in, fan_out))
    g = jnp.exp(mean + stddev * jax.random.normal(kg, (fan_out,)))
    return {"g": g, "v": w / g, "b": jnp.zeros(fan_out)}


def _dense(p, x):
    return x @ (p["g"] * p["v"]) + p["b"]


def init_model(key):
    # embed: (t, x) -> RFF dim FOURIER_DIM (scale 10)
    keys = jax.random.split(key, NUM_LAYERS + 4)
    params = {"fourier": FOURIER_SCALE * jax.random.normal(
        keys[-2], (2, FOURIER_DIM // 2))}
    dims = [FOURIER_DIM] + [HIDDEN] * NUM_LAYERS
    params["enc_u"] = _rwf_dense_init(keys[0], FOURIER_DIM, HIDDEN,
                                      RWF_MEAN, RWF_STDDEV)
    params["enc_v"] = _rwf_dense_init(keys[1], FOURIER_DIM, HIDDEN,
                                      RWF_MEAN, RWF_STDDEV)
    params["hidden"] = [
        _rwf_dense_init(keys[2 + i], dims[i], HIDDEN, RWF_MEAN, RWF_STDDEV)
        for i in range(NUM_LAYERS)
    ]
    params["out"] = _rwf_dense_init(keys[-1], HIDDEN, 1, RWF_MEAN,
                                    RWF_STDDEV)
    return params


def apply_model(params, z):  # z = (t, x)
    zb = z @ params["fourier"]
    x = jnp.concatenate([jnp.cos(zb), jnp.sin(zb)])
    u = jnp.tanh(_dense(params["enc_u"], x))
    v = jnp.tanh(_dense(params["enc_v"], x))
    for layer in params["hidden"]:
        x = jnp.tanh(_dense(layer, x))
        x = x * u + (1 - x) * v
    return _dense(params["out"], x)


# ====================================================================
# Wave problem (jaxpi protocol; analytic reference)
# ====================================================================

N_T, N_X = 200, 128
T_STAR = jnp.linspace(0.0, 1.0, N_T)
X_STAR = jnp.linspace(0.0, 1.0, N_X)
T0 = T_STAR[0]


def u_exact(t, x):
    return (jnp.sin(jnp.pi * x) * jnp.cos(C_SPEED * jnp.pi * t)
            + A_COEFF * jnp.sin(2 * C_SPEED * jnp.pi * x)
            * jnp.cos(4 * C_SPEED * jnp.pi * t))


U_REF = vmap(vmap(u_exact, (None, 0)), (0, None))(T_STAR, X_STAR)
U0 = U_REF[0, :]
DOM = jnp.array([[0.0, 1.0], [0.0, 1.0]])
M_CAUSAL = jnp.triu(jnp.ones((NUM_CHUNKS, NUM_CHUNKS)), k=1).T


# Wave 2nd derivatives peak at O((4cπ)²·A); normalize the residual by this
# so per-chunk causal losses are O(1) and CAUSAL_TOL=1.0 (paper) behaves.
RES_SCALE = (4.0 * C_SPEED * math.pi) ** 2


def u_net(params, t, x):
    return apply_model(params, jnp.stack([t, x]))[0]


def r_net(params, t, x):
    """Wave residual, matching jaxpi: (u_tt − c²·u_xx), normalized by
    RES_SCALE so per-chunk causal losses are O(1) and CAUSAL_TOL=1.0
    behaves (a scale reparametrization — the zero set is unchanged)."""
    u_tt = grad(grad(u_net, argnums=1), argnums=1)(params, t, x)
    u_xx = grad(grad(u_net, argnums=2), argnums=2)(params, t, x)
    return (u_tt - C_SPEED**2 * u_xx) / RES_SCALE


# Soft IC/BC loss blocks (paper protocol). The aux closure resamples fresh
# IC x-points and BC t-points every step (coverage through the curvature
# EMA); ICs/BCs are analytic so these take coordinates directly.

def u0_residual(params, xs):
    u_pred = vmap(u_net, (None, None, 0))(params, T0, xs)
    return vmap(lambda x: u_exact(0.0, x))(xs) - u_pred


def ut0_residual(params, xs):
    ut = vmap(lambda x: grad(u_net, argnums=1)(params, T0, x))(xs)
    return ut  # target 0


def bc_residuals(params, ts):
    bc1 = vmap(u_net, (None, 0, None))(params, ts, X_STAR[0])
    bc2 = vmap(u_net, (None, 0, None))(params, ts, X_STAR[-1])
    return bc1, bc2


def causal_l_w(params, batch):
    t_sorted = batch[:, 0].sort()
    r = vmap(r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])
    rc = r.reshape(NUM_CHUNKS, -1)
    l = jnp.mean(rc**2, axis=1)
    w = jax.lax.stop_gradient(jnp.exp(-CAUSAL_TOL * (M_CAUSAL @ l)))
    return rc, l, w


def causal_res(params, batch):
    rc, _, w = causal_l_w(params, batch)
    return (rc * jnp.sqrt(w)[:, None]).reshape(-1)


def losses(params, batch):
    bc1, bc2 = bc_residuals(params, T_STAR)
    res = causal_res(params, batch)
    return {
        "u0": jnp.mean(u0_residual(params, X_STAR) ** 2),
        "u_t0": jnp.mean(ut0_residual(params, X_STAR) ** 2),
        "bcs": jnp.mean(bc1**2) + jnp.mean(bc2**2),
        "res": jnp.mean(res**2),
    }


def weighted_loss(params, weights, batch):
    ls = losses(params, batch)
    return sum(weights[k] * ls[k] for k in ls)


def stacked_residuals(params, weights, batch, xs_ic, ts_bc):
    bc1, bc2 = bc_residuals(params, ts_bc)
    return stack_residuals(
        [u0_residual(params, xs_ic), ut0_residual(params, xs_ic),
         bc1, bc2, causal_res(params, batch)],
        [weights["u0"], weights["u_t0"], weights["bcs"], weights["bcs"],
         weights["res"]],
    )


def compute_grad_norm_weights(params, batch):
    """jaxpi's grad_norm scheme: w_k = mean_norm/(norm_k + 1e-5·mean)."""
    norms = {}
    for k in ("u0", "u_t0", "bcs", "res"):
        g = grad(lambda p: losses(p, batch)[k])(params)
        flat = jnp.concatenate([v.reshape(-1) for v in tree_leaves(g)])
        norms[k] = jnp.linalg.norm(flat)
    mean_norm = jnp.mean(jnp.stack(list(norms.values())))
    return {k: mean_norm / (n + 1e-5 * mean_norm) for k, n in norms.items()}


def rel_l2(params):
    u_pred = vmap(vmap(u_net, (None, None, 0)), (None, 0, None))(
        params, T_STAR, X_STAR)
    return jnp.linalg.norm(u_pred - U_REF) / jnp.linalg.norm(U_REF)


def sample_batch(key, n):
    return jax.random.uniform(key, (n, 2), minval=DOM[:, 0],
                              maxval=DOM[:, 1])


# ====================================================================
# Training
# ====================================================================

assert BATCH_SIZE % NUM_CHUNKS == 0
assert GNOME_AUX_BATCH % NUM_CHUNKS == 0

update_weights_fn = jax.jit(compute_grad_norm_weights)
eval_fn = jax.jit(rel_l2)
histories = {}

for opt_name in OPTIMIZERS:
    key = jax.random.PRNGKey(SEED)
    key, k_model = jax.random.split(key)
    params = init_model(k_model)
    n_params = sum(p.size for p in tree_leaves(params))
    weights = {"u0": jnp.asarray(1.0), "u_t0": jnp.asarray(1.0),
               "bcs": jnp.asarray(1.0), "res": jnp.asarray(1.0)}

    if opt_name == "gnome":
        opt = gnome(lr=GNOME_LR, betas=GNOME_BETAS,
                    shampoo_beta=GNOME_BETAS[1], eps=GNOME_EPS,
                    weight_decay=0.0,
                    precondition_frequency=GNOME_PRECOND_FREQ,
                    clip=GNOME_CLIP, warmup=GNOME_WARMUP,
                    precondition_1d=True, clip_mode=GNOME_CLIP_MODE)
        opt_state = opt.init(params)

        @jax.jit
        def train_step(params, opt_state, key, weights):
            key, k_main, k_aux, k_ic, k_bc = jax.random.split(key, 5)
            batch = sample_batch(k_main, BATCH_SIZE)
            aux_batch = sample_batch(k_aux, GNOME_AUX_BATCH)
            # Fresh analytic IC x-points and BC t-points every aux step
            # (coverage through the curvature EMA).
            xs_ic_aux = jax.random.uniform(k_ic, (GNOME_AUX_ICS,))
            ts_bc_aux = jax.random.uniform(k_bc, (GNOME_AUX_BCS,))

            def main_fn(p):
                r = stacked_residuals(p, weights, batch, X_STAR, T_STAR)
                return r, jnp.zeros_like(r)

            def aux_fn(p):
                r = stacked_residuals(p, weights, aux_batch,
                                      xs_ic_aux, ts_bc_aux)
                return r, jnp.zeros_like(r)

            return opt.step(params, opt_state, key, main_fn, aux_fn)

    else:  # soap — paper protocol
        schedule = optax.join_schedules(
            [optax.linear_schedule(0.0, SOAP_LR, SOAP_WARMUP),
             optax.exponential_decay(SOAP_LR, SOAP_DECAY_STEPS,
                                     SOAP_DECAY_RATE, staircase=False)],
            [SOAP_WARMUP],
        )
        tx = soap(learning_rate=schedule, b1=SOAP_BETAS[0],
                  b2=SOAP_BETAS[1], weight_decay=0.0,
                  precondition_frequency=SOAP_PRECOND_FREQ)
        opt_state = tx.init(params)

        @jax.jit
        def train_step(params, opt_state, key, weights):
            key, k_main = jax.random.split(key)
            batch = sample_batch(k_main, BATCH_SIZE)
            loss, grads = jax.value_and_grad(weighted_loss)(
                params, weights, batch)
            updates, new_state = tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), loss, new_state, key

    print(f"\n=== {opt_name} | params={n_params:,} | steps={STEPS} | "
          f"lr={'%.0e' % (GNOME_LR if opt_name == 'gnome' else SOAP_LR)}"
          f"{' (fixed)' if opt_name == 'gnome' else ' (warmup+exp decay)'} ===")
    hist = {"step": [], "rel_l2": [], "loss": []}
    t_start = time.time()

    for step in range(STEPS):
        params, loss, opt_state, key = train_step(params, opt_state, key,
                                                  weights)
        if step % WEIGHT_UPDATE_EVERY == 0:
            key, k_w = jax.random.split(key)
            new_w = update_weights_fn(params, sample_batch(k_w, BATCH_SIZE))
            weights = tree_map(
                lambda o, n: o * WEIGHT_MOMENTUM + (1 - WEIGHT_MOMENTUM) * n,
                weights, new_w)
        if (step + 1) % LOG_EVERY == 0:
            loss_v = float(loss)
            if not math.isfinite(loss_v):
                print(f"  DIVERGED at step {step + 1}")
                break
            rl2 = float(eval_fn(params))
            hist["step"].append(step + 1)
            hist["rel_l2"].append(rl2)
            hist["loss"].append(loss_v)
            ms = (time.time() - t_start) / (step + 1) * 1000
            wstr = ",".join(f"{k}={float(v):.1f}"
                            for k, v in weights.items())
            print(f"  step {step + 1:6d}/{STEPS}  loss={loss_v:.3e}  "
                  f"rel_l2={rl2:.3e}  w=({wstr})  {ms:.1f} ms/step",
                  flush=True)

    histories[opt_name] = hist
    if hist["rel_l2"]:
        best = min(hist["rel_l2"])
        print(f"[{opt_name}] final rel_l2={hist['rel_l2'][-1]:.3e}  "
              f"best={best:.3e}  ({time.time() - t_start:.0f}s)")

# ====================================================================
# Summary + plot
# ====================================================================

print("\n================ SUMMARY ================")
print("paper Table 1 (PirateNet-assisted): Adam 5.15e-5 | Kron 8.62e-6 | "
      "Muon 9.34e-6 | SOAP 8.05e-6")
print("paper Fig 4 (ModifiedMLP): Adam ~9.5e-5 | SOAP ~2.7e-5")
for name, h in histories.items():
    if not h["rel_l2"]:
        continue
    best = min(h["rel_l2"])
    print(f"{name:>6}: final rel_l2 {h['rel_l2'][-1]:.3e} | best {best:.3e}")
    for thr in (1e-3, 1e-4, 2.7e-5, 8.05e-6, 1e-6):
        crossed = next((s for s, r in zip(h["step"], h["rel_l2"])
                        if r <= thr), None)
        print(f"        first step ≤ {thr:.2e}: "
              f"{crossed if crossed else '—'}")

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    for name, h in histories.items():
        if h["rel_l2"]:
            plt.plot(h["step"], h["rel_l2"], label=name)
    plt.axhline(8.05e-6, ls="--", c="gray", lw=1,
                label="paper SOAP (PirateNet) 8.05e-6")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("rel L2")
    plt.title("Wave (jaxpi protocol): Gnome vs SOAP")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:  # headless environments
    print(f"(plot skipped: {e})")
