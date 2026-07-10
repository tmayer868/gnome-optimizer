"""Tests for the JAX port (gnome_jax) — the algorithm-level invariants from
jax_port_plan.md. No bit-exact PyTorch parity is asserted (RNG streams
differ by construction); these tests pin the algorithm itself:

- Phase 2: the eigenbasis diagonalizes the Kronecker factors, under jit.
- Phase 3: Gnome at a fixed lr settles onto the closed-form OLS solution.
- Phase 4: the surrogate gradient is exactly sqrt(2/K)·Jᵀε for fixed signs
  and matches the brute-force GGN in second moment; the full closure step
  (nested jax.grad PDE derivatives included) compiles under one jax.jit.
- Phase 5: stack_residuals reproduces the weighted multi-block MSE.
"""

import math

import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402

import gnome_jax  # noqa: E402
from gnome_jax.gnome import _is_preconditioner  # noqa: E402


# ----------------------------------------------------------------------
# Phase 2 — preconditioner machinery
# ----------------------------------------------------------------------


def _offdiag_ratio(mat: jnp.ndarray) -> float:
    """Relative Frobenius mass of the off-diagonal part."""
    off = mat - jnp.diag(jnp.diag(mat))
    return float(jnp.linalg.norm(off) / jnp.linalg.norm(mat))


def test_initial_eigenbasis_diagonalizes_GG():
    """After the init step, Q comes from a full eigh, so Q^T GG Q must be
    diagonal to fp accuracy (up to the relative jitter)."""
    opt = gnome_jax.gnome(lr=1e-2)
    key = jax.random.PRNGKey(0)
    params = {"w": jnp.zeros((8, 6))}
    state = opt.init(params)

    g = jax.random.normal(key, (8, 6))
    update = jax.jit(opt.update)
    _, state = update({"w": g}, {"w": g}, state, params)

    for gg, q in zip(state.GG["w"].matrices, state.Q["w"].matrices):
        rot = q.T @ gg @ q
        assert _offdiag_ratio(rot) < 1e-3


def test_qr_refresh_tracks_eigenbasis_under_jit():
    """Run many jitted steps with i.i.d. surrogate gradients; the QR power
    iteration should keep Q aligned with GG's eigenbasis (small off-diagonal
    mass), and vastly better than an arbitrary rotation."""
    opt = gnome_jax.gnome(lr=1e-3, precondition_frequency=10)
    key = jax.random.PRNGKey(1)
    params = {"w": jnp.zeros((8, 6))}
    state = opt.init(params)

    # Anisotropic gradient distribution so GG has a nontrivial eigenbasis.
    key, k1, k2 = jax.random.split(key, 3)
    mix_l = jax.random.normal(k1, (8, 8)) / math.sqrt(8) + jnp.diag(
        jnp.linspace(3.0, 0.1, 8)
    )
    mix_r = jax.random.normal(k2, (6, 6)) / math.sqrt(6) + jnp.diag(
        jnp.linspace(2.0, 0.2, 6)
    )

    update = jax.jit(opt.update)
    for _ in range(201):
        key, kg = jax.random.split(key)
        g = mix_l @ jax.random.normal(kg, (8, 6)) @ mix_r
        params, state = update({"w": g}, {"w": g}, state, params)

    for gg, q in zip(state.GG["w"].matrices, state.Q["w"].matrices):
        ratio = _offdiag_ratio(q.T @ gg @ q)
        n = gg.shape[0]
        key, kq = jax.random.split(key)
        q_rand, _ = jnp.linalg.qr(jax.random.normal(kq, (n, n)))
        rand_ratio = _offdiag_ratio(q_rand.T @ gg @ q_rand)
        assert ratio < 0.15, f"off-diagonal ratio {ratio:.3f}"
        assert ratio < 0.25 * rand_ratio, (
            f"QR basis ({ratio:.3f}) not better than random ({rand_ratio:.3f})"
        )


def test_unpreconditioned_1d_params():
    """1-D params with precondition_1d=False get [None] factors: projection
    is the identity and the update is a plain diagonal Newton step."""
    opt = gnome_jax.gnome(lr=1e-2, warmup=0, weight_decay=0.0)
    params = {"b": jnp.ones((5,))}
    state = opt.init(params)
    assert state.GG["b"].matrices == (None,)

    update = jax.jit(opt.update)
    g = jnp.full((5,), 0.5)
    params1, state = update({"b": g}, {"b": g}, state, params)  # init step
    assert jnp.allclose(params1["b"], params["b"])  # no update on step 1
    params2, state = update({"b": g}, {"b": g}, state, params1)
    assert not jnp.allclose(params2["b"], params1["b"])
    assert jnp.all(jnp.isfinite(params2["b"]))


# ----------------------------------------------------------------------
# Phase 3 — Newton step / OLS
# ----------------------------------------------------------------------


def test_ols_settles_on_closed_form_solution():
    """The pedagogical case from experiments/ols_regression.py: with a fixed
    lr, Gnome's Newton step self-anneals and settles onto the OLS minimizer.
    """
    key = jax.random.PRNGKey(42)
    n, d = 256, 8
    key, kx, kw, kn = jax.random.split(key, 4)
    X = jax.random.normal(kx, (n, d))
    w_true = jax.random.normal(kw, (d, 1))
    y = X @ w_true + 0.05 * jax.random.normal(kn, (n, 1))

    w_ols = jnp.linalg.lstsq(X, y)[0]

    opt = gnome_jax.gnome(lr=0.1, weight_decay=0.0, warmup=100)
    params = {"w": jnp.zeros((d, 1))}
    state = opt.init(params)

    aux_k = 32

    @jax.jit
    def train_step(params, state, key):
        key, kperm = jax.random.split(key)
        perm = jax.random.permutation(kperm, n)
        main_idx, aux_idx = perm[aux_k:], perm[:aux_k]
        params, loss, state, key = opt.step(
            params, state, key,
            main_fn=lambda p: (X[main_idx] @ p["w"], y[main_idx]),
            aux_fn=lambda p: (X[aux_idx] @ p["w"], y[aux_idx]),
        )
        return params, loss, state, key

    for _ in range(2000):
        params, loss, state, key = train_step(params, state, key)

    dist = float(jnp.linalg.norm(params["w"] - w_ols))
    ref = float(jnp.linalg.norm(w_ols))
    assert jnp.isfinite(loss)
    assert dist < 0.02 * ref, f"beta distance {dist:.2e} vs ||w_ols|| {ref:.2e}"


# ----------------------------------------------------------------------
# Phase 4 — surrogate + step
# ----------------------------------------------------------------------


def test_surrogate_gradient_exact_for_fixed_signs():
    """For a linear model y_hat = X @ W the surrogate gradient must equal
    sqrt(2/K) · Xᵀ ε exactly — this pins the sqrt(2) and 1/sqrt(K) scaling
    deterministically."""
    key = jax.random.PRNGKey(7)
    K, d, m = 8, 4, 3
    key, kx, kw = jax.random.split(key, 3)
    X = jax.random.normal(kx, (K, d))
    W = jax.random.normal(kw, (d, m))

    key_s = jax.random.PRNGKey(123)
    g = jax.grad(
        lambda w: gnome_jax.build_surrogate_mse(X @ w, key_s)
    )(W)

    signs = jax.random.rademacher(key_s, (K, m), dtype=W.dtype)
    expected = math.sqrt(2.0 / K) * X.T @ signs
    assert jnp.allclose(g, expected, atol=1e-5)


def test_surrogate_second_moment_matches_ggn():
    """E[g gᵀ] must equal the brute-force GGN contractions of
    (2/K) Σ_k J_kᵀ J_k: for y_hat = X W, E[g gᵀ] = (2m/K) XᵀX and
    E[gᵀ g] = (2/K) Σ_k ||x_k||² I_m."""
    key = jax.random.PRNGKey(11)
    K, d, m = 8, 4, 3
    key, kx, kw = jax.random.split(key, 3)
    X = jax.random.normal(kx, (K, d))
    W = jax.random.normal(kw, (d, m))

    grad_fn = jax.vmap(
        lambda k: jax.grad(
            lambda w: gnome_jax.build_surrogate_mse(X @ w, k)
        )(W)
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 50000)
    gs = grad_fn(keys)  # (S, d, m)

    left = jnp.einsum("sdm,sem->de", gs, gs) / gs.shape[0]
    right = jnp.einsum("sdm,sdn->mn", gs, gs) / gs.shape[0]

    left_expected = (2.0 * m / K) * X.T @ X
    right_expected = (2.0 / K) * jnp.sum(
        jnp.square(jnp.linalg.norm(X, axis=1))
    ) * jnp.eye(m)

    def rel_err(a, b):
        return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))

    assert rel_err(left, left_expected) < 0.05
    assert rel_err(right, right_expected) < 0.05


def test_main_loss_reduction():
    """sum over outputs, mean over batch — not mean over everything."""
    y_hat = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    y = jnp.zeros((2, 2))
    loss = gnome_jax.compute_main_loss(y_hat, y)
    assert jnp.allclose(loss, (1 + 4 + 9 + 16) / 2.0)


def test_full_step_jits_with_nested_grad_closure():
    """The acceptance criterion: one jax.jit over the whole step, with a
    PINN-shaped closure (nested jax.grad input derivatives through an MLP
    + stack_residuals)."""

    def mlp(p, x):  # x: scalar -> scalar
        h = jnp.tanh(x * p["w1"] + p["b1"])
        return jnp.dot(h, p["w2"]) + p["b2"]

    def pde_residual(p, xs):
        # 1D Poisson: u_xx + f, f = pi^2 sin(pi x)  (for -u'' = f)
        u_x = jax.grad(mlp, argnums=1)
        u_xx = jax.grad(lambda p, x: u_x(p, x), argnums=1)
        return jax.vmap(lambda x: u_xx(p, x))(xs) + (
            math.pi**2
        ) * jnp.sin(math.pi * xs)

    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    params = {
        "w1": 0.5 * jax.random.normal(k1, (16,)),
        "b1": jnp.zeros((16,)),
        "w2": 0.5 * jax.random.normal(k2, (16,)),
        "b2": jnp.zeros(()),
    }

    opt = gnome_jax.gnome(lr=1e-3, weight_decay=0.0, warmup=10)
    state = opt.init(params)

    xs_col = jnp.linspace(0.05, 0.95, 32)
    xs_bc = jnp.array([0.0, 1.0])

    def residuals(p, xs):
        pde = pde_residual(p, xs)
        bc = jax.vmap(lambda x: mlp(p, x))(xs_bc)
        stacked = gnome_jax.stack_residuals([pde, bc])
        return stacked, jnp.zeros_like(stacked)

    @jax.jit
    def train_step(params, state, key):
        return opt.step(
            params, state, key,
            main_fn=lambda p: residuals(p, xs_col),
            aux_fn=lambda p: residuals(p, xs_col[::2]),
        )

    key = jax.random.PRNGKey(0)
    losses = []
    for _ in range(30):
        params, loss, state, key = train_step(params, state, key)
        losses.append(float(loss))

    assert all(math.isfinite(l) for l in losses)
    leaves = jtu.tree_leaves(params)
    assert all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)
    # step 1 is basis init (no update); afterwards the loss must move.
    assert losses[-1] != losses[1]


def test_step_key_threading():
    """step must return a new key and not reuse the input key's stream."""
    opt = gnome_jax.gnome(lr=1e-3)
    params = {"w": jnp.ones((4, 2))}
    state = opt.init(params)
    key = jax.random.PRNGKey(0)
    X = jnp.eye(4, 2)

    _, _, _, key_out = opt.step(
        params, state, key,
        main_fn=lambda p: (X.T @ p["w"], jnp.zeros((2, 2))),
        aux_fn=lambda p: (X.T @ p["w"], jnp.zeros((2, 2))),
    )
    assert not jnp.array_equal(key, key_out)


# ----------------------------------------------------------------------
# Phase 5 — stack_residuals
# ----------------------------------------------------------------------


def test_stack_residuals_reproduces_weighted_mse():
    key = jax.random.PRNGKey(5)
    k1, k2, k3 = jax.random.split(key, 3)
    r1 = jax.random.normal(k1, (10,))
    r2 = jax.random.normal(k2, (4, 3))
    r3 = jax.random.normal(k3, (7, 1))
    weights = [1.0, 0.5, 2.0]

    out = gnome_jax.stack_residuals([r1, r2, r3], weights)
    assert out.shape == (10 + 12 + 7,)

    expected = sum(
        w * jnp.mean(jnp.square(r)) for w, r in zip(weights, [r1, r2, r3])
    )
    assert jnp.allclose(jnp.mean(jnp.square(out)), expected, rtol=1e-6)


def test_stack_residuals_grad_flows():
    def loss(theta):
        r1 = theta * jnp.ones((5,))
        r2 = jnp.square(theta) * jnp.ones((3,))
        out = gnome_jax.stack_residuals([r1, r2])
        return jnp.mean(jnp.square(out))

    theta = jnp.asarray(2.0)
    g = jax.grad(loss)(theta)
    # d/dθ [θ² + θ⁴] = 2θ + 4θ³
    assert jnp.allclose(g, 2 * 2.0 + 4 * 8.0)


def test_stack_residuals_validation():
    with pytest.raises(ValueError):
        gnome_jax.stack_residuals([])
    with pytest.raises(ValueError):
        gnome_jax.stack_residuals([jnp.ones(3)], weights=[1.0, 2.0])
