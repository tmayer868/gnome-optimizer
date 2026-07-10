# JAX Port Plan

A plan for porting the `gnome` optimizer (PyTorch) to JAX, so it can be used with
JAX-based PINN codebases. The target API style is
[SOAP_JAX](https://github.com/haydn-jones/SOAP_JAX), an unofficial JAX port of
the SOAP optimizer that follows optax conventions. Port from SOAP_JAX's
**current HEAD** — it fixed stale EMA updates on off-precondition steps on
01/09/2026; don't work from an older checkout.

## Motivation

The PyTorch `gnome` package is a closure-based optimizer: `step(main_closure,
aux_closure)` owns the forward passes because it needs the model outputs
`(y_hat, y)` — not a scalar loss — to build the GGN surrogate internally. JAX
optimizers follow a functional contract (`update_fn(updates, state, params) ->
(updates, new_state)`) that only sees gradients, so a faithful port cannot
follow optax exactly. Instead we ship a custom `step` that preserves Gnome's
closure contract while mirroring SOAP_JAX's preconditioner structure.

## Decisions

- **Repo layout**: Sibling `gnome_jax/` package in this repo, alongside
  `gnome/`. Keeps the PyTorch reference at hand for cross-checking.
- **Loss scope (v1)**: MSE only. Covers all four PINN experiments (Poisson,
  Burgers, Kuramoto–Sivashinsky, Navier–Stokes). CCE / `cce_hutchinson` deferred
  to a later phase — only needed for the WikiText LLM benchmark.
- **Step API**: Closures returning `(y_hat, y)`, matching the PyTorch Gnome
  exactly. The optimizer computes the main loss and the surrogate internally so
  that the surrogate scale stays consistent with the loss reduction (the
  `eps` / `clip` calibration contract from `docs/method.md` §5.1).
- **Two-level API**: a pure `update(g_main, g_aux, state, params) ->
  (new_params, new_state)` core, with `step` as a thin closure-driving wrapper
  on top. `update` is the unit-testable, jit-friendly heart; `step` is what
  PINN users call. Users with exotic training loops can drop to `update` and
  build the surrogate themselves via the exported helpers.
- **jit is a requirement, not a nice-to-have**: the acceptance criterion for
  the port is that one `jax.jit` wraps the entire `step` — closures included,
  with their nested `jax.grad` PDE derivatives — and compiles. A non-jitted
  step is 10–100x slower and no JAX user will adopt it. Every phase below
  builds under this constraint (see "jit design" notes inline).
- **RNG**: explicit `key` argument to `step`, threaded and returned. The key
  does **not** live in `GnomeState` — explicit randomness is the JAX
  convention, and it keeps state a pure function of the trajectory.
- **Faithfulness bar**: the *algorithm* must be ported faithfully — same
  surrogate construction and scaling, same preconditioner updates, same Newton
  step and clipping — but we do **not** chase bit-exact agreement with
  PyTorch. Different RNG streams make trajectory-level matching meaningless
  anyway. Faithfulness is enforced by algorithm-level invariant tests (see
  Phases 2–4) plus a qualitative PINN cross-check (Phase 6).
- **Warmup**: implement a clean linear ramp `lr * min(count / warmup, 1.0)`.
  The PyTorch version (`gnome/optimizer.py:806`) has two quirks we
  intentionally do NOT replicate: a `0.01` lr floor inside the ramp and a
  `-1` step offset (its global `_step_count` also leads the per-param
  `state["step"]` by one because the first call only initializes). These are
  incidental, not part of the method.
- **Precision under `jax_enable_x64`**: keep the Kronecker factors `GG` and
  eigenbasis `Q` in float32 working precision regardless of param dtype,
  matching the PyTorch implementation (`_eigh_safe` / the `.to(grad.dtype)`
  alignment in `_project`). Cast at the projection boundary.
- **Do not port the dead debug block** at `gnome/optimizer.py:835-838`
  (`if np.random.rand() < .00:` — probability-zero print left over from
  debugging).

## API preview

```python
# v1 — MSE only, closures returning (y_hat, y)
params, loss, state, key = gnome.step(
    params, state, key,
    main_fn=lambda p: model_apply(p, x_main, y_main),  # -> (y_hat, y)
    aux_fn=lambda p: model_apply(p, x_aux, y_aux),     # -> (y_hat, y)
)

# lower level, for exotic loops (grads computed by the caller):
params, state = gnome.update(g_main, g_aux, state, params)
```

The optimizer owns:
- the surrogate *construction* (Rademacher signs, `sqrt(2)·ε·y_hat`, the
  `1/√K` scaling), because that is its value-add and the scaling must match its
  internal loss reduction;
- the main loss computation, with Gnome's exact reduction
  (`((y_hat - y) ** 2).sum() / B`, not `mean`), so `eps` and `clip` stay
  calibrated.

The user owns:
- the batch split into main / aux slices;
- the forward-pass orchestration (the closures);
- the PRNG key (seed once, thread through).

This matches JAX conventions (functional, explicit) and keeps Gnome usable
across arbitrary PINN codebases with different batch structures (collocation
points, IC points, BC points, etc.).

## The PyTorch → JAX translation

| Concept | PyTorch (`gnome/`) | JAX (`gnome_jax/`) |
|---|---|---|
| Per-parameter preconditioner matrices | `state["GG"]` list of tensors | `Preconditioner` pytree (from SOAP_JAX) |
| Eigenbasis | `state["Q"]` list of tensors | `Preconditioner` pytree |
| First-moment EMA | `state["grad_m"]` | `exp_avg` |
| Second-moment EMA | `state["gnd_m"]` | `exp_avg_sq` |
| Step counter | `state["step"]` per param | `count` (single global) |
| RNG (Rademacher signs) | `torch.rand_like` (global) | `jax.random.rademacher(key, shape)` with an explicit key |
| `detach` / `no_grad` | `with torch.no_grad():`, `.detach()` | `jax.lax.stop_gradient` |
| In-place updates | `p.add_(...)`, `m.mul_().add_()` | Functional: return new params/state |
| Periodic eigenbasis refresh | Python `if step % freq == 0` | `jax.lax.cond(count % freq == 0, ...)` |
| First-step init special case | `state["Q"] is None` branch, early `return` | Materialize `Q` eagerly in `init`, or `lax.cond(count == 0, ...)` — decided in Phase 3 |
| `eigh` failure fallback | `try/except` → identity | Finiteness check + `jnp.where` → identity (try/except can't trace) |
| MPS CPU-bounce for `eigh`/`qr` | `_prefer_cpu`, `_eigh_safe` | Drop — JAX has no MPS backend |
| Loss reductions | `((y_hat - y) ** 2).sum() / B` | Same, on `jnp` arrays |

Note on the single global `count`: in PyTorch, per-param `state["step"]` can
lag when a param gets no gradient (`allow_unused`) and skips `_param_step`.
A global count assumes every param receives a gradient every step — true for
PINN MLPs, so safe for v1. Leave a comment in the code stating the assumption.

## Build order

### Phase 1 — Package skeleton (day 1)
- New `gnome_jax/` directory with `__init__.py`, `gnome.py`, `surrogate.py`,
  `blocks.py`.
- Update `pyproject.toml` with an optional `jax` extra: `jax`, `optax`, `chex`,
  `jaxtyping`. Keep uv-managed per `claude.md`.

### Phase 2 — Preconditioner machinery (days 2–3)
Port from SOAP_JAX (current HEAD) nearly verbatim, since Gnome inherits SOAP's
preconditioning:
- `Preconditioner` pytree class (their `soap.py`).
- `init_conditioner`, `update_preconditioner`, `project`, `project_back`,
  `get_orthogonal_matrix`, `get_orthogonal_matrix_QR`.
- Drop `merge_dims` for v1 (SOAP_JAX doesn't implement it either; PINN MLPs are
  2D so it doesn't matter).
- jit design: copy SOAP_JAX's handling of the periodic refresh (`lax.cond` on
  `count % precondition_frequency`); the QR-refresh's argsort permutation of
  `exp_avg_sq` maps to `jnp.take_along_axis`. Replace `_eigh_safe`'s
  try/except with a finiteness check + `jnp.where` identity fallback.
- **Test**: feed a known gradient, verify `Q^T GG Q` is diagonal after refresh.
  Test through a `jax.jit`-wrapped call.

### Phase 3 — Gnome state + Newton step (day 3)
- `GnomeState` NamedTuple: `count`, `exp_avg`, `exp_avg_sq` (= `gnd_m`), `GG`,
  `Q`. No `key` in state (explicit arg, per Decisions).
- Resolve the first-step init question here: prefer materializing `Q` in
  `init` (identity basis) so `update` has a single unconditional code path;
  fall back to `lax.cond(count == 0, ...)` only if eager init distorts the
  early EMA behavior.
- Pure `update(g_main, g_aux, state, params)`:
  Newton step `exp_avg / (exp_avg_sq + eps)` (un-`sqrt`, the Gnome change vs
  SOAP's `sqrt(exp_avg_sq)`), then `clip` to `±clip` in both the rotated and
  rotated-back bases (`gnome/optimizer.py:833`–`849`).
- Clean linear warmup `lr * min(count / warmup, 1.0)` (see Decisions — the
  PyTorch quirks are not replicated).
- **Test**: OLS regression through jitted `update` — Gnome should hit the
  closed-form OLS solution at a fixed LR; baselines shouldn't (the pedagogical
  case from `experiments/ols_regression.py`).

### Phase 4 — Surrogate construction + `step` (days 4–5)
- `build_surrogate_mse(y_hat_aux, key)` → scalar
  `S = (sqrt(2) / sqrt(K)) * <ε, y_hat_aux>` with
  `ε = jax.random.rademacher(key, y_hat_aux.shape, dtype)`. (JAX has
  Rademacher sampling built in; the PyTorch `rand_like < 0.5` construction at
  `gnome/optimizer.py:94` exists only to dodge an MPS kernel crash and has no
  JAX analogue.)
- `compute_main_loss(y_hat, y)` — internal, with Gnome's exact reduction
  (`((y_hat - y) ** 2).sum() / B`), so `eps` / `clip` calibration stays
  consistent (the contract from `docs/method.md` §5.1).
- `step(params, state, key, main_fn, aux_fn)` where
  `main_fn(params) -> (y_hat, y_main)` and `aux_fn(params) -> (y_hat_aux, y_aux)`.
  Inside `step`:
  1. Split `key` → `key_aux`, `key_next`.
  2. `jax.value_and_grad` of `compute_main_loss` w.r.t. params, threaded
     through `main_fn`.
  3. Build `S` from the aux output with `key_aux`, `jax.grad` of `S` for
     `g_aux`.
  4. Delegate to `update(g_main, g_aux, state, params)` — preconditioner
     update with `g_aux` (the Gnome change; SOAP uses `g_main`), Newton step
     in the rotated basis, clip, rotate back, clip again.
  5. Return `(new_params, loss, new_state, key_next)`.
- **Tests**:
  - Surrogate gradient matches a brute-force GGN on a tiny linear model
    (closed-form `2 J^T J`) — the key algorithmic-faithfulness test: it pins
    the `sqrt(2)` and `1/√K` scaling against the same invariant the PyTorch
    implementation satisfies.
  - The full `step` compiles under a single `jax.jit` with a closure that
    contains nested `jax.grad` input derivatives (mini PINN-shaped forward).

### Phase 5 — `stack_residuals` (day 5, ~10 lines)
Direct port of `gnome/blocks.py:43` — `jnp.concatenate` with
`sqrt(λ_j N / N_j)` scaling. Pure math, no JAX subtleties. Higher-order
autograd (`create_graph=True` for `u_t`, `u_xx`, ...) flows through unchanged in
JAX because `jax.grad` is always differentiable.

### Phase 6 — Validate on a PINN (days 6–7)
Port `experiments/poisson_pinn.py` to JAX (simplest of the four — Poisson
equation, MLP, collocation + BC residuals via `stack_residuals`). This is the
real test:
- The whole training step runs under one `jax.jit` (the acceptance criterion
  from Decisions).
- Gnome at a fixed LR converges.
- Run with SOAP_JAX as the baseline for cross-check.
- Qualitative cross-check against the PyTorch Gnome run on the same problem:
  same convergence shape, final rel_l2 in the same ballpark (same order of
  magnitude at matched steps/config). No trajectory- or bit-level matching —
  RNG streams differ by construction; algorithmic faithfulness is covered by
  the Phase 2–4 invariant tests.

### Phase 7 — CCE / Hutchinson (optional, later)
Only if the WikiText LLM benchmark is needed in JAX. The Fisher surrogate
(`jax.random.categorical` for label sampling) and the Rademacher-Hutchinson
surrogate (`diag(√p)(I − √p√pᵀ)` factorization of the softmax Hessian) are pure
tensor math, ~30 lines each. Skip for PINN-only use.

## Out of scope for v1

- **`merge_dims`** (conv channel merging) — not needed for PINN MLPs.
- **Micro-batch accumulation** (`gnome/optimizer.py:649`) — use `jax.lax.scan`
  if ever needed.
- **MPS CPU-bounce** (`gnome/optimizer.py:86`–`130`) — JAX has no MPS backend,
  drop entirely.
- **Bit-exact PyTorch parity** — algorithm-level invariants only (see
  Decisions / Phase 4 tests).
- **Dashboard / run logger / sweep scripts** — JAX-specific tooling is a
  separate project.
