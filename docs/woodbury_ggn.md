# Experimental Woodbury GGN

`gnome.experimental.WoodburyGGN` maintains a finite EMA of raw GGN surrogate
gradients separately for every 2D parameter tensor. It applies the damped
inverse of that empirical block using a dense solve in history space. It
reuses `Gnome._compute_loss` and `Gnome._build_surrogate` directly, including
the existing random-probe scaling. The ordinary loss gradient supplies the
optimization direction; the surrogate supplies only curvature samples.

For oldest-to-newest samples `S[0:n]`, the weights are proportional to
`beta2 ** (n-1-i)` and normalized to sum to one, including during warmup.
The optimizer caches their raw Frobenius Gram matrix and updates only the
new row/column. History tensors stay detached, in their original matrix
shapes. No parameter-space GGN is constructed or persistently flattened.

With `a = sqrt(weights)`, it solves
`(lambda * I + a[:,None] * gram * a[None,:]) c = b`, where
`b[i] = a[i] * sum(S[i] * X)`, and applies
`W -= lr * (X - sum(a[i] * c[i] * S[i])) / lambda`.
`X` is the ordinary gradient or its bias-corrected first moment.
Adaptive damping enforces a per-tensor L2 trust region by default. There is no
coordinate clipping, direction rescaling, weight decay, or structural
factorization inside a tensor block. Different tensors have independent systems.

## Trust-region damping

For each tensor, choose the smallest `lambda >= damping` satisfying

```text
||P(lambda)||_F <= R
R = trust_radius * sqrt(||W||_F**2 + 1)
P(lambda) = (C + lambda I)^-1 X
```

This matches the radius used by Gnome's current update calculation. LR scales
the bounded direction, so `||delta W||_F <= lr * R` to numerical tolerance.
Gnome's additional coordinate clipping is not part of Woodbury. This is a
damping-based step bound with a configured radius; it does not add loss-ratio
accept/reject steps or adapt the radius itself.

`trust_region=1` is the default. Larger values allow longer directions;
`trust_region=0` disables the constraint and restores fixed damping.
`woodbury_damping` is the positive damping **floor**, not the selected value
when the constraint binds. The Python equivalent is `trust_radius=1.0`, or
`None` to disable it.

The solver first tests the floor. If it binds, it brackets the solution using
`||X|| / R` as an upper bound and applies safeguarded Newton iterations to
the reciprocal-norm equation, falling back to bisection when necessary.
The previous selected damping supplies a warm start. Each trial uses the small
history system, reusing its Cholesky factor for the Newton derivative:

```text
K = a[:,None] * gram * a[None,:]
c = solve(K + lambda I, b)
||P||_F**2 = (||X||_F**2 - 2*b.T@c + c.T@K@c) / lambda**2
```

No eigendecomposition is used for selecting damping. When the norm expression
suffers cancellation, it evaluates the actual matrix norm. The reconstructed
direction is also checked; if the history-space calculation underestimated
its norm beyond tolerance, the root solve retries with direct matrix norms.
It returns the feasible bracket endpoint, with relative tolerance `1e-6` in
float64 and approximately `3.8e-6` in float32. The search is capped at 40 Newton/
bisection iterations per attempt. Tight floors can amplify float32 error in
the Woodbury subtraction; `gram_float64` also promotes that reconstruction.

## Poisson

A small initial run with fixed main points and fresh auxiliary subsets:

```bash
# Edit experiments/configs/pinns/poisson_pinn.py: optimizer='woodbury', fixed_dataset=True, hidden=8, depth=3, n_pde=32, n_bc_per_edge=4, aux_frac=0.5, steps=300, lr=0.001, woodbury_damping=0.1, beta2=0.9, warmup_steps=0, cosine_decay=1.0, float64=True, diagnostics_every=50, log_every=50
uv run -m experiments.pinns.poisson_pinn
```

This is a stability check, not a converged Poisson benchmark or a recommended
performance configuration. The first 100 steps collect curvature with fixed
parameters; the remaining 200 steps update them. The smaller `beta2=0.9`
keeps this smoke test inexpensive; the optimizer default remains `0.99`.

The model conventions are unchanged: `arch=mlp` stores one `[W|b]` matrix
per layer; `arch=fused` groups hidden layers into larger matrices. Both
work, but grouping changes the optimizer's block structure. Standalone 1D
bias parameters are rejected explicitly by this optimizer.

The relevant options are:

| Option | Default | Meaning |
|---|---:|---|
| `beta1` | `0` for Woodbury | Ordinary-gradient momentum, bias corrected |
| `beta2` | `0.99` | Curvature history decay |
| `woodbury_damping` | `1e-3` | Positive damping floor; fixed when trust region is disabled |
| `trust_region` | `1` | Per-tensor radius multiplier; `0` disables adaptive damping |
| `woodbury_warmup_steps` | `100` | Initial calls with zero applied LR; `0` disables |
| `history_tol` | `0.01` | Discarded EMA tail tolerance |
| `max_history` | derived | Optional override of the retained sample count |
| `gram_float64` | off | Double inner products, solve, and temporary reconstruction |
| `diagnostics_every` | `0` | Diagnostics cadence; zero disables |
| `diagnostics_params` | all | Optional comma-separated parameter indices |

By default, `ceil(log(history_tol) / log(beta2)) = 459` samples are retained
**per tensor**. `beta2=0` derives a one-sample history. An explicit
`max_history` replaces the derived size; weights still sum to one, but the
tail tolerance no longer determines retained EMA mass. Storage per tensor
is `O(m*d + m*m)`, where `d` is its element count. History operations cost
`O(m*d)` and each dense factorization costs `O(m^3)`. Adaptive damping adds
history-space solves for trial values; direct-norm fallbacks also reconstruct
the matrix direction. Small histories are
useful for debugging before allocating the default window on large MLPs.

Woodbury's internal curvature warmup holds parameters exactly fixed for its
first 100 calls while history, Gram caches, and optional gradient momentum
accumulate. Updates start on call 101. It skips unused solves during this
period, except when diagnostics request the hypothetical direction. Set
`woodbury_warmup_steps=N` to change the duration or `0` to disable it.
These calls count toward `steps`; a run of at most N calls makes no updates.

Poisson's existing LR schedule advances concurrently; it does not restart
after curvature warmup. `warmup_steps` still controls the separate linear
LR ramp. With defaults (100 frozen calls and a 200-step ramp), updates begin
partway through that ramp. Use `warmup_steps=0 --cosine-decay 1` to apply
the configured constant LR immediately after the frozen period. Both training
and validation logs report the applied LR, including zero during the freeze.
The Gnome `eps` control does not apply; `weight_decay` is also unused for
Woodbury. Use `woodbury_damping` for its floor. Sharing Gnome's trust radius
does not guarantee the same optimal learning rate or training performance.

With `fixed_dataset`, each auxiliary batch is freshly sampled without
replacement from the fixed interior points and each boundary edge. Random
curvature probes are refreshed each step. Without this flag, both main and
auxiliary coordinates resample from the domain. PDE/BC weighting and the
surrogate's normalization are the same as Gnome's.

Float32 is supported. `float64` uses the existing all-double CPU mode.
`gram_float64` instead retains float32 samples for float32 parameters and
promotes the inner products and reconstruction as well as the small solve.
On MPS this promoted work runs on CPU; with ordinary float32 MPS parameters,
only the small solve and diagnostic eigenspectrum move to CPU. CPU paths
were exercised here; accelerator tests skip when the devices are unavailable.

## Python API

```python
from gnome.experimental import WoodburyGGN

opt = WoodburyGGN(model.parameters(), lr=1e-3, beta1=0.0, beta2=0.99,
                  damping=1e-3, history_tol=0.01, curvature_warmup_steps=100,
                  trust_radius=1.0)

# Same independent-graph (prediction, target) closures as Gnome:
loss = opt.step(main_closure, aux_closure)
```

For callers that already compute the scalar surrogate themselves:

```python
opt.zero_grad()
loss, surrogate = model_loss_and_ggn_surrogate()
params = [p for group in opt.param_groups for p in group["params"]
          if p.requires_grad]
surrogate_grads = torch.autograd.grad(surrogate, params, retain_graph=True)
loss.backward()
opt.step(surrogate_grads=surrogate_grads)
```

Incoming samples must already have the intended GGN scale. The optimizer
adds no batch/probe normalization. Samples are cloned on insertion, so the
caller can reuse buffers safely. A `None` sample adds no history entry;
`opt.step()` uses existing curvature. With no samples, the direction is
`X / lambda`, where `lambda = max(damping, ||X||/R)` with the trust region
enabled and simply `damping` otherwise.
An explicit zero tensor is a zero curvature sample and advances the history.
The closure API handles an unused auxiliary parameter as a zero derivative,
matching Gnome. Ordinary `.grad` buffers are neither read nor overwritten
in closure mode. Standard `state_dict`/`load_state_dict` preserve the history,
Gram precision, first moment, and step counters.
Checkpoint resume also preserves progress through the frozen period and the
previous selected damping. Checkpoints predating adaptive damping retain
fixed-damping behavior when loaded.
`group["lr"]` remains scheduler-controlled; `group["applied_lr"]` reports the
effective LR of the last optimizer call.

## Diagnostics and validation

Use the existing `JsonlDiagnostics` sink or any callable accepting a record.
Poisson writes `runs/poisson_pinn/<run_id>.diag.jsonl`. Each selected tensor
logs `history_length`, `history_max`, `damping`, `grad_norm`, `x_norm`,
`direction_norm`, `direction_ratio` (`||P|| / ||X||`), `cosine`,
`k_eig_min`, `k_eig_max`, and `condition_number` of the damped history system.
`damping` now records the selected lambda; `damping_floor` is its configured
minimum. `trust_budget`, `trust_ratio` (`||P||/R`), and `trust_binding` describe
the active constraint. `trust_iterations` counts candidate norm evaluations
in the current step; `trust_fallbacks` counts cumulative direct-norm retries.
The condition number uses the selected damping. When disabled, the budget
and ratio are null and selected damping equals the floor.
`solve_failures` counts cumulative failed Cholesky factorizations;
`jitter_events` counts subsequent numerical-jitter fallbacks. Eigenvalues
are diagnostic only and are computed only when a record is due. They do not
participate in preconditioning.
Diagnostic `lr` is the applied LR; `scheduled_lr` retains the scheduler's
value, and `curvature_warmup` marks a frozen step. Direction norms describe
the hypothetical preconditioned direction even when its applied LR is zero.

Run the mathematical checks and existing Poisson regressions with:

```bash
uv run --extra dev python -m pytest -q tests/test_woodbury.py tests/test_poisson_engdw.py
```

The tests cover explicit dense inverse equivalence, matrix/flattened
equivalence, EMA weights and history sizing, incremental Gram append/drop,
momentum, empty/one-sample limits, parameter-group independence, checkpoint
round trips, generic-solve/jitter fallbacks, and Poisson integration with both
architectures, sampling modes, and float32/float64 parameters. On a tiny
fixed MLP, enumerating **all** Rademacher probes recovers each exact GGN block
without Monte Carlo uncertainty. A 200-step deterministic regression run
with curvature warmup and adaptive damping explicitly disabled
checks every update against an explicit empirical-block inverse and reduces
loss by more than 95%, with no solve failures or jitter. These establish the
implementation's correctness, not superiority to other optimizers.
Additional tests check the exact freeze/unfreeze boundary, history/momentum
accumulation, applied-LR logging, and checkpoint resume with an LR scheduler.
Adaptive-damping tests compare against independently bisected full-matrix
solves, including empty and full-rank histories, zero/orthogonal/in-span
gradients, repeated samples, binding/slack constraints, both precisions, and
checkpoint resume with momentum and independent parameter-group radii.
