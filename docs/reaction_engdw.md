# ENGD-W on the logistic reaction PINN

`experiments.pinns.reaction_pinn` supports `optimizer=engdw`. It uses the
shared solver in `experiments/baselines/engdw.py`, extracted from the existing
Poisson-5D experiment. The update is equation (5) of
[Guzmán-Cordero et al., *Improving Energy Natural Gradient Descent through
Woodbury, Momentum, and Randomization*](https://arxiv.org/html/2505.12149v2#S3):

\[
\delta = J^T(JJ^T+\lambda I)^{-1}r,\qquad
\theta \leftarrow \theta-\eta\delta.
\]

This is exact damped Gauss–Newton for the sampled residual, up to numerical
linear algebra error with the default `engdw_momentum=0`. There is no
sketching, weight decay, or learning-rate schedule. The cited paper permits nonlinear PDEs through
linearization, but does not report this reaction benchmark; this implementation
is an application of the method, not a reproduction of a published reaction result.

For `u_t - rho*u*(1-u)`, differentiating through both `u_t` and the reaction
term gives `d_theta(u_t) - rho*(1-2*u)*d_theta(u)`. The implementation uses
`torch.func` for the input derivative and `vmap(jacrev(single_residual))` for
the parameter Jacobian. All four reaction architectures and both embeddings
use the same residual contract.

Each residual block is divided by the square root of its own sample count:

\[
r = [r_{PDE}/\sqrt{N_{PDE}};\ r_{IC}/\sqrt{N_{IC}};\ r_{BC}/\sqrt{N_{BC}}].
\]

Thus `sum(r**2)` equals Gnome's training loss, the sum of block MSEs. ENGD-W's
energy is half that value; its damping is defined relative to `J.T @ J` under
this normalization. Using `stack_residuals` directly without dividing by
`sqrt(N_total)` would change the effective damping. With the periodic
embedding the BC block is omitted, as it is for Gnome.

## Run against the supplied Gnome configuration

The supplied local run
`runs/reaction_pinn/reaction_pinn_gnome_seed0_1789241936.jsonl` records:

| Setting | Value |
|---|---|
| Reaction coefficient | 5 |
| Architecture | MLP, width 50, depth 5 (four hidden layers) |
| Embedding / conditions | Raw `(t,x)`, soft IC and periodic BC |
| Parameters | 7,851, with weights and biases combined per layer |
| Samples per step | 4,000 PDE, 256 IC, 256 BC |
| Precision / device | float64 / CPU |
| Gnome training budget | 200,000 steps, 1,298.58 seconds wall time |
| Final relative L2 | 3.57876828039852e-6 |

An ENGD-W starting command with those model and sampling settings is:

```bash
# Edit experiments/configs/pinns/reaction_pinn.py: optimizer='engdw', rho=5.0, seed=0, arch='mlp', hidden=50, depth=5, embed='none', n_pde=4000, n_ic=256, n_bc=256, engdw_damping=1e-06, engdw_line_search=True, engdw_chunk=128, steps=1000, log_every=20
uv run -m experiments.pinns.reaction_pinn
```

ENGD-W automatically uses float64, on CUDA if available and otherwise CPU.
`engdw_line_search` selects the lowest same-batch loss among 13 log-spaced
step sizes from 0.001 to 1. It overrides `lr`. This is the existing Poisson
grid search, not Armijo backtracking: zero is not a candidate and it does not
guarantee descent if every trial is bad. Omitting the flag uses a fixed `lr`.
The learning rate and damping require reaction-specific tuning. In an initial
seed-0 pilot, fixed `lr=0.05 --engdw-damping 1e-6` overshot severely.

`engdw_momentum` enables SPRING-style momentum by centering the regularizer
at `mu * d_prev`:

\[
\min_d \|Jd-r\|^2 + \lambda\|d-\mu d_{prev}\|^2,\qquad
d = \mu d_{prev} + J^T(JJ^T+\lambda I)^{-1}(r-\mu Jd_{prev}).
\]

The initial direction history is zero. Each step stores the new direction
before multiplication by the fixed or line-searched step size; this is not an
EMA of natural-gradient directions. `momentum=0` preserves the original solve.
The coefficient must be in `[0, 1)` and is logged as `opt.momentum`. To compare
initial settings, add `engdw_momentum=0.9`, `0.95`, or `0.99` to the command
above and retain a zero-momentum control. These are starting experiments, not
tuned settings. The optimizer's `reset_state()` clears direction history when
reusing an optimizer after resetting model weights; constructing a new optimizer
also starts with zero history.

`engdw_chunk` controls the sample chunk used during Jacobian construction.
It does not approximate curvature or shrink the final Jacobian/kernel. At
the supplied settings, those float64 matrices alone occupy approximately
270 MiB (`4512 x 7851`) and 155 MiB (`4512 x 4512`), before factorization and
temporary storage. Match accuracy against wall time as well as steps; these
steps are much more expensive than Gnome's.

By default, both methods resample collocation points every step. Add
`engdw_fixed` to sample ENGD-W's PDE, IC, and BC training points once and
reuse them throughout training, including Jacobian construction and line
search. The Jacobian is still recomputed at the current parameters each step.
Validation residual points remain independently sampled and the analytic
evaluation grid is unchanged. Metadata records `sampling: "fixed"` for these
runs. The flag requires `optimizer=engdw`.

A common seed matches
initialization here, but does not guarantee identical later batches because
Gnome also draws auxiliary samples and probes. ENGD-W uses the entire main
batch for curvature and evaluates all line-search candidates on that batch.
The independent analytic grid supplies `rel_l2`; it does not select step sizes.

Run logs record actual optimizer settings, the selected learning rate, and
cumulative `training_seconds` excluding validation and logging. The terminal
record retains total wall time. The supplied older Gnome log has total wall
time only. Short implementation checks live under `runs/engdw_validation/`;
they are not convergence comparisons. The 100-step seed-0 sanity run with
line search, damping `1e-6`, and chunk size 128 reached relative L2
`0.0109865` in 89.09 seconds wall time on CPU. Its artifact is
`runs/engdw_validation/reaction_pinn/reaction_pinn_engdw_seed0_1789243917.jsonl`.
This run verifies end-to-end progress; it is not tuned or run to convergence.

## Verification

```bash
uv run --extra dev -m pytest tests/test_reaction_engdw.py -q
```

Tests compare the functional residual and every Jacobian row against the
existing autograd PDE/IC/BC path for all architecture/embedding combinations;
verify the analytic logistic solution; compare the Woodbury update with an
independent augmented least-squares solve; exercise chunking and line search;
and check training/logging for all reaction optimizers and the shared Poisson
solver.
