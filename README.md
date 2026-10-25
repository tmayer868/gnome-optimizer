# Gnome

**G**auss-**N**ewton **o**ptimizer via **m**atrix **e**igendecomposition — a second-order
PyTorch optimizer that extends [SOAP](https://arxiv.org/abs/2409.11321) into a true
Gauss-Newton method. It keeps SOAP's Kronecker-factored eigenbasis but builds it from an
unbiased estimate of the Generalized Gauss-Newton (GGN) matrix instead of the empirical
Fisher, and takes a diagonal Newton step in that basis instead of an Adam step. The result
is a step size that vanishes as the residual does — so on regression problems it reaches the
exact minimum at a *fixed* learning rate, with no decay schedule.

This README is about **running the code**. For how the method works, see
[`docs/method.md`](docs/method.md).

## Install

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Clone the repo and sync:

```bash
git clone https://github.com/tmayer868/gnome-optimizer
cd gnome-optimizer
uv sync                        # the optimizer only (torch + numpy)
```

The optimizer core needs only `torch` and `numpy`. The experiments need extra packages,
installed as optional-dependency groups:

| Extra | Command | For |
|---|---|---|
| *(base)* | `uv sync` | the `gnome` optimizer package |
| `experiments` | `uv sync --extra experiments` | the regression + PINN benchmarks (scipy, matplotlib) |
| `llm` | `uv sync --extra llm` | the WikiText GPT benchmark (datasets, transformers) |
| `nanogpt` | `uv sync --extra nanogpt` | the Modded-NanoGPT FineWeb benchmark (huggingface-hub) |
| `vision` | `uv sync --extra vision` | image classification, including CUB-200-2011 (torchvision, Pillow) |
| `dev` | `uv sync --extra dev` | tests + notebooks (pytest, jupyter) |

Requires Python ≥ 3.10. The code auto-selects a device: CUDA if available, else Apple MPS,
else CPU.

## Using the optimizer

Gnome is closure-based. Unlike a normal optimizer, it computes the loss *internally* (it
needs to build the curvature surrogate from the model outputs), so your closures return the
model's `(y_hat, y)` — **not** a scalar loss. You pass two: a `main_closure` over the full
batch, and an `aux_closure` over a small disjoint slice used to estimate curvature.

```python
import torch
from gnome import Gnome

model = ...  # any nn.Module
opt = Gnome(model.parameters(), lr=1e-2, loss="mse")

for x, y in loader:
    idx = torch.randperm(x.shape[0])[:10]      # small auxiliary slice for curvature

    def main_closure():
        return model(x), y                     # (y_hat, y) on the full batch

    def aux_closure():
        return model(x[idx]), y[idx]           # (y_hat, y) on the K-sample aux slice

    loss = opt.step(main_closure, aux_closure) # returns the main loss
```

For binary or multi-label classification, pass raw logits and floating-point targets with
the same shape, and select the BCE-Hutchinson loss:

```python
opt = Gnome(model.parameters(), lr=1e-3, loss="bce_hutchinson")

def main_closure():
    return model(x), targets.float()       # logits and targets shaped [B, ...]

def aux_closure():
    return model(x_aux), targets_aux.float()
```

This mode uses binary cross-entropy with logits, summed over each sample's output
coordinates and averaged over the batch. Its curvature surrogate uses independent
Rademacher probes scaled by the square root of the Bernoulli output Hessian.

Key arguments:

- `loss` — `"mse"` (regression), `"bce_hutchinson"` (binary cross-entropy with logits and
  a Rademacher surrogate), `"cce"` (softmax cross-entropy, Fisher surrogate), or
  `"cce_hutchinson"` (softmax cross-entropy, lower-variance Rademacher surrogate).
- `lr` — Gnome owns no schedule; it is a stock `torch.optim.Optimizer`, so any
  `torch.optim.lr_scheduler` drives it as it would AdamW. On MSE the Gauss-Newton step
  self-anneals as the residual shrinks, so decay is optional there in a way it isn't for
  gradient-RMS methods — a warmup is still worth having while the eigenbasis is cold.
- `K` (auxiliary batch size) — the number of samples in the aux slice you pass to
  `aux_closure` (10 in the example above). Controls the curvature estimate's variance, not
  bias. It is *not* a constructor argument — Gnome reads K from the aux batch you provide.
- `eps` — curvature damping (larger → closer to gradient descent).
- `trust_radius` — relative L2 update budget:
  `trust_radius * sqrt(p.square().sum() + initial_sq_norm)`, before multiplying
  by the learning rate. The squared initialization norm is captured when the
  parameter first participates in an optimizer step and stored in its checkpoint.
  Entirely zero tensors use `initial_sq_norm = 1.0`. Pass `None` to disable
  the trust constraint.

The reference norm is calculated once per parameter and reused. Older optimizer
checkpoints without it acquire the reference on the next step. For zero-initialized
tensors, the fallback is the expected squared norm of a length-N vector with
Xavier variance `1/N`, assuming a hypothetical square layer. It adds no
hyperparameter or layer lookup and does not change the initial parameter values.

For multi-block losses (e.g. PINNs — a PDE residual plus boundary/initial terms),
`gnome.stack_residuals([r1, r2, ...])` folds the blocks into a single flat residual vector
whose `mean(·²)` is the weighted loss, and hands the MSE surrogate the right per-block GGN.
See the PINN experiments and [`docs/method.md`](docs/method.md) §8 for details.

### AdamW parameter groups inside Gnome

Set `update_rule="adamw"` on a group to update it with AdamW directly inside
Gnome. For example, to use AdamW for a model's input embedding:

```python
embedding = model.embed.weight
opt = Gnome([
    {"params": [p for p in model.parameters() if p is not embedding],
     "update_rule": "gnome"},
    {"params": [embedding], "update_rule": "adamw", "lr": 1e-3,
     "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0},
], lr=1e-3, loss="cce_hutchinson")
```

The existing `opt.step(main_closure, aux_closure)` API stays the same, including
microbatch accumulation. Both groups receive the main gradients; only Gnome
groups receive curvature gradients and preconditioner state. AdamW groups use
bias-corrected gradient and squared-gradient EMAs, decoupled weight decay, and
update immediately on the first step. Gnome's trust radius, curvature damping,
coordinate clipping, and curvature diagnostics apply only to Gnome groups.
Optional global main-gradient clipping covers both rules.

AdamW moments are FP32 (FP64 for double parameters), including after checkpoint
loading. Keep parameters in FP32 too if small updates would round away in BF16.
All groups share one optimizer checkpoint and work with standard LR schedulers.
Each group inherits constructor settings unless overridden, so specify AdamW's
`betas` and `eps` explicitly when they should differ from Gnome's. Parameters,
including tied weights, must appear only once. Choose each group's update rule
at construction. An AdamW-only Gnome optimizer does not need an auxiliary closure.

## Running the experiments

Edit the experiment's Python `CONFIG` dictionary under
[`experiments/configs/`](experiments/configs/README.md), then run its module.
The config directory mirrors the runners: `pinns/`, `resnets/`, `transformers/`,
and `tabular/`, with root-level experiments at the config root. Scientific
notation and Python expressions are supported. Commands now accept only
`--config path/to/custom.py` and `--help`; settings formerly passed as flags
are dictionary keys with underscores. The examples below indicate which
entries to edit before running.

Each experiment is a module under `experiments/`, grouped into `pinns`, `resnets`, and
`transformers` where applicable, and run from the repo root. Every run streams an
append-only JSONL log to `runs/<experiment>/<run_id>.jsonl` (see below).

**Regression** (runs on CPU/MPS/GPU; needs the `experiments` extra for plots):

```bash
# Edit experiments/configs/ols_regression.py: optimizer='gnome', lr=0.1
uv run python -m experiments.ols_regression
```

`optimizer` is `gnome`, `soap`, or `adamw`. This is the pedagogical example: Gnome reaches
the closed-form least-squares solution at a fixed LR, while the baselines stall unless given
a decay schedule.

**PINNs** (need the `experiments` extra; references are generated or downloaded
automatically on first run):

```bash
# Edit experiments/configs/pinns/poisson_pinn.py: optimizer='gnome', steps=50000
uv run python -m experiments.pinns.poisson_pinn
# Edit experiments/configs/pinns/burgers_pinn.py: optimizer='gnome', steps=75000
uv run python -m experiments.pinns.burgers_pinn
# Edit experiments/configs/pinns/kuramoto_sivashinsky_pinn.py: optimizer='gnome', steps=100000
uv run python -m experiments.pinns.kuramoto_sivashinsky_pinn
# Edit experiments/configs/pinns/navier_stokes_pinn.py: optimizer='gnome', steps=200000
uv run python -m experiments.pinns.navier_stokes_pinn
```

Burgers uses a float64 Cole–Hopf reference for `rel_l2`. Pass `float64` for
double-precision training and inference as well. The additional `rel_l2_jaxpi`
metric uses the original JAXPI dataset. See the
[reference documentation](experiments/reference_solutions/README.md) for
accuracy checks and differences from historical spectral-reference results.

Allen–Cahn can switch precision during a run while preserving optimizer state
and the learning-rate schedule:

```bash
# Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', float64_after=30000
uv run -m experiments.pinns.allen_cahn_pinn
```

This runs 30,000 steps in FP32 and the remaining steps in FP64. On Apple MPS,
the switch moves training to CPU. Use `float64` to start in FP64; the two
options are mutually exclusive. The switch is recorded in the run log.

Allen–Cahn also supports `fixed_dataset` with every optimizer:

```bash
# Edit experiments/configs/pinns/allen_cahn_pinn.py: optimizer='gnome', fixed_dataset=True
uv run -m experiments.pinns.allen_cahn_pinn
```

This reuses one randomly sampled PDE/IC/BC dataset throughout training.
Gnome draws fresh auxiliary subsets without replacement within each block;
`aux_frac` must be in `(0,1]`, and auxiliary sizes are capped at the main
block sizes. Validation residual points still resample. Fixed coordinates
are preserved when switching precision with `float64_after`.

Each takes `optimizer=gnome|soap|adamw`. **Schedule protocol:** every optimizer, Gnome
included, gets the same linear-warmup + cosine-decay schedule, so the comparison is over the
update rule and not over who was handed a schedule. `cosine_decay` is the final-LR
fraction and defaults to `0.0` — decay all the way to zero. Pass `cosine_decay=1.0` for
warmup-then-constant, which is where Gnome is happiest on MSE since its step already
self-anneals. So a plain run already gets the standard cosine decay:

```bash
# Edit experiments/configs/pinns/poisson_pinn.py: optimizer='soap', steps=50000
uv run python -m experiments.pinns.poisson_pinn
```

Set `cosine_decay=1` to disable decay entirely (raw SOAP/AdamW), or e.g. `0.1` to decay to
10% of the peak LR.

**Poisson with fixed training points:** add `fixed_dataset` to reuse one
sampled PDE/boundary batch throughout training with any optimizer:

```bash
# Edit experiments/configs/pinns/poisson_pinn.py: optimizer='gnome', fixed_dataset=True
uv run -m experiments.pinns.poisson_pinn
```

Gnome and WoodburyGGN draw fresh auxiliary subsets from that fixed batch each step, uniformly
without replacement within the PDE points and each boundary edge. `aux_frac`
sets the subset sizes and must be in `(0,1]` for this mode. PDE/BC loss weights
are preserved, and curvature probes remain random.
Validation residual points still resample, and relative L2 uses the analytical
reference grid. Omitting the flag retains per-step training resampling.

**Poisson with ENGD-W:** the 2D Poisson benchmark accepts the shared exact
sample-space Gauss–Newton baseline:

```bash
# Edit experiments/configs/pinns/poisson_pinn.py: optimizer='engdw', n_pde=128, n_bc_per_edge=16, engdw_line_search=True, steps=1000
uv run -m experiments.pinns.poisson_pinn
```

ENGD-W automatically uses float64 on CUDA or CPU (MPS falls back to CPU),
with equal-weight PDE and boundary MSEs. `engdw_damping` defaults to `1e-6`;
`engdw_momentum` defaults to zero. It uses a fixed `lr` or optional same-batch
grid line search, without warmup, cosine decay, or weight decay. Points resample
each step; `fixed_dataset` (or the existing `engdw_fixed`) reuses the training set while validation stays
independent. Both `arch=mlp` and `arch=fused` are supported. Start with small
collocation batches: the dense sample Gram has `n_pde + 4*n_bc_per_edge` rows.
`engdw_chunk` limits differentiation workspace, not that matrix's size.

**Burgers with ENGD-W:** the same solver and options are available for the
nonlinear Burgers PDE, with equal-weight PDE, initial-condition, and boundary MSEs:

```bash
# Edit experiments/configs/pinns/burgers_pinn.py: optimizer='engdw', n_pde=128, n_ic=32, n_bc=16, engdw_line_search=True, steps=1000
uv run --extra experiments -m experiments.pinns.burgers_pinn
```

This automatically uses float64 on CUDA or CPU (CPU on an MPS machine).
`engdw_damping=1e-6`, `engdw_momentum=0`, `engdw_chunk=32`, and optional
`engdw_fixed` match Poisson. `fixed_dataset` works for every Burgers optimizer;
Gnome samples fresh auxiliary subsets of the fixed PDE/IC/BC blocks. Without
line search, `lr` is constant; warmup, cosine decay, and weight decay are ignored
by ENGD-W. The dense solve has `n_pde + n_ic + 2*n_bc` rows because both boundaries
share one BC loss block. Chunking reduces Jacobian workspace, not the solve size.
Reference evaluation is unchanged; `no_jaxpi_ref` skips the optional external
reference download. Final-step evaluation runs even when `log_every=0`.

**Poisson with experimental WoodburyGGN:** `optimizer=woodbury` uses the
existing Gnome surrogate with a finite EMA history of raw matrix gradients
and an exact damped Woodbury solve per parameter tensor:

```bash
# Edit experiments/configs/pinns/poisson_pinn.py: optimizer='woodbury', fixed_dataset=True, hidden=8, depth=3, n_pde=32, n_bc_per_edge=4, aux_frac=0.5, steps=300, lr=0.001, woodbury_damping=0.1, beta2=0.9, warmup_steps=0, cosine_decay=1.0, float64=True, diagnostics_every=50, log_every=50
uv run -m experiments.pinns.poisson_pinn
```

Import it as `from gnome.experimental import WoodburyGGN`. The defaults are
`beta1=0`, `beta2=0.99`, `damping=1e-3`, and `history_tol=0.01` (459 samples
per tensor). `max_history` overrides the derived window; `gram_float64`
promotes inner products and the small solve. `trust_region=1` adaptively
raises damping above `woodbury_damping` to bound each direction's L2 norm,
using Gnome's radius `trust_region * sqrt(||W||² + 1)`. Set `trust_region=0`
for fixed damping. There is no coordinate clipping or weight decay.
The optimizer holds parameters fixed for the first 100 calls
while collecting curvature; `woodbury_warmup_steps` changes this duration
(`0` disables). The separate `warmup_steps` LR ramp advances concurrently.
See [the API, diagnostics, and validation notes](docs/woodbury_ggn.md)
before scaling up the initial stability run.

**Reaction with ENGD-W:** `experiments.pinns.reaction_pinn` also accepts
`optimizer=engdw`, using an exact damped Gauss–Newton solve in sample space:

```bash
# Edit experiments/configs/pinns/reaction_pinn.py: optimizer='engdw', rho=5.0, engdw_line_search=True, engdw_damping=1e-06, engdw_chunk=128, steps=1000
uv run -m experiments.pinns.reaction_pinn
```

This baseline uses float64 and fixed damping, with either a fixed learning rate
or a same-batch grid line search. See [the comparison protocol](docs/reaction_engdw.md)
for residual normalization, settings matching the Gnome run, and validation.

**Convection across a parameter range, then fine-tuning:**

```bash
uv run -m experiments.pinns.convection_family_pinn --optimizer gnome \
    --beta-range 1 200 --beta 200 \
    --stage1-steps 30000 --stage2-steps 20000 --cosine-decay 1
```

Both stage arguments enable a model with inputs `(t, x, beta)`, with beta
normalized to `[-1, 1]` using the fixed range bounds. Stage 1 samples beta uniformly
across the range for every PDE, initial-condition, and boundary-condition point.
Stage 2 fixes beta at the target. Optimizer state carries across the transition;
one LR schedule spans the sum of the stage budgets, which replace `steps`.
The example uses warmup then constant LR; omit `cosine_decay=1` for cosine decay
over the full run. Either stage budget can be zero for a control run.

In staged mode, the default range is `[1, 200]` and default target is `200`.
The original `(t, x)` benchmark remains in `experiments.pinns.convection_pinn`,
with its beta default of `40`. Family runs write to `runs/convection_family_pinn/`. Logs identify each stage; `rel_l2` and PDE/IC/BC validation losses always
refer to the target. Additional `rel_l2_beta_*` metrics score the endpoints and
midpoint, including at the stage transition and final step. Training loss in stage 1
averages over the family, while stage 2 training loss covers only the target.

**2D complex Ginzburg–Landau, trained globally in time:**

```bash
uv run --extra experiments -m experiments.pinns.ginzburg_landau \
    --optimizer gnome --time-interval 0.2 --depth 5 --hidden 128 \
    --n-pde 8192 --n-ic 2048 --steps 100000
```

One two-output shared MLP solves the entire physical interval `[time_start,T]`
at once. `--time-interval` sets the upper bound (default `1.0`); `--time-start`
sets the lower bound (default `0.0`), with `0 <= time_start < T <= 1`.
For example, `--time-start 0.4 --time-interval 0.8` trains on `[0.4,0.8]`,
using the cached reference at t=0.4 as its initial condition. Both spatial
directions use the shared hard periodic embedding. The objective is the sum
of the four block MSEs `pde_u`, `pde_v`, `ic_u`, `ic_v`. Choose `gnome`, `soap`,
`adamw`, `adam`, or `engdw`. `width` aliases `hidden`; depth counts affine
layers, so depth 5 means four hidden layers and the output layer.

`ic_weight` sets a fixed weight on both IC blocks:
`loss = pde + ic_weight * ic`. It defaults to `1.0`; try `ic_weight=10` or
`ic_weight=100` to emphasize the initial condition. All optimizers use this
same weighted objective, including Gnome's curvature pass and ENGD-W's solve.
Logged `ic`, `ic_u`, and `ic_v` remain unweighted; `ic_weighted` and `objective`
report the weighted IC contribution and total validation objective.

A separately generated complex128 Fourier/ETDRK4 reference is cached over
`[0,1]` and reused across horizons. Chunked evaluation reports combined-complex
`rel_l2` plus `rel_l2_u` and `rel_l2_v`. `log_every` controls residual logging;
`eval_every` controls the more expensive relative errors (default 200).
The final step always reports both. Runs also save field comparisons at five
times, amplitude and wrapped-phase errors, and training curves. Use `--no-plots`
for a diagnostic run. See [benchmark details](docs/ginzburg_landau.md) for
reference validation, precision options, and a horizon sweep.

**WikiText-103 GPT** (needs the `llm` extra + a GPU; downloads the dataset on first run):

```bash
# Edit experiments/configs/transformers/wikitext_gpt.py: optimizer='gnome_hutchinson', max_steps=30000
uv run python -m experiments.transformers.wikitext_gpt
```

`optimizer` is `gnome_hutchinson`, `gnome_fisher`, `soap`, or `adamw`. Cross-entropy
gradients don't vanish at the optimum, so here *every* optimizer (Gnome included) uses a
cosine schedule.

**Modded-NanoGPT optimization benchmark** (Track 3; automatic CUDA/MPS/CPU selection):

```bash
# Tiny offline development check, using MPS when available.
# Edit experiments/configs/transformers/modded_nanogpt.py: preset='dev', synthetic=True
uv run -m experiments.transformers.modded_nanogpt

# Full benchmark model/data on one CUDA GPU, with gradient accumulation.
# Edit experiments/configs/transformers/modded_nanogpt.py: optimizer='muon', download_shards=20, steps=3250, save_checkpoint=True
uv run --extra nanogpt -m experiments.transformers.modded_nanogpt
```

Choose `muon`, `gnome_hutchinson`, `gnome_fisher`, `soap`, or `adamw`.
The full preset preserves the upstream model, FineWeb token stream, global
batch, and validation budget. Initialization follows WikiText: std 0.02
embeddings/linear weights, depth-scaled residual projections, zero biases,
and unit normalization gains. Gnome uses an additional curvature pass, so its
results are research comparisons and **do not qualify for the official track's
single-pass rule**. The development preset is also distinct from the benchmark.
See [protocol, Thunder Compute commands, and validation notes](docs/modded_nanogpt.md).

**CUB-200-2011 bird classification** (needs the `vision` extra):

```bash
# Edit experiments/configs/resnets/cub200.py: optimizer='gnome_hutchinson', download=True, save_checkpoint=True
uv run --extra vision -m experiments.resnets.cub200
```

The [official dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)
contains 200 bird species. `download` retrieves the approximately 1.2 GB archive
from [CaltechDATA](https://data.caltech.edu/records/65de6-vp158), verifies its MD5,
and extracts it under `experiments/data`. For existing data, `data_dir` accepts
the `CUB_200_2011` directory or its parent; subsequent runs need no download flag.

The experiment uses the official train/test split and only species labels.
It trains from scratch with the shared GELU ResNet trunk, a 7×7 stride-2 stem
and max-pool for larger images, and GroupNorm by default. Defaults are ResNet-18,
224×224 crops, batch size 32, and 100 epochs. CUB owns its augmentation policy:
random resized crops (area fraction 0.2–1.0, aspect ratio 3/4–4/3), horizontal
flips (p=0.5), and color jitter (brightness/contrast 0.2, saturation 0.1, no hue
shift). Crops resize to `image_size` (224 by default). Evaluation uses
resize/center-crop. `no_augment` disables all three random transformations.
The full policy is recorded in run metadata.
All images use fixed ImageNet channel normalization.

To fine-tune an ImageNet-pretrained ResNet-18:

```bash
# Edit experiments/configs/resnets/cub200.py: optimizer='gnome_hutchinson', model='resnet18_pretrained', save_checkpoint=True
uv run --extra vision -m experiments.resnets.cub200
```

`arch` is an alias for `model`. This option uses torchvision's standard
ReLU ResNet-18 with `IMAGENET1K_V1` weights, retains BatchNorm by default, and
replaces the classifier with a fresh 200-class head. All layers are fine-tuned.
Weights download automatically once into the PyTorch cache, independently of
the dataset's `download` flag. Explicit `norm=gn` converts BatchNorm to
GroupNorm, copying affine parameters and discarding running statistics.
Run metadata records the weight source and normalization. Existing model
choices continue to train from scratch with GroupNorm by default.

Choose `gnome_hutchinson`, `gnome_fisher`, `soap`, or `adamw`; all receive the
same warmup/cosine schedule. All support GroupNorm or BatchNorm. Gnome draws its auxiliary
samples from the intact main batch. Its auxiliary forward uses batch statistics
without updating BatchNorm's running mean, variance, or batch counter; only the
main forward updates those buffers. CUB owns its optimizer factory and
convolution-factor partition settings, with no dependency on the CIFAR-100 module.
`merge_dims` selects `none`, `greedy`, `spatial`, or `patch` partitions.
`model=resnet12` selects a smaller model; `workers=4` enables parallel image
loading. `max_steps=10 --warmup-steps 0` is useful for a short diagnostic run.

Runs write to `runs/cub200/`. Each epoch reports test cross-entropy, top-1 and
top-5 as `val` metrics; metadata explicitly identifies the official test split.
`train_eval_loss`, `train_eval_top1`, and `train_eval_top5` score the training
set with the same deterministic evaluation transform, exposing the generalization
gap. Accuracy values are fractions. `training_seconds` excludes evaluation time;
the final record also includes total wall time. `save_checkpoint` saves the
final model, optimizer state, and configuration alongside the JSONL. The test
set does not select the saved checkpoint or control early stopping.

**Covertype tabular classification** (no additional dependencies):

```bash
# Edit experiments/configs/tabular/covertype.py: optimizer='gnome', save_checkpoint=True
uv run python -u -m experiments.tabular.covertype
```

Choices are `gnome` (Hutchinson), `gnome_hutchinson`, `gnome_fisher`, `adamw`,
`soap`, and `muon`. A width-256 residual MLP trains with a fixed stratified
70/15/15 split, training-only feature standardization, and warmup + cosine decay.
The dataset downloads once and is cached locally. Device selection prefers CUDA,
then MPS, then CPU. See [docs/covertype.md](docs/covertype.md) for commands,
optimizer routing, and the split protocol.

**Otto Group product classification** (same tabular model and optimizer choices):

```bash
# Edit experiments/configs/tabular/otto.py: optimizer='gnome', save_checkpoint=True
uv run python -u -m experiments.tabular.otto
```

Automatically downloads the labeled data from OpenML, or accepts Kaggle
`train.csv` via `data_file`. The 93 count features receive `log1p` and
training-only standardization; the nine-class model reports cross-entropy and
accuracy on a fixed 70/15/15 split. See [docs/otto.md](docs/otto.md).

Pass `--help` to any experiment for the full argument list.

### CUB-200 vision transformer (Python configuration)

The ViT experiment supports AdamW, SOAP, Gnome, and Muon. Edit
[`experiments/configs/transformers/cub200_vit.py`](experiments/configs/transformers/cub200_vit.py)
to select the optimizer and configure the model, dataset, training, and outputs:

```bash
uv run --extra vision -m experiments.transformers.cub200_vit
```

Set `data.data_dir` to your CUB directory or set `data.download` to `True`.
The default is a ViT-Tiny-sized model trained from scratch. Muon updates the
encoder's attention/MLP matrices, with AdamW for embeddings, normalization,
biases, and the output head. Each optimizer has its own settings dictionary.
The only runner option is `--config path/to/run.py` for a different complete
config. See [the experiment documentation](docs/cub200_vit.md) for details.

## Run artifacts

A run writes one append-only JSONL file: a `meta` line (run identity, hyperparameters,
environment), one `train`/`val` line per logged step with a nested `metrics` dict, and a
terminal `end` line. It is safe to `tail -f` while training, and a killed run only truncates
its last line. Read runs back for analysis:

```python
from experiments.common import load_run

run = load_run("runs/poisson_pinn/<run_id>.jsonl")
steps, rel_l2 = run.series("val", "rel_l2")   # a metric over steps
run.final("val", "rel_l2")                     # last logged value
run.best("val", "rel_l2")                      # best (min by default)
```

## Repository layout

```
gnome/           the optimizer package (Gnome, stack_residuals)
experiments/     runnable benchmarks
  common/        shared run logger, device selection, LR schedules
  baselines/     the SOAP baseline
docs/            methodology write-ups
runs/            JSONL run artifacts (git-ignored)
```

## Documentation

- [`docs/method.md`](docs/method.md) — the algorithm: SOAP → Gnome, the GGN surrogate
  construction, per-loss output-Hessian square roots, and the multi-block (PINN) extension.
- [`docs/variance.md`](docs/variance.md) — variance of the curvature surrogate as a function
  of the auxiliary batch size.

## License

MIT — see [`LICENSE`](LICENSE).
