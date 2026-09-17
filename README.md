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

## Running the experiments

Each experiment is a module under `experiments/`, grouped into `pinns`, `resnets`, and
`transformers` where applicable, and run from the repo root. Every run streams an
append-only JSONL log to `runs/<experiment>/<run_id>.jsonl` (see below).

**Regression** (runs on CPU/MPS/GPU; needs the `experiments` extra for plots):

```bash
uv run python -m experiments.ols_regression --optimizer gnome --lr 0.1
```

`--optimizer` is `gnome`, `soap`, or `adamw`. This is the pedagogical example: Gnome reaches
the closed-form least-squares solution at a fixed LR, while the baselines stall unless given
a decay schedule.

**PINNs** (need the `experiments` extra; references are generated or downloaded
automatically on first run):

```bash
uv run python -m experiments.pinns.poisson_pinn                --optimizer gnome --steps 50000
uv run python -m experiments.pinns.burgers_pinn                --optimizer gnome --steps 75000
uv run python -m experiments.pinns.kuramoto_sivashinsky_pinn   --optimizer gnome --steps 100000
uv run python -m experiments.pinns.navier_stokes_pinn          --optimizer gnome --steps 200000
```

Burgers uses a float64 Cole–Hopf reference for `rel_l2`. Pass `--float64` for
double-precision training and inference as well. The additional `rel_l2_jaxpi`
metric uses the original JAXPI dataset. See the
[reference documentation](experiments/reference_solutions/README.md) for
accuracy checks and differences from historical spectral-reference results.

Each takes `--optimizer gnome|soap|adamw`. **Schedule protocol:** every optimizer, Gnome
included, gets the same linear-warmup + cosine-decay schedule, so the comparison is over the
update rule and not over who was handed a schedule. `--cosine-decay` is the final-LR
fraction and defaults to `0.0` — decay all the way to zero. Pass `--cosine-decay 1.0` for
warmup-then-constant, which is where Gnome is happiest on MSE since its step already
self-anneals. So a plain run already gets the standard cosine decay:

```bash
uv run python -m experiments.pinns.poisson_pinn --optimizer soap --steps 50000
```

Set `--cosine-decay 1` to disable decay entirely (raw SOAP/AdamW), or e.g. `0.1` to decay to
10% of the peak LR.

**Reaction with ENGD-W:** `experiments.pinns.reaction_pinn` also accepts
`--optimizer engdw`, using an exact damped Gauss–Newton solve in sample space:

```bash
uv run -m experiments.pinns.reaction_pinn --optimizer engdw --rho 5 \
    --engdw-line-search --engdw-damping 1e-6 --engdw-chunk 128 --steps 1000
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
one LR schedule spans the sum of the stage budgets, which replace `--steps`.
The example uses warmup then constant LR; omit `--cosine-decay 1` for cosine decay
over the full run. Either stage budget can be zero for a control run.

In staged mode, the default range is `[1, 200]` and default target is `200`.
The original `(t, x)` benchmark remains in `experiments.pinns.convection_pinn`,
with its beta default of `40`. Family runs write to `runs/convection_family_pinn/`. Logs identify each stage; `rel_l2` and PDE/IC/BC validation losses always
refer to the target. Additional `rel_l2_beta_*` metrics score the endpoints and
midpoint, including at the stage transition and final step. Training loss in stage 1
averages over the family, while stage 2 training loss covers only the target.

**WikiText-103 GPT** (needs the `llm` extra + a GPU; downloads the dataset on first run):

```bash
uv run python -m experiments.transformers.wikitext_gpt --optimizer gnome_hutchinson --max-steps 30000
```

`--optimizer` is `gnome_hutchinson`, `gnome_fisher`, `soap`, or `adamw`. Cross-entropy
gradients don't vanish at the optimum, so here *every* optimizer (Gnome included) uses a
cosine schedule.

**CUB-200-2011 bird classification** (needs the `vision` extra):

```bash
uv run --extra vision -m experiments.resnets.cub200 \
    --optimizer gnome_hutchinson --download --save-checkpoint
```

The [official dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)
contains 200 bird species. `--download` retrieves the approximately 1.2 GB archive
from [CaltechDATA](https://data.caltech.edu/records/65de6-vp158), verifies its MD5,
and extracts it under `experiments/data`. For existing data, `--data-dir` accepts
the `CUB_200_2011` directory or its parent; subsequent runs need no download flag.

The experiment uses the official train/test split and only species labels.
It trains from scratch with the shared GELU ResNet trunk, a 7×7 stride-2 stem
and max-pool for larger images, and GroupNorm by default. Defaults are ResNet-18,
224×224 crops, batch size 32, and 100 epochs. CUB owns its augmentation policy:
random resized crops (area fraction 0.2–1.0, aspect ratio 3/4–4/3), horizontal
flips (p=0.5), and color jitter (brightness/contrast 0.2, saturation 0.1, no hue
shift). Crops resize to `--image-size` (224 by default). Evaluation uses
resize/center-crop. `--no-augment` disables all three random transformations.
The full policy is recorded in run metadata.
All images use fixed ImageNet channel normalization.

To fine-tune an ImageNet-pretrained ResNet-18:

```bash
uv run --extra vision -m experiments.resnets.cub200 \
    --optimizer gnome_hutchinson --arch resnet18_pretrained --save-checkpoint
```

`--arch` is an alias for `--model`. This option uses torchvision's standard
ReLU ResNet-18 with `IMAGENET1K_V1` weights, retains BatchNorm by default, and
replaces the classifier with a fresh 200-class head. All layers are fine-tuned.
Weights download automatically once into the PyTorch cache, independently of
the dataset's `--download` flag. Explicit `--norm gn` converts BatchNorm to
GroupNorm, copying affine parameters and discarding running statistics.
Run metadata records the weight source and normalization. Existing model
choices continue to train from scratch with GroupNorm by default.

Choose `gnome_hutchinson`, `gnome_fisher`, `soap`, or `adamw`; all receive the
same warmup/cosine schedule. All support GroupNorm or BatchNorm. Gnome draws its auxiliary
samples from the intact main batch. Its auxiliary forward uses batch statistics
without updating BatchNorm's running mean, variance, or batch counter; only the
main forward updates those buffers. CUB owns its optimizer factory and
convolution-factor partition settings, with no dependency on the CIFAR-100 module.
`--merge-dims` selects `none`, `greedy`, `spatial`, or `patch` partitions.
`--model resnet12` selects a smaller model; `--workers 4` enables parallel image
loading. `--max-steps 10 --warmup-steps 0` is useful for a short diagnostic run.

Runs write to `runs/cub200/`. Each epoch reports test cross-entropy, top-1 and
top-5 as `val` metrics; metadata explicitly identifies the official test split.
`train_eval_loss`, `train_eval_top1`, and `train_eval_top5` score the training
set with the same deterministic evaluation transform, exposing the generalization
gap. Accuracy values are fractions. `training_seconds` excludes evaluation time;
the final record also includes total wall time. `--save-checkpoint` saves the
final model, optimizer state, and configuration alongside the JSONL. The test
set does not select the saved checkpoint or control early stopping.

Pass `--help` to any experiment for the full argument list.

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
