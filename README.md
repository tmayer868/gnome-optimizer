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

model = ...  # any nn.Module. Under loss="cce", use LayerNorm/GroupNorm, not BatchNorm.
opt = Gnome(model.parameters(), lr=1e-2, loss="mse", aux_batch_size=10)

for x, y in loader:
    idx = torch.randperm(x.shape[0])[:10]      # small auxiliary slice for curvature

    def main_closure():
        return model(x), y                     # (y_hat, y) on the full batch

    def aux_closure():
        return model(x[idx]), y[idx]           # (y_hat, y) on the K-sample aux slice

    loss = opt.step(main_closure, aux_closure) # returns the main loss
```

Key arguments:

- `loss` — `"mse"` (regression), `"cce"` (softmax cross-entropy, Fisher surrogate), or
  `"cce_hutchinson"` (cross-entropy, lower-variance Rademacher surrogate).
- `lr` — for MSE, Gnome self-anneals, so a fixed learning rate is expected (no schedule).
- `aux_batch_size` (`K`) — samples used for the curvature estimate (default 10). Controls
  variance, not bias.
- `eps` — curvature damping (larger → closer to gradient descent).

For multi-block losses (e.g. PINNs — a PDE residual plus boundary/initial terms),
`gnome.stack_residuals([r1, r2, ...])` folds the blocks into a single flat residual vector
whose `mean(·²)` is the weighted loss, and hands the MSE surrogate the right per-block GGN.
See the PINN experiments and [`docs/method.md`](docs/method.md) §8 for details.

## Running the experiments

Each experiment is a module under `experiments/`, run from the repo root. Every run streams
an append-only JSONL log to `runs/<experiment>/<run_id>.jsonl` (see below).

**Regression** (runs on CPU/MPS/GPU; needs the `experiments` extra for plots):

```bash
uv run python -m experiments.ols_regression --optimizer gnome --lr 0.1
```

`--optimizer` is `gnome`, `soap`, or `adamw`. This is the pedagogical example: Gnome reaches
the closed-form least-squares solution at a fixed LR, while the baselines stall unless given
a decay schedule.

**PINNs** (need the `experiments` extra; the Burgers and Navier-Stokes references are
downloaded automatically on first run):

```bash
uv run python -m experiments.poisson_pinn                --optimizer gnome --steps 50000
uv run python -m experiments.burgers_pinn                --optimizer gnome --steps 75000
uv run python -m experiments.kuramoto_sivashinsky_pinn   --optimizer gnome --steps 100000
uv run python -m experiments.navier_stokes_pinn          --optimizer gnome --steps 200000
```

Each takes `--optimizer gnome|soap|adamw`. **Schedule protocol:** Gnome runs at a fixed
learning rate on these MSE losses (it self-anneals, so `--cosine-decay` is ignored for it).
The SOAP/AdamW baselines get the standard treatment through `--cosine-decay`, a final-LR
fraction that defaults to `0.0` — decay all the way to zero. So a plain baseline run already
gets the standard cosine decay:

```bash
uv run python -m experiments.poisson_pinn --optimizer soap --steps 50000
```

Set `--cosine-decay 1` to disable decay entirely (raw SOAP/AdamW), or e.g. `0.1` to decay to
10% of the peak LR.

**WikiText-103 GPT** (needs the `llm` extra + a GPU; downloads the dataset on first run):

```bash
uv run python -m experiments.wikitext_gpt --optimizer gnome_hutchinson --max-steps 30000
```

`--optimizer` is `gnome_hutchinson`, `gnome_fisher`, `soap`, or `adamw`. Cross-entropy
gradients don't vanish at the optimum, so here *every* optimizer (Gnome included) uses a
cosine schedule.

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
