# Otto Group product classification

```bash
# Edit experiments/configs/tabular/otto.py: optimizer='gnome', save_checkpoint=True
uv run python -u -m experiments.tabular.otto
# Edit experiments/configs/tabular/otto.py: optimizer='adamw'
uv run python -u -m experiments.tabular.otto
# Edit experiments/configs/tabular/otto.py: optimizer='soap'
uv run python -u -m experiments.tabular.otto
# Edit experiments/configs/tabular/otto.py: optimizer='muon'
uv run python -u -m experiments.tabular.otto
```

No additional dependencies or Kaggle credentials are needed. By default the
runner downloads the labeled training set from
[OpenML dataset 45548, version 1](https://www.openml.org/d/45548), caches it in
`experiments/data/otto/otto.arff`, and verifies its pinned SHA-256. The source is
the [Otto Group Product Classification Challenge](https://www.kaggle.com/competitions/otto-group-product-classification-challenge):
61,878 labeled products, 93 count features, and nine product classes.

For a local Kaggle download, place `train.csv`, `train.csv.gz`, or `train.csv.zip`
in `experiments/data/otto`, or pass an explicit path:

```bash
# Edit experiments/configs/tabular/otto.py: optimizer='gnome', data_file='/path/to/train.csv'
uv run python -u -m experiments.tabular.otto
```

A ZIP must contain exactly one `train.csv` (possibly within a subdirectory).
Dense OpenML `.arff` files are also accepted via `data_file`. Features are
selected in `feat_1` through `feat_93` order by column name; `id` is excluded.
Targets `Class_1` through `Class_9` map to integers 0–8. The loader rejects
unlabeled files, duplicate IDs, malformed columns, and negative/nonfinite counts.

## Evaluation and preprocessing

The Kaggle competition test file is unlabeled. This experiment instead splits
the **labeled training rows** into stratified 70% training, 15% validation, and
15% held-out test, using `split_seed=42`. Per-class training and validation
counts are rounded down; test receives the remainder. On the complete dataset
the counts are 43,310 / 9,277 / 9,291. These are local held-out metrics, not Kaggle
leaderboard scores. `seed` controls model initialization and training randomness
independently of the split.

By default, count features receive `log1p`, followed by standardization using
only training means and population standard deviations. Constant features use
a scale of one. `log1p=False` keeps raw counts before standardization. Neither
preprocessing mode uses validation or test statistics.

Cross-entropy (multiclass log loss) is the primary metric; top-1 accuracy is also
reported. Validation runs after each epoch, including a shortened final epoch.
The held-out test runs once on the final model. There is no initial validation,
early stopping, or test-based checkpoint selection.

## Model and optimizers

Otto and Covertype share `experiments/tabular/common.py`. The default model is a
423,177-parameter residual MLP: 93 inputs, width 256, three LayerNorm/GELU residual
blocks, final LayerNorm, and a nine-class head. `width` and `blocks` resize it.

| Optimizer | Updates | Default learning rate |
| --- | --- | --- |
| `gnome` / `gnome_hutchinson` | Hutchinson Gnome on all parameters | 0.001 |
| `gnome_fisher` | Fisher-sampling Gnome on all parameters | 0.001 |
| `adamw` | AdamW on all parameters | 0.001 |
| `soap` | Repository SOAP on all parameters | 0.003 |
| `muon` | Muon on residual-block matrices; AdamW on input/output layers, biases and norms | 0.02 / 0.001 |

These are starting learning rates, not tuned results. `lr` overrides the main
rate; `aux_lr` controls Muon's companion AdamW. Shared defaults are 20 epochs,
batch size 512, 30 warmup steps, cosine decay to `min_lr_frac=0.1`, and weight
decay 0.01 on all parameters. Gnome uses a 128-row auxiliary subset sampled
without replacement from the main batch, `trust_radius=0.15`, and
`norm_free=False`. Main-batch ordering uses a separate random generator so
Gnome's extra random probes do not change the example order.

All Covertype optimizer controls are available, including `eps`, `beta1`,
`beta2`, `shampoo_beta`, `trust_region`, `precondition_frequency`,
`precondition_1d`, `norm_free`, and `max_grad_norm`. The core optimizers
are reused unchanged. See [Covertype's optimizer details](covertype.md).

Device selection prefers CUDA, then MPS, then CPU; `device` overrides it.
Model computation and parameters use FP32. As in NanoGPT/Covertype, Muon's
internal matrix orthogonalization uses BF16 on CUDA and FP32 on MPS/CPU.

## Logs, checkpoints, and short runs

```bash
# Edit experiments/configs/tabular/otto.py: optimizer='gnome', device='mps', max_steps=100, warmup_steps=10, log_every=10
uv run python -u -m experiments.tabular.otto
```

`max_steps` shortens the training budget and cosine schedule. For a check shorter
than the default warmup, set `warmup_steps=0` or another smaller value.

Logs go to `runs/otto/*.jsonl`: configuration and dataset hash, per-step loss/LR
and synchronized timing, validation loss/top-1, and final held-out test metrics.
The final record includes `best_val_loss`, `final_val_loss`, and `test_loss`.
Top-1 values are fractions. Training time excludes evaluation.

`save_checkpoint` writes a `.pt` beside the log with model and optimizer states,
configuration, split indices, and preprocessing statistics. To reconstruct
inputs, select the 93 ordered features, apply `torch.log1p` when
`checkpoint['preprocessing']['log1p']` is true, then subtract its `mean` and
divide by its `std`. Checkpoints save the final model; there is no resume CLI.
