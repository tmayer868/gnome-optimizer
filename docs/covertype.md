# Covertype optimizer benchmark

```bash
# Edit experiments/configs/tabular/covertype.py: optimizer='gnome', save_checkpoint=True
uv run python -u -m experiments.tabular.covertype
# Edit experiments/configs/tabular/covertype.py: optimizer='adamw'
uv run python -u -m experiments.tabular.covertype
# Edit experiments/configs/tabular/covertype.py: optimizer='soap'
uv run python -u -m experiments.tabular.covertype
# Edit experiments/configs/tabular/covertype.py: optimizer='muon'
uv run python -u -m experiments.tabular.covertype
```

No extra dependencies are required beyond the project's Torch and NumPy. The first
run downloads the approximately 11 MB compressed dataset to
`experiments/data/covertype`; subsequent runs reuse it. The archive's SHA-256 is
verified before parsing. Downloads print a start message; loading, training, and
evaluation print flushed progress messages.

## Data and model

[UCI Covertype](https://archive.ics.uci.edu/dataset/31/covertype), contributed by
Jock A. Blackard (1998), contains 581,012 rows, 54 features and seven forest-cover
classes. Data is obtained from the same archive mirror and checksum used by
[scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_covtype.py).
Labels are converted from 1–7 to 0–6 for cross-entropy.

This experiment uses a **custom stratified 70/15/15 train/validation/test split**,
not the historical Covertype split. Each class is independently shuffled; train
and validation counts are rounded down and test receives the remainder.
`split_seed=42` fixes this split independently of `seed`, which controls model
initialization, batch ordering, and stochastic optimizer behavior. Comparisons
must keep the split seed fixed.

Only the first ten continuous features are standardized, using training means
and population standard deviations. Constant features use a scale of one. The
44 binary indicators remain unchanged. Validation runs after each epoch and after
a shortened final epoch; test runs once on the final model. No initial validation
pass, early stopping, or test-based model selection is performed.

The default MLP has an input projection to width 256, GELU, three residual blocks
(LayerNorm → Linear → GELU → Linear), a final LayerNorm and seven-class head.
Each residual branch is scaled by `1/sqrt(blocks)`. Layers use PyTorch's standard
initialization, identically seeded across optimizers. `width` and `blocks`
control size. Training uses FP32 on auto-selected CUDA, MPS, or CPU; `device`
can override selection. The reused Muon implementation performs its internal
matrix orthogonalization in BF16 on CUDA and FP32 on MPS/CPU, as in NanoGPT.

## Optimizers and schedule

| Choice | Parameter updates | Default LR |
| --- | --- | --- |
| `gnome` / `gnome_hutchinson` | Gnome with `cce_hutchinson` on all parameters | 0.001 |
| `gnome_fisher` | Gnome with `cce` on all parameters | 0.001 |
| `adamw` | AdamW on all parameters | 0.001 |
| `soap` | Repository SOAP on all parameters | 0.003 |
| `muon` | Repository Muon on residual-block matrices; AdamW on input/head matrices, biases and norms | 0.02 / 0.001 |

These learning rates are starting points, not tuned benchmark results. `lr`
overrides the primary optimizer LR; `aux_lr` sets Muon's companion AdamW LR.
`weight_decay` defaults to 0.01 for every parameter group, including biases and
norms. Gnome/SOAP use beta2=0.99; AdamW and Muon's companion use 0.999. Beta1=0.9,
epsilon=1e-8, shampoo-beta=0.99, and precondition-frequency=10 are configurable.
Muon momentum defaults to 0.95.

Gnome's auxiliary batch is sampled without replacement **from the main batch**,
defaulting to 128 of 512 rows; a smaller final batch uses all available rows.
Main batch ordering has its own random generator, so extra Gnome probes cannot
change the training-example order relative to other optimizers. Gnome receives
`trust_radius=0.15` and `norm_free=False` by default; `trust_region=0` passes
`None`. The runner uses the current core Gnome update unchanged. `precondition_1d`
enables vector rotation for Gnome/SOAP. `max_grad_norm` optionally clips main
gradients for every optimizer; it does not clip Gnome's auxiliary surrogate.

All optimizer groups share linear warmup for 30 steps followed by cosine decay
to `min_lr_frac=0.1`. Muon's two learning rates retain their ratio. Defaults are
20 epochs and batch size 512. `max_steps` truncates training and defines a shorter
schedule; use `warmup_steps=0` for very short checks. Gnome and SOAP retain their
native first-step preconditioner initialization behavior.

## Outputs and short checks

```bash
# Edit experiments/configs/tabular/covertype.py: optimizer='gnome', device='mps', max_steps=100, warmup_steps=10, log_every=10
uv run python -u -m experiments.tabular.covertype
```

Runs write `runs/covertype/*.jsonl`, including configuration, each step's loss/LR
and synchronized step time, epoch validation loss/top-1, and final test loss/top-1.
Accuracy is stored as a fraction. Training time excludes validation and test.
`save_checkpoint` stores the final model, all optimizer states, configuration,
split indices, and preprocessing statistics alongside the JSONL as `.pt`.
The checkpoint supports inspection/reconstruction; the runner has no resume CLI.

Compare both held-out accuracy versus steps and versus training time: Gnome's
extra curvature pass and the different preconditioners have different costs.
