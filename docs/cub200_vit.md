# CUB-200 vision transformer

This experiment compares AdamW, SOAP, Gnome, and Muon on CUB-200-2011.
All run settings live in [`experiments/configs/transformers/cub200_vit.py`](../experiments/configs/transformers/cub200_vit.py).
The file defines a `CONFIG` dictionary and supports Python expressions,
comments, and scientific notation such as `1e-3` and `1e-8`.
The runner accepts only `--config`; there are no hyperparameter CLI overrides.

```bash
uv sync --extra vision
uv run --extra vision -m experiments.transformers.cub200_vit
```

Before the first run, set `data.data_dir` to the extracted `CUB_200_2011`
directory or its parent. Alternatively, set `data.download` to `True` to
download and extract the checksummed archive using the existing CUB data helper.
Downloading is disabled in the supplied config.

To choose an optimizer, edit the top-level `optimizer` field to `adamw`,
`soap`, `gnome`, or `muon`. Each has a separate settings block under
`optimizers`, so switching algorithms preserves their individual learning rates
and other hyperparameters. The supplied values are starting settings, not tuned
CUB results.

For separate runs, copy the complete Python config file, edit it, and select it:

```bash
uv run --extra vision -m experiments.transformers.cub200_vit --config experiments/configs/transformers/my_cub_run.py
```

The default config is located relative to the runner. Paths *inside* a config
are relative to the working directory; normally run from the repository root.
The runner executes the selected file and reads its `CONFIG` dictionary.
Missing, unknown, and invalid settings produce an error instead of
silently falling back to another config. Every optimizer block is validated,
including those that are not selected.

## Model and data

The default is a randomly initialized ViT with 224-pixel images, 16-pixel
patches, 12 encoder blocks, width 192, 3 attention heads, and MLP width 768
(ViT-Tiny-sized). It uses torchvision's
[`VisionTransformer`](https://docs.pytorch.org/vision/stable/models/vision_transformer.html):
patch projection, learned position and class-token embeddings, pre-LayerNorm
attention/MLP blocks, and a 200-class head. All layers train in FP32. No
pretrained weights are loaded.

`model` controls resolution, patch size, depth, width, heads, MLP width,
dropout, attention dropout, LayerNorm epsilon, and head initialization standard
deviation. The head uses a normal initialization with standard deviation 0.02
by default, allowing backbone gradients on the first pass. Set `head_init_std`
to zero for torchvision's zero-head convention. Other initialization follows
torchvision. The number of output classes is fixed by CUB.

The experiment reuses the existing CUB metadata reader and official split:
5,994 training images and 5,794 test images. It uses species labels only.
Training augmentation is random resized crop, horizontal flip, and color jitter;
evaluation uses resize and center crop. `data` contains their parameters and
the normalization statistics. Disable `data.augment` to use deterministic
evaluation transforms during training as well.

Shuffle, augmentation, and Gnome auxiliary selection use separate random
streams, including when `training.workers` is zero. The seed controls model
initialization and these data streams. This isolates data selection and crops
from additional random draws used by Gnome's curvature probes.

## Optimizer routing

| Selected optimizer | Parameters it updates |
| --- | --- |
| AdamW | All parameters |
| SOAP | All parameters |
| Gnome | All parameters |
| Muon | Attention QKV/output and MLP weight matrices within encoder blocks |
| Muon's companion AdamW | Patch projection, position and class-token embeddings, all normalization parameters, all biases, and the output head |

The Muon and AdamW sets are disjoint and cover the model. The experiment reuses
the repository's portable Muon implementation from the NanoGPT benchmark.
Its momentum setting is `optimizers.muon.mu`. Configure its companion through
`optimizers.muon.adamw`, including an independent learning rate, betas, epsilon,
and weight decay. Both optimizers are stepped and checkpointed together.

Gnome uses `cce_hutchinson` by default; change `optimizers.gnome.loss` to `cce`
for Fisher sampling. Its main closure uses the entire training batch and its
auxiliary closure uses a fresh subset without replacement, capped at the current
batch size. `aux_batch_size` therefore also works on the final partial batch.
Set `optimizers.gnome.use_soap_basis` to `True` to build the basis from the
main gradient while keeping the surrogate curvature estimate; it defaults to `False`.
Set `trust_radius` to `None` to disable it. The runner uses the current local
Gnome implementation without modifying it.

## Training and outputs

`training` controls seed, device, epochs, training/evaluation batch sizes,
workers, optional total-step cap, warmup, cosine floor, and optional global
gradient clipping. `device` may be `auto`, `cpu`, `mps`, or `cuda`.
`max_steps: None` trains all epochs; `max_grad_norm: None` disables clipping.
When enabled, clipping covers all parameters, including both Muon groups.

All optimizers use linear warmup followed by cosine decay. Muon and its companion
AdamW receive the same multiplier while retaining their distinct base learning
rates. `min_lr_frac: 1.0` means constant LR after warmup. Warmup must be shorter
than the run; for a tiny smoke run, set `warmup_steps` to zero. Gnome and SOAP
use their existing first-call preconditioner initialization behavior.

`output` controls the runs directory, console logging interval (`0` disables
periodic step prints), evaluation frequency in epochs, final checkpoint saving,
and quiet mode. Evaluation always runs after the final update, even when a
step cap stops training mid-epoch or the epoch does not match the evaluation
frequency. Loss, top-1, and top-5 are sample-weighted, including partial batches.
Accuracy values are fractions in `[0, 1]`.

Runs write `runs/cub200_vit/<run_id>.jsonl`, including a snapshot of the full
config, resolved device and data path, actual parameter routing, per-step loss
and learning rate(s), and evaluation metrics. As in the existing CUB runner,
the official **test** split is recorded using `val` events for viewer
compatibility; it is explicitly identified in metadata. There is no separate
validation split or best-test checkpoint selection.

If `save_checkpoint` is `True`, the final sibling `.pt` file contains the model,
all optimizer states, config, metadata, step, and epoch. It can reconstruct the
model via `build_model(checkpoint["config"])` and load its state dict; it is not
an automatic resume interface. Non-finite training/evaluation metrics mark the
run incomplete and use the project's divergence exit code.

```bash
uv run -m experiments.viewer
uv run python scripts/plot_runs.py -e cub200_vit -m top1
```
