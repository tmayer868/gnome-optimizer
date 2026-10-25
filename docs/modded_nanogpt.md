# Modded-NanoGPT optimization experiment

This experiment ports the [Track 3 optimization benchmark](https://github.com/KellerJordan/modded-nanogpt/tree/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization).
The pinned reference is revision `4ea6b937337a4889b8cfe3f38a93d120048d8f71`.
The unmodified source and MIT license are in
`experiments/transformers/modded_nanogpt_reference/`.

The published tuned Muon + auxiliary AdamW baseline (result #36) reports
3.2787 validation loss at 3,250 steps over ten trials. Our `optimizer=muon`
uses its architecture, optimizer settings, LR schedule, and reference
zero-projection initialization described below.
The port has not yet been validated with a full CUDA training run; a local
plumbing check is not a reproduction of that result.

## Protocol

| Setting | Benchmark default |
|---|---|
| Data | `kjj0/fineweb10B-gpt2`, official pretokenized FineWeb shards |
| Vocabulary | 50,304 output classes |
| Model | 12 layers, width 768, 6 attention heads of width 128 |
| Context | 1,024 tokens |
| Global training batch | 524,288 tokens = 512 sequences |
| Validation | Fixed first 10,485,760 tokens from the validation stream |
| Target | Validation cross-entropy below 3.28 |
| Schedule | Constant LR, then linear decay over the final 70%; no warmup |
| Default budget | 3,250 steps, fixed in advance |

The architecture uses trainable RMSNorm gains, Q/K RMS normalization,
half-truncated rotary embeddings, attention scale 0.12, ReLU-squared MLPs,
untied input/output embeddings, and the reference soft-capped logits.
Weights use the reference initialization below; biases start at zero.
The portable model's forward loss and gradients are checked against the saved
upstream architecture with identical weights loaded into both models.

All optimizers see the same sequential batches. Training drops shard tails
using the reference boundary rule, then advances to the next shard. It never
wraps: insufficient tokens fail before allocating the model. Validation always
uses the same slice and includes partial evaluation microbatches with token
weighting. There is no validation-based early stopping or checkpoint selection.

## Gnome comparisons and eligibility

The official track prohibits additional forward/backward passes. Existing
Gnome computes an additional curvature pass, so **Gnome results from this runner
do not qualify for that leaderboard**. They are useful comparisons on the same
model/data/global batch, with extra compute explicitly recorded. Changing the
optimizer to meet that rule is separate research work.

The official statistical criterion is
`(3.28 - mean_final_loss) * sqrt(number_of_trials) >= 0.004`, using
non-cherry-picked trials and a common stopping step. A single threshold crossing
does not establish an accepted result. Submissions also require all source code
inline in the upstream logfile format. This runner instead produces the project's
JSONL logs plus a source ZIP, and marks every run `submission_ready=false`.

Gnome Fisher/Hutchinson and SOAP apply to all model parameters by default.
Gnome can instead use an internal AdamW group for input embeddings, as described
below. Gnome and
SOAP spend their first call initializing preconditioners; that call counts toward
the step and data budgets. Their starting hyperparameters are exploratory,
not claimed to be tuned benchmark baselines. The standalone AdamW option is
also an exploratory baseline, not a reproduction of an accepted Adam record.

### Gnome with AdamW embeddings

Enable `embedding_optimizer=adamw` with either Gnome optimizer to assign only
`model.embed.weight` to an AdamW group inside the same Gnome instance. The output
head is untied and stays with Gnome, as do the hidden layers and RMSNorm gains.
`embedding_optimizer=same` is the default and preserves the all-Gnome behavior.
SOAP and the other optimizer choices retain their existing parameter assignments.

The AdamW group has independent settings: `embed_lr` (default 0.001),
`embed_beta1` (0.9), `embed_beta2` (0.999), `embed_eps` (1e-8), and
`embed_weight_decay` (0). These are exploratory settings, not tuned results.
For Muon, `embed_lr` keeps its existing default of 0.7; its other auxiliary
AdamW settings are unchanged.

Both groups use the same accumulated token-mean main gradient. Only Gnome
parameters receive auxiliary curvature gradients and allocate preconditioner
state. AdamW embeddings start updating on step one, while Gnome's first call
initializes its basis. The warmup/cooldown multiplier applies to both groups'
individual learning rates. Logs include the embedding settings and scheduled
`embedding_lr`; the single Gnome checkpoint contains both groups' state.

Use `fp32_embedding` for CUDA BF16 runs so small embedding updates accumulate
in FP32 weights. AdamW's moments use FP32 regardless of parameter storage, but
FP32 moments alone do not prevent rounding when writing back to BF16 weights.

For example, add these flags to a Gnome run:

```bash
--embedding-optimizer adamw --embed-lr 0.001 --fp32-embedding
```

Muon applies only to hidden matrices, with the reference's separate AdamW groups
for token embeddings, output weights, and all 1D parameters. It retains summed
token gradients, as upstream does. Other optimizers use token-mean gradients,
matching Gnome's curvature normalization. This choice is logged. Clipping is off
by default; `max_grad_norm` enables it for any optimizer.

## Local MPS development

Offline check (no optional dependencies or data downloads):

```bash
# Edit experiments/configs/transformers/modded_nanogpt.py: preset='dev', synthetic=True, optimizer='gnome_hutchinson'
uv run -m experiments.transformers.modded_nanogpt
```

Automatic device selection prefers CUDA, then MPS, then CPU. `device=mps`,
`device=cuda`, and `device=cpu` force a device and fail if unavailable.
Auto precision is FP32 on MPS/CPU and BF16 on CUDA hardware supporting it.
Use `dtype=float32` to force FP32. BF16 mode mirrors upstream: the embedding
is BF16, other parameters remain FP32, and linear operations cast their weights
to activation precision; logits and cross-entropy are FP32. MPS/CPU Muon uses
FP32 Newton–Schulz iterations, while CUDA uses the reference BF16 iterations.
Gnome/SOAP already handle unsupported MPS eigendecompositions on CPU.

For Gnome BF16 runs, `fp32_embedding` keeps the embedding parameter and its
optimizer state in FP32, casting the gathered activations to BF16 for the model.
Initialization is drawn in FP32. Without this flag, embedding weights are then
cast to the compute dtype; with the flag, the FP32 draws are retained.
This lets small updates accumulate
instead of rounding away on every BF16 parameter update. Other parameter
tensors were already stored in FP32. The option is explicit so the default
Muon reference path retains its original parameter precision.

### Reference initialization

All optimizer choices use the initialization strategy from the pinned
`modded_nanogpt_reference/train_gpt_simple.py`:

- Token embeddings: zero-mean Gaussian with std 1.
- Attention and MLP output projections, and the language-model output head: zero.
- Other linear weights: zero-mean Gaussian with std `sqrt(0.33 / fan_in)`.
- Biases: zero. RMSNorm gains: one.
- The output head remains untied. `head_init_std` defaults to zero and can
  override just that matrix; the residual output projections remain zero.

This restores the benchmark strategy in place of the previous WikiText-style
initialization. No additional flag is needed. Initial logits are zero, giving
uniform token probabilities; gradients reach earlier layers as the zero
projections begin to learn. Gnome's update normalization preserves zero updates
without dividing by zero. Tests compare the FP32 initialization against the
saved reference initialization code and exercise the first training updates.

Logs record `initialization=reference`, `head_init_std`, `residual_init_std=0`,
and `reference_initialization_matches=true` for the default zero head. A nonzero
head override sets the match flag to false. The flag describes the initialization
strategy, not bitwise equality across devices or dtypes: draws are made in FP32
before an optional embedding cast. The architecture/data-budget match is recorded
separately, and Gnome's extra backward pass still falls outside the track's rules.
For Gnome BF16 runs, `fp32_embedding` remains useful for preserving small updates.

`preset=dev` uses two layers of width 128, context 64, 256 training tokens per
step, 512 validation tokens, and five steps. Synthetic mode also shrinks the
vocabulary to 256. These runs are explicitly marked as having a different
configuration from the benchmark. They verify code paths, not benchmark quality.

For a local run with real data:

```bash
# Edit experiments/configs/transformers/modded_nanogpt.py: preset='dev', download_shards=1, microbatch_size=1, steps=20
uv run --extra nanogpt -m experiments.transformers.modded_nanogpt
```

That download includes one training shard and one validation shard (about
200 MB each). Downloads require an explicit flag; existing shards are checked
for a valid header and expected byte count. Data defaults to the git-ignored
`experiments/data/fineweb10B/` directory. Header/size checks do not independently
authenticate custom data: use the official downloader for comparisons.

## Thunder Compute run

On a provisioned CUDA machine, copy/clone this project and install `uv`. No
Thunder-specific API or cloud provisioning is required. Use one GPU; this runner
does not implement multi-rank training, and rejects multi-rank `torchrun`.
Gradient accumulation preserves the global batch on one GPU. Choose enough
VRAM for model/optimizer state; reducing microbatch size only reduces activation
memory. Gnome/SOAP also store curvature factors and rotated states.

```bash
uv sync --extra nanogpt

# Cache 20 training shards (2B tokens) and the validation shard.
# Edit experiments/configs/transformers/modded_nanogpt.py: download_shards=20, download_only=True
uv run --extra nanogpt -m experiments.transformers.modded_nanogpt

# Muon reference optimizer settings and initialization.
# Edit experiments/configs/transformers/modded_nanogpt.py: optimizer='muon', steps=3250, microbatch_size=4, seed=0, save_checkpoint=True
uv run --extra nanogpt -m experiments.transformers.modded_nanogpt

# Gnome research comparison, including its additional curvature pass.
# Edit experiments/configs/transformers/modded_nanogpt.py: optimizer='gnome_hutchinson', steps=3250, microbatch_size=4, aux_batch_size=1, lr=0.001, fp32_embedding=True, seed=0, save_checkpoint=True
uv run --extra nanogpt -m experiments.transformers.modded_nanogpt
```

The upstream quickstart pins PyTorch 2.11 and reports a compiled A100 issue with
2.10. Use 2.11 for an initial upstream-version comparison (for example add
`--with 'torch==2.11.0'` to `uv run`). Compilation is opt-in via `compile`,
CUDA only. This port runs eagerly by default. Changing microbatch size preserves
tokens and the objective, although floating-point accumulation order can affect
numerical results. There is no claim of bitwise identity across devices.

Use multiple seeds and the same preselected step budget when assessing target
attainment. Longer runs may need more shards; request them with
`download_shards=N` (up to 103). An existing upstream cache can be used with
`data_dir=/path/to/fineweb10B`.

## Learning-rate schedules

`lr_schedule=linear` (default) preserves the reference schedule: optional
linear warmup, a constant learning rate, and linear decay during the final
`cooldown_frac` of training.

For linear warmup followed immediately by cosine decay, use:

```bash
--lr 0.002 --warmup-steps 30 --lr-schedule cosine --min-lr-frac 0.10
```

Warmup uses `lr / warmup_steps` on the first update and reaches `lr` on update
30 in this example. Cosine decay spans the remaining updates, reaching exactly
`0.10 * lr = 0.0002` on the last training update. `cooldown_frac` is ignored
in cosine mode; there is no constant-LR plateau. Without warmup, cosine starts
at the configured LR (a single-update run uses the final floor).

`min_lr_frac` defaults to 0.1 and must be between 0 and 1. It affects only
cosine decay, not warmup or the linear schedule. Every optimizer group receives
the same multiplier relative to its own initial LR. Schedule settings are saved
in the run configuration, and training logs contain the applied learning rate.

## Artifacts and controls

Runs write to `runs/modded_nanogpt/` and appear in `experiments.viewer`.
`scripts/sweep.py` accepts the experiment name `modded_nanogpt`.

Each JSONL captures the full resolved configuration, actual device/precision,
upstream revision, data filenames/counts, source hashes, token loss, validation
loss/perplexity, main/auxiliary token counts, and synchronized training time.
Training time excludes validation and model setup, but includes data loading,
optimizer work, and compilation encountered during training. The recorded
forward/backward token ratio describes additional input-token processing, not
an exact FLOP measurement. Gnome's auxiliary sequences are sampled without
replacement from the intact main batch using a separate RNG.

A companion source ZIP captures the experiment and optimizer code, including
uncommitted local changes. `save_checkpoint` saves final model/optimizer states
and configuration; automatic training resume is not implemented.
Use `--help` for learning rates, damping, momentum, preconditioner size/frequency,
trust radius, evaluation cadence, data paths, and architecture overrides.
Architecture/data-budget overrides are reflected in
`reference_configuration_matches`; this flag is not a certification of submission
eligibility or a verification of data provenance.
