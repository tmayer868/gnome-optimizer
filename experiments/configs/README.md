# Experiment configuration

Each runnable module under `experiments/` has a Python config at the same
relative path here. For example:

| Runner | Config |
| --- | --- |
| `experiments.viewer` | `experiments/configs/viewer.py` |
| `experiments.pinns.allen_cahn_pinn` | `experiments/configs/pinns/allen_cahn_pinn.py` |
| `experiments.pinns.heat5d_pinn` (four spatial dimensions plus time) | `experiments/configs/pinns/heat5d_pinn.py` |
| `experiments.resnets.cub200` | `experiments/configs/resnets/cub200.py` |
| `experiments.transformers.cub200_vit` | `experiments/configs/transformers/cub200_vit.py` |
| `experiments.tabular.covertype` | `experiments/configs/tabular/covertype.py` |

The reference generator also has a matching config. Model, baseline, data-loading, and shared helper modules do not run
experiments themselves and do not need configs. The vendored NanoGPT reference
snapshot remains unchanged; use the project's `modded_nanogpt` runner to train.

Edit the file's `CONFIG` dictionary and run the module without setting flags:

```bash
uv run --extra experiments -m experiments.pinns.poisson_pinn
uv run --extra vision -m experiments.resnets.cub200
uv run --extra vision -m experiments.transformers.cub200_vit
```

PINN configs keep experiment settings (steps, architecture, collocation counts,
precision, and logging) at the top level. Each supported optimizer has its own
block under `optimizers`, selected by the top-level `optimizer` key. Learning
rates, schedules, auxiliary sampling, and solver-specific settings live in those
blocks, so switching optimizers also switches their hyperparameters:

```python
CONFIG = {
    # ...other settings from the experiment's supplied config...
    "optimizer": "gnome",
    "fixed_dataset": True,
    "steps": 50_000,
    "hidden": 64,
    "depth": 5,
    "optimizers": {
        "gnome": {"lr": 1e-3, "eps": 1e-6},  # ...remaining Gnome settings...
        "soap": {"lr": 5e-4},                # ...remaining SOAP settings...
        "adamw": {"lr": 1e-4},               # ...remaining AdamW settings...
        # ...other supported optimizer blocks from the supplied config...
    },
}
```

For example, edit `CONFIG["optimizers"]["gnome"]["lr"]` to tune only Gnome.
Allen–Cahn, Burgers 2D, convection, reaction, and Heat 5D enable
`CONFIG["optimizers"]["gnome"]["partition_residuals"] = True`. Each auxiliary
PDE/IC/BC residual block gets a separate curvature contribution, computed in one
batched backward. Partition sizes follow the actual auxiliary batch on every
step, omit inactive boundary blocks, and include both boundary-value and
boundary-derivative residuals for Allen–Cahn. Set this option to `False` to use
one surrogate over the entire stacked residual. Other optimizers ignore it;
run metadata records whether partitioning is active.

Reaction and convection support top-level `fixed_dataset=True` for every
optimizer. Gnome draws fresh auxiliary subsets of the fixed main dataset;
validation remains independent. Their existing `optimizers["engdw"]["engdw_fixed"]`
option still enables fixed sampling for ENGD-W, even when `fixed_dataset=False`.
Existing field names are retained, including `trust_region`, `beta1`, `beta2`,
and solver-prefixed names such as `engdw_damping`. The `adamw+lbfgs` block
contains both the AdamW phase settings and the `lbfgs_*` refinement settings.
Each block exposes the settings that the runner already allowed for that
optimizer; fixed optimizer defaults remain in the runner.

The ViT also uses named optimizer blocks, along with model/data/training groups.
Other configs use flat keys matching the former argument names, with underscores
instead of dashes. All config files support scientific notation, comments, and
expressions. The Allen–Cahn reference generator has no optimizer and stays flat.

Categorical cross-entropy experiments expose `use_soap_basis=False` for Gnome.
Set it to `True` to build the basis from the main loss gradient while retaining
the surrogate-based curvature estimate. For the ViT, this setting is nested at
`CONFIG["optimizers"]["gnome"]["use_soap_basis"]`; other CCE runners use a
top-level key. The NanoGPT Kaggle config inherits it from the base config.

Keep the full dictionary when copying a config; the abbreviated example above
shows syntax only. `None` retains automatic selection where the old runner
supported it, such as optimizer-dependent defaults in NanoGPT and tabular
experiments. Boolean values are `True`/`False`; negative names
remain literal (`no_augment=True` disables augmentation in the ResNet CUB runner).
The old `--arch` alias for ResNet CUB is the `model` dictionary key.

To use a different file:

```bash
uv run --extra experiments -m experiments.pinns.poisson_pinn --config /path/to/my_poisson.py
```

`--config` and `--help` are the only experiment CLI options. Unknown keys,
missing settings, wrong types, and invalid choices are rejected. Existing
experiment-specific checks still run after loading. Defaults are located
relative to the experiment module, independent of the working directory;
relative data/output paths inside the config use the working directory.

For Python callers and tests, ConfigParser-based runners also accept a
dictionary of overrides: `poisson_pinn.parse_args({"steps": 10, "lr": 1e-3})`.
This merges with that experiment's config and applies the same validation.
PINNs also accept nested overrides, for example
`poisson_pinn.parse_args({"optimizer": "soap", "optimizers": {"soap": {"lr": 5e-4}}})`.
PINN files must use the nested layout, including all supplied optimizer blocks;
the flat Python override API remains available for existing callers.
The ViT continues to take its full nested config through `train(config)`.

`scripts/sweep.py` merges each run's overrides into its base config and saves
the resulting complete file under `runs/sweeps/<sweep_id>/`. It launches the
runner with `--config` pointing to that snapshot. Nested overrides work for
the PINNs and ViT, and both `True` and `False` are preserved. For PINN sweeps,
flat optimizer overrides such as `lr=1e-3` are written into the selected
optimizer's block; nested overrides can customize multiple blocks explicitly.
`--dry-run` previews the
sweep without creating files. `scripts/run_kaggle_nanogpt.sh` uses
`transformers/modded_nanogpt_kaggle.py`.

When adding a flat-config experiment, add its defaults here, register field
types/choices in `experiments/common/config_schema.py`, and construct
`ConfigParser(__file__)` in the runner. Keep conditional defaults and checks
with the experiment's parsing function.
