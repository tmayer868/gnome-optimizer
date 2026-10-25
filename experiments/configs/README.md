# Experiment configuration

Each runnable module under `experiments/` has a Python config at the same
relative path here. For example:

| Runner | Config |
| --- | --- |
| `experiments.ols_regression` | `experiments/configs/ols_regression.py` |
| `experiments.pinns.allen_cahn_pinn` | `experiments/configs/pinns/allen_cahn_pinn.py` |
| `experiments.resnets.cub200` | `experiments/configs/resnets/cub200.py` |
| `experiments.transformers.cub200_vit` | `experiments/configs/transformers/cub200_vit.py` |
| `experiments.tabular.covertype` | `experiments/configs/tabular/covertype.py` |

The reference generator, GGN diagnostic, and run viewer also have matching
configs. Model, baseline, data-loading, and shared helper modules do not run
experiments themselves and do not need configs. The vendored NanoGPT reference
snapshot remains unchanged; use the project's `modded_nanogpt` runner to train.

Edit the file's `CONFIG` dictionary and run the module without setting flags:

```bash
uv run --extra experiments -m experiments.pinns.poisson_pinn
uv run --extra vision -m experiments.resnets.cub200
uv run --extra vision -m experiments.transformers.cub200_vit
```

Most configs use flat keys matching the former argument names, with underscores
instead of dashes. For example, `--aux-frac 0.05` becomes `"aux_frac": 0.05`.
The ViT retains its existing nested dictionary. All config files support
scientific notation, comments, and expressions:

```python
CONFIG = {
    # ...other settings from the experiment's supplied config...
    "optimizer": "gnome",
    "lr": 1e-3,
    "eps": 1e-6,
    "fixed_dataset": True,
    "steps": 50_000,
}
```

Categorical cross-entropy experiments expose `use_soap_basis=False` for Gnome.
Set it to `True` to build the basis from the main loss gradient while retaining
the surrogate-based curvature estimate. For the ViT, this setting is nested at
`CONFIG["optimizers"]["gnome"]["use_soap_basis"]`; other CCE runners use a
top-level key. The NanoGPT Kaggle config inherits it from the base config.

Keep the full dictionary when copying a config; the abbreviated example above
shows syntax only. `None` retains automatic selection where the old runner
supported it, such as optimizer-dependent defaults in NanoGPT and tabular
experiments. Lists such as LDC's `re_stages` and `stage_steps` are Python lists,
not comma-separated strings. Boolean values are `True`/`False`; negative names
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

For Python callers and tests, migrated flat-config runners also accept a
dictionary of overrides: `poisson_pinn.parse_args({"steps": 10, "lr": 1e-3})`.
This merges with that experiment's config and applies the same validation.
The ViT continues to take its full nested config through `train(config)`.

`scripts/sweep.py` merges each run's overrides into its base config and saves
the resulting complete file under `runs/sweeps/<sweep_id>/`. It launches the
runner with `--config` pointing to that snapshot. Nested overrides work for
the ViT, and both `True` and `False` are preserved. `--dry-run` previews the
sweep without creating files. `scripts/run_kaggle_nanogpt.sh` uses
`transformers/modded_nanogpt_kaggle.py`.

When adding a flat-config experiment, add its defaults here, register field
types/choices in `experiments/common/config_schema.py`, and construct
`ConfigParser(__file__)` in the runner. Keep conditional defaults and checks
with the experiment's parsing function.
