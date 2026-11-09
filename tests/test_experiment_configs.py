import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.common.config import (
    CONFIGS_DIR, EXPERIMENTS_DIR, SCHEMAS, ConfigParser,
    default_config_path, load_config_dict, write_config,
)
from experiments.common.config_schema import pinn_optimizer_fields


@pytest.mark.parametrize("key", sorted(SCHEMAS))
def test_mirrored_default_and_custom_config_loads(key, tmp_path, monkeypatch):
    module = importlib.import_module("experiments." + key.replace("/", "."))
    source = EXPERIMENTS_DIR / (key + ".py")
    assert default_config_path(source) == CONFIGS_DIR / (key + ".py")
    config = load_config_dict(default_config_path(source))
    config_path = write_config(tmp_path / "copied.py", config)
    # Defaults remain discoverable even when the process is in another folder.
    monkeypatch.chdir(tmp_path)
    if key == "viewer":
        parser = ConfigParser(source)
        default, custom = parser.parse_args([]), parser.parse_args(["--config", str(config_path)])
    else:
        default = module.parse_args([])
        custom = module.parse_args(["--config", str(config_path)])
    assert vars(custom) == vars(default)
    assert set(vars(custom)) == set(SCHEMAS[key]["fields"])
    with pytest.raises(SystemExit) as error:
        ConfigParser(source).parse_args(["--unknown-setting", "1"])
    assert error.value.code == 2


def test_python_expressions_paths_and_config_isolation(tmp_path):
    from experiments.tabular import otto

    config = load_config_dict(default_config_path(otto.__file__))
    config["data_dir"] = tmp_path / "birds and things"
    path = write_config(tmp_path / "with spaces.py", config)
    with path.open("a") as file:
        file.write("CONFIG['lr'] = 2 * 1e-3\nCONFIG['log1p'] = False\n")
    args = otto.parse_args(["--config", str(path)])
    assert args.lr == 2e-3 and args.log1p is False
    assert args.data_dir == config["data_dir"]
    loaded = load_config_dict(path)
    loaded["lr"] = 100
    assert load_config_dict(path)["lr"] == 2e-3


@pytest.mark.parametrize("change", [
    {"lr": "1e-3"}, {"lr": float("inf")}, {"steps": True},
    {"optimizer": "unknown"}, {"fixed_dataset": "False"}, {"stpes": 20},
])
def test_config_validation_rejects_invalid_overrides(change):
    from experiments.pinns import poisson_pinn

    with pytest.raises(SystemExit):
        poisson_pinn.parse_args(change)


def test_incomplete_file_and_exclusive_precision(tmp_path):
    from experiments.pinns import allen_cahn_pinn

    path = write_config(tmp_path / "incomplete.py", {"optimizer": "gnome"})
    with pytest.raises(SystemExit):
        allen_cahn_pinn.parse_args(["--config", str(path)])
    with pytest.raises(SystemExit):
        allen_cahn_pinn.parse_args({"float64": True, "float64_after": 10})
    path.write_text("SETTINGS = {}\n")
    with pytest.raises(SystemExit):
        allen_cahn_pinn.parse_args(["--config", str(path)])


def test_optimizer_dependent_defaults():
    from experiments.pinns import poisson_pinn
    from experiments.transformers import modded_nanogpt

    assert poisson_pinn.parse_args({"optimizer": "woodbury", "beta1": None}).beta1 == 0
    args = modded_nanogpt.parse_args({"preset": "dev", "synthetic": True, "optimizer": "muon",
                                     "steps": None, "lr": None})
    assert args.vocab_size == 256 and args.steps == 5 and args.lr == 0.025


def test_vit_config_moved_without_flattening():
    from experiments.transformers import cub200_vit

    assert cub200_vit.DEFAULT_CONFIG == CONFIGS_DIR / "transformers" / "cub200_vit.py"
    config = cub200_vit.load_config()
    assert {"model", "training", "data", "optimizers"} <= config.keys()
    assert not (CONFIGS_DIR / "cub200_vit.py").exists()


PINN_SELECTIONS = [
    (key, optimizer)
    for key in SCHEMAS if pinn_optimizer_fields(key)
    for optimizer in pinn_optimizer_fields(key)
]


@pytest.mark.parametrize("key,optimizer", PINN_SELECTIONS)
def test_pinn_optimizer_selection_from_file(key, optimizer, tmp_path):
    config = load_config_dict(CONFIGS_DIR / (key + ".py"))
    config["optimizer"] = optimizer
    # Distinct LRs ensure the selected block wins, regardless of dict order.
    for i, (name, block) in enumerate(config["optimizers"].items(), 1):
        block["engd_lr" if name == "engd" else "lr"] = i * 1e-4
    path = write_config(tmp_path / "pinn.py", config)
    parser = ConfigParser(EXPERIMENTS_DIR / (key + ".py"))
    args = vars(parser.parse_args(["--config", str(path)]))
    for name, value in config["optimizers"][optimizer].items():
        assert args[name] == value
    assert args["hidden"] == config["hidden"]
    assert args["depth"] == config["depth"]
    assert "lr" not in config and "warmup_steps" not in config


@pytest.mark.parametrize("case", ["missing_block", "missing_field", "extra_block",
                                  "misplaced_field", "wrong_type", "not_dict", "flat_lr"])
def test_pinn_nested_config_validation(case, tmp_path):
    from experiments.pinns import poisson_pinn

    config = load_config_dict(default_config_path(poisson_pinn.__file__))
    blocks = config["optimizers"]
    if case == "missing_block":
        del blocks["soap"]
    elif case == "missing_field":
        del blocks["gnome"]["lr"]
    elif case == "extra_block":
        blocks["sgd"] = {"lr": 1e-3}
    elif case == "misplaced_field":
        blocks["adamw"]["trust_region"] = 1.0
    elif case == "wrong_type":
        blocks["soap"]["lr"] = "1e-3"  # Inactive blocks are validated too.
    elif case == "not_dict":
        blocks["soap"] = None
    else:
        config["lr"] = 1e-3
    path = write_config(tmp_path / "invalid.py", config)
    with pytest.raises(SystemExit):
        poisson_pinn.parse_args(["--config", str(path)])


def test_pinn_nested_overrides_and_inactive_fixed_dataset():
    from experiments.pinns import poisson_pinn

    args = poisson_pinn.parse_args({
        "optimizer": "soap",
        "optimizers": {"soap": {"lr": 2e-4, "warmup_steps": 7},
                       "gnome": {"lr": 9e-2}, "engdw": {"engdw_fixed": True}},
    })
    assert args.lr == 2e-4 and args.warmup_steps == 7
    assert not args.engdw_fixed and not args.fixed_dataset


def test_pinn_sweep_overrides_selected_optimizer(tmp_path):
    from scripts import sweep
    from experiments.pinns import poisson_pinn

    defaults = load_config_dict(default_config_path(poisson_pinn.__file__))
    config = sweep.run_config(dict(experiment="poisson_pinn", optimizer="soap", lr=3e-4,
                                   optimizers={"soap": {"warmup_steps": 9}}))
    assert config["optimizers"]["gnome"] == defaults["optimizers"]["gnome"]
    assert config["optimizers"]["soap"]["lr"] == 3e-4
    path = write_config(tmp_path / "sweep.py", config)
    args = poisson_pinn.parse_args(["--config", str(path)])
    assert args.optimizer == "soap" and args.lr == 3e-4 and args.warmup_steps == 9
    with pytest.raises(ValueError, match="Duplicate"):
        sweep.run_config(dict(experiment="poisson_pinn", lr=1e-3,
                              optimizers={"gnome": {"lr": 1e-2}}))


def test_sweep_materializes_complete_nested_config(tmp_path):
    from scripts import sweep
    from experiments.transformers import cub200_vit

    run = dict(experiment="cub200_vit", optimizer="muon", data={"augment": False},
               optimizers={"muon": {"lr": 2e-2}}, training={"epochs": 7})
    config = sweep.run_config(run)
    assert config["data"]["augment"] is False and config["training"]["epochs"] == 7
    assert "batch_size" in config["training"] and "adamw" in config["optimizers"]["muon"]
    path = write_config(tmp_path / "run.py", config)
    assert cub200_vit.load_config(path) == config
    assert sweep.build_cmd(run, path)[-2:] == ["--config", str(path)]
    with pytest.raises(ValueError, match="Unknown config key"):
        sweep.run_config({"experiment": "poisson_pinn", "stpes": 1})


def test_kaggle_config_loads():
    from experiments.transformers import modded_nanogpt

    path = CONFIGS_DIR / "transformers" / "modded_nanogpt_kaggle.py"
    args = modded_nanogpt.parse_args(["--config", str(path)])
    assert args.download_shards == 24 and args.steps == 4200
    assert args.fp32_embedding and args.save_checkpoint


@pytest.mark.parametrize("key,optimizer,updates", [
    *[("pinns/poisson_pinn", optimizer,
       dict(steps=2, hidden=3, depth=2, n_pde=4, n_bc_per_edge=2,
            log_every=0, float64=True))
      for optimizer in ("gnome", "soap", "adamw", "engdw", "woodbury")],
])
def test_entrypoint_trains_from_custom_config(key, optimizer, updates, tmp_path):
    config = load_config_dict(CONFIGS_DIR / (key + ".py"))
    if "optimizers" in config:
        block = config["optimizers"][optimizer]
        if "warmup_steps" in block:
            block["warmup_steps"] = 0
        block["lr"] = 2e-4
    config.update(updates, optimizer=optimizer, quiet=True, runs_dir=str(tmp_path / "runs"))
    path = write_config(tmp_path / "smoke.py", config)
    result = subprocess.run([sys.executable, "-m", "experiments." + key.replace("/", "."),
                             "--config", str(path)], cwd=EXPERIMENTS_DIR.parent,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    logs = list((tmp_path / "runs").rglob("*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert records[0]["optimizer"] == optimizer
    assert records[-1]["type"] == "end" and records[-1]["completed"]
