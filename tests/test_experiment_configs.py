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
    assert set(vars(custom)) == set(config)
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


def test_lists_and_optimizer_dependent_defaults():
    from experiments.pinns import ldc_pinn, poisson_pinn
    from experiments.transformers import modded_nanogpt

    args = ldc_pinn.parse_args({"re_stages": [100, 400], "stage_steps": [10, 20],
                               "block_weights": [1, 1, 10, 100, 100]})
    assert args.re_stages == [100, 400] and args.stage_steps == [10, 20]
    assert args.block_weights == [1, 1, 10, 100, 100]
    with pytest.raises(SystemExit):
        ldc_pinn.parse_args({"re_stages": "100,400"})
    assert poisson_pinn.parse_args({"optimizer": "woodbury", "beta1": None}).beta1 == 0
    args = modded_nanogpt.parse_args({"preset": "dev", "synthetic": True, "optimizer": "muon"})
    assert args.vocab_size == 256 and args.steps == 5 and args.lr == 0.025


def test_vit_config_moved_without_flattening():
    from experiments.transformers import cub200_vit

    assert cub200_vit.DEFAULT_CONFIG == CONFIGS_DIR / "transformers" / "cub200_vit.py"
    config = cub200_vit.load_config()
    assert {"model", "training", "data", "optimizers"} <= config.keys()
    assert not (CONFIGS_DIR / "cub200_vit.py").exists()


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


@pytest.mark.parametrize("key,updates", [
    ("ols_regression", dict(epochs=1, n_train=8, n_val=8, dim=2, batch_size=4, warmup_steps=0)),
    ("pinns/poisson_pinn", dict(steps=2, hidden=3, depth=2, n_pde=4, n_bc_per_edge=2,
                               warmup_steps=0, log_every=0, float64=True)),
])
def test_entrypoint_trains_from_custom_config(key, updates, tmp_path):
    config = load_config_dict(CONFIGS_DIR / (key + ".py"))
    config.update(updates, optimizer="adamw", quiet=True, runs_dir=str(tmp_path / "runs"))
    path = write_config(tmp_path / "smoke.py", config)
    result = subprocess.run([sys.executable, "-m", "experiments." + key.replace("/", "."),
                             "--config", str(path)], cwd=EXPERIMENTS_DIR.parent,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    logs = list((tmp_path / "runs").rglob("*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text().splitlines()]
    assert records[0]["optimizer"] == "adamw"
    assert records[-1]["type"] == "end" and records[-1]["completed"]
