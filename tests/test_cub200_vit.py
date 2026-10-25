import json
import math
from pathlib import Path

import pytest
import torch
from PIL import Image

from experiments.transformers import cub200_vit as vit


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(tmp_path):
    config = vit.load_config()
    config["model"].update(image_size=16, patch_size=8, num_layers=1, num_heads=2,
                           hidden_dim=8, mlp_dim=16)
    config["training"].update(device="cpu", epochs=2, batch_size=2, eval_batch_size=2,
                              max_steps=3, warmup_steps=0, min_lr_frac=1.0)
    config["output"].update(runs_dir=str(tmp_path / "runs"), quiet=True, eval_every_epochs=10)
    config["data"].update(data_dir=str(tmp_path), augment=False)
    for name in ("soap", "gnome"):
        config["optimizers"][name]["precondition_frequency"] = 1
    return config


@pytest.fixture
def cub_root(config):
    root = Path(config["data"]["data_dir"]) / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    images, labels, splits = [], [], []
    for index in range(1, 9):
        name = f"{index}.png"
        Image.new("RGB", (20 + index, 32), (index * 20, 30, 100)).save(root / "images" / name)
        images.append(f"{index} {name}\n")
        labels.append(f"{index} {200 if index == 8 else index}\n")
        splits.append(f"{index} {int(index <= 5)}\n")
    for filename, rows in zip(("images.txt", "image_class_labels.txt", "train_test_split.txt"),
                              (images, labels, splits)):
        (root / filename).write_text("".join(reversed(rows)))
    return root


def test_default_config_model_and_cli(capsys):
    config = vit.load_config()
    model = vit.build_model(config).eval()
    with torch.no_grad():
        size = config["model"]["image_size"]
        logits = model(torch.randn(1, 3, size, size))
    assert logits.shape == (1, 200)
    assert model.heads.head.weight.std().item() == pytest.approx(config["model"]["head_init_std"], rel=0.03)
    with pytest.raises(SystemExit) as error:
        vit.main(["--help"])
    assert error.value.code == 0
    assert "--config" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        vit.main(["--optimizer", "muon"])


@pytest.mark.parametrize("path,value", [
    (("optimizer",), "sgd"),
    (("model", "patch_size"), 7),
    (("model", "num_heads"), 3),
    (("model", "dropout"), 1),
    (("training", "batch_size"), True),
    (("training", "max_steps"), 0),
    (("training", "max_grad_norm"), 0),
    (("data", "download"), "false"),
    (("data", "normalize_std"), [1, 0, 1]),
    (("optimizers", "gnome", "loss"), "mse"),
    (("optimizers", "gnome", "aux_batch_size"), 0),
    (("optimizers", "soap", "precondition_frequency"), 0),
    (("optimizers", "muon", "adamw", "lr"), float("nan")),
    (("output", "eval_every_epochs"), 0),
])
def test_invalid_config_fails_early(config, path, value):
    section = config
    for key in path[:-1]:
        section = section[key]
    section[path[-1]] = value
    with pytest.raises(ValueError):
        vit.validate_config(config)


def test_python_config_expressions_and_validation(config, tmp_path):
    path = tmp_path / "custom.py"
    path.write_text(f"CONFIG = {config!r}\nCONFIG['optimizers']['gnome']['lr'] = 1e-3\n")
    config["optimizers"]["gnome"]["lr"] = 1e-3
    assert vit.load_config(path) == config
    config["training"]["batch_szie"] = 4
    path.write_text(f"CONFIG = {config!r}\n")
    with pytest.raises(ValueError, match="unknown keys.*batch_szie"):
        vit.load_config(path)
    del config["training"]["batch_szie"]
    del config["optimizers"]["muon"]["adamw"]
    path.write_text(f"CONFIG = {config!r}\n")
    with pytest.raises(ValueError, match="missing keys.*adamw"):
        vit.load_config(path)
    path.write_text("optimizer = 'muon'\n")
    with pytest.raises(ValueError, match="must define a CONFIG dictionary"):
        vit.load_config(path)
    path.write_text("CONFIG = []\n")
    with pytest.raises(ValueError, match="must be a dictionary"):
        vit.load_config(path)


def test_muon_routes_each_parameter_once_and_updates_both_groups(config):
    config["optimizer"] = "muon"
    model = vit.build_model(config)
    muon, adamw = vit.build_optimizers(model, config)
    by_id = {id(p): name for name, p in model.named_parameters()}
    muon_names = {by_id[id(p)] for group in muon.param_groups for p in group["params"]}
    adamw_names = {by_id[id(p)] for group in adamw.param_groups for p in group["params"]}
    prefix = "encoder.layers.encoder_layer_0."
    assert muon_names == {prefix + name for name in (
        "self_attention.in_proj_weight", "self_attention.out_proj.weight", "mlp.0.weight", "mlp.3.weight")}
    assert muon_names.isdisjoint(adamw_names)
    assert muon_names | adamw_names == set(dict(model.named_parameters()))
    assert {"class_token", "encoder.pos_embedding", "conv_proj.weight", "conv_proj.bias",
            "heads.head.weight", "heads.head.bias", "encoder.ln.weight", "encoder.ln.bias"} <= adamw_names
    assert all(name in adamw_names for name in by_id.values() if "bias" in name or ".ln_" in name)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    loss = vit.train_step(model, [muon, adamw], torch.randn(2, 3, 16, 16),
                          torch.tensor([0, 199]), config, torch.Generator())
    assert math.isfinite(loss)
    for name in (prefix + "self_attention.in_proj_weight", "conv_proj.weight", "heads.head.weight",
                 "encoder.pos_embedding", "class_token", "encoder.ln.weight", "heads.head.bias"):
        assert not torch.equal(before[name], dict(model.named_parameters())[name]), name


@pytest.mark.parametrize("optimizer", ["adamw", "soap", "gnome", "muon"])
def test_training_logs_final_partial_batch_and_checkpoint(config, cub_root, optimizer):
    config["optimizer"] = optimizer
    path = vit.train(config)
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    meta = records[0]["hyperparameters"]
    assert meta["config"] == config
    assert meta["n_train"] == 5 and meta["n_test"] == 3
    assert meta["evaluation_split"] == "official_test"
    assert meta["pretrained"] is False and meta["dtype"] == "float32"
    updates = [r for r in records if r["type"] == "train"]
    assert [r["step"] for r in updates] == [1, 2, 3]
    assert all(math.isfinite(r["metrics"]["loss"]) for r in updates)
    evaluations = [r for r in records if r["type"] == "val"]
    assert len(evaluations) == 1 and evaluations[0]["step"] == 3
    assert 0 <= evaluations[0]["metrics"]["top1"] <= evaluations[0]["metrics"]["top5"] <= 1
    assert records[-1]["completed"] is True
    checkpoint = torch.load(Path(path).with_suffix(".pt"), weights_only=False)
    restored = vit.build_model(checkpoint["config"])
    restored.load_state_dict(checkpoint["model"])
    optimizers = vit.build_optimizers(restored, checkpoint["config"])
    assert len(checkpoint["optimizers"]) == (2 if optimizer == "muon" else 1)
    for opt, state in zip(optimizers, checkpoint["optimizers"]):
        opt.load_state_dict(state)
        assert opt.state
    assert checkpoint["step"] == 3
    before = restored.heads.head.weight.detach().clone()
    assert math.isfinite(vit.train_step(restored, optimizers, torch.randn(2, 3, 16, 16),
                                       torch.tensor([0, 199]), config, torch.Generator()))
    assert not torch.equal(before, restored.heads.head.weight)


def test_muon_schedule_scales_both_lrs(config, cub_root):
    config["optimizer"] = "muon"
    config["training"].update(max_steps=3, warmup_steps=0, min_lr_frac=0.1)
    config["output"]["save_checkpoint"] = False
    path = vit.train(config)
    updates = [json.loads(line)["metrics"] for line in Path(path).read_text().splitlines()
               if json.loads(line)["type"] == "train"]
    for metrics, multiplier in zip(updates, (1.0, 0.55, 0.1)):
        assert metrics["lr"] == pytest.approx(config["optimizers"]["muon"]["lr"] * multiplier)
        assert metrics["adamw_lr"] == pytest.approx(config["optimizers"]["muon"]["adamw"]["lr"] * multiplier)
    assert not Path(path).with_suffix(".pt").exists()


def test_gnome_fisher_and_auxiliary_subset(config):
    config["optimizer"] = "gnome"
    config["optimizers"]["gnome"]["loss"] = "cce"
    config["optimizers"]["gnome"]["aux_batch_size"] = 2
    model = vit.build_model(config)
    optimizers = vit.build_optimizers(model, config)
    images, labels = torch.randn(3, 3, 16, 16), torch.tensor([0, 12, 199])
    seen = []
    handle = model.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].clone()))
    for _ in range(2):
        assert math.isfinite(vit.train_step(model, optimizers, images, labels, config, torch.Generator()))
    handle.remove()
    assert [len(batch) for batch in seen] == [3, 2, 3, 2]
    for image in seen[1]:
        assert any(torch.equal(image, candidate) for candidate in images)
    assert not torch.equal(seen[1][0], seen[1][1])


def test_augmentation_stream_is_independent_of_optimizer_randomness(config):
    config["data"]["augment"] = True
    transform_a, _ = vit.build_transforms(config)
    transform_b, _ = vit.build_transforms(config)
    image = Image.fromarray(torch.randint(0, 256, (40, 30, 3), dtype=torch.uint8).numpy())
    for _ in range(3):
        state = torch.get_rng_state()
        transformed = transform_a(image)
        assert torch.equal(state, torch.get_rng_state())
        torch.rand(1000)  # extra optimizer random draws must not change crops
        assert torch.equal(transformed, transform_b(image))


def test_divergence_marks_run_incomplete(config, cub_root, monkeypatch):
    monkeypatch.setattr(vit, "train_step", lambda *args: float("nan"))
    with pytest.raises(SystemExit) as error:
        vit.train(config)
    assert error.value.code == vit.DIVERGED_EXIT
    path, = (Path(config["output"]["runs_dir"]) / vit.EXPERIMENT).glob("*.jsonl")
    end = json.loads(path.read_text().splitlines()[-1])
    assert end["completed"] is False and end["diverged"] is True
    assert not path.with_suffix(".pt").exists()
