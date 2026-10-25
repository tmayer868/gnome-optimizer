from config_helpers import config_overrides

import json
import math
from copy import deepcopy

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.resnets import cub200


@pytest.fixture
def cub_root(tmp_path):
    """Tiny real image files with deliberately shuffled annotation rows."""
    root = tmp_path / "CUB_200_2011"
    (root / "images" / "bird").mkdir(parents=True)
    images, labels, splits = [], [], []
    for image_id in range(1, 10):
        name = f"bird/{image_id}.jpg"
        # Include grayscale input and class 200 to exercise both conversions.
        Image.new("L", (40 + image_id, 48), color=image_id * 20).save(root / "images" / name)
        images.append(f"{image_id} {name}\n")
        labels.append(f"{image_id} {200 if image_id == 9 else image_id}\n")
        splits.append(f"{image_id} {int(image_id in (2, 4, 6, 8, 9))}\n")
    (root / "images.txt").write_text("".join(reversed(images)))
    (root / "image_class_labels.txt").write_text("".join(labels[3:] + labels[:3]))
    (root / "train_test_split.txt").write_text("".join(splits))
    return root


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_official_id_join_rgb_and_deterministic_eval(cub_root):
    assert cub200.prepare_data(str(cub_root)) == cub_root
    assert cub200.prepare_data(str(cub_root.parent)) == cub_root
    _, transform = cub200.build_transforms(32)
    train = cub200.CUB200(cub_root, train=True, transform=transform)
    test = cub200.CUB200(cub_root, train=False, transform=transform)
    assert train.image_ids == [2, 4, 6, 8, 9]
    assert test.image_ids == [1, 3, 5, 7]
    assert set(train.image_ids).isdisjoint(test.image_ids)
    image, label = train[-1]
    assert label == 199 and image.shape == (3, 32, 32)
    assert torch.equal(image, train[-1][0]) and torch.isfinite(image).all()
    (cub_root / "image_class_labels.txt").write_text("1 1\n")
    with pytest.raises(ValueError, match="same non-empty image IDs"):
        cub200.CUB200(cub_root, train=True)


def test_missing_data_and_optional_download(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="--download"):
        cub200.prepare_data(str(tmp_path))
    calls = []

    def download(url, destination, **kwargs):
        calls.append((url, kwargs))
        folder = tmp_path / "CUB_200_2011"
        folder.mkdir()
        for name in cub200.METADATA_FILES:
            (folder / name).write_text("1 1\n")

    monkeypatch.setattr("torchvision.datasets.utils.download_and_extract_archive", download)
    cub200.prepare_data(str(tmp_path), download=True)
    cub200.prepare_data(str(tmp_path), download=True)
    assert calls == [(cub200.DATASET_URL, {
        "filename": "CUB_200_2011.tgz", "md5": cub200.DATASET_MD5,
    })]


def test_metrics_weight_partial_batches_and_restore_mode():
    logits = torch.tensor([[9., 4., 3., 2., 1., 0.]] * 5)
    labels = torch.tensor([0, 0, 1, 5, 5])
    model = nn.Identity().train()
    metrics = cub200.evaluate(
        model, DataLoader(TensorDataset(logits, labels), batch_size=2), torch.device("cpu")
    )
    assert model.training
    assert metrics["top1"] == pytest.approx(2 / 5)
    assert metrics["top5"] == pytest.approx(3 / 5)
    assert metrics["loss"] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())


def test_default_resnet_supports_full_resolution():
    model = cub200.build_model("resnet18").eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 200)
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())


@pytest.mark.parametrize("norm", ["bn", "gn"])
def test_pretrained_weights_head_gradients_and_restore(monkeypatch, norm):
    from torchvision.models import ResNet18_Weights, resnet18

    # Mock only the download boundary: exercise torchvision's real weight loader.
    source = resnet18(weights=None)
    with torch.no_grad():
        source.bn1.weight.fill_(0.75)
        source.bn1.running_mean.fill_(0.25)
    state = source.state_dict()
    calls = []

    def get_weights(weights, *args, **kwargs):
        calls.append(weights)
        return state

    monkeypatch.setattr(ResNet18_Weights, "get_state_dict", get_weights)
    model = cub200.build_model("resnet18_pretrained", norm)
    assert calls == [ResNet18_Weights.IMAGENET1K_V1]
    assert torch.equal(model.conv1.weight, source.conv1.weight)
    assert torch.equal(model.bn1.weight, source.bn1.weight)
    assert model.fc.in_features == 512 and model.fc.out_features == 200
    assert isinstance(model.relu, nn.ReLU)
    assert all(p.requires_grad for p in model.parameters())
    if norm == "bn":
        assert isinstance(model.bn1, nn.BatchNorm2d)
        assert torch.equal(model.bn1.running_mean, source.bn1.running_mean)
    else:
        assert isinstance(model.bn1, nn.GroupNorm)
        assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())
    logits = model(torch.randn(2, 3, 64, 64))
    assert logits.shape == (2, 200)
    nn.functional.cross_entropy(logits, torch.tensor([0, 199])).backward()
    for parameter in (model.conv1.weight, model.fc.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    restored = cub200.build_model("resnet18_pretrained", norm, load_pretrained=False)
    restored.load_state_dict(model.state_dict())
    assert len(calls) == 1  # checkpoint reconstruction does not fetch weights
    model.eval()
    restored.eval()
    inputs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), restored(inputs))


def test_arch_alias_and_normalization_defaults():
    assert cub200.parse_args(config_overrides(cub200, ["--optimizer", "adamw"])).norm == "gn"
    for flag in ("--arch", "--model"):
        args = cub200.parse_args(config_overrides(cub200, ["--optimizer", "gnome_hutchinson", flag, "resnet18_pretrained"]))
        assert args.model == "resnet18_pretrained" and args.norm == "bn"
        assert cub200.parse_args(config_overrides(cub200, [
            "--optimizer", "adamw", flag, "resnet18_pretrained", "--norm", "gn",
        ])).norm == "gn"


def test_jitter_changes_appearance():
    # A uniform colored image makes crop/flip inert, isolating color changes.
    image = Image.new("RGB", (80, 80), color=(130, 80, 30))
    jitter_only, evaluation = cub200.build_transforms(64)
    torch.manual_seed(17)
    jittered = jitter_only(image)
    assert not torch.allclose(jittered, evaluation(image))


def test_augmentation_draws_are_fresh_and_no_augment_is_deterministic(cub_root):
    training, _ = cub200.build_transforms(32)
    dataset = cub200.CUB200(cub_root, train=True, transform=training)
    torch.manual_seed(19)
    views = [dataset[0][0] for _ in range(4)]
    assert all(view.shape == (3, 32, 32) and torch.isfinite(view).all() for view in views)
    assert any(not torch.equal(views[0], view) for view in views[1:])
    torch.manual_seed(19)
    assert torch.equal(views[0], dataset[0][0])
    disabled, evaluation = cub200.build_transforms(32, augment=False)
    with Image.open(dataset.samples[0][0]) as image:
        image = image.convert("RGB")
        assert torch.equal(disabled(image), evaluation(image))
        assert torch.equal(disabled(image), disabled(image))


@pytest.mark.parametrize("momentum", [0.1, None])
def test_aux_batchnorm_preserves_buffers_and_training_gradients(momentum):
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4, momentum=momentum)).train()
    model(torch.randn(4, 3, 8, 8))  # main forward updates running statistics
    reference = deepcopy(model)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    inputs = torch.randn(2, 3, 8, 8) + 5
    with cub200.freeze_bn_running_stats(model):
        actual = model(inputs)
        assert model[1].training
    expected = reference(inputs)
    torch.testing.assert_close(actual, expected)
    # Backward occurs after the context exits, just as in Gnome.step.
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for parameter, other in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, other.grad)
        assert torch.isfinite(parameter.grad).all()
    for name, value in model.named_buffers():
        assert torch.equal(value, buffers[name])
    assert model[1].track_running_stats
    model(inputs)
    assert model[1].num_batches_tracked.item() == 2


def test_aux_batchnorm_restores_tracking_on_error_and_preserves_modes():
    model = nn.Sequential(nn.BatchNorm2d(4), nn.BatchNorm2d(4, track_running_stats=False),
                          nn.BatchNorm2d(4).eval())
    with pytest.raises(RuntimeError, match="forward failed"):
        with cub200.freeze_bn_running_stats(model):
            raise RuntimeError("forward failed")
    assert [layer.track_running_stats for layer in model] == [True, False, True]
    assert [layer.training for layer in model] == [True, True, False]


@pytest.mark.parametrize("norm", ["gn", "bn"])
@pytest.mark.parametrize("optimizer", ["gnome_fisher", "gnome_hutchinson", "soap", "adamw"])
def test_training_all_optimizers_and_checkpoint(cub_root, tmp_path, optimizer, norm):
    args = cub200.parse_args(config_overrides(cub200, [
        "--optimizer", optimizer, "--data-dir", str(cub_root),
        "--device", "cpu", "--model", "resnet8", "--image-size", "32",
        "--norm", norm,
        "--epochs", "2", "--batch-size", "2", "--aux-batch-size", "1",
        "--max-steps", "4", "--warmup-steps", "0",
        "--save-checkpoint", "--quiet", "--runs-dir", str(tmp_path / "runs"),
    ]))
    path = cub200.train(args)
    with open(path) as file:
        records = [json.loads(line) for line in file]
    meta = records[0]["hyperparameters"]
    assert meta["evaluation_split"] == "official_test" and not meta["pretrained"]
    assert meta["n_train"] == 5 and meta["n_test"] == 4
    assert meta["augmentation"]["enabled"]
    assert meta["augmentation"]["color_jitter"]["hue"] == 0
    assert meta["augmentation"]["crop_scale"] == [0.2, 1.0]
    assert meta["augmentation"]["crop_ratio"] == [3 / 4, 4 / 3]
    assert meta["augmentation"]["policy"] == "cub_crop_flip_jitter_v2"
    assert len([r for r in records if r["type"] == "train"]) == 4
    vals = [r for r in records if r["type"] == "val"]
    assert [r["step"] for r in vals] == [3, 4]  # includes singleton training batch
    for record in vals:
        metrics = record["metrics"]
        assert math.isfinite(metrics["loss"]) and math.isfinite(metrics["train_eval_loss"])
        assert 0 <= metrics["top1"] <= metrics["top5"] <= 1
    assert records[-1]["completed"]
    checkpoint = torch.load(path.replace(".jsonl", ".pt"), map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 4 and checkpoint["optimizer"]["state"]
    restored = cub200.build_model("resnet8", norm)
    restored.load_state_dict(checkpoint["model"])
    if norm == "bn":
        for layer in restored.modules():
            if isinstance(layer, nn.BatchNorm2d):
                assert layer.num_batches_tracked.item() == 4  # one main forward per step


@pytest.mark.parametrize("optimizer", ["gnome_fisher", "gnome_hutchinson", "soap", "adamw"])
def test_cli_accepts_batchnorm(optimizer):
    assert cub200.parse_args(config_overrides(cub200, ["--optimizer", optimizer, "--norm", "bn"])).norm == "bn"
