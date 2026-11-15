from config_helpers import config_overrides

import gzip
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from experiments.tabular import covertype as cv


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_data():
    rng = torch.Generator().manual_seed(17)
    x = torch.randn(140, 54, generator=rng)
    x[:, :10] = 100 + 30 * x[:, :10]
    x[:, 10:] = (x[:, 10:] > 0).float()
    x[:, 0] = 5  # constant numeric column
    y = torch.arange(7).repeat_interleave(20)
    return x, y


def test_download_checksum_cache_and_label_conversion(tmp_path, monkeypatch):
    x, y = fixture_data()
    buffer = io.StringIO()
    np.savetxt(buffer, np.column_stack((x.numpy(), y.numpy() + 1)), delimiter=",")
    payload = gzip.compress(buffer.getvalue().encode())
    monkeypatch.setattr(cv, "DATA_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def download(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(payload)

    monkeypatch.setattr(cv.urllib.request, "urlopen", download)
    for _ in range(2):
        actual_x, actual_y = cv.load_data(tmp_path)
        torch.testing.assert_close(actual_x, x)
        assert torch.equal(actual_y, y)
    assert len(calls) == 1
    (tmp_path / "covtype.data.gz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        cv.load_data(tmp_path)


def test_failed_download_does_not_poison_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cv.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"broken"))
    with pytest.raises(ValueError, match="checksum"):
        cv.load_data(tmp_path)
    assert not list(tmp_path.iterdir())


def test_split_disjoint_stratified_reproducible_and_train_only_scaling():
    x, y = fixture_data()
    splits, prep = cv.split_and_standardize(x, y)
    indices = prep["indices"]
    assert sorted(torch.cat(list(indices.values())).tolist()) == list(range(len(y)))
    assert [len(pair[1]) for pair in splits.values()] == [98, 21, 21]
    for _, labels in splits.values():
        assert len(torch.unique(labels)) == 7
    for name, (features, labels) in splits.items():
        assert torch.equal(features[:, 10:], x[indices[name], 10:])
        assert torch.equal(labels, y[indices[name]])
    torch.testing.assert_close(splits["train"][0][:, :10].mean(0), torch.zeros(10), atol=1e-6, rtol=0)
    torch.testing.assert_close(splits["train"][0][:, 1:10].std(0, correction=0), torch.ones(9))
    assert torch.isfinite(splits["train"][0]).all()
    # Perturb held-out features: fitted statistics/training features cannot change.
    perturbed = x.clone()
    perturbed[torch.cat((indices["val"], indices["test"])), :10] += 10000
    changed, changed_prep = cv.split_and_standardize(perturbed, y)
    assert torch.equal(changed_prep["mean"], prep["mean"])
    assert torch.equal(changed["train"][0], splits["train"][0])
    torch.manual_seed(999)
    _, again = cv.split_and_standardize(x, y)
    assert all(torch.equal(indices[k], again["indices"][k]) for k in indices)
    _, different = cv.split_and_standardize(x, y, split_seed=1)
    assert not torch.equal(indices["train"], different["indices"]["train"])


def test_muon_parameter_coverage_and_schedule_ratios():
    args = cv.parse_args(config_overrides(cv, ["--optimizer", "muon", "--lr", ".03", "--aux-lr", ".002"]))
    model = cv.ResidualMLP(16, 1)
    muon, adamw = cv.build_optimizers(model, args)
    matrix_ids = {id(p) for group in muon.param_groups for p in group["params"]}
    aux_ids = {id(p) for group in adamw.param_groups for p in group["params"]}
    assert matrix_ids == {id(p) for p in model.blocks.parameters() if p.ndim == 2}
    assert not matrix_ids & aux_ids
    assert matrix_ids | aux_ids == {id(p) for p in model.parameters()}
    assert {id(model.input.weight), id(model.head.weight), id(model.norm.weight)} <= aux_ids
    assert muon.param_groups[0]["lr"] == .03
    assert adamw.param_groups[0]["lr"] == .002


def test_metrics_weight_partial_batch_and_restore_mode():
    logits = torch.tensor([[3., 0., -1.]] * 5)
    labels = torch.tensor([0, 0, 1, 2, 2])
    model = torch.nn.Identity().train()
    result = cv.evaluate(model, logits, labels, 2, torch.device("cpu"))
    assert model.training
    assert result["top1"] == pytest.approx(0.4)
    assert result["loss"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())


@pytest.mark.parametrize("optimizer", cv.OPTIMIZERS)
def test_training_all_optimizers_checkpoint_and_shared_batch_order(tmp_path, monkeypatch, optimizer):
    monkeypatch.setattr(cv, "load_data", lambda _: fixture_data())
    args = cv.parse_args(config_overrides(cv, [
        "--optimizer", optimizer, "--device", "cpu", "--width", "16", "--blocks", "1",
        "--epochs", "2", "--batch-size", "32", "--aux-batch-size", "8",
        "--eval-batch-size", "11", "--max-steps", "5", "--warmup-steps", "1",
        "--precondition-frequency", "2", "--save-checkpoint", "--quiet",
        "--runs-dir", str(tmp_path),
    ]))
    torch.manual_seed(args.seed)
    initial = {name: p.clone() for name, p in cv.ResidualMLP(16, 1).named_parameters()}
    batches = []
    train_step = cv.train_step

    def record_step(model, optimizers, x, y, args, rng):
        batches.append(x.clone())
        return train_step(model, optimizers, x, y, args, rng)

    monkeypatch.setattr(cv, "train_step", record_step)
    path = cv.train(args)
    with open(path) as file:
        records = [json.loads(line) for line in file]
    training = [r for r in records if r["type"] == "train"]
    assert len(training) == 5
    assert training[0]["metrics"]["lr"] == pytest.approx(args.lr)
    assert training[-1]["metrics"]["lr"] == pytest.approx(args.lr * args.min_lr_frac)
    assert [r["step"] for r in records if r["type"] == "val"] == [4, 5]
    assert [r["step"] for r in records if r["type"] == "test"] == [5]
    assert records[-1]["completed"] and records[-1]["training_seconds"] > 0
    ckpt = torch.load(path.replace(".jsonl", ".pt"), weights_only=True)
    assert ckpt["step"] == 5 and all(opt["state"] for opt in ckpt["optimizers"])
    assert len(ckpt["optimizers"]) == (2 if optimizer == "muon" else 1)
    assert all(torch.isfinite(p).all() for p in ckpt["model"].values())
    for name, p in initial.items():
        assert not torch.equal(p, ckpt["model"][name]), name
    expected_splits, _ = cv.split_and_standardize(*fixture_data())
    rng = torch.Generator().manual_seed(args.seed)
    expected_batches = []
    for _ in range(2):
        for idx in torch.randperm(98, generator=rng).split(32):
            expected_batches.append(expected_splits["train"][0][idx])
    assert all(torch.equal(a, b) for a, b in zip(batches, expected_batches[:5]))
    assert len(batches[3]) == 2  # auxiliary sampling also tolerates partial batches
    model = cv.ResidualMLP(16, 1)
    model.load_state_dict(ckpt["model"])
    result = cv.evaluate(model, *expected_splits["test"], 11, torch.device("cpu"))
    assert result["loss"] == pytest.approx(records[-1]["test_loss"])


@pytest.mark.parametrize("flags", [
    ["--lr", "nan"], ["--batch-size", "0"], ["--min-lr-frac", "1.1"],
    ["--eps", "0"], ["--max-steps", "2"], ["--aux-batch-size", "513"],
])
def test_cli_rejects_invalid_settings(flags):
    with pytest.raises(SystemExit):
        cv.parse_args(config_overrides(cv, flags))
