from config_helpers import config_overrides

import csv
import gzip
import hashlib
import io
import json
import zipfile

import pytest
import torch

from experiments.tabular import otto


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tables(tmp_path):
    x = torch.randint(0, 20, (180, 93), generator=torch.Generator().manual_seed(1)).float()
    x[:, 0] = 7  # exercise constant-feature scaling
    y = torch.arange(9).repeat_interleave(20)
    header = ["id", *otto.FEATURE_NAMES, "target"]
    # Deliberately reverse the feature columns; the loader must restore named order.
    order = [0, *range(93, 0, -1), 94]
    header = [header[i] for i in order]
    rows = [[str(100000 + i), *[str(int(v)) for v in x[i]], otto.CLASS_NAMES[y[i]]]
            for i in range(len(y))]
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerow(header)
    writer.writerows([[row[i] for i in order] for row in rows])
    csv_text = text.getvalue()
    (tmp_path / "train.csv").write_text(csv_text)
    arff_header = "@relation otto\n" + "".join(f"@attribute '{n}' numeric\n" for n in header[:-1])
    arff_header += "@attribute target {Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9}\n@data\n"
    (tmp_path / "otto.arff").write_text(arff_header + csv_text.split("\n", 1)[1])
    with gzip.open(tmp_path / "train.csv.gz", "wt") as file:
        file.write(csv_text)
    with zipfile.ZipFile(tmp_path / "train.csv.zip", "w") as archive:
        archive.writestr("nested/train.csv", csv_text)
    return tmp_path, x, y


@pytest.mark.parametrize("filename", ["train.csv", "train.csv.gz", "train.csv.zip", "otto.arff"])
def test_formats_feature_order_and_ids_excluded(tables, filename):
    path, x, y = tables
    actual_x, actual_y = otto.load_data(path, path / filename)
    assert torch.equal(x, actual_x) and torch.equal(y, actual_y)
    assert actual_x.shape == (180, 93)


def test_download_validation_cache_and_local_precedence(tables, tmp_path, monkeypatch):
    folder, _, _ = tables
    payload = (folder / "otto.arff").read_bytes()
    monkeypatch.setattr(otto, "DATA_SHA256", hashlib.sha256(payload).hexdigest())
    calls = []

    def download(*args, **kwargs):
        calls.append(args)
        return io.BytesIO(payload)

    monkeypatch.setattr(otto.urllib.request, "urlopen", download)
    assert otto.prepare_data(folder) == folder / "train.csv"
    assert not calls
    cache = tmp_path / "cache"
    assert otto.prepare_data(cache) == cache / "otto.arff"
    assert otto.prepare_data(cache) == cache / "otto.arff"
    assert len(calls) == 1
    (cache / "otto.arff").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        otto.prepare_data(cache)
    monkeypatch.setattr(otto, "DATA_SHA256", "bad")
    with pytest.raises(ValueError, match="checksum"):
        otto.prepare_data(tmp_path / "bad")
    assert not list((tmp_path / "bad").iterdir())


@pytest.mark.parametrize("issue", ["unlabeled", "duplicate_id", "negative", "nan", "label", "short", "duplicate_column"])
def test_reject_invalid_training_data(tables, issue):
    folder, _, _ = tables
    path = folder / "train.csv"
    with path.open() as file:
        rows = list(csv.reader(file))
    if issue == "unlabeled":
        rows = [r[:-1] for r in rows]
    elif issue == "duplicate_id":
        rows[2][0] = rows[1][0]
    elif issue in ("negative", "nan"):
        rows[1][1] = "-1" if issue == "negative" else "nan"
    elif issue == "label":
        rows[1][-1] = "Class_10"
    elif issue == "short":
        rows[1].pop()
    else:
        rows[0][1] = rows[0][2]
    with path.open("w") as file:
        csv.writer(file).writerows(rows)
    with pytest.raises(ValueError):
        otto.load_data(folder)


def test_log_transform_train_only_stats_and_split(tables):
    _, x, y = tables
    splits, prep = otto.split_and_standardize(x, y)
    indices = prep["indices"]
    train_x = torch.log1p(x[indices["train"]])
    torch.testing.assert_close(prep["mean"], train_x.mean(0))
    assert prep["log1p"] and prep["num_continuous"] == 93
    assert [len(v[1]) for v in splits.values()] == [126, 27, 27]
    assert sorted(torch.cat(list(indices.values())).tolist()) == list(range(len(y)))
    for _, labels in splits.values():
        assert len(torch.unique(labels)) == 9
    changed = x.clone()
    changed[torch.cat((indices["val"], indices["test"]))] += 10000
    changed_splits, changed_prep = otto.split_and_standardize(changed, y)
    assert torch.equal(changed_splits["train"][0], splits["train"][0])
    assert torch.equal(prep["mean"], changed_prep["mean"])
    _, no_log = otto.split_and_standardize(x, y, log1p=False)
    torch.testing.assert_close(no_log["mean"], x[indices["train"]].mean(0))
    assert not no_log["log1p"] and not torch.equal(prep["mean"], no_log["mean"])
    assert torch.isfinite(splits["train"][0]).all()
    assert otto.parse_args(config_overrides(otto, ["--no-log1p"])).log1p is False


@pytest.mark.parametrize("optimizer", otto.OPTIMIZERS)
def test_training_checkpoint_and_final_test(tables, tmp_path, optimizer):
    folder, x, y = tables
    args = otto.parse_args(config_overrides(otto, [
        "--optimizer", optimizer, "--device", "cpu", "--data-file", str(folder / "train.csv"),
        "--width", "16", "--blocks", "1", "--batch-size", "40", "--aux-batch-size", "8",
        "--eval-batch-size", "11", "--epochs", "2", "--max-steps", "5", "--warmup-steps", "1",
        "--precondition-frequency", "2", "--runs-dir", str(tmp_path / "runs"),
        "--quiet", "--save-checkpoint",
    ]))
    torch.manual_seed(args.seed)
    initial = {name: p.clone() for name, p in otto.ResidualMLP(16, 1).named_parameters()}
    path = otto.train(args)
    with open(path) as file:
        records = [json.loads(line) for line in file]
    meta = records[0]["hyperparameters"]
    assert records[0]["experiment"] == "otto"
    assert meta["n_train"] == 126 and meta["n_val"] == meta["n_test"] == 27
    assert meta["feature_names"] == otto.FEATURE_NAMES and meta["class_names"] == otto.CLASS_NAMES
    assert meta["feature_transform"] == "log1p"
    assert meta["evaluation_split"] == "held_out_labeled_training_rows"
    assert [r["step"] for r in records if r["type"] == "val"] == [4, 5]
    assert [r["step"] for r in records if r["type"] == "test"] == [5]
    assert records[-1]["completed"] and records[-1]["best_val_loss"] <= records[-1]["final_val_loss"]
    ckpt = torch.load(path.replace(".jsonl", ".pt"), weights_only=True)
    assert ckpt["step"] == 5 and ckpt["preprocessing"]["log1p"]
    assert len(ckpt["optimizers"]) == (2 if optimizer == "muon" else 1)
    assert all(opt["state"] for opt in ckpt["optimizers"])
    assert all(torch.isfinite(p).all() for p in ckpt["model"].values())
    for name, p in initial.items():
        assert not torch.equal(p, ckpt["model"][name]), name
    model = otto.ResidualMLP(16, 1)
    model.load_state_dict(ckpt["model"])
    idx = ckpt["preprocessing"]["indices"]["test"]
    features = (torch.log1p(x[idx]) - ckpt["preprocessing"]["mean"]) / ckpt["preprocessing"]["std"]
    result = otto.evaluate(model, features, y[idx], 11, torch.device("cpu"))
    assert result["loss"] == pytest.approx(records[-1]["test_loss"])
