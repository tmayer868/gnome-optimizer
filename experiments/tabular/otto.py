"""Otto Group product classification with Gnome, AdamW, SOAP, or Muon.

    # Edit experiments/configs/tabular/otto.py: optimizer='gnome'
    uv run python -u -m experiments.tabular.otto

Downloads the labeled training set from OpenML, or reads Kaggle train.csv with
--data-file. See docs/otto.md for the evaluation and preprocessing protocol.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import io
from pathlib import Path
import shlex
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch

from experiments.tabular import common
from experiments.tabular.common import OPTIMIZERS, build_optimizers, evaluate, train_step


NUM_FEATURES, NUM_CLASSES = 93, 9
FEATURE_NAMES = [f"feat_{i}" for i in range(1, NUM_FEATURES + 1)]
CLASS_NAMES = [f"Class_{i}" for i in range(1, NUM_CLASSES + 1)]
DATA_URL = "https://openml.org/data/v1/download/22116516/Otto-Group-Product-Classification-Challenge.arff"
DATA_SHA256 = "bdd6b4397d7b2175b4b2626af2af662b88f11739b7d3f1b75acc838f6dcb3d1d"
OPENML_ID = 45548


def prepare_data(data_dir: Path, data_file: Path | None = None):
    """Prefer an explicit/local Kaggle file; otherwise cache pinned OpenML data."""
    if data_file is not None:
        if not data_file.is_file():
            raise FileNotFoundError(f"Otto training data not found: {data_file}")
        return data_file
    for name in ("train.csv", "train.csv.gz", "train.csv.zip"):
        path = data_dir / name
        if path.is_file():
            return path
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "otto.arff"
    if not path.exists():
        print("[otto] downloading labeled training data from OpenML (~12 MB)...", flush=True)
        with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".download", delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with temporary.open("wb") as out, urllib.request.urlopen(DATA_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != DATA_SHA256:
                raise ValueError("Otto download checksum mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError(f"Otto checksum mismatch: remove {path} and retry")
    return path


@contextmanager
def open_table(path):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if Path(n).name == "train.csv"]
            if len(members) != 1:
                raise ValueError("Otto ZIP must contain exactly one train.csv")
            with archive.open(members[0]) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig") as file:
                yield file
    elif path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as file:
            yield file
    else:
        with path.open(encoding="utf-8-sig", newline="") as file:
            yield file


def read_table(path):
    """Read dense OpenML ARFF or Kaggle CSV, selecting features by name.

    IDs identify rows only. Missing/negative/nonfinite counts, duplicate IDs,
    and unlabeled competition test files are rejected instead of silently used.
    """
    print(f"[otto] loading {path}...", flush=True)
    is_arff = ".arff" in path.suffixes
    with open_table(path) as file:
        if is_arff:
            columns = []
            for line in file:
                stripped = line.strip()
                if stripped.lower().startswith("@attribute "):
                    columns.append(shlex.split(stripped)[1])
                elif stripped.lower() == "@data":
                    break
            rows = csv.reader(line for line in file if line.strip() and not line.lstrip().startswith("%"))
        else:
            rows = csv.reader(file)
            columns = next(rows, [])
        expected = {"id", "target", *FEATURE_NAMES}
        if len(columns) != len(expected) or set(columns) != expected:
            raise ValueError("Expected labeled Otto data: id, feat_1 through feat_93, target. "
                             "Use train.csv; Kaggle test.csv has no labels.")
        feature_cols = [columns.index(name) for name in FEATURE_NAMES]
        id_col, target_col = columns.index("id"), columns.index("target")
        class_index = {name: i for i, name in enumerate(CLASS_NAMES)}
        features, labels, ids = [], [], set()
        for row in rows:
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError("Otto row has the wrong number of columns")
            row_id = row[id_col].strip()
            if not row_id or row_id in ids:
                raise ValueError("Otto product IDs must be nonempty and unique")
            ids.add(row_id)
            label = row[target_col].strip().strip("'\"")
            if label not in class_index:
                raise ValueError(f"Invalid Otto target {label!r}; expected Class_1 through Class_9")
            features.append(np.asarray([row[i] for i in feature_cols], dtype=np.float32))
            labels.append(class_index[label])
    if not features:
        raise ValueError("Otto data is empty")
    x = np.stack(features)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Otto features must be finite, nonnegative counts")
    return torch.from_numpy(x), torch.tensor(labels, dtype=torch.long)


def load_data(data_dir: Path, data_file: Path | None = None):
    return read_table(prepare_data(data_dir, data_file))


def split_and_standardize(x, y, split_seed=42, *, log1p=True):
    return common.split_and_standardize(x, y, split_seed, num_classes=NUM_CLASSES,
                                        num_continuous=NUM_FEATURES, log1p=log1p)


class ResidualMLP(common.ResidualMLP):
    def __init__(self, width=256, blocks=3):
        super().__init__(width, blocks, num_features=NUM_FEATURES, num_classes=NUM_CLASSES)


def train(args):
    path = prepare_data(args.data_dir, args.data_file)
    return common.train(
        args, experiment="otto", load_data=lambda _: read_table(path),
        split_data=lambda x, y, seed: split_and_standardize(x, y, seed, log1p=args.log1p),
        build_model=ResidualMLP, step_fn=train_step,
        dataset_metadata=dict(dataset_file=str(path.resolve()),
                              dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              dataset_origin="Kaggle Otto Group labeled training set",
                              evaluation_split="held_out_labeled_training_rows",
                              feature_names=FEATURE_NAMES, class_names=CLASS_NAMES,
                              primary_metric="cross_entropy", feature_transform="log1p" if args.log1p else "identity"),
    )


def parse_args(argv=None):
    from experiments.common.config import ConfigParser
    return common.parse_args(ConfigParser(__file__), argv)


if __name__ == "__main__":
    train(parse_args())
