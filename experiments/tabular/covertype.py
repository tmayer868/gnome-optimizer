"""Covertype classification with a residual MLP and Gnome/AdamW/SOAP/Muon.

    # Edit experiments/configs/tabular/covertype.py: optimizer='gnome'
    uv run python -m experiments.tabular.covertype

Downloads and caches the UCI data on first use. See docs/covertype.md for
the split protocol, optimizer parameter routing, and reproducibility details.
"""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import torch

from experiments.tabular import common
from experiments.tabular.common import OPTIMIZERS, build_optimizers, evaluate, train_step


# Same archive and checksum used by scikit-learn's fetch_covtype.
DATA_URL = "https://ndownloader.figshare.com/files/5976039"
DATA_SHA256 = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
NUM_FEATURES, NUM_CLASSES, NUM_CONTINUOUS = 54, 7, 10


def load_data(data_dir: Path):
    """Return raw float32 features and zero-based int64 labels, on CPU."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "covtype.data.gz"
    if not archive.exists():
        print("[covertype] downloading dataset (~11 MB)...", flush=True)
        # A failed download never becomes the reusable archive.
        with tempfile.NamedTemporaryFile(dir=data_dir, suffix=".download") as tmp:
            with urllib.request.urlopen(DATA_URL, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    tmp.write(chunk)
            tmp.flush()
            if hashlib.sha256(Path(tmp.name).read_bytes()).hexdigest() != DATA_SHA256:
                raise ValueError("Covertype download checksum mismatch")
            archive.write_bytes(Path(tmp.name).read_bytes())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError(f"Covertype checksum mismatch: remove {archive} and retry")
    print("[covertype] loading dataset...", flush=True)
    with gzip.open(archive, "rt") as file:
        raw = np.loadtxt(file, delimiter=",", dtype=np.float32, ndmin=2)
    if raw.shape[1] != NUM_FEATURES + 1 or not np.isfinite(raw).all():
        raise ValueError("Expected finite Covertype rows with 54 features and a label")
    labels = raw[:, -1]
    if not np.isin(labels, np.arange(1, NUM_CLASSES + 1)).all():
        raise ValueError("Covertype labels must be integers in [1, 7]")
    return torch.from_numpy(raw[:, :-1].copy()), torch.from_numpy(labels.astype(np.int64) - 1)


def split_and_standardize(x, y, split_seed=42):
    return common.split_and_standardize(x, y, split_seed, num_classes=NUM_CLASSES,
                                        num_continuous=NUM_CONTINUOUS)


class ResidualMLP(common.ResidualMLP):
    def __init__(self, width=256, blocks=3):
        super().__init__(width, blocks, num_features=NUM_FEATURES, num_classes=NUM_CLASSES)


def train(args):
    return common.train(args, experiment="covertype", load_data=load_data,
                        split_data=split_and_standardize, build_model=ResidualMLP,
                        dataset_metadata={"dataset_sha256": DATA_SHA256}, step_fn=train_step)


def parse_args(argv=None):
    from experiments.common.config import ConfigParser
    return common.parse_args(ConfigParser(__file__), argv)


if __name__ == "__main__":
    train(parse_args())
