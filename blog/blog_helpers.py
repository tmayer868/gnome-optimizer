"""Helpers for the Gnome blog: locate run artifacts and style them consistently.

The blog reads the same append-only ``.jsonl`` run files the experiments write
(via :mod:`experiments.common.run_log`), so every figure is regenerated from the
real logs — nothing is hand-transcribed.
"""

from __future__ import annotations

import glob
import os

from experiments.common import load_run, RunRecord

# Repo root = parent of this file's directory.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(REPO, "runs")

# Consistent per-optimizer styling across every figure.
STYLE = {
    "gnome":            dict(color="#0b7285", label="Gnome"),
    "gnome_fisher":     dict(color="#0b7285", label="Gnome (Fisher)"),
    "gnome_hutchinson": dict(color="#12b886", label="Gnome (Hutchinson)"),
    "soap":             dict(color="#e8590c", label="SOAP"),
    "adamw":            dict(color="#868e96", label="AdamW"),
}


def find_runs(experiment: str, runs_dir: str = RUNS) -> list[RunRecord]:
    """Load every run under ``runs/{experiment}/``."""
    paths = sorted(glob.glob(os.path.join(runs_dir, experiment, "*.jsonl")))
    return [load_run(p) for p in paths]


def latest(
    experiment: str,
    optimizer: str,
    where=None,
    completed_only: bool = False,
    runs_dir: str = RUNS,
) -> RunRecord:
    """Most recent run for ``(experiment, optimizer)`` matching an optional
    hyperparameter predicate ``where(hyperparameters) -> bool``.

    Selecting on hyperparameters lets a figure pin, e.g., the *raw* (no-decay)
    baseline vs the cosine-decayed one when both were run. ``completed_only``
    skips runs that never wrote an ``end`` record (killed / crashed / still
    running), so a figure never plots a truncated curve against finished ones.
    """
    candidates = []
    for r in find_runs(experiment, runs_dir):
        if r.optimizer != optimizer:
            continue
        if completed_only and not r.completed:
            continue
        if where is not None and not where(r.hyperparameters):
            continue
        candidates.append(r)
    if not candidates:
        raise FileNotFoundError(
            f"no run found for {experiment}/{optimizer}"
            + (" matching predicate" if where else "")
        )
    return max(candidates, key=lambda r: r.meta.get("start_time", 0))
