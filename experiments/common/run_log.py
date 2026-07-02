"""Streaming run logger for Gnome benchmark experiments.

One run writes a single append-only JSONL file at
``runs/{experiment}/{run_id}.jsonl``. Every training/validation event is one
self-describing line, so the file is:

* **monitorable in real time** — ``tail -f runs/.../<run_id>.jsonl`` shows
  progress as it happens, and each line parses on its own;
* **crash-safe** — a killed run only ever truncates the final partial line;
  everything before it is intact;
* **cheap** — each log call appends one line and flushes (O(1)), instead of
  re-serializing a growing document.

Schema — the first line is a ``meta`` record with run identity, hyperparameters,
and environment; then one record per logged event; then a final ``end`` record::

    {"type":"meta","experiment":"burgers_pinn","optimizer":"gnome","seed":0,
     "run_id":"...","hyperparameters":{...},"git_sha":"...",
     "torch_version":"2.12.0","python_version":"3.14.3","start_time":1782...}
    {"type":"train","step":200,"metrics":{"pde":1.04e-2,"rel_l2":1.08}}
    {"type":"val","step":15000,"epoch":null,"metrics":{"rel_l2":2.08e-1}}
    {"type":"end","completed":true,"wall_time_seconds":16843.2}

``metrics`` is an open-ended dict, so a new experiment can log any metric names
(``rel_l2``, ``pde``, ``top1``, ``perplexity``, ...) without a schema change.

Read runs back with :func:`load_run`, which returns a :class:`RunRecord` for
plotting and aggregation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ----------------------------------------------------------------------
# Environment capture (written once, into the meta line)
# ----------------------------------------------------------------------

def _git_sha() -> Optional[str]:
    """Return the current commit SHA, or None if not inside a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _torch_version() -> Optional[str]:
    try:
        import torch
        return torch.__version__
    except Exception:
        return None


def make_run_id(experiment: str, optimizer: str, seed: int) -> str:
    """Construct a unique-ish run id from identity fields and a UNIX timestamp."""
    return f"{experiment}_{optimizer}_seed{seed}_{int(time.time())}"


def _coerce(value: Any) -> Any:
    """Make a metric value JSON-serializable.

    Torch tensors and numpy scalars (anything with ``.item()``) collapse to a
    Python scalar; ints/floats/bools/strings/None pass through untouched.
    """
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------

class RunLogger:
    """Append-only JSONL logger for a single training run.

    Opens (and creates) the run file on construction, immediately writing the
    ``meta`` line, and appends one line per :meth:`log` call. Call
    :meth:`finish` when done — or use it as a context manager, which finishes
    automatically (marking the run incomplete if an exception propagates).

    Args:
        experiment: Experiment slug; also the subdirectory under ``runs_dir``.
        optimizer: Optimizer slug (``"gnome"``, ``"soap"``, ``"adamw"``, ...).
        seed: Random seed for the run.
        hyperparameters: Dict of all run hyperparameters (optimizer args,
            schedule, model config, ...). Stored verbatim in the meta line.
        runs_dir: Root directory for run files. Default ``"runs"``.
        run_id: Override the generated run id (also the filename stem).
        path: Override the full output path (ignores ``runs_dir``/``run_id``).
    """

    def __init__(
        self,
        experiment: str,
        optimizer: str,
        seed: int,
        hyperparameters: dict,
        runs_dir: str = "runs",
        run_id: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.experiment = experiment
        self.optimizer = optimizer
        self.seed = seed
        self.run_id = run_id or make_run_id(experiment, optimizer, seed)
        self.path = path or os.path.join(
            runs_dir, experiment, f"{self.run_id}.jsonl"
        )
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        self._start = time.monotonic()
        self._finished = False
        self._file = open(self.path, "a")

        self._write({
            "type": "meta",
            "experiment": experiment,
            "optimizer": optimizer,
            "seed": seed,
            "run_id": self.run_id,
            "hyperparameters": hyperparameters,
            "git_sha": _git_sha(),
            "torch_version": _torch_version(),
            "python_version": sys.version.split()[0],
            "start_time": time.time(),
        })

    # -- core write ----------------------------------------------------

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log(
        self,
        kind: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **metrics: Any,
    ) -> None:
        """Append one event line of type ``kind`` with a nested metrics dict."""
        record: dict[str, Any] = {"type": kind}
        if step is not None:
            record["step"] = int(step)
        if epoch is not None:
            record["epoch"] = int(epoch)
        record["metrics"] = {k: _coerce(v) for k, v in metrics.items()}
        self._write(record)

    def log_train(self, step: int, **metrics: Any) -> None:
        """Record train-side metrics at global ``step``."""
        self.log("train", step=step, **metrics)

    def log_val(
        self, step: int, epoch: Optional[int] = None, **metrics: Any
    ) -> None:
        """Record validation metrics at global ``step`` (optionally ``epoch``)."""
        self.log("val", step=step, epoch=epoch, **metrics)

    # -- lifecycle -----------------------------------------------------

    def finish(self, completed: bool = True, **summary: Any) -> str:
        """Write the terminal ``end`` record and close the file.

        Idempotent — a second call is a no-op. Any ``summary`` keyword args
        (e.g. best/final metrics) are merged into the end record for cheap
        aggregation without re-reading the whole file. Returns the run path.
        """
        if not self._finished:
            self._write({
                "type": "end",
                "completed": completed,
                "wall_time_seconds": time.monotonic() - self._start,
                **{k: _coerce(v) for k, v in summary.items()},
            })
            self._file.close()
            self._finished = True
        return self.path

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(completed=exc_type is None)


# ----------------------------------------------------------------------
# Reader
# ----------------------------------------------------------------------

@dataclass
class RunRecord:
    """A parsed run: the meta line, ordered events, and the end record."""

    meta: dict
    events: list[dict] = field(default_factory=list)
    end: Optional[dict] = None
    path: Optional[str] = None

    @property
    def experiment(self) -> Any:
        return self.meta.get("experiment")

    @property
    def optimizer(self) -> Any:
        return self.meta.get("optimizer")

    @property
    def seed(self) -> Any:
        return self.meta.get("seed")

    @property
    def hyperparameters(self) -> dict:
        return self.meta.get("hyperparameters", {})

    @property
    def completed(self) -> bool:
        return bool(self.end and self.end.get("completed"))

    def series(self, kind: str, metric: str) -> tuple[list, list]:
        """Return ``(steps, values)`` for one metric among ``kind`` events."""
        steps, values = [], []
        for e in self.events:
            if e.get("type") == kind:
                m = e.get("metrics", {})
                if metric in m:
                    steps.append(e.get("step"))
                    values.append(m[metric])
        return steps, values

    def final(self, kind: str, metric: str) -> Any:
        """Last logged value of ``metric`` among ``kind`` events (or None)."""
        _, values = self.series(kind, metric)
        return values[-1] if values else None

    def best(self, kind: str, metric: str, mode: str = "min") -> Any:
        """Best (min/max) value of ``metric`` among ``kind`` events (or None)."""
        _, values = self.series(kind, metric)
        if not values:
            return None
        return (min if mode == "min" else max)(values)


def load_run(path: str) -> RunRecord:
    """Parse a ``.jsonl`` run file into a :class:`RunRecord`.

    Tolerant of a truncated final line (a crashed run): a trailing partial
    record is skipped rather than raising.
    """
    meta: dict = {}
    events: list[dict] = []
    end: Optional[dict] = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from a killed run
            kind = rec.get("type")
            if kind == "meta":
                meta = rec
            elif kind == "end":
                end = rec
            else:
                events.append(rec)
    return RunRecord(meta=meta, events=events, end=end, path=path)
