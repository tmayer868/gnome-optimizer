"""State metrics and output sinks for explicitly requested diagnostics.

Experiments own the reporting cadence and destination::

    sink = JsonlDiagnostics("run.diag.jsonl")
    for step in range(steps):
        optimizer.step(main_closure, aux_closure)
        if (step + 1) % 100 == 0:
            for record in optimizer.get_diagnostics():
                sink(record)
    sink.close()

get_diagnostics() reports the stored LM damping by default. Supply a list of
read-only callables for other persistent-state measurements::

    optimizer.get_diagnostics([
        lm_damping,
        lambda c: {"curvature_ema_max": c["state"]["gnd_m"].max()},
    ])

Each callable receives p, state, group, step, and param in a context dictionary
and returns named scalar tensors or Python scalars. Values describe the current
stored state; temporary update tensors are unavailable.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Callable, Iterable, Optional, TextIO


def lm_damping(context: dict) -> dict:
    """Actual lambda used on the last update; None during basis initialization."""
    return {"lam": context["state"].get("lm_lambda")}


_DEFAULT_FIELDS = ("lam",)


class PrintDiagnostics:
    """Print one line per record.

    Args:
        fields: Record keys to show, in order. Defaults to the stored damping;
            pass your own to show custom metrics.
        params: If given, only emit for these parameter indices. A 40-tensor
            model at ``diagnostics_every=100`` is 40 lines every 100 steps,
            which is usually more than you want — ``params=[0]`` or a couple
            of representative layers reads much better.
        stream: Defaults to stderr, so diagnostics stay separable from
            training output on stdout.
        prefix: Leading tag on each line.
    """

    def __init__(
        self,
        fields: Optional[Iterable[str]] = None,
        params: Optional[Iterable[int]] = None,
        stream: Optional[TextIO] = None,
        prefix: str = "[gnome]",
    ):
        self.fields = tuple(fields) if fields is not None else _DEFAULT_FIELDS
        self.params = None if params is None else set(params)
        self.stream = stream if stream is not None else sys.stderr
        self.prefix = prefix

    def __call__(self, rec: dict) -> None:
        if self.params is not None and rec["param"] not in self.params:
            return
        shape = "x".join(str(d) for d in rec["shape"])
        body = "  ".join(
            f"{k}={_fmt(rec.get(k))}" for k in self.fields
        )
        print(
            f"{self.prefix} step {rec['step']:>7d}  "
            f"p{rec['param']:02d} [{shape}]  {body}",
            file=self.stream,
            flush=True,
        )


class JsonlDiagnostics:
    """Append each record to a JSONL file, one JSON object per line.

    Use this when you want to plot the curvature history afterwards rather
    than read it scroll past. Pairs with ``pandas.read_json(path, lines=True)``.

    The file handle is opened on first record and held open; call
    :meth:`close` when done, or use it as a context manager.
    """

    def __init__(self, path: str, params: Optional[Iterable[int]] = None):
        self.path = path
        self.params = None if params is None else set(params)
        self._fh: Optional[TextIO] = None

    def __call__(self, rec: dict) -> None:
        if self.params is not None and rec["param"] not in self.params:
            return
        if self._fh is None:
            self._fh = open(self.path, "a")
        # shape is a tuple; JSON turns it into a list either way.
        # allow_nan=False turns a missed non-finite into a loud error rather
        # than a file that only Python can read back.
        self._fh.write(json.dumps(_json_safe(rec), allow_nan=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlDiagnostics":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CollectDiagnostics:
    """Accumulate records in a list, for poking at in a notebook or a test.

    Unbounded by default — set ``max_records`` if a long run would otherwise
    eat memory.
    """

    def __init__(self, max_records: Optional[int] = None,
                 params: Optional[Iterable[int]] = None):
        self.records: list[dict] = []
        self.max_records = max_records
        self.params = None if params is None else set(params)

    def __call__(self, rec: dict) -> None:
        if self.params is not None and rec["param"] not in self.params:
            return
        if self.max_records is not None and len(self.records) >= self.max_records:
            return
        self.records.append(rec)

    def __len__(self) -> int:
        return len(self.records)


def multi(*sinks: Callable[[dict], None]) -> Callable[[dict], None]:
    """Fan one record out to several sinks: ``multi(PrintDiagnostics(), jsonl)``."""
    def _fan(rec: dict) -> None:
        for s in sinks:
            s(rec)
    return _fan


def _json_safe(rec: dict) -> dict:
    """Map non-finite floats to ``None``.

    Custom metrics may use NaN for "not applicable",
    but ``json.dumps`` writes that as a bare ``NaN`` token, which is not valid
    JSON — Python reads it back, ``jq`` and most other parsers do not. ``null``
    round-trips everywhere, and pandas reads it back as NaN regardless.
    """
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in rec.items()
    }


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3e}"
    return str(v)
