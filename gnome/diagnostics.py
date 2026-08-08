"""Sinks for Gnome's internal-state diagnostics.

These are for the optimizer's *internals* — curvature spectrum, damping,
step geometry — not for training metrics. Training metrics belong in your
experiment's own logger; this is the channel for answering questions like
"is the trust region binding?", "what fraction of the curvature estimate is
below eps?", "did v_max blow up right before the loss did?".

Pass any callable taking a record dict::

    from gnome import Gnome, PrintDiagnostics

    opt = Gnome(model.parameters(), lr=1e-3, loss='mse',
                diagnostics=PrintDiagnostics(), diagnostics_every=1000)

Nothing is computed unless a record is due, so leaving ``diagnostics=None``
(the default) costs one integer comparison per parameter per step.

**What gets measured is separate from where it goes.** ``diagnostics=`` is the
sink; ``metrics=`` is a ``{name: fn}`` dict saying what to compute. Adding a
statistic needs no change to the optimizer::

    from gnome import Gnome, DEFAULT_METRICS

    opt = Gnome(..., diagnostics=PrintDiagnostics(),
                metrics={**DEFAULT_METRICS,
                         "v_cond": lambda c: c["v"].max() / c["v"].min(),
                         "gg_trace": lambda c: c["state"]["GG"][0].diag().sum()})

Metric context ``c``
--------------------
Each metric receives this dict and must return a **0-dim tensor** (see
:data:`DEFAULT_METRICS` for why).

``v``           bias-corrected curvature — the diagonal GGN estimate in the
                rotated eigenbasis, i.e. the estimated GGN eigenvalues
``m``           bias-corrected gradient EMA, same basis
``update``      the damped Newton step ``m / (v + lam)``, pre-projection.
                Basis-independent in norm, since Q is orthonormal
``lam``         effective LM damping. A 0-dim tensor when the trust region is
                active; the Python float ``eps`` when ``trust_radius=None``
``eps``         curvature floor (Python float)
``lr``          learning rate applied this step
``trust_radius`` the group's trust radius, or ``None``
``p``           the parameter tensor itself
``state``       Gnome's per-parameter state — the escape hatch. Holds the raw
                pre-bias-correction EMAs ``gnd_m`` / ``grad_m``, the Kronecker
                factors ``GG``, the eigenbasis ``Q``, and ``lm_lambda``.
                Careful: ``GG`` and ``Q`` are *lists* whose entries are ``[]``
                for modes carrying no factor (1D params under
                ``precondition_1d=False``, or any mode wider than
                ``max_precond_dim``). A metric touching them runs for every
                parameter, so guard it — ``len(c["state"]["GG"][0]) and ...``
                — or it will hit an empty list on the biases
``group``       the parameter group (all hyperparameters)
``step``        global optimizer step
``param``       parameter index

Record fields
-------------
Metric names become record fields directly. Alongside them, these are always
present and need no reduction:
``step``              global optimizer step (one per ``step()`` call)
``param``             index of the parameter in ``step()``'s iteration order,
                      which matches ``[p for p in model.parameters()
                      if p.requires_grad]`` for the usual construction
``shape``, ``numel``  the parameter's shape and element count
``lr``                learning rate applied on this step
``eps``, ``lam``      the curvature floor and the effective LM damping
``lam_binding``       ``lam > eps``, i.e. the trust region actually bound this
                      step rather than passing through an undamped Newton step

From :data:`DEFAULT_METRICS` — present only while those metrics are in use:

``v_min``, ``v_max``, ``v_mean``
                      curvature (the diagonal GGN estimate) in the rotated
                      eigenbasis, after bias correction — these are the
                      estimated GGN eigenvalues
``v_frac_below_eps``  fraction of coordinates whose curvature sits under
                      ``eps``. Only meaningful when ``eps`` is in the same
                      range as the curvature — set ``eps`` far below
                      ``v_min`` and this reads 0.0 for the whole run because
                      it genuinely is zero, not because it rounded
``v_frac_below_lam``  fraction of coordinates whose curvature sits under the
                      *effective* damping ``lam``. Usually the one you want:
                      the update is ``m / (v + lam)``, so these are the
                      directions where damping outweighs curvature and the
                      step is gradient-descent-like rather than Newton-like.
                      Equals ``v_frac_below_eps`` exactly when
                      ``trust_radius=None``; above it whenever the trust
                      region binds and pushes ``lam`` past ``eps``
``grad_rms``          RMS of the bias-corrected gradient EMA
``update_rms``        RMS of the update, in units of ``lr``. Basis-independent
                      (the Q matrices are orthonormal), so it is also the RMS
                      of the final parameter update
``trust_ratio``       ``update_rms / trust_radius`` — the fraction of the
                      trust-region budget spent; ~1.0 means the constraint is
                      active. ``None`` when ``trust_radius=None``. Derived
                      from ``update_rms``, so it disappears if you drop that
                      metric
"""

from __future__ import annotations

import json
import sys
from typing import Callable, Iterable, Optional, TextIO


# ----------------------------------------------------------------------
# What gets measured
# ----------------------------------------------------------------------

DEFAULT_METRICS = {
    # Curvature: the diagonal GGN estimate in the rotated eigenbasis, so
    # these entries are the estimated GGN eigenvalues.
    "v_min": lambda c: c["v"].min(),
    "v_max": lambda c: c["v"].max(),
    "v_mean": lambda c: c["v"].mean(),
    # Fraction of directions the eps floor is holding up. Only informative
    # when eps is in the same range as the curvature.
    "v_frac_below_eps":
        lambda c: (c["v"] < c["eps"]).to(c["v"].dtype).mean(),
    # Fraction where the *effective* damping outweighs the curvature. The
    # update is m / (v + lam), so these coordinates get a gradient-descent-like
    # step rather than a Newton one — the Newton-to-GD mix, measured.
    "v_frac_below_lam":
        lambda c: (c["v"] < c["lam"]).to(c["v"].dtype).mean(),
    "grad_rms": lambda c: c["m"].square().mean().sqrt(),
    # Basis-independent: Q is orthonormal, so this is also the RMS of the
    # final parameter update, in units of lr.
    "update_rms": lambda c: c["update"].square().mean().sqrt(),
}
"""Default ``{name: fn}`` metric set. Pass your own via ``Gnome(metrics=...)``.

Each ``fn`` takes the context dict described in this module's docstring and
returns a **0-dim tensor**. Returning a Python float raises: every metric is
stacked into one host transfer, and a pre-converted scalar would force its own
device sync. Keep the reduction on-device — ``c["v"].max()``, not
``float(c["v"].max())``.

Extend rather than replace, so the built-in fields keep working::

    metrics={**DEFAULT_METRICS,
             "v_cond":   lambda c: c["v"].max() / c["v"].min().clamp_min(1e-30),
             "gnd_m_max": lambda c: c["state"]["gnd_m"].max()}

Note ``trust_ratio`` is derived from ``update_rms``, so dropping that metric
also drops ``trust_ratio`` from the record.
"""


_DEFAULT_FIELDS = (
    "v_min", "v_max", "v_mean", "v_frac_below_eps", "v_frac_below_lam",
    "lam", "grad_rms", "update_rms", "trust_ratio",
)


class PrintDiagnostics:
    """Print one line per record.

    Args:
        fields: Record keys to show, in order. Defaults to the curvature and
            step-geometry summary; pass your own to narrow it.
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
        self._fh.write(json.dumps(rec) + "\n")
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


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3e}"
    return str(v)
