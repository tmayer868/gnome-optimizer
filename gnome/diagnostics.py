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

``v``           bias-corrected curvature: the EMA of the squared rotated
                surrogate gradient, which estimates ``diag(Q^T H Q)`` — the
                *diagonal of the rotated GGN*. Not the eigenvalues of H: those
                coincide only when Q diagonalizes H, i.e. only when rho = 0,
                and rho is precisely what measures that gap
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
                      the rotated-GGN diagonal ``diag(Q^T H Q)``, after bias
                      correction. These are eigenvalues of H only in the
                      rho = 0 limit
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
``captured_energy``   ``‖diag(Q_Gᵀ H Q_G)‖²_F`` — curvature energy Gnome's
                      eigenbasis actually captures on the diagonal
``energy_ratio``      ``captured / ‖M_G‖²_F``. ``>1`` is the factorization
                      gap — the curvature eigenvalues do not factorize, so
                      re-estimating the diagonal beats any Kronecker-product
                      model of the GGN (i.e. beats KFAC; Shampoo factorizes
                      loss gradients, a different matrix, so it is not what
                      this compares against). ``<1`` means ``rho_proxy`` is a
                      valid upper bound; ``≈1`` means the model is accurate.
                      ``NaN`` for 1D params. ``kron_energy`` itself is
                      recoverable as ``captured_energy / energy_ratio``
``rho_proxy``         ``sqrt(1 - captured/kron)`` — Kronecker-model proxy for
                      the off-diagonal mass fraction ``ρ(Q_G)``. 0 means the
                      basis diagonalizes the curvature exactly; near 1 means
                      most of it sits off-diagonal. Conservative, but a hard 0
                      is ambiguous (also what the clamp returns when
                      ``captured > kron``) — read it with ``energy_ratio``

The eigenbasis-quality group is unreliable for roughly the first 50-75 steps:
``GG`` is an EMA from zero carrying no bias correction while ``v`` is
bias-corrected, so ``kron`` is understated early and ``energy_ratio`` inflated.
During that window ``rho_proxy`` clamps to ``0.0``, which is indistinguishable
from a genuine zero — a Kronecker-model-free ρ for the early transient needs a
dedicated measurement script, not a per-step metric (see the note above
:data:`DEFAULT_METRICS`).
"""

from __future__ import annotations

import json
import math
import sys
from typing import Callable, Iterable, Optional, TextIO


# ----------------------------------------------------------------------
# Eigenbasis quality
# ----------------------------------------------------------------------
#
# How much of the curvature does Gnome's basis actually diagonalize? The
# rotated curvature ``v`` is the diagonal Gnome keeps; the Kronecker model
# carries the total. Their ratio bounds the off-diagonal mass Gnome discards.


def _nan_like(t):
    """0-dim NaN matching ``t``'s dtype/device — the 'not applicable' value."""
    return t.sum().new_full((), float("nan"))


def _kron_energy(c):
    """``‖M_G‖²_F``, or ``None`` when there is no separable model.

    ``M_G = (Â ⊗ B̂) / tr(H)`` is diagonal in ``Q_G``, so its Frobenius norm
    equals its diagonal norm; when the Kronecker model is accurate this
    approximates ``‖H‖²_F``. Uses ``‖A ⊗ B‖²_F = ‖A‖²_F · ‖B‖²_F``, so no
    Kronecker product is formed.

    Only the first two modes are used — all of them for an MLP's 2D weights,
    but a >2D parameter (conv) has a factor per mode and the rest are ignored.

    Internal helper, not a metric: it returns ``None`` rather than a tensor
    for the inapplicable cases.
    """
    v, GG = c["v"], c["state"]["GG"]
    # 1D params have no separable-basis question at all. The len(GG) < 2 test
    # is a second guard: with merge_dims a 2D param can collapse to a single
    # factor, so v.dim() alone does not guarantee GG[1] exists. Empty entries
    # mean the mode was skipped (precondition_1d=False, or > max_precond_dim).
    if v.dim() < 2 or len(GG) < 2 or len(GG[0]) == 0 or len(GG[1]) == 0:
        return None
    A, B = GG[0], GG[1]
    trH = A.trace().clamp_min(1e-30)
    return (A * A).sum() * (B * B).sum() / (trH * trH)


def _captured_energy(c):
    """``‖diag(Q_Gᵀ H Q_G)‖²_F`` — curvature energy Gnome's basis captures."""
    return (c["v"] ** 2).sum()


def _energy_ratio(c):
    """``captured / kron_energy``.

    | ``> 1`` — factorization gap: the curvature eigenvalues do not
      factorize, so re-estimating the diagonal in the basis beats any
      Kronecker-*product* model of the GGN. That bounds KFAC; Shampoo
      factorizes loss-gradient outer products, a different matrix entirely,
      and SOAP also re-estimates its diagonal, so neither is what this
      compares against
    | ``< 1`` — ``rho_proxy`` is valid as an upper bound on true ρ
    | ``≈ 1`` — Kronecker model accurate, proxy ≈ true ρ
    | ``NaN`` — 1D param (no separable basis question)

    Expect ``> 1`` for the first tens of steps regardless: ``GG`` is an EMA
    from zero with no bias correction while ``v`` is bias-corrected, so
    ``kron`` is understated early and the ratio is inflated.
    """
    kron = _kron_energy(c)
    if kron is None:
        return _nan_like(c["v"])
    return (c["v"] ** 2).sum() / kron.clamp_min(1e-30)


def _rho_proxy(c):
    """Proxy ρ: ``sqrt(1 - captured / kron)``, clamped to 0 when captured > kron.

    Uses ``‖M_G‖`` as a stand-in for ``‖H‖``. Biased when the Kronecker model
    is poor, but conservatively: understating ``‖M_G‖`` overstates ρ rather
    than under. 0 means the basis diagonalizes the curvature exactly; near 1
    means most of the curvature sits off-diagonal.

    A hard 0 is ambiguous — it is also what the clamp returns whenever
    ``captured > kron``, so read it alongside ``energy_ratio``.
    """
    kron = _kron_energy(c)
    if kron is None:
        return _nan_like(c["v"])
    return (1.0 - (c["v"] ** 2).sum() / kron.clamp_min(1e-30)).clamp_min(0.0).sqrt()


# A note on what is deliberately *not* here: a fourth-moment estimate of
# trace(H^2), which would give rho without the Kronecker model. It needs an EMA
# of ||g_s||^4 updated on every step, i.e. optimizer state maintained purely to
# be observed — unlike v/GG/Q/lam, which exist because the update needs them.
# That cost is paid whether or not anyone is looking, and the estimator is
# biased anyway: the Gaussian identity assumes a fixed H, while an EMA averages
# over a window in which the curvature is moving. Measuring it properly means
# freezing the parameters and drawing independent surrogate samples there,
# which belongs in a dedicated script rather than the hot path.


# ----------------------------------------------------------------------
# What gets measured
# ----------------------------------------------------------------------

DEFAULT_METRICS = {
    # Curvature: diag(Q^T H Q), the rotated-GGN diagonal. Eigenvalues of H
    # only when Q diagonalizes H (rho = 0).
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
    # Eigenbasis quality — how much curvature Gnome's basis diagonalizes.
    # kron_energy is not itself a metric: it is recoverable as
    # captured_energy / energy_ratio, and returning None for the
    # inapplicable cases keeps it out of the record cleanly.
    "captured_energy": _captured_energy,
    "energy_ratio": _energy_ratio,
    "rho_proxy": _rho_proxy,
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


# captured_energy is an intermediate of the ratios, so it is in the record but
# not printed. The line is wide with everything else on it — pass fields= to
# narrow it (e.g. fields=("rho_proxy", "energy_ratio", "trust_ratio")).
_DEFAULT_FIELDS = (
    "v_min", "v_max", "v_mean", "v_frac_below_eps", "v_frac_below_lam",
    "lam", "grad_rms", "update_rms", "trust_ratio",
    "energy_ratio", "rho_proxy",
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

    Metrics use NaN for "not applicable" (``rho_proxy`` on a 1D parameter, say),
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
