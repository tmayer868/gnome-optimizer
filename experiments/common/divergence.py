"""Divergence detection for training loops.

A run whose loss goes non-finite (NaN/Inf) is dead — every subsequent step is
wasted compute. Experiments call :func:`diverged` each step and, on a hit,
finish the artifact with ``diverged=True`` and exit with :data:`DIVERGED_EXIT`
so a batch/sweep runner can distinguish a blown-up hyperparameter corner from a
real crash.
"""

from __future__ import annotations

import math

# Reserved process exit code for "training diverged" — distinct from a crash
# (exit 1/2). The sweep runner treats it as an expected outcome, not a failure.
DIVERGED_EXIT = 3

# A *finite* loss this large is still unambiguously a blow-up for every
# benchmark here — healthy MSE/CCE losses stay well under ~1e4 even at init, so
# there is no false-positive risk, and it catches runs that explode to a
# huge-but-finite value without ever hitting NaN/Inf.
BLOWUP_THRESHOLD = 1e8


def diverged(*values: float) -> bool:
    """True if any value is non-finite (NaN/Inf) or has blown up past
    ``BLOWUP_THRESHOLD`` — i.e. training has diverged and every further step is
    wasted compute."""
    for v in values:
        v = float(v)
        if not math.isfinite(v) or abs(v) > BLOWUP_THRESHOLD:
            return True
    return False
