"""Shared on-disk cache for reproducible experiment reference solutions."""

from __future__ import annotations

import shutil
from pathlib import Path
from collections.abc import Iterable


REFERENCE_SOLUTIONS_DIR = Path(__file__).resolve().parent


def reference_path(filename: str) -> str:
    """Return an absolute path inside the shared reference-solution cache."""
    path = Path(filename)
    if path.name != filename or filename in {"", ".", ".."}:
        raise ValueError(f"reference filename must be a basename; got {filename!r}")
    return str(REFERENCE_SOLUTIONS_DIR / filename)


def cached_reference_path(
    filename: str,
    legacy_paths: Iterable[str] = (),
) -> str:
    """Return the shared path, migrating the first legacy cache found.

    Existing projects historically cached references under ``experiments/data``
    or at the repository root. Copying once preserves those downloads while all
    subsequent reads and writes use this tracked, experiment-independent cache.
    """
    destination = Path(reference_path(filename))
    if destination.is_file():
        return str(destination)
    for legacy_path in legacy_paths:
        source = Path(legacy_path)
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            break
    return str(destination)


__all__ = ["REFERENCE_SOLUTIONS_DIR", "cached_reference_path", "reference_path"]
