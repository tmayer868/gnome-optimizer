"""Shared input embeddings for PINN experiments.

Every embedding follows the coordinate-wise model contract used by the shared
MLPs: ``embed(t, x)``, ``embed(x, y)``, and so on. Each exposes ``out_dim`` so
the following affine layer can be constructed without a dummy forward pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


class ConcatEmbedding(nn.Module):
    """Return input coordinates unchanged, concatenated along feature dim 1."""

    def __init__(self, n_inputs: int):
        super().__init__()
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be positive, got {n_inputs}")
        self.n_inputs = n_inputs
        self.out_dim = n_inputs

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        self._check_coords(coords)
        return torch.cat(coords, dim=1)

    def _check_coords(self, coords: tuple[torch.Tensor, ...]) -> None:
        if len(coords) != self.n_inputs:
            raise ValueError(
                f"expected {self.n_inputs} coordinate tensors, got {len(coords)}"
            )


class PeriodicEmbedding(nn.Module):
    """Replace selected coordinates with fixed periodic harmonics.

    With the defaults, ``forward(t, x)`` returns

    ``[t, cos(k*x), sin(k*x), ..., cos(n*k*x), sin(n*k*x)]``.

    ``periodic_dims`` selects which coordinate positions are replaced. Raw
    coordinates not listed there pass through in their original position. For
    example, ``periodic_dims=(0, 1)`` embeds both ``x`` and ``y`` for a fully
    periodic 2-D problem. Harmonics are interleaved cosine then sine for each
    mode, and coordinates retain their input order.
    """

    def __init__(
        self,
        n_inputs: int = 2,
        *,
        n_harmonics: int = 1,
        wavenumber: float = math.pi,
        periodic_dims: Sequence[int] = (1,),
    ):
        super().__init__()
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be positive, got {n_inputs}")
        if n_harmonics < 1:
            raise ValueError(
                f"n_harmonics must be positive, got {n_harmonics}"
            )
        if not math.isfinite(wavenumber) or wavenumber <= 0.0:
            raise ValueError(
                f"wavenumber must be finite and positive, got {wavenumber}"
            )

        dims = tuple(periodic_dims)
        if not dims:
            raise ValueError("periodic_dims must contain at least one index")
        if len(set(dims)) != len(dims):
            raise ValueError(f"periodic_dims contains duplicates: {dims}")
        if any(dim < 0 or dim >= n_inputs for dim in dims):
            raise ValueError(
                f"periodic_dims must be in [0, {n_inputs}), got {dims}"
            )

        self.n_inputs = n_inputs
        self.n_harmonics = n_harmonics
        self.wavenumber = float(wavenumber)
        self.periodic_dims = dims
        self._periodic_dim_set = frozenset(dims)
        self.out_dim = (
            n_inputs - len(dims) + 2 * n_harmonics * len(dims)
        )
        # Keep mode numbers integral and apply ``wavenumber`` after the input
        # tensor, so float64 coordinates do not inherit a float32-rounded pi
        # from a buffer created under the default dtype.
        self.register_buffer(
            "harmonic_numbers", torch.arange(1, n_harmonics + 1)
        )

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        if len(coords) != self.n_inputs:
            raise ValueError(
                f"expected {self.n_inputs} coordinate tensors, got {len(coords)}"
            )

        features: list[torch.Tensor] = []
        for index, coord in enumerate(coords):
            if index not in self._periodic_dim_set:
                features.append(coord)
                continue
            phase = coord * self.wavenumber * self.harmonic_numbers
            # (N, n, 2) -> (N, 2*n): cos(kx), sin(kx), cos(2kx), ...
            harmonics = torch.stack(
                [torch.cos(phase), torch.sin(phase)], dim=-1
            ).flatten(start_dim=1)
            features.append(harmonics)
        return torch.cat(features, dim=1)


class TrainableFourierEmbedding(nn.Module):
    """Trainable random Fourier features over concatenated coordinates.

    For raw coordinates ``z``, returns ``[sin(zB), cos(zB)]`` with
    ``B ~ N(0, scale^2)`` at initialization. ``B`` is an ``nn.Parameter`` and
    therefore trains with the rest of the model. Set ``include_input=True``
    to prepend the raw coordinates to the Fourier features.
    """

    def __init__(
        self,
        n_inputs: int,
        embed_dim: int = 256,
        scale: float = 10.0,
        *,
        include_input: bool = False,
    ):
        super().__init__()
        if n_inputs < 1:
            raise ValueError(f"n_inputs must be positive, got {n_inputs}")
        if embed_dim < 2 or embed_dim % 2 != 0:
            raise ValueError(
                f"embed_dim must be a positive even number, got {embed_dim}"
            )
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale must be finite and positive, got {scale}")

        self.n_inputs = n_inputs
        self.embed_dim = embed_dim
        self.scale = float(scale)
        self.include_input = include_input
        self.out_dim = embed_dim + (n_inputs if include_input else 0)
        self.B = nn.Parameter(torch.randn(n_inputs, embed_dim // 2) * scale)

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        if len(coords) != self.n_inputs:
            raise ValueError(
                f"expected {self.n_inputs} coordinate tensors, got {len(coords)}"
            )
        raw = torch.cat(coords, dim=1)
        projection = raw @ self.B
        fourier = torch.cat(
            [torch.sin(projection), torch.cos(projection)], dim=1
        )
        if self.include_input:
            return torch.cat([raw, fourier], dim=1)
        return fourier


# Backward-compatible name for external callers; experiments use the explicit
# ``ConcatEmbedding`` name so all embedding imports point at this module.
ConcatEmbed = ConcatEmbedding


__all__ = [
    "ConcatEmbed",
    "ConcatEmbedding",
    "PeriodicEmbedding",
    "TrainableFourierEmbedding",
]
