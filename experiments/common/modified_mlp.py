"""Modified MLP (Wang, Teng & Perdikaris 2021) — shared by the PINN benchmarks.

Single source of truth for the gated architecture used by ``allen_cahn``,
``kdv``, ``kuramoto_sivashinsky``, ``ldc`` and ``wave``. Each of those files
supplies its own input embedding and (where needed) an output transform; the
trunk lives here so the encoder layout stays identical across benchmarks.

Two encoders gate every hidden layer. The gate is written in the
algebraically equivalent form ``h = v + h·(u - v)`` via one fused
``addcmul`` (rather than ``h·u + (1-h)·v``, three elementwise kernels and
three autograd nodes).

**The two encoders are one fused ``Linear`` producing ``2·hidden`` features,
not two ``Linear``s.** Besides saving a GEMM launch, this is load-bearing for
the optimizer study: Gnome preconditions per parameter *tensor*, so the
split form gives ``u`` and ``v`` two independent Kronecker factor pairs and
two independent trust regions, while the fused form gives them one of each.
Concretely, for the fused weight the output-side factor ``G Gᵀ`` is

    [ Gᵤ Gᵤᵀ   Gᵤ Gᵥᵀ ]
    [ Gᵥ Gᵤᵀ   Gᵥ Gᵥᵀ ]

whose diagonal blocks are exactly the two split preconditioners and whose
off-diagonal blocks carry the u–v coupling the split form cannot represent —
coupling that is real, since the gates combine multiplicatively in the same
expression. The input-side factor conversely becomes ``GᵤᵀGᵤ + GᵥᵀGᵥ``, one
shared basis where the split form had two, and the trust-region solve
(``_lm_lambda``, once per tensor) now sets one damping level for both
encoders instead of two. So fusing is *not* optimizer-neutral: it is the same
function of the inputs, trained along a different trajectory. Do not "simplify"
this back into two ``Linear``s — it would silently change what is being
benchmarked, and differently in different files.

Architecture only — no random weight factorization, Fourier features or
grad-norm balancing (jaxpi-pipeline pieces, deliberately not ported).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


class ConcatEmbed(nn.Module):
    """``[c₁, …, c_n]`` — the raw coordinates, concatenated. No parameters.

    The identity embedding, for problems with no periodicity to encode.
    """

    def __init__(self, n_coords: int):
        super().__init__()
        self.out_dim = n_coords

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        return torch.cat(coords, dim=1)


class ModifiedMLP(nn.Module):
    """Gated MLP over an input embedding: ``coords → out_features``.

    ``embed`` is any module exposing an ``out_dim`` attribute and accepting
    the problem's coordinate tensors positionally (``embed(t, x)``,
    ``embed(x, y)``, …); ``forward`` passes its own arguments straight
    through, so the coordinate count and order are the embedding's business.

    ``depth`` is the gated-hidden-layer count. ``out_transform``, if given, is
    called as ``out_transform(out, *coords)`` and lets a benchmark impose a
    hard boundary condition (e.g. the ``sin(πx)·N`` Dirichlet transform).
    """

    def __init__(
        self,
        embed: nn.Module,
        hidden: int = 256,
        depth: int = 4,
        out_features: int = 1,
        out_transform: Optional[Callable[..., torch.Tensor]] = None,
    ):
        super().__init__()
        assert depth >= 1
        self.embed = embed
        d = embed.out_dim
        # Fused u/v encoder: one matmul, chunked into the two gates. See the
        # module docstring before changing this.
        self.enc_uv = nn.Linear(d, 2 * hidden)
        self.layers = nn.ModuleList(
            [nn.Linear(d if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out = nn.Linear(hidden, out_features)
        self.out_transform = out_transform

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        z = self.embed(*coords)

        uv = torch.tanh(self.enc_uv(z))
        enc_a, enc_b = uv.chunk(2, dim=-1)
        w = enc_a - enc_b  # computed once; gate becomes enc_b + h*w

        h = z
        for layer in self.layers:
            h = torch.tanh(layer(h))
            h = torch.addcmul(enc_b, h, w)  # == h*enc_a + (1-h)*enc_b

        out = self.out(h)
        if self.out_transform is not None:
            out = self.out_transform(out, *coords)
        return out
