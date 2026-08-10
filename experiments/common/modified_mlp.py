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

``FusedMLP`` at the bottom pushes the same idea one level further, grouping
runs of ``hidden -> hidden`` layers into shared tensors — ``fuse_every``
layers each, from one-per-layer up to the whole stack. See its docstring; it
is an open experiment, not a recommendation.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        self.enc_uv = FusedLinear(d, 2 * hidden)
        self.layers = nn.ModuleList(
            [FusedLinear(d if i == 0 else hidden, hidden) for i in range(depth)]
        )
        self.out = FusedLinear(hidden, out_features)
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


class FusedLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        """
        This class fuses the bias and weight tensors
        into a single tensor. This groups the tensors when
        we apply the gnome/soap algorithm to them.

        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # [W | b], shape (out_dim, in_dim + 1)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim + 1))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight[:, :-1], self.weight[:, -1])


class FusedMLP(nn.Module):
    """Plain tanh MLP whose ``k`` hidden layers are grouped into shared
    parameter tensors, ``fuse_every`` consecutive layers per tensor.

    ``embed -> hidden``, then ``k = depth - 2`` layers ``hidden -> hidden``,
    then ``hidden -> out_features``. ``depth`` counts affine layers, matching
    the ``MLP`` classes in the experiment files, so ``--depth`` means the same
    thing whichever arch is selected.

    Where ``FusedLinear`` merges ``[W | b]`` for one layer, this merges whole
    runs of consecutive layers. With ``k = 8``::

        fuse_every=1:  8 tensors of (  hidden, hidden+1)   <- the control
        fuse_every=2:  4 tensors of (2*hidden, hidden+1)
        fuse_every=4:  2 tensors of (4*hidden, hidden+1)
        fuse_every=8:  1 tensor  of (8*hidden, hidden+1)   <- fully fused

    Within a chunk, block ``i`` is layer ``i``. ``fuse_every=0`` (or anything
    ``>= k``) means the whole stack in one tensor. If ``fuse_every`` does not
    divide ``k`` the last chunk is short — ``k=5, fuse_every=2`` gives
    ``[2, 2, 1]`` — so the sweep is not restricted to divisors of the depth.

    **Every setting is the same function from the same initialization.** The
    weights are drawn one ``(hidden, hidden+1)`` block at a time in layer
    order (see ``_init_blocks``), so the RNG stream does not depend on the
    chunking and ``fuse_every=1`` and ``fuse_every=k`` start from bit-identical
    weights. The only thing that varies is how many parameter tensors the
    optimizer sees — which is the whole point, since Gnome preconditions per
    tensor.

    What raising ``fuse_every`` does to Gnome, concretely:

    * **Output-side factor.** ``G Gᵀ`` grows to ``(c·hidden)²`` for a chunk of
      ``c`` layers, with a ``c × c`` block structure whose off-diagonal blocks
      ``G_i G_jᵀ`` carry curvature coupling *between layers*. That coupling is
      real — layer i's gradient depends on the weights of every layer after
      it — and the split form cannot represent it at all. It is a different
      kind of coupling from the ``u``/``v`` one above: sequential rather than
      multiplicative, and mediated by the intervening nonlinearities. Note it
      is only captured *within* a chunk: ``fuse_every=2`` on 8 layers models
      the 1-2, 3-4, 5-6, 7-8 couplings and none of the others.
    * **Input-side factor.** ``GᵀG`` becomes ``sum_i G_iᵀ G_i`` over the
      chunk, one shared ``(hidden+1)²`` basis where the split form had ``c``.
      This is the side that could plausibly *hurt*: layers at different depths
      see activations with different covariances, so their eigenvectors need
      not agree, and the shared basis is a compromise between them. Chunking
      bounds how far apart the compromised layers can be.
    * **Trust region.** One ``lambda`` per chunk instead of ``c``. The
      aggregate budget is unchanged — ``trust_radius·sqrt(c·P)`` fused versus
      ``c`` tensors each held to ``trust_radius·sqrt(P)``, same total l2 — but
      fusing lets the step spend it unevenly across the chunk, and conversely
      lets one badly-conditioned layer set the damping for its neighbours.
    * **Cost — not monotonic, and it gets *faster* before it gets slower.**
      Optimizer time is roughly ``A·(k/c) + B·k·c²·hidden³``. The first term
      is per-*tensor* work that runs every step (two projections, the EMA
      updates, the 3-iteration ``_lm_lambda`` solve — all launch-bound on
      small tensors, and on MPS each Kronecker factor also costs a
      device→host→device round-trip because ``qr`` is routed through CPU).
      The second is the eigenbasis refresh, which at
      ``precondition_frequency=10`` runs one step in ten. So at modest width
      the ``k/c`` term dominates and raising ``fuse_every`` is a *speedup*:
      measured ~8% going from ``c=1`` to ``c=2`` at ``hidden=64, depth=10``,
      with the forward pass unchanged (same ``k`` matmuls either way — the
      win is entirely optimizer-side). The ``c²·hidden³`` term takes over as
      width grows, so the sign flips somewhere above ``hidden=64``; at
      ``hidden=256`` a fully-fused stack means one 1024² decomposition
      against four 256², and that is not free. Memory for ``GG`` is
      ``O(k·c·hidden²)`` and rises with ``c`` throughout — no crossover there.

    Whether the extra within-chunk curvature is worth the shared input basis
    and the shared trust region is an empirical question — hence the knob.
    First data point (2D Poisson, ``hidden=64 depth=10``, one seed): ``c=2``
    beat ``c=1`` beat fully-fused, so the useful coupling is the adjacent-layer
    kind and the peak is at small ``c``. Note this did *not* track rho — rho
    rose sharply from ``c=1`` to ``c=2`` while the optimizer did better, so
    off-diagonal mass left behind is not by itself a measure of step quality.
    """

    def __init__(
        self,
        embed: nn.Module,
        hidden: int = 64,
        depth: int = 5,
        out_features: int = 1,
        fuse_every: int = 0,
        activation: Callable[[], nn.Module] = nn.Tanh,
        out_transform: Optional[Callable[..., torch.Tensor]] = None,
    ):
        super().__init__()
        assert depth >= 2
        if fuse_every < 0:
            raise ValueError(f"fuse_every must be >= 0, got {fuse_every}")
        self.embed = embed
        self.hidden = hidden
        self.k = depth - 2
        d = embed.out_dim

        self.chunks = self._chunk_sizes(self.k, fuse_every)
        self.fuse_every = fuse_every

        self.inp = FusedLinear(d, hidden)
        # A ParameterList rather than bare attributes so the chunks iterate in
        # position: nn.Module yields its own _parameters before descending into
        # submodules, so a directly-assigned Parameter would jump ahead of
        # ``inp`` and shift every parameter index in the diagnostics and rho
        # logs relative to the fuse_every=1 run it is being compared against.
        self.mid = nn.ParameterList(
            [nn.Parameter(self._init_blocks(c, hidden)) for c in self.chunks]
        )
        self.out = FusedLinear(hidden, out_features)
        # Held as a module (not called functionally) so it matches the shared
        # ``MLP``'s ``activation=`` contract and shows up in repr().
        self.act = activation()
        self.out_transform = out_transform

    @staticmethod
    def _chunk_sizes(k: int, fuse_every: int) -> list[int]:
        """Layer counts per fused tensor. ``0`` or ``>= k`` means one chunk.

        A short final chunk is fine and deliberate: restricting the knob to
        divisors of ``k`` would make the interesting settings depend on the
        depth, which is not a variable this experiment wants to move.
        """
        if k == 0:
            return []
        c = k if fuse_every == 0 else min(fuse_every, k)
        return [min(c, k - i) for i in range(0, k, c)]

    @staticmethod
    def _init_blocks(c: int, hidden: int) -> torch.Tensor:
        """A ``(c*hidden, hidden+1)`` chunk, initialized one layer-block at a
        time so the RNG stream is independent of the chunking and every
        ``fuse_every`` starts from identical weights.

        Each block draws with exactly the call shape ``FusedLinear`` would have
        used. One ``uniform_`` over the whole chunk would match on CPU but not
        necessarily on CUDA/MPS, where the Philox offset advances per call —
        and an init that silently differs by backend or by chunking would
        confound the one variable this class exists to isolate.
        """
        w = torch.empty(c * hidden, hidden + 1)
        bound = 1.0 / math.sqrt(hidden)
        for i in range(c):
            nn.init.uniform_(w[i * hidden:(i + 1) * hidden], -bound, bound)
        return w

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        h = self.act(self.inp(self.embed(*coords)))

        for W, c in zip(self.mid, self.chunks):
            for i in range(c):
                blk = W[i * self.hidden:(i + 1) * self.hidden]
                h = self.act(F.linear(h, blk[:, :-1], blk[:, -1]))

        out = self.out(h)
        if self.out_transform is not None:
            out = self.out_transform(out, *coords)
        return out


class MLP(nn.Module):
    """Plain MLP over an input embedding: ``coords → out_features``.

    The single source of truth for the non-gated baseline net, which every
    ``*_pinn.py`` previously defined for itself. Same contract as
    :class:`ModifiedMLP` — ``embed`` exposes ``out_dim`` and takes the
    problem's coordinates positionally, ``out_transform(out, *coords)``
    imposes a hard BC — so the two are drop-in swappable behind an ``--arch``
    flag. Use :class:`ConcatEmbed` where the old local class did a bare
    ``torch.cat``.

    ``depth`` counts affine layers, matching every call site it replaces:
    ``depth=5`` is 4 hidden layers plus the output layer.

    Built from :class:`FusedLinear`, not ``nn.Linear``, so each layer's
    ``[W | b]`` is one parameter tensor. That is **not** cosmetic: it changes
    what Gnome and SOAP precondition over, since biases are no longer separate
    1-D tensors. Runs are not comparable across the switch.
    """

    def __init__(
        self,
        embed: nn.Module,
        hidden: int = 64,
        depth: int = 5,
        out_features: int = 1,
        activation: Callable[[], nn.Module] = nn.Tanh,
        out_transform: Optional[Callable[..., torch.Tensor]] = None,
    ):
        super().__init__()
        assert depth >= 2
        self.embed = embed
        layers: list[nn.Module] = [FusedLinear(embed.out_dim, hidden),
                                   activation()]
        for _ in range(depth - 2):
            layers += [FusedLinear(hidden, hidden), activation()]
        layers += [FusedLinear(hidden, out_features)]
        self.net = nn.Sequential(*layers)
        self.out_transform = out_transform

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        out = self.net(self.embed(*coords))
        if self.out_transform is not None:
            out = self.out_transform(out, *coords)
        return out


class FusedModifiedMLP(nn.Module):
    """:class:`ModifiedMLP` with its gated hidden layers grouped into shared
    parameter tensors — ``FusedMLP``'s chunking applied to the gated trunk.

    Same function as :class:`ModifiedMLP` for every ``fuse_every``; only the
    parameter-tensor grouping changes. ``fuse_every=1`` reproduces
    ``ModifiedMLP`` exactly, weight for weight (see below), and is the control
    for every larger setting.

    **The fusable run is ``k = depth - 1``, not ``depth - 2``.** In the gated
    architecture the first hidden layer consumes the *embedding*
    (``d -> hidden``) rather than a hidden state, so it has a different input
    width and cannot share a tensor with the rest. It stays as its own
    ``first``; the remaining ``depth - 1`` layers are ``hidden -> hidden`` and
    are what gets chunked. ``depth`` keeps ``ModifiedMLP``'s meaning: the
    gated-hidden-layer count.

    Init is bit-identical to ``ModifiedMLP`` under the same seed. Both draw in
    the order ``enc_uv``, first layer, hidden layers in sequence, ``out``, and
    the chunk tensors are filled one ``(hidden, hidden+1)`` block at a time
    with ``FusedLinear``'s bound — so the RNG stream does not depend on the
    chunking. See :meth:`FusedMLP._init_blocks`.

    The ``u``/``v`` encoder is left alone. It is ``(2*hidden, d+1)`` — a
    different input width again — so it cannot join a chunk; it is already
    fused across ``u`` and ``v``, which is the module docstring's subject.

    One reason to expect fusion to be *cheaper* here than in the plain
    ``FusedMLP``: every gated layer's output passes through the same
    ``enc_b + h*(enc_a - enc_b)``, so the layers' output-side statistics are
    shaped by a shared gate rather than drifting independently with depth.
    That should make their curvature eigenbases more alike, and the shared
    input-side factor correspondingly less of a compromise. Untested.

    Activation is ``tanh`` throughout, matching ``ModifiedMLP`` — the gate
    wants ``u, v`` bounded, so it is not a free knob here the way it is in
    :class:`FusedMLP`.
    """

    def __init__(
        self,
        embed: nn.Module,
        hidden: int = 256,
        depth: int = 4,
        out_features: int = 1,
        fuse_every: int = 0,
        out_transform: Optional[Callable[..., torch.Tensor]] = None,
    ):
        super().__init__()
        assert depth >= 1
        if fuse_every < 0:
            raise ValueError(f"fuse_every must be >= 0, got {fuse_every}")
        self.embed = embed
        self.hidden = hidden
        self.fuse_every = fuse_every
        d = embed.out_dim

        # Draw order matches ModifiedMLP exactly, so seeds line up.
        self.enc_uv = FusedLinear(d, 2 * hidden)
        self.first = FusedLinear(d, hidden)
        self.chunks = FusedMLP._chunk_sizes(depth - 1, fuse_every)
        self.mid = nn.ParameterList(
            [nn.Parameter(FusedMLP._init_blocks(c, hidden))
             for c in self.chunks]
        )
        self.out = FusedLinear(hidden, out_features)
        self.out_transform = out_transform

    def forward(self, *coords: torch.Tensor) -> torch.Tensor:
        z = self.embed(*coords)

        uv = torch.tanh(self.enc_uv(z))
        enc_a, enc_b = uv.chunk(2, dim=-1)
        w = enc_a - enc_b  # computed once; gate becomes enc_b + h*w

        h = torch.addcmul(enc_b, torch.tanh(self.first(z)), w)
        for W, c in zip(self.mid, self.chunks):
            for i in range(c):
                blk = W[i * self.hidden:(i + 1) * self.hidden]
                h = torch.tanh(F.linear(h, blk[:, :-1], blk[:, -1]))
                h = torch.addcmul(enc_b, h, w)

        out = self.out(h)
        if self.out_transform is not None:
            out = self.out_transform(out, *coords)
        return out
