"""Small CIFAR-style ResNet family (GELU + GroupNorm/BN).

Shared by the image-regression benchmarks (``cifar_rotation``, ``utkface``).
Three depths in one lineage — a 3×3 stride-1 stem into a few stages of He
basic blocks, global-average-pool, Linear head — with the weight-layer count
equal to the ``resnetN`` number, so resnet12 sits squarely between resnet8
and resnet18:

    resnet8   ~77K params
    resnet12  ~0.9M params   (custom, the geometric middle)
    resnet18  ~11M params
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer_from_name(norm: str, gn_groups: int, channel_counts):
    """Resolve ``norm`` to a ``norm_layer(num_channels)`` callable."""
    if norm == "gn":
        for c in channel_counts:
            if c % gn_groups != 0:
                raise ValueError(
                    f"gn_groups={gn_groups} does not divide channel count {c}"
                )
        return partial(nn.GroupNorm, gn_groups)
    if norm == "bn":
        return nn.BatchNorm2d
    raise ValueError(f"norm={norm!r}; expected 'gn' or 'bn'")


class _BasicBlock(nn.Module):
    """He CIFAR-ResNet basic block: two 3×3 convs + norm, GELU, residual.

    Uses a 1×1 projection shortcut when the spatial or channel dim changes,
    and GELU activations to match the rest of the optimizer-benchmark family.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, norm_layer) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm2 = norm_layer(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                norm_layer(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.gelu(h + self.shortcut(x))


# Each spec is one CIFAR-ResNet: a 3×3 stride-1 stem into `len(channels)`
# stages, `blocks[i]` basic blocks per stage at `channels[i]` (first block of a
# stage carries `strides[i]`), global-average-pool, Linear head. The
# weight-layer count = 1 (stem) + 2·sum(blocks) + 1 (head) == the "resnetN"
# number, so resnet12 sits squarely between resnet8 and resnet18.
_SPECS = {
    #            channels             blocks       strides       gn_groups
    "resnet8":  ((16, 32, 64),        (1, 1, 1),    (1, 2, 2),    8),
    "resnet12": ((48, 96, 192),       (2, 2, 1),    (1, 2, 2),    8),
    "resnet18": ((64, 128, 256, 512), (2, 2, 2, 2), (1, 2, 2, 2), 32),
}

MODEL_NAMES = tuple(_SPECS)  # ("resnet8", "resnet12", "resnet18")


def build_model(name: str, num_outputs: int = 1, norm: str = "gn") -> nn.Module:
    """Build one of ``resnet8`` / ``resnet12`` / ``resnet18`` (see ``_SPECS``)."""
    if name not in _SPECS:
        raise ValueError(f"unknown model {name!r}; choose from {list(_SPECS)}")
    channels, blocks, strides, gn_groups = _SPECS[name]
    norm_layer = _norm_layer_from_name(norm, gn_groups, channels)
    c0 = channels[0]
    layers = [nn.Conv2d(3, c0, 3, stride=1, padding=1, bias=False),
              norm_layer(c0), nn.GELU()]
    in_ch = c0
    for out_ch, n_blocks, stride in zip(channels, blocks, strides):
        for b in range(n_blocks):
            layers.append(_BasicBlock(in_ch, out_ch, stride if b == 0 else 1, norm_layer))
            in_ch = out_ch
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_ch, num_outputs)]
    return nn.Sequential(*layers)
