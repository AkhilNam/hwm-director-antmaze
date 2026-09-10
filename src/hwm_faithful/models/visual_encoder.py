"""Learned Level-1 visual encoder (original ``MeNet6`` / ``d4rl_a`` path).

Original sources (read-only, SHA ``e197375``):

- ``pldm/models/encoders/encoders.py``: ``ENCODER_LAYERS_CONFIG["d4rl_a"]``,
  ``MeNet6``, ``build_backbone``
- ``pldm/models/utils.py``: ``build_conv`` (GroupNorm + ReLU on all but the
  last conv; last conv has no norm/activation)

The YAML field ``backbone_width_factor: 2`` is unused in original Python.
This module matches the **executed** ``d4rl_a`` topology with RGB input
channels (original ``build_conv`` overrides the table's first in-channels
of 6 with the actual image channel count, which is 3).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from hwm_faithful.shapes import FEATURE_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, VISUAL_CHANNELS

# (out_channels, kernel, stride, padding). First-layer in-channels are
# IMAGE_CHANNELS, not the unused 6 in the original table.
D4RL_A_LAYERS: tuple[tuple[int, int, int, int], ...] = (
    (16, 5, 1, 0),
    (32, 5, 2, 0),
    (32, 3, 1, 0),
    (32, 3, 1, 1),
    (16, 1, 1, 0),
)

GROUP_FACTOR = 4  # original build_conv default


class _ConvNormRelu(nn.Module):
    """Conv2d + GroupNorm(out // 4) + ReLU, matching original ``build_conv``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = nn.GroupNorm(out_channels // GROUP_FACTOR, out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class VisualEncoder(nn.Module):
    """RGB ``[B, 3, 98, 98]`` → spatial features ``[B, 16, 43, 43]``.

    Intermediate spatial sizes (H=W): 98 → 94 → 45 → 43 → 43 → 43.
    The last 1x1 conv has no GroupNorm and no ReLU, as in original
    ``build_conv(..., last_layer_act_norm=False)``.
    """

    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = IMAGE_CHANNELS
        for i, (out_channels, kernel, stride, padding) in enumerate(D4RL_A_LAYERS):
            is_last = i == len(D4RL_A_LAYERS) - 1
            if is_last:
                blocks.append(
                    nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
                )
            else:
                blocks.append(
                    _ConvNormRelu(
                        in_channels, out_channels, kernel, stride, padding
                    )
                )
            in_channels = out_channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _validate_image(image)
        x = image
        for block in self.blocks:
            x = block(x)
        return x

    def intermediate_shapes(
        self, image: torch.Tensor | None = None
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Return ``(name, shape)`` after each block, including the input."""
        if image is None:
            image = torch.zeros(1, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        _validate_image(image)
        shapes: list[tuple[str, tuple[int, ...]]] = [
            ("input", tuple(image.shape))
        ]
        x = image
        for i, block in enumerate(self.blocks):
            x = block(x)
            name = f"block{i}_{_block_kind(block)}"
            shapes.append((name, tuple(x.shape)))
        return shapes


def _block_kind(block: nn.Module) -> str:
    if isinstance(block, _ConvNormRelu):
        conv = block.conv
        return (
            f"conv{conv.kernel_size[0]}_s{conv.stride[0]}_gn_relu"
            f"_c{conv.out_channels}"
        )
    if isinstance(block, nn.Conv2d):
        return (
            f"conv{block.kernel_size[0]}_s{block.stride[0]}"
            f"_c{block.out_channels}"
        )
    return type(block).__name__


def _validate_image(image: torch.Tensor) -> None:
    if image.ndim != 4:
        raise ValueError(
            f"image must have shape [B, {IMAGE_CHANNELS}, {IMAGE_SIZE}, "
            f"{IMAGE_SIZE}], got ndim={image.ndim} shape={tuple(image.shape)}"
        )
    b, c, h, w = image.shape
    if c != IMAGE_CHANNELS or h != IMAGE_SIZE or w != IMAGE_SIZE:
        
        raise ValueError(
            f"image must have shape [B, {IMAGE_CHANNELS}, {IMAGE_SIZE}, "
            f"{IMAGE_SIZE}], got {tuple(image.shape)}"
        )
    if b < 1:
        raise ValueError(f"batch size must be >= 1, got {b}")


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def layer_sequence_description() -> Sequence[str]:
    return (
        "Conv2d(3, 16, k=5, s=1, p=0) + GroupNorm(4) + ReLU  -> [B, 16, 94, 94]",
        "Conv2d(16, 32, k=5, s=2, p=0) + GroupNorm(8) + ReLU -> [B, 32, 45, 45]",
        "Conv2d(32, 32, k=3, s=1, p=0) + GroupNorm(8) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 32, k=3, s=1, p=1) + GroupNorm(8) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 16, k=1, s=1, p=0)  (no GN, no ReLU)    -> [B, 16, 43, 43]",
    )


assert VISUAL_CHANNELS == D4RL_A_LAYERS[-1][0]
assert FEATURE_SIZE == 43
