"""Spatial expansion of proprioception (original ``Expander2D`` / ``id_expand``).

Original: ``pldm/models/utils.py::Expander2D`` used by
``MeNet6._build_proprio_encoder`` when ``late_proprio_cfg.encoder_arch`` is
``id_expand``. Each scalar channel is repeated over the visual feature map
``(FEATURE_SIZE, FEATURE_SIZE)``. There are no trainable parameters.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_faithful.shapes import FEATURE_SIZE, PROPRIO_DIM


class ProprioExpander(nn.Module):
    """``[B, 2]`` velocity → ``[B, 2, 43, 43]`` by spatial broadcast.

    Matches original ``Expander2D(w=43, h=43)``: unsqueeze to
    ``[B, 2, 1, 1]`` then ``repeat`` over height and width. Every spatial
    location of channel ``c`` equals ``proprio[:, c]``.
    """

    def __init__(
        self,
        proprio_dim: int = PROPRIO_DIM,
        height: int = FEATURE_SIZE,
        width: int = FEATURE_SIZE,
    ) -> None:
        super().__init__()
        if proprio_dim < 1:
            raise ValueError(f"proprio_dim must be >= 1, got {proprio_dim}")
        if height < 1 or width < 1:
            raise ValueError(f"height/width must be >= 1, got {height}, {width}")
        self.proprio_dim = int(proprio_dim)
        self.height = int(height)
        self.width = int(width)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        _validate_proprio(proprio, self.proprio_dim)
        # [B, C, 1, 1] -> [B, C, H, W]
        return proprio.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.height, self.width
        ).contiguous()


def _validate_proprio(proprio: torch.Tensor, proprio_dim: int) -> None:
    if proprio.ndim != 2 or proprio.shape[-1] != proprio_dim:
        raise ValueError(
            f"proprio must have shape [B, {proprio_dim}], got {tuple(proprio.shape)}"
        )
    if proprio.shape[0] < 1:
        raise ValueError(f"batch size must be >= 1, got {proprio.shape[0]}")
