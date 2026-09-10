"""Level-2 predictor action encoder: ``action_encoder_arch = 8-64-8``.

This is **not** the posterior. Original ``ConvPredictor`` builds:

    Linear(8, 64) + ReLU
    Linear(64, 8)          # last ReLU removed
    Expander2D(43, 43)     # 0 params → [B, 8, 43, 43]

The first ``8`` in the arch string is the input width (``z_dim``), not a
hidden size prepended by ``build_mlp``. Concat with fused H then uses 8
spatial channels, so predictor in-channels are 18+8=26.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from hwm_faithful.models.action_encoder import PrimitiveActionEncoder
from hwm_faithful.shapes import FEATURE_SIZE, Z_DIM


class Level2ActionEncoderOutput(NamedTuple):
    vector: torch.Tensor  # [B, 8] after MLP
    spatial: torch.Tensor  # [B, 8, 43, 43] after Expander2D


class Level2ActionEncoder(nn.Module):
    """Maps ``z`` before the (not-yet-implemented) L2 conv predictor."""

    def __init__(
        self,
        arch: str = "8-64-8",
        height: int = FEATURE_SIZE,
        width: int = FEATURE_SIZE,
    ) -> None:
        super().__init__()
        dims = [int(x) for x in arch.split("-")]
        if len(dims) < 2:
            raise ValueError(f"action_encoder_arch must have ≥2 widths, got {arch!r}")
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.pop()  # original removes the last ReLU
        self.mlp = nn.Sequential(*layers)
        self.in_dim = dims[0]
        self.out_dim = dims[-1]
        self.expander = PrimitiveActionEncoder(
            action_dim=self.out_dim, height=height, width=width
        )

    def forward(self, z: torch.Tensor) -> Level2ActionEncoderOutput:
        if z.ndim != 2 or z.shape[-1] != self.in_dim:
            raise ValueError(
                f"z must be [B, {self.in_dim}], got {tuple(z.shape)}"
            )
        vector = self.mlp(z)
        spatial = self.expander(vector)
        return Level2ActionEncoderOutput(vector=vector, spatial=spatial)

    def encode_sequence(self, z: torch.Tensor) -> Level2ActionEncoderOutput:
        """``z`` ``[T, B, 8]`` → vector ``[T, B, 8]``, spatial ``[T, B, 8, 43, 43]``."""
        if z.ndim != 3 or z.shape[-1] != self.in_dim:
            raise ValueError(f"z sequence must be [T, B, {self.in_dim}], got {tuple(z.shape)}")
        t, b, _ = z.shape
        out = self.forward(z.reshape(t * b, self.in_dim))
        return Level2ActionEncoderOutput(
            vector=out.vector.view(t, b, self.out_dim),
            spatial=out.spatial.view(t, b, self.out_dim, *out.spatial.shape[-2:]),
        )


assert Z_DIM == 8
