"""Level-1 encoder: visual features fused with expanded proprioception.

Original path for Diverse Maze L1 (``late_proprio_cfg.fuse = true``):

    obs = MeNet6.layers(image)                    # [B, 16, 43, 43]
    late_proprio = Expander2D(proprio_vel)        # [B,  2, 43, 43]
    encodings = cat([obs, late_proprio], dim=1)   # [B, 18, 43, 43]

See ``pldm/models/encoders/encoders.py::MeNet6.forward``. First 16 channels
are visual; last 2 channels are proprio. No encoder LayerNorm is applied
(``final_ln`` entries are ``Identity`` for this config).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from hwm_faithful.models.proprio import ProprioExpander
from hwm_faithful.models.visual_encoder import VisualEncoder
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    PROPRIO_DIM,
    VISUAL_CHANNELS,
)


class Level1EncoderOutput(NamedTuple):
    """Split and fused L1 representation.

    ``visual`` is original ``obs_component``.
    ``proprio_map`` is original ``proprio_component``.
    ``fused`` is original ``encodings`` (channel-concat of the two).
    """

    visual: torch.Tensor
    proprio_map: torch.Tensor
    fused: torch.Tensor


class Level1Encoder(nn.Module):
    """``(image, proprio_vel) → fused [B, 18, 43, 43]``.

    This is ``ρ`` / ``E`` for faithful HWM. It is not the identity encoder
    used by the raw-state baseline in ``hwm_director``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.proprio_expander = ProprioExpander(
            proprio_dim=PROPRIO_DIM,
            height=FEATURE_SIZE,
            width=FEATURE_SIZE,
        )

    def forward(
        self, image: torch.Tensor, proprio: torch.Tensor
    ) -> Level1EncoderOutput:
        visual = self.visual_encoder(image)
        proprio_map = self.proprio_expander(proprio)
        if visual.shape[0] != proprio_map.shape[0]:
            raise ValueError(
                "image and proprio batch sizes must match, got "
                f"{visual.shape[0]} vs {proprio_map.shape[0]}"
            )
        fused = torch.cat([visual, proprio_map], dim=1)
        return Level1EncoderOutput(
            visual=visual, proprio_map=proprio_map, fused=fused
        )


def expected_shapes(batch_size: int) -> dict[str, tuple[int, ...]]:
    return {
        "image": (batch_size, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE),
        "proprio": (batch_size, PROPRIO_DIM),
        "visual": (batch_size, VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        "proprio_map": (batch_size, PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE),
        "fused": (batch_size, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
    }
