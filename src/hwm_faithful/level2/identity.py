"""Level-2 identity backbone: concat L1 visual + proprio maps.

Original ``IdentityEncoder`` (``backbone.arch = identity_encoder``) does
**not** mean raw-state identity. Its input is already the learned L1
representation:

    visual  [B, 16, 43, 43]   ← L1 ``obs_component``
    proprio [B,  2, 43, 43]   ← L1 ``proprio_component`` (already expanded)
    fused   [B, 18, 43, 43]   ← channel concat, no extra expander, 0 params

YAML ``channels: 16`` / ``final_ln: false`` / late proprio ``id_expand``
are unused inside ``IdentityEncoder.forward``; proprio is passed in already
spatial from frozen L1.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.shapes import FEATURE_SIZE, FUSED_CHANNELS, PROPRIO_DIM, VISUAL_CHANNELS


class Level2IdentityEncoder(nn.Module):
    """0-parameter concat matching original ``IdentityEncoder``."""

    def forward(
        self,
        visual: torch.Tensor | None = None,
        proprio_map: torch.Tensor | None = None,
        fused: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if fused is not None:
            if fused.ndim not in (4, 5) or fused.shape[-3] != FUSED_CHANNELS:
                raise ValueError(
                    f"fused must have {FUSED_CHANNELS} channels, got {tuple(fused.shape)}"
                )
            return fused
        if visual is None or proprio_map is None:
            raise ValueError("provide fused, or both visual and proprio_map")
        if visual.shape[-3:] != (VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
            raise ValueError(
                f"visual must end with [{VISUAL_CHANNELS}, {FEATURE_SIZE}, "
                f"{FEATURE_SIZE}], got {tuple(visual.shape)}"
            )
        if proprio_map.shape[-3:] != (PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE):
            raise ValueError(
                f"proprio_map must end with [{PROPRIO_DIM}, {FEATURE_SIZE}, "
                f"{FEATURE_SIZE}], got {tuple(proprio_map.shape)}"
            )
        return torch.cat([visual, proprio_map], dim=-3)


def fused_from_l1_components(fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split fused L1 maps the way HJEPA feeds IdentityEncoder."""
    return split_fused(fused)
