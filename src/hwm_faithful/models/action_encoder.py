"""Primitive action spatial expander (original ``action_encoder_arch = id``).

Original: ``ConvPredictor`` in ``pldm/models/predictors/conv_predictors.py``
uses ``Expander2D(w=43, h=43)`` when ``action_encoder_arch`` is ``id`` (or
empty). Primitive maze actions ``[B, 2]`` are broadcast to
``[B, 2, 43, 43]`` and concatenated with the fused Level-1 state on the
channel axis. There are no trainable parameters and no MLP on the action.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_faithful.shapes import ACTION_DIM, FEATURE_SIZE


class PrimitiveActionEncoder(nn.Module):
    """``[B, 2]`` primitive action → ``[B, 2, 43, 43]`` by spatial broadcast.

    Matches original ``Expander2D``. Channel ``c`` is constant over H×W and
    equal to ``action[:, c]``.
    """

    def __init__(
        self,
        action_dim: int = ACTION_DIM,
        height: int = FEATURE_SIZE,
        width: int = FEATURE_SIZE,
    ) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if height < 1 or width < 1:
            raise ValueError(f"height/width must be >= 1, got {height}, {width}")
        self.action_dim = int(action_dim)
        self.height = int(height)
        self.width = int(width)

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        _validate_action(action, self.action_dim)
        return (
            action.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.height, self.width)
            .contiguous()
        )


def _validate_action(action: torch.Tensor, action_dim: int) -> None:
    if action.ndim != 2 or action.shape[-1] != action_dim:
        raise ValueError(
            f"action must have shape [B, {action_dim}], got {tuple(action.shape)}"
        )
    if action.shape[0] < 1:
        raise ValueError(f"batch size must be >= 1, got {action.shape[0]}")
