"""Level-1 residual conv predictor (original ``conv2`` / ``d4rl_b_p``).

Original sources (read-only, SHA ``e197375``):

- ``pldm/models/predictors/conv_predictors.py``: ``ConvPredictor``,
  ``ConvPredictorConfig["d4rl_b_p"]``
- ``pldm/models/utils.py``: ``build_conv`` (GroupNorm + ReLU on all but the
  last conv). Predictor uses ``group_factor=8``, unlike the encoder's 4.
- ``pldm/models/predictors/sequence_predictor.py``: ``forward_multiple``
  recursive rollout; ``_separate_obs_proprio_from_fused_repr``
- Diverse Maze YAML: ``predictor.residual = true``,
  ``action_encoder_arch = id``, ``z_dim = 0``

One-step semantics (fused residual, all 18 channels):

    action_map = expand(a)                         # [B, 2, 43, 43]
    x = cat([h, action_map], dim=1)                # [B, 20, 43, 43]
    delta = conv(x)                                # [B, 18, 43, 43]
    h_next = h + delta                             # residual on fused h
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn

from hwm_faithful.models.action_encoder import PrimitiveActionEncoder
from hwm_faithful.shapes import (
    ACTION_DIM,
    FEATURE_SIZE,
    FUSED_CHANNELS,
    PREDICTOR_IN_CHANNELS,
    PROPRIO_DIM,
    VISUAL_CHANNELS,
)

# (out_channels, kernel, stride, padding). First-layer in-channels are
# PREDICTOR_IN_CHANNELS (20), matching the d4rl_b_p table.
D4RL_B_P_LAYERS: tuple[tuple[int, int, int, int], ...] = (
    (32, 3, 1, 1),
    (32, 3, 1, 1),
    (18, 3, 1, 1),
)

# Original ConvPredictor.build_conv(..., group_factor=8).
PREDICTOR_GROUP_FACTOR = 8


class Level1PredictorOutput(NamedTuple):
    """One-step L1 prediction, matching original ``SingleStepPredictorOutput``.

    ``fused`` is original ``prediction`` (full 18-channel residual output).
    ``obs_component`` is the first 16 channels.
    ``proprio_component`` is the last 2 channels.
    ``delta`` is the conv output *before* residual addition.
    """

    fused: torch.Tensor
    obs_component: torch.Tensor
    proprio_component: torch.Tensor
    delta: torch.Tensor


class Level1RolloutOutput(NamedTuple):
    """Open-loop recursive rollout, matching original ``PredictorOutput``.

    Time is leading. Index 0 is the input state ``h0`` (not a prediction).
    Indices ``1..T`` are recursive predictions. Shapes are
    ``[T+1, B, C, 43, 43]``.
    """

    fused: torch.Tensor
    obs_component: torch.Tensor
    proprio_component: torch.Tensor


class _ConvNormRelu(nn.Module):
    """Conv2d + GroupNorm(out // 8) + ReLU, matching predictor ``build_conv``."""

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
        self.norm = nn.GroupNorm(
            out_channels // PREDICTOR_GROUP_FACTOR, out_channels
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Level1Predictor(nn.Module):
    """``h_t [B,18,43,43], a_t [B,2] → ĥ_{t+1} [B,18,43,43]``.

    Residual is added to the **full fused** state (all 18 channels), as in
    ``ConvPredictor.forward``, not visual-only.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = PrimitiveActionEncoder(
            action_dim=ACTION_DIM,
            height=FEATURE_SIZE,
            width=FEATURE_SIZE,
        )
        blocks: list[nn.Module] = []
        in_channels = PREDICTOR_IN_CHANNELS
        for i, (out_channels, kernel, stride, padding) in enumerate(D4RL_B_P_LAYERS):
            is_last = i == len(D4RL_B_P_LAYERS) - 1
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

    def compute_delta(self, h: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Conv residual (no addition). Original ``self.layers(cat(h, a_map))``."""
        _validate_fused(h)
        action_map = self.action_encoder(action)
        if h.shape[0] != action_map.shape[0]:
            raise ValueError(
                "h and action batch sizes must match, got "
                f"{h.shape[0]} vs {action_map.shape[0]}"
            )
        x = torch.cat([h, action_map], dim=1)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, h: torch.Tensor, action: torch.Tensor) -> Level1PredictorOutput:
        delta = self.compute_delta(h, action)
        fused = h + delta
        obs, proprio = split_fused(fused)
        return Level1PredictorOutput(
            fused=fused,
            obs_component=obs,
            proprio_component=proprio,
            delta=delta,
        )

    def rollout(self, h0: torch.Tensor, actions: torch.Tensor) -> Level1RolloutOutput:
        """Recursive open-loop rollout matching ``SequencePredictor.forward_multiple``.

        Parameters
        ----------
        h0:
            ``[B, 18, 43, 43]`` first fused state (original ``state_encs[0]``).
        actions:
            ``[T, B, 2]`` primitive actions (original ``actions`` of length T).

        Returns
        -------
        fused of shape ``[T+1, B, 18, 43, 43]``. Index 0 is ``h0``; index
        ``t+1`` is ``f(h_t, a_t)`` using the *predicted* ``h_t``, not
        ground-truth intermediates.
        """
        _validate_fused(h0)
        _validate_action_sequence(actions, h0.shape[0])
        fused_steps = [h0]
        h = h0
        for t in range(actions.shape[0]):
            h = self.forward(h, actions[t]).fused
            fused_steps.append(h)
        fused = torch.stack(fused_steps, dim=0)
        obs, proprio = split_fused(fused)
        return Level1RolloutOutput(
            fused=fused, obs_component=obs, proprio_component=proprio
        )

    def intermediate_shapes(
        self, h: torch.Tensor | None = None, action: torch.Tensor | None = None
    ) -> list[tuple[str, tuple[int, ...]]]:
        if h is None:
            h = torch.zeros(1, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
        if action is None:
            action = torch.zeros(1, ACTION_DIM)
        _validate_fused(h)
        action_map = self.action_encoder(action)
        x = torch.cat([h, action_map], dim=1)
        shapes: list[tuple[str, tuple[int, ...]]] = [
            ("h", tuple(h.shape)),
            ("action_map", tuple(action_map.shape)),
            ("concat", tuple(x.shape)),
        ]
        for i, block in enumerate(self.blocks):
            x = block(x)
            shapes.append((f"block{i}", tuple(x.shape)))
        fused = h + x
        shapes.append(("residual_sum", tuple(fused.shape)))
        return shapes


def split_fused(fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split fused channels the way original ``_separate_obs_proprio_from_fused_repr`` does.

    Last ``PROPRIO_DIM`` channels are proprio; the rest are visual. Works for
    both ``[B, 18, H, W]`` and ``[T, B, 18, H, W]`` (channel axis is ``-3``).
    """
    if fused.ndim not in (4, 5) or fused.shape[-3] != FUSED_CHANNELS:
        raise ValueError(
            f"fused must have {FUSED_CHANNELS} channels at axis -3, "
            f"got shape {tuple(fused.shape)}"
        )
    obs = fused[..., :VISUAL_CHANNELS, :, :]
    proprio = fused[..., VISUAL_CHANNELS:, :, :]
    return obs, proprio


def _validate_fused(h: torch.Tensor) -> None:
    if h.ndim != 4:
        raise ValueError(
            f"h must have shape [B, {FUSED_CHANNELS}, {FEATURE_SIZE}, "
            f"{FEATURE_SIZE}], got ndim={h.ndim} shape={tuple(h.shape)}"
        )
    b, c, height, width = h.shape
    if c != FUSED_CHANNELS or height != FEATURE_SIZE or width != FEATURE_SIZE:
        raise ValueError(
            f"h must have shape [B, {FUSED_CHANNELS}, {FEATURE_SIZE}, "
            f"{FEATURE_SIZE}], got {tuple(h.shape)}"
        )
    if b < 1:
        raise ValueError(f"batch size must be >= 1, got {b}")


def _validate_action_sequence(actions: torch.Tensor, batch_size: int) -> None:
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"actions must have shape [T, B, {ACTION_DIM}], got {tuple(actions.shape)}"
        )
    if actions.shape[0] < 1:
        raise ValueError(f"rollout T must be >= 1, got {actions.shape[0]}")
    if actions.shape[1] != batch_size:
        raise ValueError(
            "h0 and actions batch sizes must match, got "
            f"{batch_size} vs {actions.shape[1]}"
        )


def layer_sequence_description() -> Sequence[str]:
    return (
        "cat(h [B,18,43,43], expand(a) [B,2,43,43]) -> [B, 20, 43, 43]",
        "Conv2d(20, 32, k=3, s=1, p=1) + GroupNorm(4) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 32, k=3, s=1, p=1) + GroupNorm(4) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 18, k=3, s=1, p=1)  (no GN, no ReLU)    -> [B, 18, 43, 43]",
        "residual: h_next = h + delta  (all 18 fused channels)",
    )


assert D4RL_B_P_LAYERS[-1][0] == FUSED_CHANNELS
assert PREDICTOR_IN_CHANNELS == 20
