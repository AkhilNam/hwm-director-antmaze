"""Level-2 residual conv predictor ``f_H`` (original ``conv2`` / ``l2_d4rl_e_p``).

Original sources (SHA ``e197375``):

- ``ConvPredictorConfig["l2_d4rl_e_p"]``
- ``ConvPredictor.forward``: ``cat(H, action_encoder(z))`` then residual
- ``build_conv(..., group_factor=8)``: GN+ReLU on all but the last conv
- table first in-channels ``42`` are **overridden** to
  ``16 + 2 + 8 = 26`` (obs + proprio + encoded-z map)

YAML ``predictor_ln: true`` constructs an unused ``LayerNorm`` on
``SequencePredictor``; ``ConvPredictor.forward`` never applies it. We omit it.

One-step:

    z_map = expand(MLP_8-64-8(z))              # [B, 8, 43, 43]
    x = cat([H, z_map], dim=1)                 # [B, 26, 43, 43]
    delta = conv(x)                            # [B, 18, 43, 43]
    H_next = H + delta                         # residual on all 18 channels
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import nn

from hwm_faithful.level2.action_encoder import Level2ActionEncoder
from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.shapes import (
    FEATURE_SIZE,
    FUSED_CHANNELS,
    L2_PREDICTOR_IN_CHANNELS,
    Z_DIM,
)

# (out_channels, kernel, stride, padding). First in-channels are 26, not 42.
L2_D4RL_E_P_LAYERS: tuple[tuple[int, int, int, int], ...] = (
    (32, 5, 1, 2),
    (32, 5, 1, 2),
    (32, 5, 1, 2),
    (32, 5, 1, 2),
    (18, 5, 1, 2),
)

PREDICTOR_GROUP_FACTOR = 8


class Level2PredictorOutput(NamedTuple):
    fused: torch.Tensor
    obs_component: torch.Tensor
    proprio_component: torch.Tensor
    delta: torch.Tensor
    z_map: torch.Tensor


class Level2RolloutOutput(NamedTuple):
    """Time-leading. Index 0 is ``H0``; ``1..T`` are recursive predictions."""

    fused: torch.Tensor
    obs_component: torch.Tensor
    proprio_component: torch.Tensor


class _ConvNormRelu(nn.Module):
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


class Level2Predictor(nn.Module):
    """``f_H(H_t, z_t) → Ĥ_{t+1}``. Accepts ``z`` directly (planning path)."""

    def __init__(self) -> None:
        super().__init__()
        self.action_encoder = Level2ActionEncoder()
        blocks: list[nn.Module] = []
        in_channels = L2_PREDICTOR_IN_CHANNELS
        for i, (out_channels, kernel, stride, padding) in enumerate(
            L2_D4RL_E_P_LAYERS
        ):
            is_last = i == len(L2_D4RL_E_P_LAYERS) - 1
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

    def compute_delta(
        self, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_h(h)
        _validate_z(z, h.shape[0])
        z_map = self.action_encoder(z).spatial
        x = torch.cat([h, z_map], dim=1)
        for block in self.blocks:
            x = block(x)
        return x, z_map

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> Level2PredictorOutput:
        delta, z_map = self.compute_delta(h, z)
        fused = h + delta
        obs, proprio = split_fused(fused)
        return Level2PredictorOutput(
            fused=fused,
            obs_component=obs,
            proprio_component=proprio,
            delta=delta,
            z_map=z_map,
        )

    def rollout(self, h0: torch.Tensor, z_seq: torch.Tensor) -> Level2RolloutOutput:
        """Open-loop recursive rollout. ``z_seq`` is ``[T, B, 8]``."""
        _validate_h(h0)
        _validate_z_sequence(z_seq, h0.shape[0])
        steps = [h0]
        h = h0
        for t in range(z_seq.shape[0]):
            h = self.forward(h, z_seq[t]).fused
            steps.append(h)
        fused = torch.stack(steps, dim=0)
        obs, proprio = split_fused(fused)
        return Level2RolloutOutput(
            fused=fused, obs_component=obs, proprio_component=proprio
        )


def _validate_h(h: torch.Tensor) -> None:
    if h.ndim != 4 or h.shape[1:] != (FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError(
            f"H must be [B, {FUSED_CHANNELS}, {FEATURE_SIZE}, {FEATURE_SIZE}], "
            f"got {tuple(h.shape)}"
        )
    if h.shape[0] < 1:
        raise ValueError(f"batch size must be >= 1, got {h.shape[0]}")


def _validate_z(z: torch.Tensor, batch_size: int) -> None:
    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must be [B, {Z_DIM}], got {tuple(z.shape)}")
    if z.shape[0] != batch_size:
        raise ValueError(
            f"H/z batch mismatch: {batch_size} vs {z.shape[0]}"
        )


def _validate_z_sequence(z_seq: torch.Tensor, batch_size: int) -> None:
    if z_seq.ndim != 3 or z_seq.shape[-1] != Z_DIM:
        raise ValueError(f"z_seq must be [T, B, {Z_DIM}], got {tuple(z_seq.shape)}")
    if z_seq.shape[0] < 1:
        raise ValueError(f"rollout T must be >= 1, got {z_seq.shape[0]}")
    if z_seq.shape[1] != batch_size:
        raise ValueError(
            f"H0/z_seq batch mismatch: {batch_size} vs {z_seq.shape[1]}"
        )


def layer_sequence_description() -> Sequence[str]:
    return (
        "z [B,8] -> Linear 8-64-ReLU-8 -> expand [B, 8, 43, 43]",
        "cat(H [B,18,43,43], z_map [B,8,43,43]) -> [B, 26, 43, 43]",
        "Conv2d(26, 32, k=5, s=1, p=2) + GroupNorm(4) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 32, k=5, s=1, p=2) + GroupNorm(4) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 32, k=5, s=1, p=2) + GroupNorm(4) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 32, k=5, s=1, p=2) + GroupNorm(4) + ReLU -> [B, 32, 43, 43]",
        "Conv2d(32, 18, k=5, s=1, p=2)  (no GN, no ReLU)    -> [B, 18, 43, 43]",
        "residual: H_next = H + delta  (all 18 fused channels)",
    )


assert L2_D4RL_E_P_LAYERS[-1][0] == FUSED_CHANNELS
assert L2_PREDICTOR_IN_CHANNELS == 26
