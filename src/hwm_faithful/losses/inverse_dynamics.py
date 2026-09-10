"""Inverse dynamics (IDM) as executed by original ``IDMObjective``.

Original: ``pldm/objectives/idm.py`` with YAML
``arch='conv'``, ``arch_subclass='a'``, ``use_pred=false``.

Input is **encoded fused** states, not predictions:

    curr = h_gt[:-1]     # [T-1, B, 18, 43, 43]
    next = h_gt[1:]
    x = cat(curr, next, dim=2)   # [T-1, B, 36, 43, 43]
    x = flatten time×batch       # [(T-1)*B, 36, 43, 43]
    a_hat = conv_idm(x)          # [(T-1)*B, 2]
    loss = MSE(a_hat, actions.flatten(0, 1))
    total = coeff * loss

No detach. Encoder gradients flow; the predictor does not (``use_pred=false``).

Conv topology is original ``build_conv`` with default ``group_factor=4``:

    Conv 36→32 k=3 s=1 p=1 + GN(8) + ReLU
    MaxPool 2
    Conv 32→32 k=3 s=1 p=1 + GN(8) + ReLU
    MaxPool 2
    Conv 32→32 k=3 s=1 p=1 + GN(8) + ReLU
    Flatten + Linear(3200, 2)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from hwm_faithful.losses.coefficients import IDM_COEFF
from hwm_faithful.shapes import ACTION_DIM, FEATURE_SIZE, FUSED_CHANNELS

IDM_GROUP_FACTOR = 4
# 43 → 21 → 10 after two kernel=2 stride=2 pools.
IDM_FC_IN = 32 * 10 * 10  # 3200


class InverseDynamicsInfo(NamedTuple):
    total: torch.Tensor
    action_loss: torch.Tensor


class _ConvNormRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm = nn.GroupNorm(out_channels // IDM_GROUP_FACTOR, out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class InverseDynamics(nn.Module):
    """``concat(h_t, h_{t+1}) → a_t`` conv IDM, subclass ``a``."""

    def __init__(self) -> None:
        super().__init__()
        in_ch = FUSED_CHANNELS * 2
        self.block0 = _ConvNormRelu(in_ch, 32)
        self.pool0 = nn.MaxPool2d(2, 2, 0)
        self.block1 = _ConvNormRelu(32, 32)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.block2 = _ConvNormRelu(32, 32)
        self.fc = nn.Linear(IDM_FC_IN, ACTION_DIM)

    def forward(self, fused_pair: torch.Tensor) -> torch.Tensor:
        """``fused_pair`` is ``[N, 36, 43, 43]`` → ``[N, 2]``."""
        if fused_pair.ndim != 4 or fused_pair.shape[1:] != (
            FUSED_CHANNELS * 2,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ):
            raise ValueError(
                f"IDM input must be [N, {FUSED_CHANNELS * 2}, {FEATURE_SIZE}, "
                f"{FEATURE_SIZE}], got {tuple(fused_pair.shape)}"
            )
        x = self.block0(fused_pair)
        x = self.pool0(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = x.flatten(1)
        if x.shape[-1] != IDM_FC_IN:
            raise ValueError(f"IDM flatten dim {x.shape[-1]} != {IDM_FC_IN}")
        return self.fc(x)


def idm_loss(idm: InverseDynamics, fused_gt: torch.Tensor, actions: torch.Tensor) -> InverseDynamicsInfo:
    """IDM on encoded fused sequence. ``fused_gt [T,B,18,43,43]``, ``actions [T-1,B,2]``."""
    if fused_gt.ndim != 5 or fused_gt.shape[2:] != (
        FUSED_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError(
            f"fused_gt must be [T, B, {FUSED_CHANNELS}, {FEATURE_SIZE}, "
            f"{FEATURE_SIZE}], got {tuple(fused_gt.shape)}"
        )
    t, b = fused_gt.shape[:2]
    if t < 2:
        raise ValueError(f"IDM needs T>=2, got {t}")
    if actions.shape != (t - 1, b, ACTION_DIM):
        raise ValueError(
            f"actions must have shape [{t - 1}, {b}, {ACTION_DIM}], got {tuple(actions.shape)}"
        )
    curr = fused_gt[:-1]
    nxt = fused_gt[1:]
    paired = torch.cat([curr, nxt], dim=2).flatten(0, 1)
    pred = idm(paired)
    target = actions.flatten(0, 1)
    action_loss = F.mse_loss(pred, target, reduction="mean")
    return InverseDynamicsInfo(total=IDM_COEFF * action_loss, action_loss=action_loss)
