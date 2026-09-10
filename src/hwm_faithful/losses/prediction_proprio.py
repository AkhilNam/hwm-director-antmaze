"""PredictionProprio as executed by original ``PredictionObjective``.

Original: ``pldm/objectives/prediction.py`` with ``pred_attr='proprio'``.

Compares **spatial** proprio maps, not reduced ``[B, 2]`` vectors:

    target = proprio_gt[1:]      # [T-1, B, 2, 43, 43]
    pred   = proprio_pred[1:]
    loss   = (target - pred).pow(2).mean()
    total  = global_coeff * loss

No detach. No extra decoder. Index 0 (copied ``h0``) is excluded.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from hwm_faithful.losses.coefficients import PRED_PROPRIO_COEFF
from hwm_faithful.losses.vicreg_obs import prediction_mse


class PredictionProprioInfo(NamedTuple):
    total: torch.Tensor
    pred_loss: torch.Tensor


def prediction_proprio(
    proprio_gt: torch.Tensor,
    proprio_pred: torch.Tensor,
) -> PredictionProprioInfo:
    if proprio_gt.ndim != 5 or proprio_pred.ndim != 5:
        raise ValueError(
            f"proprio maps must be [T, B, 2, H, W], got "
            f"{tuple(proprio_gt.shape)} and {tuple(proprio_pred.shape)}"
        )
    if proprio_gt.shape != proprio_pred.shape:
        raise ValueError(
            f"proprio_gt/pred shape mismatch: {tuple(proprio_gt.shape)} vs "
            f"{tuple(proprio_pred.shape)}"
        )
    if proprio_gt.shape[0] < 2:
        raise ValueError(f"PredictionProprio needs T>=2, got {proprio_gt.shape[0]}")
    pred_loss = prediction_mse(proprio_gt[1:], proprio_pred[1:])
    return PredictionProprioInfo(
        total=PRED_PROPRIO_COEFF * pred_loss, pred_loss=pred_loss
    )
