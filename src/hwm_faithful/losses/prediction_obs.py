"""PredictionObs as executed by original ``PredictionObjective`` (``pred_attr='obs'``).

Original: ``pldm/objectives/prediction.py``.

    target = obs_gt[1:]      # [T-1, B, 16, 43, 43]
    pred   = obs_pred[1:]
    loss   = (target - pred).pow(2).mean()
    total  = global_coeff * loss

No detach. Index 0 (copied ``H0``) is excluded. Released L2
``prediction_obs.global_coeff = 2.416154262252218``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from hwm_faithful.losses.coefficients import PRED_OBS_COEFF
from hwm_faithful.losses.vicreg_obs import prediction_mse


class PredictionObsInfo(NamedTuple):
    total: torch.Tensor
    pred_loss: torch.Tensor


def prediction_obs(obs_gt: torch.Tensor, obs_pred: torch.Tensor) -> PredictionObsInfo:
    if obs_gt.ndim != 5 or obs_pred.ndim != 5:
        raise ValueError(
            f"obs maps must be [T, B, 16, H, W], got "
            f"{tuple(obs_gt.shape)} and {tuple(obs_pred.shape)}"
        )
    if obs_gt.shape != obs_pred.shape:
        raise ValueError(
            f"obs_gt/pred shape mismatch: {tuple(obs_gt.shape)} vs "
            f"{tuple(obs_pred.shape)}"
        )
    if obs_gt.shape[0] < 2:
        raise ValueError(f"PredictionObs needs T>=2, got {obs_gt.shape[0]}")
    pred_loss = prediction_mse(obs_gt[1:], obs_pred[1:])
    return PredictionObsInfo(total=PRED_OBS_COEFF * pred_loss, pred_loss=pred_loss)
