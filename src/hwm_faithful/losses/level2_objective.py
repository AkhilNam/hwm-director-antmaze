"""Released Level-2 objectives: PredictionObs + PredictionProprio only.

Original ``objectives_l2.objectives`` in ``large_diverse_25maps_l2.yaml``.
VICReg YAML fields are present but **not** in the objectives list. No KL, no IDM.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from hwm_faithful.losses.prediction_obs import prediction_obs
from hwm_faithful.losses.prediction_proprio import prediction_proprio
from hwm_faithful.models.conv_predictor import split_fused
from hwm_faithful.shapes import FEATURE_SIZE, FUSED_CHANNELS, PROPRIO_DIM, VISUAL_CHANNELS


class Level2ObjectiveOutput(NamedTuple):
    total: torch.Tensor
    prediction_obs: torch.Tensor
    prediction_obs_raw: torch.Tensor
    prediction_proprio: torch.Tensor
    prediction_proprio_raw: torch.Tensor


def level2_prediction_losses(
    h_gt: torch.Tensor,
    h_pred: torch.Tensor,
) -> Level2ObjectiveOutput:
    """MSE on visual and proprio maps from index 1 onward, then YAML coeffs."""
    if h_gt.shape != h_pred.shape:
        raise ValueError(
            f"h_gt/h_pred mismatch: {tuple(h_gt.shape)} vs {tuple(h_pred.shape)}"
        )
    if h_gt.ndim != 5 or h_gt.shape[2:] != (
        FUSED_CHANNELS,
        FEATURE_SIZE,
        FEATURE_SIZE,
    ):
        raise ValueError(
            f"fused sequences must be [T, B, {FUSED_CHANNELS}, {FEATURE_SIZE}, "
            f"{FEATURE_SIZE}], got {tuple(h_gt.shape)}"
        )
    if h_gt.shape[0] < 2:
        raise ValueError(f"L2 losses need T>=2, got {h_gt.shape[0]}")
    obs_gt, proprio_gt = split_fused(h_gt)
    obs_pred, proprio_pred = split_fused(h_pred)
    if obs_gt.shape[2] != VISUAL_CHANNELS or proprio_gt.shape[2] != PROPRIO_DIM:
        raise ValueError("unexpected obs/proprio split")
    obs = prediction_obs(obs_gt, obs_pred)
    prop = prediction_proprio(proprio_gt, proprio_pred)
    return Level2ObjectiveOutput(
        total=obs.total + prop.total,
        prediction_obs=obs.total,
        prediction_obs_raw=obs.pred_loss,
        prediction_proprio=prop.total,
        prediction_proprio_raw=prop.pred_loss,
    )
