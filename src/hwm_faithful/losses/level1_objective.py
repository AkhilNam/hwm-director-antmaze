"""Aggregate Level-1 objectives: VICRegObs + IDM + PredictionProprio.

Matches ``Trainer`` summing ``loss_info.total_loss`` over
``objectives_l1.objectives`` in ``pldm/train.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from hwm_faithful.losses.inverse_dynamics import InverseDynamics, idm_loss
from hwm_faithful.losses.prediction_proprio import prediction_proprio
from hwm_faithful.losses.vicreg_obs import vicreg_obs
from hwm_faithful.shapes import ACTION_DIM, FEATURE_SIZE, FUSED_CHANNELS, PROPRIO_DIM, VISUAL_CHANNELS


class Level1ObjectiveOutput(NamedTuple):
    total: torch.Tensor
    vicreg_obs: torch.Tensor
    vicreg_sim: torch.Tensor
    vicreg_std: torch.Tensor
    vicreg_cov: torch.Tensor
    vicreg_std_t: torch.Tensor
    idm: torch.Tensor
    idm_action: torch.Tensor
    prediction_proprio: torch.Tensor
    prediction_proprio_raw: torch.Tensor


class Level1Objective(nn.Module):
    """Released L1 objective bundle. Only IDM has trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.idm = InverseDynamics()

    def forward(
        self,
        *,
        obs_gt: torch.Tensor,
        obs_pred: torch.Tensor,
        proprio_gt: torch.Tensor,
        proprio_pred: torch.Tensor,
        fused_gt: torch.Tensor,
        actions: torch.Tensor,
    ) -> Level1ObjectiveOutput:
        _validate_sequence(obs_gt, obs_pred, proprio_gt, proprio_pred, fused_gt, actions)
        vic = vicreg_obs(obs_gt, obs_pred)
        idm_info = idm_loss(self.idm, fused_gt, actions)
        prop = prediction_proprio(proprio_gt, proprio_pred)
        total = vic.total + idm_info.total + prop.total
        return Level1ObjectiveOutput(
            total=total,
            vicreg_obs=vic.total,
            vicreg_sim=vic.sim,
            vicreg_std=vic.std,
            vicreg_cov=vic.cov,
            vicreg_std_t=vic.std_t,
            idm=idm_info.total,
            idm_action=idm_info.action_loss,
            prediction_proprio=prop.total,
            prediction_proprio_raw=prop.pred_loss,
        )


def _validate_sequence(
    obs_gt: torch.Tensor,
    obs_pred: torch.Tensor,
    proprio_gt: torch.Tensor,
    proprio_pred: torch.Tensor,
    fused_gt: torch.Tensor,
    actions: torch.Tensor,
) -> None:
    t, b = obs_gt.shape[:2]
    if obs_gt.shape[2:] != (VISUAL_CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError(f"obs_gt spatial/channel mismatch: {tuple(obs_gt.shape)}")
    if proprio_gt.shape[2:] != (PROPRIO_DIM, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError(
            f"proprio_gt spatial/channel mismatch: {tuple(proprio_gt.shape)}"
        )
    if fused_gt.shape != (t, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE):
        raise ValueError(f"fused_gt shape {tuple(fused_gt.shape)} != {(t, b, FUSED_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)}")
    if actions.shape != (t - 1, b, ACTION_DIM):
        raise ValueError(
            f"actions must be [{t - 1}, {b}, {ACTION_DIM}], got {tuple(actions.shape)}"
        )
    if obs_pred.shape != obs_gt.shape or proprio_pred.shape != proprio_gt.shape:
        raise ValueError("pred/gt visual or proprio shapes do not match")
