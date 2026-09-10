"""Level-2 world model: posterior (train-time z) + ``f_H``.

Training:

    action_chunk → posterior → z → f_H(H, z)

Planning (later):

    MPPI proposes z → f_H(H, z)

``f_H`` therefore accepts ``z`` directly. L1 stays frozen outside this module.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn

from hwm_faithful.level2.posterior import Level2ActionPosterior, PosteriorOutput
from hwm_faithful.level2.predictor import Level2Predictor, Level2RolloutOutput
from hwm_faithful.shapes import (
    FUSED_CHANNELS,
    FEATURE_SIZE,
)


class Level2TrainForward(NamedTuple):
    h_gt: torch.Tensor
    h_pred: torch.Tensor
    z: torch.Tensor
    posterior_mu: torch.Tensor
    posterior_std: torch.Tensor
    rollout: Level2RolloutOutput
    posterior: PosteriorOutput


class Level2WorldModel(nn.Module):
    """Train-time L2 graph without optimizer or L2 MPPI."""

    def __init__(self) -> None:
        super().__init__()
        self.posterior = Level2ActionPosterior()
        self.predictor = Level2Predictor()

    def predict(self, h: torch.Tensor, z: torch.Tensor):
        return self.predictor(h, z)

    def rollout(self, h0: torch.Tensor, z_seq: torch.Tensor) -> Level2RolloutOutput:
        return self.predictor.rollout(h0, z_seq)

    def forward_train(
        self,
        h_gt: torch.Tensor,
        action_chunks: torch.Tensor,
    ) -> Level2TrainForward:
        """GT fused ``[T+1, B, 18, 43, 43]`` + chunks ``[T, B, 10, 2]``.

        Rollout is open-loop from ``h_gt[0]`` (not teacher-forced).
        """
        if h_gt.ndim != 5 or h_gt.shape[2:] != (
            FUSED_CHANNELS,
            FEATURE_SIZE,
            FEATURE_SIZE,
        ):
            raise ValueError(
                f"h_gt must be [T+1, B, {FUSED_CHANNELS}, {FEATURE_SIZE}, "
                f"{FEATURE_SIZE}], got {tuple(h_gt.shape)}"
            )
        t_states = h_gt.shape[0]
        if t_states < 2:
            raise ValueError(f"need at least 2 L2 states, got {t_states}")
        t_trans = t_states - 1
        if action_chunks.shape[0] != t_trans:
            raise ValueError(
                f"action_chunks T={action_chunks.shape[0]} != "
                f"n_transitions={t_trans}"
            )
        if action_chunks.shape[1] != h_gt.shape[1]:
            raise ValueError("h_gt/action_chunks batch mismatch")
        post = self.posterior.encode_sequence(action_chunks)
        rolled = self.predictor.rollout(h_gt[0], post.z)
        return Level2TrainForward(
            h_gt=h_gt,
            h_pred=rolled.fused,
            z=post.z,
            posterior_mu=post.mu,
            posterior_std=post.std,
            rollout=rolled,
            posterior=post,
        )
