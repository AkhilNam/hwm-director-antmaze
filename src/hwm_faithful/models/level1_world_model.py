"""Level-1 world model: encoder plus residual conv predictor.

Wraps M1 ``Level1Encoder`` and M2 ``Level1Predictor``. Does not include
training losses. ``z_dim = 0`` at Level 1: the predictor is conditioned
only on primitive 2D actions.
"""

from __future__ import annotations

import torch
from torch import nn

from hwm_faithful.models.conv_predictor import (
    Level1Predictor,
    Level1PredictorOutput,
    Level1RolloutOutput,
)
from hwm_faithful.models.level1_encoder import Level1Encoder, Level1EncoderOutput


class Level1WorldModel(nn.Module):
    """``E`` + ``f_L`` for faithful HWM Level 1.

    ``encode`` is the M1 observation path. ``predict`` / ``rollout`` are the
    M2 residual conv dynamics. Recursive rollout uses predicted states, not
    ground-truth intermediates (original ``forward_multiple``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Level1Encoder()
        self.predictor = Level1Predictor()

    def encode(
        self, image: torch.Tensor, proprio: torch.Tensor
    ) -> Level1EncoderOutput:
        return self.encoder(image, proprio)

    def predict(
        self, h: torch.Tensor, action: torch.Tensor
    ) -> Level1PredictorOutput:
        return self.predictor(h, action)

    def rollout(
        self, h0: torch.Tensor, actions: torch.Tensor
    ) -> Level1RolloutOutput:
        return self.predictor.rollout(h0, actions)

    def encode_and_predict(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[Level1EncoderOutput, Level1PredictorOutput]:
        encoded = self.encode(image, proprio)
        predicted = self.predict(encoded.fused, action)
        return encoded, predicted
