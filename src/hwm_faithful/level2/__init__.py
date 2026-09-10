"""Level-2 temporal abstraction, latent ``z``, and residual predictor ``f_H``.

``z`` is the L2 macro-action. ``f_H(H, z) → Ĥ_next`` is the L2 world model.
"""

from hwm_faithful.level2.action_encoder import (
    Level2ActionEncoder,
    Level2ActionEncoderOutput,
)
from hwm_faithful.level2.freeze import freeze_module
from hwm_faithful.level2.identity import Level2IdentityEncoder
from hwm_faithful.level2.posterior import Level2ActionPosterior, PosteriorOutput
from hwm_faithful.level2.predictor import (
    Level2Predictor,
    Level2PredictorOutput,
    Level2RolloutOutput,
)
from hwm_faithful.level2.temporal_abstraction import (
    Level2Inputs,
    Level2TemporalConfig,
    build_level2_inputs,
    chunk_level2_actions,
    flatten_action_chunk,
    subsample_level2_states,
)
from hwm_faithful.level2.world_model import Level2TrainForward, Level2WorldModel

__all__ = [
    "Level2ActionEncoder",
    "Level2ActionEncoderOutput",
    "Level2ActionPosterior",
    "Level2IdentityEncoder",
    "Level2Inputs",
    "Level2Predictor",
    "Level2PredictorOutput",
    "Level2RolloutOutput",
    "Level2TemporalConfig",
    "Level2TrainForward",
    "Level2WorldModel",
    "PosteriorOutput",
    "build_level2_inputs",
    "chunk_level2_actions",
    "flatten_action_chunk",
    "freeze_module",
    "subsample_level2_states",
]
