"""Level-2 input construction and latent action ``z`` (M5).

``z`` is the L2 **macro-action**, inferred at train time from a 10-step
primitive chunk. It is not a subgoal, not a state, and not a policy head.
Planning later samples ``z`` with L2 MPPI and does not use this posterior.
"""

from hwm_faithful.level2.action_encoder import (
    Level2ActionEncoder,
    Level2ActionEncoderOutput,
)
from hwm_faithful.level2.freeze import freeze_module
from hwm_faithful.level2.identity import Level2IdentityEncoder
from hwm_faithful.level2.posterior import Level2ActionPosterior, PosteriorOutput
from hwm_faithful.level2.temporal_abstraction import (
    Level2Inputs,
    Level2TemporalConfig,
    build_level2_inputs,
    chunk_level2_actions,
    flatten_action_chunk,
    subsample_level2_states,
)

__all__ = [
    "Level2ActionEncoder",
    "Level2ActionEncoderOutput",
    "Level2ActionPosterior",
    "Level2IdentityEncoder",
    "Level2Inputs",
    "Level2TemporalConfig",
    "PosteriorOutput",
    "build_level2_inputs",
    "chunk_level2_actions",
    "flatten_action_chunk",
    "freeze_module",
    "subsample_level2_states",
]
