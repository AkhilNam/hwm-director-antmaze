"""Faithful HWM (Diverse Maze), implemented independently of original HWM_PLDM.

M1: Level-1 observation encoder. M2: Level-1 residual conv predictor.
M3: Level-1 objectives (VICRegObs, IDM, PredictionProprio).
M4: Level-1 MPPI planner (``pi_L``).
M5: Level-2 skip-10 inputs and latent action ``z``.
The simplified raw-state baseline remains in ``hwm_director`` and is not
imported here.
"""

from hwm_faithful.losses import Level1Objective, Level1ObjectiveOutput
from hwm_faithful.models.action_encoder import PrimitiveActionEncoder
from hwm_faithful.models.conv_predictor import (
    Level1Predictor,
    Level1PredictorOutput,
    Level1RolloutOutput,
)
from hwm_faithful.models.level1_encoder import Level1Encoder, Level1EncoderOutput
from hwm_faithful.models.level1_world_model import Level1WorldModel
from hwm_faithful.models.proprio import ProprioExpander
from hwm_faithful.models.visual_encoder import VisualEncoder
from hwm_faithful.planning.level1_planner import Level1MPPIPlanner, Level1PlanResult
from hwm_faithful.planning.config import Level1MPPIConfig
from hwm_faithful.level2 import (
    Level2ActionEncoder,
    Level2ActionPosterior,
    Level2IdentityEncoder,
    build_level2_inputs,
)

__all__ = [
    "Level1Objective",
    "Level1ObjectiveOutput",
    "Level1Encoder",
    "Level1EncoderOutput",
    "Level1Predictor",
    "Level1PredictorOutput",
    "Level1RolloutOutput",
    "Level1WorldModel",
    "PrimitiveActionEncoder",
    "ProprioExpander",
    "VisualEncoder",
    "Level1MPPIConfig",
    "Level1MPPIPlanner",
    "Level1PlanResult",
    "Level2ActionEncoder",
    "Level2ActionPosterior",
    "Level2IdentityEncoder",
    "build_level2_inputs",
]
