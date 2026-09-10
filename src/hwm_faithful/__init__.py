"""Faithful HWM (Diverse Maze), implemented independently of original HWM_PLDM.

M1: Level-1 observation encoder. M2: Level-1 residual conv predictor.
M3: Level-1 objectives (VICRegObs, IDM, PredictionProprio).
M4: Level-1 MPPI planner (``pi_L``).
M5: Level-2 skip-10 inputs and latent action ``z``.
M6: Level-2 residual predictor ``f_H``.
M7: Level-2 MPPI planner ``pi_H`` and hierarchical handoff.
M8: Diverse Maze data, normalization, goal, and eval interface.
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
from hwm_faithful.planning.config import Level1MPPIConfig, Level2MPPIConfig
from hwm_faithful.planning.hierarchical_planner import (
    HierarchicalPlanResult,
    HierarchicalPlanner,
)
from hwm_faithful.planning.level1_planner import Level1MPPIPlanner, Level1PlanResult
from hwm_faithful.planning.level2_planner import Level2MPPIPlanner, Level2PlanResult
from hwm_faithful.losses.level2_objective import (
    Level2ObjectiveOutput,
    level2_prediction_losses,
)
from hwm_faithful.data import (
    DiverseMazeEnvAdapter,
    DiverseMazeOfflineDataset,
    MazeNormalizer,
    encode_goal,
    load_probe_dataset,
    load_starts_targets,
    success_from_distance,
)
from hwm_faithful.level2 import (
    Level2ActionEncoder,
    Level2ActionPosterior,
    Level2IdentityEncoder,
    Level2Predictor,
    Level2WorldModel,
    build_level2_inputs,
)

__all__ = [
    "Level1Objective",
    "Level1ObjectiveOutput",
    "Level2ObjectiveOutput",
    "level2_prediction_losses",
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
    "Level2MPPIConfig",
    "Level2MPPIPlanner",
    "Level2PlanResult",
    "HierarchicalPlanner",
    "HierarchicalPlanResult",
    "Level2ActionEncoder",
    "Level2ActionPosterior",
    "Level2IdentityEncoder",
    "Level2Predictor",
    "Level2WorldModel",
    "build_level2_inputs",
    "DiverseMazeEnvAdapter",
    "DiverseMazeOfflineDataset",
    "MazeNormalizer",
    "encode_goal",
    "load_probe_dataset",
    "load_starts_targets",
    "success_from_distance",
]
