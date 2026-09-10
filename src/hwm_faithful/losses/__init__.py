"""Faithful Level-1 training objectives (M3)."""

from hwm_faithful.losses.inverse_dynamics import InverseDynamics
from hwm_faithful.losses.level1_objective import Level1Objective, Level1ObjectiveOutput
from hwm_faithful.losses.level2_objective import (
    Level2ObjectiveOutput,
    level2_prediction_losses,
)

__all__ = [
    "InverseDynamics",
    "Level1Objective",
    "Level1ObjectiveOutput",
    "Level2ObjectiveOutput",
    "level2_prediction_losses",
]
