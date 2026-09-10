"""Faithful Level-1 training objectives (M3)."""

from hwm_faithful.losses.inverse_dynamics import InverseDynamics
from hwm_faithful.losses.level1_objective import Level1Objective, Level1ObjectiveOutput

__all__ = [
    "InverseDynamics",
    "Level1Objective",
    "Level1ObjectiveOutput",
]
