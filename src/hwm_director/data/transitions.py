"""One-step transition ``(state, action, next_state)`` plus a task ``goal``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hwm_director.data.state import GOAL_DIM, STATE_DIM

ACTION_DIM = 8


@dataclass
class Transition:
    """One environment step.

    Expected shapes
    ---------------
    state:      (107,)
    action:     (8,)
    next_state: (107,)
    goal:       (2,)
    """

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    goal: np.ndarray

    def validate(self) -> None:
        """Check that all fields have the expected 1-D shapes.

        Raises
        ------
        ValueError
            If any field has the wrong shape.
        """
        expected = {
            "state": (STATE_DIM,),
            "action": (ACTION_DIM,),
            "next_state": (STATE_DIM,),
            "goal": (GOAL_DIM,),
        }
        for name, shape in expected.items():
            actual = np.asarray(getattr(self, name)).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")
