"""Convert AntMaze Dict observations into state and goal vectors.

AntMaze_UMaze-v5 returns a Dict with three arrays:

- ``achieved_goal`` (2,): current torso x/y position.
- ``observation`` (105,): Ant-v5 body state (height, orientation, joints,
  velocities, contact forces). This does **not** include x/y.
- ``desired_goal`` (2,): final task target x/y. Kept separate from ``state``.

Baseline layout:

    state = concatenate(achieved_goal, observation)  # shape (107,)
    goal  = desired_goal                             # shape (2,)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

STATE_DIM = 107
GOAL_DIM = 2
OBSERVATION_DIM = 105
ACHIEVED_GOAL_DIM = 2


class ExtractedState(NamedTuple):
    """Vectors derived from one raw env Dict observation."""

    state: np.ndarray
    goal: np.ndarray


def extract_state_and_goal(observation: dict) -> ExtractedState:
    """Build ``state`` and ``goal`` from a raw AntMaze observation dict.

    ``state`` is ``[achieved_goal, observation]`` with shape ``(107,)``.
    ``goal`` is ``desired_goal`` with shape ``(2,)``.
    """
    achieved_goal = np.asarray(observation["achieved_goal"], dtype=np.float64)
    body = np.asarray(observation["observation"], dtype=np.float64)
    goal = np.asarray(observation["desired_goal"], dtype=np.float64)
    state = np.concatenate([achieved_goal, body], axis=0)
    return ExtractedState(state=state, goal=goal)
