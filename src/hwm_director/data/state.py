"""Convert AntMaze Dict observations into state and goal vectors.

Default research environment: ``AntMaze_UMaze-v4`` (Minari
``D4RL/antmaze/umaze-v1``). The raw observation is a Dict with three arrays:

- ``achieved_goal`` (2,): current torso x/y position.
- ``observation`` (27,): Ant-v4 proprioception. This does **not** include
  torso x/y. See ``ant_v4_qpos_qvel_from_state`` for the layout.
- ``desired_goal`` (2,): final task target x/y. Kept separate from ``state``.

Baseline layout (dimensions come from the constants below; do not scatter
magic numbers through the rest of the stack):

    state = concatenate(achieved_goal, observation)  # (STATE_DIM,)
    goal  = desired_goal                             # (GOAL_DIM,)

There are no contact-force channels in this representation. Ant-v5's 105-D
observation (including 78 ``cfrc_ext`` dims) is not used by the default
experiment.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

ACHIEVED_GOAL_DIM = 2
OBSERVATION_DIM = 27
STATE_DIM = ACHIEVED_GOAL_DIM + OBSERVATION_DIM  # 29
GOAL_DIM = 2

# Ant-v4 MuJoCo sizes. ``observation`` stores qpos[2:] then qvel.
QPOS_DIM = 15
QVEL_DIM = 14
OBS_QPOS_TAIL_DIM = QPOS_DIM - ACHIEVED_GOAL_DIM  # 13


class ExtractedState(NamedTuple):
    """Vectors derived from one raw env Dict observation."""

    state: np.ndarray
    goal: np.ndarray


def extract_state_and_goal(observation: dict) -> ExtractedState:
    """Build ``state`` and ``goal`` from a raw AntMaze observation dict.

    ``state`` is ``[achieved_goal, observation]`` with shape ``(STATE_DIM,)``.
    ``goal`` is ``desired_goal`` with shape ``(GOAL_DIM,)``.
    """
    achieved_goal = np.asarray(observation["achieved_goal"], dtype=np.float64)
    body = np.asarray(observation["observation"], dtype=np.float64)
    goal = np.asarray(observation["desired_goal"], dtype=np.float64)
    if achieved_goal.shape != (ACHIEVED_GOAL_DIM,):
        raise ValueError(
            f"achieved_goal has shape {achieved_goal.shape}, "
            f"expected ({ACHIEVED_GOAL_DIM},)"
        )
    if body.shape != (OBSERVATION_DIM,):
        raise ValueError(
            f"observation has shape {body.shape}, expected ({OBSERVATION_DIM},)"
        )
    if goal.shape != (GOAL_DIM,):
        raise ValueError(f"desired_goal has shape {goal.shape}, expected ({GOAL_DIM},)")
    state = np.concatenate([achieved_goal, body], axis=0)
    return ExtractedState(state=state, goal=goal)


def ant_v4_qpos_qvel_from_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct Ant-v4 ``qpos`` / ``qvel`` from the 29-D baseline state.

    This mapping is **exact** for Gymnasium Ant-v4 with
    ``exclude_current_positions_from_observation=True`` (the AntMaze_UMaze-v4
    default). It is not an approximation and does not invent missing channels.

    Ant-v4 simulator sizes
    ----------------------
    - ``qpos`` has length 15: ``[x, y, z, quaternion(4), 8 hinge joints]``.
    - ``qvel`` has length 14: ``[linvel(3), angvel(3), 8 hinge velocities]``.

    Observation layout (27-D, no x/y, no contact forces)
    ----------------------------------------------------
    Gymnasium Ant-v4 concatenates ``qpos[2:]`` (13) then ``qvel`` (14).
    AntMaze stores torso x/y separately as ``achieved_goal``.

    Therefore the baseline state is:

        state[0:2]   = achieved_goal = qpos[0:2]   (x, y)
        state[2:15]  = observation[0:13] = qpos[2:]
        state[15:29] = observation[13:27] = qvel

    so

        qpos = state[0:15]
        qvel = state[15:29]

    Do not use this helper for Ant-v5: that observation includes 78 contact-
    force dimensions that are **not** a function of ``qpos``/``qvel`` alone.
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state has shape {state.shape}, expected ({STATE_DIM},)")
    qpos = np.array(state[:QPOS_DIM], copy=True)
    qvel = np.array(state[QPOS_DIM : QPOS_DIM + QVEL_DIM], copy=True)
    return qpos, qvel
