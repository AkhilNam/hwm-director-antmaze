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
    state:      (STATE_DIM,)   # 29 = achieved_goal (2) + Ant-v4 observation (27)
    action:     (ACTION_DIM,)  # 8
    next_state: (STATE_DIM,)
    goal:       (GOAL_DIM,)    # 2, desired_goal

    ``episode_id`` labels the rollout that produced this step. It is an int,
    not an array, and is used only for trajectory-level train/val splits.

    ``qpos`` / ``qvel`` are optional copies of the MuJoCo configuration at
    ``s_t`` (before ``action``). For Minari Ant-v4 they are filled from the
    exact mapping in ``ant_v4_qpos_qvel_from_state`` (or from episode infos
    when those arrays are present). They are not part of the learned state.
    """

    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    goal: np.ndarray
    episode_id: int
    qpos: np.ndarray | None = None
    qvel: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.qpos is not None:
            self.qpos = np.array(self.qpos, copy=True)
        if self.qvel is not None:
            self.qvel = np.array(self.qvel, copy=True)

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
