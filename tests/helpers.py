"""Shared test helpers. Dimensions come from the research constants."""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import GOAL_DIM, OBSERVATION_DIM, STATE_DIM
from hwm_director.data.transitions import ACTION_DIM, Transition


def make_transition(
    *,
    episode_id: int = 0,
    state: np.ndarray | None = None,
    next_state: np.ndarray | None = None,
    action: np.ndarray | None = None,
    goal: np.ndarray | None = None,
    qpos: np.ndarray | None = None,
    qvel: np.ndarray | None = None,
) -> Transition:
    return Transition(
        state=np.zeros(STATE_DIM, dtype=np.float64) if state is None else state,
        action=(
            np.zeros(ACTION_DIM, dtype=np.float32) if action is None else action
        ),
        next_state=(
            np.zeros(STATE_DIM, dtype=np.float64)
            if next_state is None
            else next_state
        ),
        goal=np.zeros(GOAL_DIM, dtype=np.float64) if goal is None else goal,
        episode_id=episode_id,
        qpos=qpos,
        qvel=qvel,
    )


def fake_antmaze_observation() -> dict:
    return {
        "observation": np.arange(OBSERVATION_DIM, dtype=np.float64),
        "achieved_goal": np.array([1.5, -2.0], dtype=np.float64),
        "desired_goal": np.array([3.0, 4.0], dtype=np.float64),
    }
