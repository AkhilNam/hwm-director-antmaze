"""AntMaze environment factory (Gymnasium-Robotics)."""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics

DEFAULT_ENV_ID = "AntMaze_UMaze-v5"


def make_antmaze(env_id: str = DEFAULT_ENV_ID, **kwargs) -> gym.Env:
    """Register Gymnasium-Robotics envs and construct an AntMaze environment."""
    gym.register_envs(gymnasium_robotics)
    return gym.make(env_id, **kwargs)
