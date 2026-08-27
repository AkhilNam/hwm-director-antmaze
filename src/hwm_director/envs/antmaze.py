"""AntMaze environment factory (Gymnasium-Robotics / Minari).

The default *research* environment is ``AntMaze_UMaze-v4``, matching Minari
``D4RL/antmaze/umaze-v1``. Experiments tied to that offline dataset should
prefer ``recover_minari_environment`` so the Gymnasium spec is the one stored
with the data.

``make_antmaze`` remains a generic constructor. ``AntMaze_UMaze-v5`` can still
be requested explicitly; it is not the default and is not compatible with the
29-D Minari baseline state.
"""

from __future__ import annotations

import gymnasium as gym
import gymnasium_robotics

DEFAULT_ENV_ID = "AntMaze_UMaze-v4"
DEFAULT_DATASET_ID = "D4RL/antmaze/umaze-v1"


def make_antmaze(env_id: str = DEFAULT_ENV_ID, **kwargs) -> gym.Env:
    """Register Gymnasium-Robotics envs and construct an AntMaze environment."""
    gym.register_envs(gymnasium_robotics)
    return gym.make(env_id, **kwargs)


def recover_minari_environment(
    dataset_id: str = DEFAULT_DATASET_ID,
    *,
    eval_env: bool = False,
    download: bool = True,
):
    """Return the Gymnasium env stored with a Minari dataset.

    Parameters
    ----------
    dataset_id:
        Minari id, default ``D4RL/antmaze/umaze-v1``.
    eval_env:
        If True, recover the evaluation environment spec when the dataset
        provides one.
    download:
        Forwarded to ``minari.load_dataset``.
    """
    import gymnasium as gym
    import gymnasium_robotics
    import minari

    gym.register_envs(gymnasium_robotics)
    dataset = minari.load_dataset(dataset_id, download=download)
    return dataset.recover_environment(eval_env=eval_env)
