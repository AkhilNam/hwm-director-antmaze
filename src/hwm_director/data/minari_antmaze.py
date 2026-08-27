"""Load Minari AntMaze episodes into ``Transition`` lists.

Default dataset: ``D4RL/antmaze/umaze-v1`` (AntMaze_UMaze-v4).

Each Minari environment step becomes one ``Transition``:

    state      = [achieved_goal_t, observation_t]      # (29,)
    action     = actions[t]                            # (8,)
    next_state = [achieved_goal_{t+1}, observation_{t+1}]
    goal       = desired_goal_t                        # (2,)
    episode_id = Minari episode id (or enumerate index)

Episode boundaries and temporal order are preserved: a transition never
uses ``next_state`` from a different episode, and steps stay in dataset
order.

``qpos`` / ``qvel``
-------------------
Minari's published observation space for this dataset is a Dict of
``achieved_goal``, ``desired_goal``, and the 27-D Ant-v4 vector. That is
enough to restore MuJoCo state **exactly** via
``ant_v4_qpos_qvel_from_state`` (see ``state.py``). If an episode's
``infos`` already stores ``qpos``/``qvel`` arrays of the right length,
those copies are used instead of the reconstruction.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hwm_director.data.state import (
    ACHIEVED_GOAL_DIM,
    GOAL_DIM,
    OBSERVATION_DIM,
    extract_state_and_goal,
    ant_v4_qpos_qvel_from_state,
)
from hwm_director.data.transitions import ACTION_DIM, Transition
from hwm_director.envs.antmaze import DEFAULT_DATASET_ID
from hwm_director.models.encoder import IdentityEncoder

DEFAULT_MINARI_DATASET_ID = DEFAULT_DATASET_ID


def _as_2d(array: np.ndarray, last_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1 and last_dim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[-1] != last_dim:
        raise ValueError(f"{name} has shape {arr.shape}, expected (*, {last_dim})")
    return arr


def _dict_time_series(observations: dict, n_obs: int | None = None) -> dict[str, np.ndarray]:
    achieved = _as_2d(
        observations["achieved_goal"], ACHIEVED_GOAL_DIM, "achieved_goal"
    )
    body = _as_2d(observations["observation"], OBSERVATION_DIM, "observation")
    desired = _as_2d(observations["desired_goal"], GOAL_DIM, "desired_goal")
    if achieved.shape[0] != body.shape[0] or body.shape[0] != desired.shape[0]:
        raise ValueError(
            "observation dict time lengths differ: "
            f"achieved_goal={achieved.shape[0]}, observation={body.shape[0]}, "
            f"desired_goal={desired.shape[0]}"
        )
    if n_obs is not None and achieved.shape[0] != n_obs:
        raise ValueError(
            f"observation time length {achieved.shape[0]} != expected {n_obs}"
        )
    return {
        "achieved_goal": achieved,
        "observation": body,
        "desired_goal": desired,
    }


def _sequence_of_dicts(observations: Sequence[dict]) -> dict[str, np.ndarray]:
    stacked = {
        "achieved_goal": np.stack(
            [np.asarray(o["achieved_goal"], dtype=np.float64) for o in observations]
        ),
        "observation": np.stack(
            [np.asarray(o["observation"], dtype=np.float64) for o in observations]
        ),
        "desired_goal": np.stack(
            [np.asarray(o["desired_goal"], dtype=np.float64) for o in observations]
        ),
    }
    return _dict_time_series(stacked)


def unpack_episode_observations(observations: Any) -> dict[str, np.ndarray]:
    """Normalize Minari observations to time-major Dict arrays."""
    if isinstance(observations, dict):
        return _dict_time_series(observations)
    if isinstance(observations, (list, tuple)):
        return _sequence_of_dicts(observations)
    raise TypeError(
        f"Unsupported Minari observations type {type(observations)!r}; "
        "expected dict of arrays or a sequence of dicts"
    )


def _info_vector_series(infos: Any, key: str, expected_dim: int) -> np.ndarray | None:
    """Return a ``(T, dim)`` array from episode infos, or None if absent."""
    if infos is None:
        return None
    if isinstance(infos, dict):
        if key not in infos:
            return None
        arr = np.asarray(infos[key])
        if arr.ndim == 1 and arr.shape[0] == expected_dim:
            return arr.reshape(1, expected_dim)
        if arr.ndim == 2 and arr.shape[1] == expected_dim:
            return arr
        return None
    if isinstance(infos, (list, tuple)) and infos:
        first = infos[0]
        if isinstance(first, dict) and key in first:
            stacked = np.stack([np.asarray(item[key]) for item in infos])
            if stacked.ndim == 2 and stacked.shape[1] == expected_dim:
                return stacked
    return None


def _qpos_qvel_for_step(
    state: np.ndarray,
    infos: Any,
    t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Prefer stored simulator state; otherwise use the documented Ant-v4 map."""
    from hwm_director.data.state import QPOS_DIM, QVEL_DIM

    qpos_series = _info_vector_series(infos, "qpos", QPOS_DIM)
    qvel_series = _info_vector_series(infos, "qvel", QVEL_DIM)
    if qpos_series is not None and qvel_series is not None and t < qpos_series.shape[0]:
        return (
            np.array(qpos_series[t], dtype=np.float64, copy=True),
            np.array(qvel_series[t], dtype=np.float64, copy=True),
        )
    return ant_v4_qpos_qvel_from_state(state)


def episode_id_from_minari(episode: Any, fallback: int) -> int:
    """Use ``episode.id`` when it is an int; otherwise ``fallback``."""
    episode_id = getattr(episode, "id", None)
    if episode_id is None:
        return int(fallback)
    return int(episode_id)


def transitions_from_minari_episode(
    episode: Any,
    episode_id: int | None = None,
    *,
    fallback_episode_id: int = 0,
    encoder: IdentityEncoder | None = None,
) -> list[Transition]:
    """Convert one Minari episode into temporally ordered ``Transition``s.

    Requires ``len(observations) == len(actions) + 1`` (standard Minari:
    observations include the terminal / truncated state).
    """
    if encoder is None:
        encoder = IdentityEncoder()
    if episode_id is None:
        episode_id = episode_id_from_minari(episode, fallback_episode_id)

    obs = unpack_episode_observations(episode.observations)
    actions = _as_2d(np.asarray(episode.actions, dtype=np.float64), ACTION_DIM, "actions")
    n_obs = int(obs["achieved_goal"].shape[0])
    n_act = int(actions.shape[0])
    if n_obs != n_act + 1:
        raise ValueError(
            "Minari episode must have one more observation than actions "
            f"(got n_obs={n_obs}, n_actions={n_act}). Refusing to invent "
            "a missing next_state or to drop a boundary step."
        )

    infos = getattr(episode, "infos", None)
    transitions: list[Transition] = []
    for t in range(n_act):
        extracted = extract_state_and_goal(
            {
                "achieved_goal": obs["achieved_goal"][t],
                "observation": obs["observation"][t],
                "desired_goal": obs["desired_goal"][t],
            }
        )
        extracted_next = extract_state_and_goal(
            {
                "achieved_goal": obs["achieved_goal"][t + 1],
                "observation": obs["observation"][t + 1],
                "desired_goal": obs["desired_goal"][t + 1],
            }
        )
        state = encoder.encode(extracted.state)
        next_state = encoder.encode(extracted_next.state)
        qpos, qvel = _qpos_qvel_for_step(state, infos, t)
        transition = Transition(
            state=state,
            action=np.asarray(actions[t], dtype=np.float64),
            next_state=next_state,
            goal=extracted.goal,
            episode_id=int(episode_id),
            qpos=qpos,
            qvel=qvel,
        )
        transition.validate()
        transitions.append(transition)
    return transitions


def transitions_from_minari_episodes(
    episodes: Sequence[Any],
    *,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
) -> list[Transition]:
    """Convert many Minari episodes, preserving episode order and ids."""
    out: list[Transition] = []
    n_eps = 0
    for i, episode in enumerate(episodes):
        if max_episodes is not None and n_eps >= max_episodes:
            break
        if max_transitions is not None and len(out) >= max_transitions:
            break
        ep_id = episode_id_from_minari(episode, i)
        converted = transitions_from_minari_episode(
            episode, episode_id=ep_id, fallback_episode_id=i
        )
        if max_transitions is not None:
            remaining = max_transitions - len(out)
            if remaining <= 0:
                break
            # Keep whole remaining steps from this episode up to the cap, but
            # never attach a truncated step that would invent a next episode.
            converted = converted[:remaining]
        out.extend(converted)
        n_eps += 1
    return out


def load_minari_transitions(
    dataset_id: str = DEFAULT_MINARI_DATASET_ID,
    *,
    max_episodes: int | None = None,
    max_transitions: int | None = None,
    download: bool = True,
) -> list[Transition]:
    """Load a Minari dataset and convert episodes to ``Transition``s."""
    import minari

    dataset = minari.load_dataset(dataset_id, download=download)
    return transitions_from_minari_episodes(
        dataset.iterate_episodes(),
        max_episodes=max_episodes,
        max_transitions=max_transitions,
    )
