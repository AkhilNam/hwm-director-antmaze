"""Tests for manager training split and baselines."""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import STATE_DIM
from hwm_director.models.director_manager import DirectorManager
from hwm_director.training.train_manager import train_director_manager
from tests.helpers import make_transition


def _episode(episode_id: int, length: int) -> list:
    steps = []
    for t in range(length):
        state = np.zeros(STATE_DIM)
        next_state = np.zeros(STATE_DIM)
        state[0] = float(t)
        next_state[0] = float(t + 1)
        steps.append(
            make_transition(
                episode_id=episode_id,
                state=state,
                next_state=next_state,
                goal=np.array([50.0, 0.0]),
            )
        )
    return steps


def test_train_manager_episode_split_and_metrics() -> None:
    transitions = _episode(0, 16) + _episode(1, 16)
    metrics = train_director_manager(
        transitions,
        model=DirectorManager(hidden_dims=(8,)),
        horizon_k=4,
        val_fraction=0.5,
        seed=0,
        batch_size=8,
        epochs=2,
    )
    assert metrics["n_train_episodes"] == 1
    assert metrics["n_val_episodes"] == 1
    assert set(metrics["train_episode_ids"]).isdisjoint(metrics["val_episode_ids"])
    assert metrics["n_train_examples"] > 0
    assert metrics["n_val_examples"] > 0
    assert np.isfinite(metrics["val_mse"])
    assert np.isfinite(metrics["val_xy_euclidean"])
    assert np.isfinite(metrics["current_position_val_euclidean"])
    assert np.isfinite(metrics["final_goal_val_euclidean"])
    assert metrics["horizon_k"] == 4
