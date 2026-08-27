"""Tests for WorkerDataset future-subgoal construction."""

from __future__ import annotations

import numpy as np

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.data.worker_dataset import WorkerDataset, normalize_subgoal
from hwm_director.training.train_dynamics import split_episode_indices
from tests.helpers import make_transition


def _episode(episode_id: int, length: int) -> list:
    steps = []
    for t in range(length):
        state = np.zeros(STATE_DIM)
        next_state = np.zeros(STATE_DIM)
        state[0] = float(episode_id)
        state[1] = float(t)
        next_state[0] = float(episode_id)
        next_state[1] = float(t + 1)
        steps.append(
            make_transition(
                episode_id=episode_id,
                state=state,
                next_state=next_state,
                action=np.full(8, t, dtype=np.float32),
            )
        )
    return steps


def test_future_subgoal_stays_in_episode() -> None:
    transitions = _episode(0, 6) + _episode(1, 6)
    dataset = WorkerDataset(transitions, horizon_k=10)
    for example in dataset.examples:
        assert example.subgoal[0] == example.state[0]


def test_future_index_greater_than_current() -> None:
    dataset = WorkerDataset(_episode(0, 8), horizon_k=5)
    for example in dataset.examples:
        assert example.subgoal[1] > example.state[1]
        assert example.offset >= 1


def test_future_offset_at_most_k() -> None:
    horizon_k = 3
    dataset = WorkerDataset(_episode(0, 12), horizon_k=horizon_k)
    for example in dataset.examples:
        assert 1 <= example.offset <= horizon_k
        assert example.subgoal[1] - example.state[1] == example.offset


def test_worker_split_episodes_do_not_overlap() -> None:
    transitions = _episode(0, 5) + _episode(1, 5) + _episode(2, 5) + _episode(3, 5)
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=0.25, seed=0
    )
    train_eps = {transitions[int(i)].episode_id for i in train_idx}
    val_eps = {transitions[int(i)].episode_id for i in val_idx}
    assert train_eps.isdisjoint(val_eps)


def test_subgoal_normalization_uses_training_xy_stats_only() -> None:
    rng = np.random.default_rng(0)
    train_states = rng.normal(loc=10.0, scale=2.0, size=(64, STATE_DIM))
    val_states = rng.normal(loc=0.0, scale=1.0, size=(64, STATE_DIM))
    normalizer = StateNormalizer().fit(train_states)
    xy = normalizer.mean[:2].copy()
    out = normalize_subgoal(xy, normalizer)
    np.testing.assert_allclose(out, np.zeros(2), atol=1e-6)
    val_mean_xy = np.mean(val_states[:, :2], axis=0)
    leaked = (xy - val_mean_xy) / np.maximum(np.std(val_states[:, :2], axis=0), 1e-8)
    assert np.linalg.norm(out - leaked) > 0.5
