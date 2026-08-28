"""Tests for exactly-K manager BC examples."""

from __future__ import annotations

import numpy as np

from hwm_director.data.manager_dataset import DirectorManagerDataset
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.training.train_dynamics import split_episode_indices
from tests.helpers import make_transition


def _episode(episode_id: int, length: int, goal_x: float = 10.0) -> list:
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
                goal=np.array([goal_x, 0.0], dtype=np.float64),
            )
        )
    return steps


def test_manager_target_stays_in_same_episode() -> None:
    horizon_k = 4
    transitions = _episode(0, 12) + _episode(1, 12, goal_x=20.0)
    dataset = DirectorManagerDataset(transitions, horizon_k=horizon_k)
    assert dataset.examples
    for example in dataset.examples:
        assert example.target_subgoal[0] == example.state[0]
        assert example.final_goal[0] in (10.0, 20.0)


def test_manager_skips_examples_with_fewer_than_k_steps() -> None:
    horizon_k = 10
    short = _episode(0, 5)
    long = _episode(1, 12)
    short_ds = DirectorManagerDataset(short, horizon_k=horizon_k)
    long_ds = DirectorManagerDataset(long, horizon_k=horizon_k)
    assert len(short_ds) == 0
    assert len(long_ds) == 12 - 10 + 1
    for example in long_ds.examples:
        assert example.t + horizon_k <= 12
        assert example.target_subgoal[1] - example.state[1] == horizon_k


def test_manager_split_has_no_episode_leakage() -> None:
    transitions = _episode(0, 12) + _episode(1, 12) + _episode(2, 12) + _episode(3, 12)
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=0.25, seed=0
    )
    train_eps = {transitions[int(i)].episode_id for i in train_idx}
    val_eps = {transitions[int(i)].episode_id for i in val_idx}
    assert train_eps.isdisjoint(val_eps)
    train_ds = DirectorManagerDataset(
        [transitions[int(i)] for i in train_idx], horizon_k=4
    )
    val_ds = DirectorManagerDataset(
        [transitions[int(i)] for i in val_idx], horizon_k=4
    )
    assert set(train_ds.episode_ids.tolist()).isdisjoint(set(val_ds.episode_ids.tolist()))


def test_manager_normalization_uses_train_stats_only() -> None:
    rng = np.random.default_rng(0)
    train_states = rng.normal(loc=10.0, scale=2.0, size=(64, STATE_DIM))
    val_states = rng.normal(loc=0.0, scale=1.0, size=(64, STATE_DIM))
    normalizer = StateNormalizer().fit(train_states)
    train_xy_mean = train_states[:, :2].mean(axis=0)
    dataset_states = []
    for t in range(12):
        state = np.zeros(STATE_DIM)
        next_state = np.zeros(STATE_DIM)
        state[:2] = train_xy_mean
        next_state[:2] = train_xy_mean + 1.0
        dataset_states.append(
            make_transition(episode_id=0, state=state, next_state=next_state)
        )
    dataset = DirectorManagerDataset(
        dataset_states, horizon_k=4, normalizer=normalizer
    )
    # normalized current xy of a train-mean state should be ~0
    np.testing.assert_allclose(dataset.states[0, :2].numpy(), np.zeros(2), atol=1e-5)
    val_mean = val_states[:, :2].mean(axis=0)
    leaked = (train_xy_mean - val_mean) / np.maximum(val_states[:, :2].std(axis=0), 1e-8)
    assert np.linalg.norm(dataset.states[0, :2].numpy() - leaked) > 0.5
