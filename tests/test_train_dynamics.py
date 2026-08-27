"""Tests for episode-level split, no-change baseline, and x/y MSE."""

from __future__ import annotations

import numpy as np
import pytest

from hwm_director.data.state import STATE_DIM
from hwm_director.training.train_dynamics import (
    next_position_mse,
    no_change_baseline_mse,
    split_episode_indices,
)
from tests.helpers import make_transition


def _transitions_for_episodes(lengths: list[int]):
    transitions = []
    for episode_id, length in enumerate(lengths):
        for _ in range(length):
            transitions.append(make_transition(episode_id=episode_id))
    return transitions


def test_split_episode_no_leakage() -> None:
    transitions = _transitions_for_episodes([5, 5, 5, 5, 5])
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=0.2, seed=0
    )
    train_eps = {transitions[int(i)].episode_id for i in train_idx}
    val_eps = {transitions[int(i)].episode_id for i in val_idx}
    assert train_eps.isdisjoint(val_eps)


def test_split_episode_complete_coverage() -> None:
    transitions = _transitions_for_episodes([3, 4, 5, 2])
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=0.25, seed=0
    )
    combined = np.concatenate([train_idx, val_idx])
    assert len(combined) == len(transitions)
    assert len(set(combined.tolist())) == len(transitions)
    assert set(combined.tolist()) == set(range(len(transitions)))


def test_split_episode_whole_trajectories() -> None:
    transitions = _transitions_for_episodes([4, 4, 4])
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=0.34, seed=1
    )
    assigned = {}
    for name, indices in (("train", train_idx), ("val", val_idx)):
        for i in indices:
            ep = transitions[int(i)].episode_id
            assigned.setdefault(ep, name)
            assert assigned[ep] == name


def test_split_episode_reproducible() -> None:
    transitions = _transitions_for_episodes([2] * 10)
    a_train, a_val = split_episode_indices(transitions, val_fraction=0.2, seed=0)
    b_train, b_val = split_episode_indices(transitions, val_fraction=0.2, seed=0)
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_val, b_val)


def test_split_episode_seed_changes_assignment() -> None:
    transitions = _transitions_for_episodes([2] * 20)
    train0, val0 = split_episode_indices(transitions, val_fraction=0.2, seed=0)
    train1, val1 = split_episode_indices(transitions, val_fraction=0.2, seed=1)
    assert not np.array_equal(np.sort(train0), np.sort(train1)) or not np.array_equal(
        np.sort(val0), np.sort(val1)
    )


def test_split_episode_single_episode_raises() -> None:
    transitions = _transitions_for_episodes([8])
    with pytest.raises(ValueError, match="2 unique episodes"):
        split_episode_indices(transitions, val_fraction=0.2, seed=0)


def test_no_change_baseline_mse() -> None:
    states = np.zeros((5, STATE_DIM))
    next_states = np.zeros((5, STATE_DIM))
    next_states[:, 0] = 2.0
    mse = no_change_baseline_mse(states, next_states)
    expected = (2.0**2) / STATE_DIM
    assert abs(mse - expected) < 1e-8


def test_next_position_mse() -> None:
    pred = np.zeros((3, STATE_DIM))
    target = np.zeros((3, STATE_DIM))
    pred[:, 0] = 2.0
    mse = next_position_mse(pred, target)
    assert abs(mse - 2.0) < 1e-8


def test_dynamics_normalizer_fit_on_train_episodes_only() -> None:
    """Mean of the fitted normalizer must come from train episodes, not val."""
    from hwm_director.models.dynamics_low import LowLevelDynamicsModel
    from hwm_director.training.train_dynamics import train_low_level_dynamics

    train_ep = []
    val_ep = []
    for _ in range(4):
        state = np.full(STATE_DIM, 10.0)
        next_state = np.full(STATE_DIM, 10.5)
        train_ep.append(make_transition(episode_id=0, state=state, next_state=next_state))
        vstate = np.full(STATE_DIM, -10.0)
        vnext = np.full(STATE_DIM, -9.5)
        val_ep.append(make_transition(episode_id=1, state=vstate, next_state=vnext))
    metrics = train_low_level_dynamics(
        train_ep + val_ep,
        model=LowLevelDynamicsModel(hidden_dims=(8,)),
        val_fraction=0.5,
        seed=0,
        batch_size=4,
        epochs=1,
        lr=1e-3,
    )
    train_ids = set(metrics["train_episode_ids"])
    val_ids = set(metrics["val_episode_ids"])
    assert train_ids.isdisjoint(val_ids)
    mean0 = float(metrics["normalizer"].mean[0])
    if train_ids == {0}:
        assert abs(mean0 - 10.0) < 1e-6
    else:
        assert abs(mean0 - (-10.0)) < 1e-6

