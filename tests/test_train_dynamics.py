"""Tests for episode-level split, no-change baseline, and x/y MSE."""

from __future__ import annotations

import numpy as np
import pytest

from hwm_director.data.transitions import Transition
from hwm_director.training.train_dynamics import (
    next_position_mse,
    no_change_baseline_mse,
    split_episode_indices,
)


def _transitions_for_episodes(lengths: list[int]) -> list[Transition]:
    transitions: list[Transition] = []
    for episode_id, length in enumerate(lengths):
        for _ in range(length):
            transitions.append(
                Transition(
                    state=np.zeros(107),
                    action=np.zeros(8, dtype=np.float32),
                    next_state=np.zeros(107),
                    goal=np.zeros(2),
                    episode_id=episode_id,
                )
            )
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
    states = np.zeros((5, 107))
    next_states = np.zeros((5, 107))
    next_states[:, 0] = 2.0
    mse = no_change_baseline_mse(states, next_states)
    expected = (2.0**2) / 107
    assert abs(mse - expected) < 1e-8


def test_next_position_mse() -> None:
    pred = np.zeros((3, 107))
    target = np.zeros((3, 107))
    pred[:, 0] = 2.0
    mse = next_position_mse(pred, target)
    assert abs(mse - 2.0) < 1e-8
