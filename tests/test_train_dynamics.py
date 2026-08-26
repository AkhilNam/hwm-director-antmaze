"""Tests for split, no-change baseline, and x/y MSE."""

from __future__ import annotations

import numpy as np

from hwm_director.training.train_dynamics import (
    next_position_mse,
    no_change_baseline_mse,
    split_indices,
)


def test_split_indices_no_overlap() -> None:
    train_idx, val_idx = split_indices(100, val_fraction=0.2, seed=0)
    assert len(train_idx) + len(val_idx) == 100
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(100))


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
