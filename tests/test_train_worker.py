"""Tests for worker training helpers."""

from __future__ import annotations

import numpy as np

from hwm_director.training.train_worker import zero_action_mse


def test_zero_action_mse() -> None:
    actions = np.zeros((5, 8))
    actions[:, 0] = 2.0
    mse = zero_action_mse(actions)
    expected = (2.0**2) / 8
    assert abs(mse - expected) < 1e-8
