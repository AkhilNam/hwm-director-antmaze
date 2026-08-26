"""Tests for GoalConditionedWorker shapes and action bounds."""

from __future__ import annotations

import torch

from hwm_director.models.worker import GoalConditionedWorker


def test_worker_output_shape_batched() -> None:
    model = GoalConditionedWorker(hidden_dims=(32, 32))
    state = torch.zeros(4, 107)
    subgoal = torch.zeros(4, 2)
    action = model(state, subgoal)
    assert action.shape == (4, 8)


def test_worker_actions_in_unit_box() -> None:
    model = GoalConditionedWorker(hidden_dims=(16,))
    state = torch.randn(8, 107)
    subgoal = torch.randn(8, 2)
    action = model(state, subgoal)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
