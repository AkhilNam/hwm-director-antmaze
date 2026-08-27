"""Tests for GoalConditionedWorker shapes and action bounds."""

from __future__ import annotations

import torch

from hwm_director.data.state import ACHIEVED_GOAL_DIM, STATE_DIM
from hwm_director.data.transitions import ACTION_DIM
from hwm_director.models.worker import GoalConditionedWorker


def test_worker_output_shape_batched() -> None:
    model = GoalConditionedWorker(hidden_dims=(32, 32))
    state = torch.zeros(4, STATE_DIM)
    subgoal = torch.zeros(4, ACHIEVED_GOAL_DIM)
    action = model(state, subgoal)
    assert action.shape == (4, ACTION_DIM)


def test_worker_actions_in_unit_box() -> None:
    model = GoalConditionedWorker(hidden_dims=(16,))
    state = torch.randn(8, STATE_DIM)
    subgoal = torch.randn(8, ACHIEVED_GOAL_DIM)
    action = model(state, subgoal)
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)
    assert action.shape[-1] == ACTION_DIM
