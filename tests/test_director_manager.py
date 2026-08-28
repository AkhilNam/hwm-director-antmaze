"""Tests for DirectorManager shapes and relative subgoals."""

from __future__ import annotations

import torch

from hwm_director.data.state import ACHIEVED_GOAL_DIM, GOAL_DIM, STATE_DIM
from hwm_director.models.director_manager import DirectorManager


def test_manager_input_is_state_plus_final_goal() -> None:
    model = DirectorManager(hidden_dims=(16,))
    state = torch.zeros(4, STATE_DIM)
    goal = torch.zeros(4, GOAL_DIM)
    assert state.shape[-1] + goal.shape[-1] == 31
    subgoal = model(state, goal, clamp=False)
    assert subgoal.shape == (4, ACHIEVED_GOAL_DIM)


def test_manager_output_is_2d_subgoal() -> None:
    model = DirectorManager(hidden_dims=(8,))
    state = torch.zeros(STATE_DIM)
    state[0] = 1.0
    state[1] = 2.0
    goal = torch.zeros(GOAL_DIM)
    subgoal = model(state, goal, clamp=False)
    assert subgoal.shape == (2,)


def test_manager_is_relative_to_current_xy() -> None:
    model = DirectorManager(hidden_dims=(8,), max_subgoal_distance=100.0)
    torch.manual_seed(0)
    state = torch.zeros(STATE_DIM)
    state[0] = 10.0
    state[1] = -4.0
    goal = torch.ones(GOAL_DIM)
    with torch.no_grad():
        delta = model.net(torch.cat([state, goal], dim=-1))
        subgoal = model(state, goal, clamp=False)
    torch.testing.assert_close(subgoal, state[:2] + delta)
