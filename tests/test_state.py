"""Tests for AntMaze state extraction."""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import STATE_DIM, extract_state_and_goal


def _fake_antmaze_observation() -> dict:
    return {
        "observation": np.arange(105, dtype=np.float64),
        "achieved_goal": np.array([1.5, -2.0], dtype=np.float64),
        "desired_goal": np.array([3.0, 4.0], dtype=np.float64),
    }


def test_extracted_state_shape_is_107() -> None:
    extracted = extract_state_and_goal(_fake_antmaze_observation())
    assert extracted.state.shape == (STATE_DIM,)


def test_extracted_goal_shape_is_2() -> None:
    raw = _fake_antmaze_observation()
    extracted = extract_state_and_goal(raw)
    assert extracted.goal.shape == (2,)
    np.testing.assert_array_equal(extracted.goal, raw["desired_goal"])


def test_state_concatenates_xy_then_body() -> None:
    raw = _fake_antmaze_observation()
    extracted = extract_state_and_goal(raw)
    np.testing.assert_array_equal(extracted.state[:2], raw["achieved_goal"])
    np.testing.assert_array_equal(extracted.state[2:], raw["observation"])
