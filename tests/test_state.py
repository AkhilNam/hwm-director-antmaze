"""Tests for AntMaze state extraction."""

from __future__ import annotations

import numpy as np
import pytest

from hwm_director.data.state import (
    ACHIEVED_GOAL_DIM,
    GOAL_DIM,
    OBSERVATION_DIM,
    QPOS_DIM,
    QVEL_DIM,
    STATE_DIM,
    ant_v4_qpos_qvel_from_state,
    extract_state_and_goal,
)
from tests.helpers import fake_antmaze_observation


def test_state_dim_is_29() -> None:
    assert ACHIEVED_GOAL_DIM == 2
    assert OBSERVATION_DIM == 27
    assert STATE_DIM == 29
    assert GOAL_DIM == 2
    assert STATE_DIM == ACHIEVED_GOAL_DIM + OBSERVATION_DIM


def test_extracted_state_shape_is_state_dim() -> None:
    extracted = extract_state_and_goal(fake_antmaze_observation())
    assert extracted.state.shape == (STATE_DIM,)


def test_extracted_goal_shape_is_2() -> None:
    raw = fake_antmaze_observation()
    extracted = extract_state_and_goal(raw)
    assert extracted.goal.shape == (GOAL_DIM,)
    np.testing.assert_array_equal(extracted.goal, raw["desired_goal"])


def test_state_concatenates_xy_then_body() -> None:
    raw = fake_antmaze_observation()
    extracted = extract_state_and_goal(raw)
    np.testing.assert_array_equal(extracted.state[:2], raw["achieved_goal"])
    np.testing.assert_array_equal(extracted.state[2:], raw["observation"])


def test_ant_v4_qpos_qvel_mapping() -> None:
    raw = fake_antmaze_observation()
    extracted = extract_state_and_goal(raw)
    qpos, qvel = ant_v4_qpos_qvel_from_state(extracted.state)
    assert qpos.shape == (QPOS_DIM,)
    assert qvel.shape == (QVEL_DIM,)
    np.testing.assert_array_equal(qpos[:2], raw["achieved_goal"])
    np.testing.assert_array_equal(qpos[2:], raw["observation"][:13])
    np.testing.assert_array_equal(qvel, raw["observation"][13:])


def test_extract_rejects_v5_observation_dim() -> None:
    raw = fake_antmaze_observation()
    raw["observation"] = np.zeros(105)
    with pytest.raises(ValueError, match="observation"):
        extract_state_and_goal(raw)
