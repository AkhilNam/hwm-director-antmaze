"""Tests for Transition."""

from __future__ import annotations

import numpy as np
import pytest

from hwm_director.data.state import STATE_DIM
from hwm_director.data.transitions import ACTION_DIM, Transition
from tests.helpers import make_transition


def test_transition_fields_have_expected_shapes() -> None:
    transition = make_transition(
        episode_id=3,
        next_state=np.ones(STATE_DIM),
        goal=np.array([1.0, 2.0]),
    )
    transition.validate()
    assert transition.state.shape == (STATE_DIM,)
    assert transition.action.shape == (ACTION_DIM,)
    assert transition.next_state.shape == (STATE_DIM,)
    assert transition.goal.shape == (2,)
    assert transition.episode_id == 3
    assert transition.qpos is None
    assert transition.qvel is None


def test_qpos_qvel_are_copied() -> None:
    qpos = np.arange(15, dtype=np.float64)
    qvel = np.arange(14, dtype=np.float64)
    transition = make_transition(
        next_state=np.ones(STATE_DIM),
        goal=np.array([1.0, 2.0]),
        qpos=qpos,
        qvel=qvel,
    )
    qpos[0] = 999.0
    qvel[0] = 999.0
    assert transition.qpos[0] == 0.0
    assert transition.qvel[0] == 0.0
    transition.qpos[1] = -1.0
    assert qpos[1] == 1.0


def test_validate_rejects_wrong_shape() -> None:
    transition = Transition(
        state=np.zeros(10),
        action=np.zeros(ACTION_DIM, dtype=np.float32),
        next_state=np.ones(STATE_DIM),
        goal=np.array([1.0, 2.0]),
        episode_id=0,
    )
    with pytest.raises(ValueError, match="state"):
        transition.validate()
