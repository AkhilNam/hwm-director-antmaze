"""Tests for Transition."""

from __future__ import annotations

import numpy as np
import pytest

from hwm_director.data.transitions import Transition


def test_transition_fields_have_expected_shapes() -> None:
    transition = Transition(
        state=np.zeros(107),
        action=np.zeros(8, dtype=np.float32),
        next_state=np.ones(107),
        goal=np.array([1.0, 2.0]),
        episode_id=3,
    )
    transition.validate()
    assert transition.state.shape == (107,)
    assert transition.action.shape == (8,)
    assert transition.next_state.shape == (107,)
    assert transition.goal.shape == (2,)
    assert transition.episode_id == 3


def test_validate_rejects_wrong_shape() -> None:
    transition = Transition(
        state=np.zeros(10),
        action=np.zeros(8, dtype=np.float32),
        next_state=np.ones(107),
        goal=np.array([1.0, 2.0]),
        episode_id=0,
    )
    with pytest.raises(ValueError, match="state"):
        transition.validate()
