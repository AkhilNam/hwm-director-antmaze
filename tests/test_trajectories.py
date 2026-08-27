"""Tests for grouping transitions by episode."""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import STATE_DIM
from hwm_director.data.trajectories import group_by_episode
from tests.helpers import make_transition


def _step(episode_id: int, mark: float):
    state = np.zeros(STATE_DIM)
    state[0] = mark
    return make_transition(episode_id=episode_id, state=state)


def test_group_by_episode_membership_and_order() -> None:
    transitions = [
        _step(1, 0.0),
        _step(1, 1.0),
        _step(2, 10.0),
        _step(1, 2.0),
        _step(2, 11.0),
    ]
    grouped = group_by_episode(transitions)
    assert set(grouped) == {1, 2}
    assert [t.state[0] for t in grouped[1]] == [0.0, 1.0, 2.0]
    assert [t.state[0] for t in grouped[2]] == [10.0, 11.0]
    ids = [id(t) for ts in grouped.values() for t in ts]
    assert len(ids) == len(set(ids)) == len(transitions)
