"""Tests for transition statistics."""

from __future__ import annotations

import numpy as np

from hwm_director.data.statistics import compute_state_statistics, summarize_transitions
from hwm_director.data.transitions import Transition


def test_state_statistics_arrays_have_shape_107() -> None:
    states = np.zeros((5, 107))
    stats = compute_state_statistics(states)
    for key in ("mean", "std", "min", "max"):
        assert stats[key].shape == (107,)


def test_summarize_transitions_reports_counts_and_shapes() -> None:
    state = np.zeros(107)
    next_state = np.zeros(107)
    next_state[0] = 3.0
    next_state[1] = 4.0
    transitions = [
        Transition(
            state=state,
            action=np.zeros(8, dtype=np.float32),
            next_state=next_state,
            goal=np.zeros(2),
            episode_id=0,
        )
    ]
    summary = summarize_transitions(transitions)
    assert summary["n_transitions"] == 1
    assert summary["state_shape"] == (107,)
    assert summary["action_shape"] == (8,)
    assert summary["mean_abs_xy_delta"] == 5.0
