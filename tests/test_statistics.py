"""Tests for transition statistics."""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import STATE_DIM
from hwm_director.data.statistics import compute_state_statistics, summarize_transitions
from hwm_director.data.transitions import ACTION_DIM
from tests.helpers import make_transition


def test_state_statistics_arrays_have_state_dim() -> None:
    states = np.zeros((5, STATE_DIM))
    stats = compute_state_statistics(states)
    for key in ("mean", "std", "min", "max"):
        assert stats[key].shape == (STATE_DIM,)


def test_summarize_transitions_reports_counts_and_shapes() -> None:
    next_state = np.zeros(STATE_DIM)
    next_state[0] = 3.0
    next_state[1] = 4.0
    transitions = [make_transition(next_state=next_state)]
    summary = summarize_transitions(transitions)
    assert summary["n_transitions"] == 1
    assert summary["state_shape"] == (STATE_DIM,)
    assert summary["action_shape"] == (ACTION_DIM,)
    assert summary["mean_abs_xy_delta"] == 5.0
