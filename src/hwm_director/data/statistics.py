"""Summaries over collected states.

``state`` layout:

    index 0: x (from achieved_goal)
    index 1: y (from achieved_goal)
    index 2: start of the 105-D Ant observation
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hwm_director.data.state import STATE_DIM
from hwm_director.data.transitions import ACTION_DIM, Transition


def compute_state_statistics(states: np.ndarray) -> dict[str, np.ndarray]:
    """Per-dimension mean, std, min, and max of stacked states.

    Parameters
    ----------
    states:
        Array of shape ``(N, 107)``.

    Returns
    -------
    dict
        Keys ``mean``, ``std``, ``min``, ``max``. Each value has shape ``(107,)``.
    """
    states = np.asarray(states)
    return {
        "mean": np.mean(states, axis=0),
        "std": np.std(states, axis=0),
        "min": np.min(states, axis=0),
        "max": np.max(states, axis=0),
    }


def summarize_transitions(transitions: Sequence[Transition]) -> dict:
    """Collection summary: count, shapes, and mean x/y displacement.

    ``mean_abs_xy_delta`` is the mean Euclidean distance between
    ``state[:2]`` and ``next_state[:2]``.
    """
    n_transitions = len(transitions)
    if n_transitions == 0:
        return {
            "n_transitions": 0,
            "state_shape": (STATE_DIM,),
            "action_shape": (ACTION_DIM,),
            "mean_abs_xy_delta": float("nan"),
        }

    xy_deltas = np.asarray(
        [np.linalg.norm(t.next_state[:2] - t.state[:2]) for t in transitions],
        dtype=np.float64,
    )
    first = transitions[0]
    return {
        "n_transitions": n_transitions,
        "state_shape": tuple(np.asarray(first.state).shape),
        "action_shape": tuple(np.asarray(first.action).shape),
        "mean_abs_xy_delta": float(np.mean(xy_deltas)),
    }
