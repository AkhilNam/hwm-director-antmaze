"""Behavior-cloning examples for the goal-conditioned worker ``pi_L``.

For a step ``t`` in an episode, the worker should imitate ``action_t`` while
conditioned on a **local** subgoal: the x/y of a later state in the **same**
episode,

    g_tau = s_{t+k}[:2]    k in {1, ..., K}

``K`` is the low-level horizon and later matches the high-level interval.

Never take ``t+k`` from another episode. If fewer than ``k`` steps remain,
skip that ``(t, k)`` pair — do not wrap into the next id.

Along a trajectory of ``L`` transitions the positions are

    s_0 = traj[0].state
    s_{i+1} = traj[i].next_state   (same as traj[i+1].state when i+1 < L)

so ``k=1`` uses ``traj[t].next_state[:2]`` and ``k > 1`` uses
``traj[t + k - 1].next_state[:2]``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import ACHIEVED_GOAL_DIM, STATE_DIM
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import ACTION_DIM, Transition

DEFAULT_HORIZON_K = 10


class WorkerExample(NamedTuple):
    """One BC sample for ``pi_L(s_t, g_tau) -> a_t``."""

    state: np.ndarray  # (107,)
    subgoal: np.ndarray  # (2,)
    action: np.ndarray  # (8,)
    offset: int  # k >= 1


def normalize_subgoal(xy: np.ndarray, normalizer: StateNormalizer) -> np.ndarray:
    """Normalize x/y with the **state** normalizer's first two dimensions.

    Parameters
    ----------
    xy:
        ``(2,)`` or ``(N, 2)``.
    normalizer:
        Already ``fit`` on **training** states of shape ``(N, 107)``.
    """
    if normalizer.mean is None or normalizer.std is None:
        raise ValueError("fit() must be called before normalize_subgoal()")
    mean_xy = normalizer.mean[:ACHIEVED_GOAL_DIM]
    std_xy = normalizer.std[:ACHIEVED_GOAL_DIM]
    return (xy - mean_xy) / std_xy


def denormalize_subgoal(xy: np.ndarray, normalizer: StateNormalizer) -> np.ndarray:
    """Inverse of ``normalize_subgoal`` (raw maze x/y)."""
    if normalizer.mean is None or normalizer.std is None:
        raise ValueError("fit() must be called before denormalize_subgoal()")
    mean_xy = normalizer.mean[:ACHIEVED_GOAL_DIM]
    std_xy = normalizer.std[:ACHIEVED_GOAL_DIM]
    return xy * std_xy + mean_xy


def _future_xy(traj: Sequence[Transition], t: int, k: int) -> np.ndarray:
    """x/y of ``s_{t+k}`` from ``traj[t + k - 1].next_state``."""
    return np.asarray(traj[t + k - 1].next_state[:ACHIEVED_GOAL_DIM])


class WorkerDataset(Dataset):
    """Indexable ``(state, subgoal, action)`` tensors for ``pi_L``.

    Each item is a dict of float32 tensors:

    - ``state``: ``(107,)``
    - ``subgoal``: ``(2,)``
    - ``action``: ``(8,)``  (not normalized; already in [-1, 1])
    """

    def __init__(
        self,
        transitions: Sequence[Transition],
        horizon_k: int = DEFAULT_HORIZON_K,
        normalizer: StateNormalizer | None = None,
    ) -> None:
        if horizon_k < 1:
            raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
        self.horizon_k = int(horizon_k)
        self.normalizer = normalizer
        self.examples = self._build_examples(list(transitions))
        if not self.examples:
            self.states = torch.zeros(0, STATE_DIM, dtype=torch.float32)
            self.subgoals = torch.zeros(0, ACHIEVED_GOAL_DIM, dtype=torch.float32)
            self.actions = torch.zeros(0, ACTION_DIM, dtype=torch.float32)
            return
        self.states = torch.as_tensor(
            np.stack([e.state for e in self.examples]), dtype=torch.float32
        )
        self.subgoals = torch.as_tensor(
            np.stack([e.subgoal for e in self.examples]), dtype=torch.float32
        )
        self.actions = torch.as_tensor(
            np.stack([e.action for e in self.examples]), dtype=torch.float32
        )

    def _build_examples(self, transitions: list[Transition]) -> list[WorkerExample]:
        """Build BC pairs from future x/y in the same episode."""
        examples: list[WorkerExample] = []
        for traj in group_by_episode(transitions).values():
            n_steps = len(traj)
            for t in range(n_steps):
                max_k = n_steps - t
                last_k = min(self.horizon_k, max_k)
                for k in range(1, last_k + 1):
                    state = np.asarray(traj[t].state)
                    subgoal = _future_xy(traj, t, k)
                    action = np.asarray(traj[t].action)
                    if self.normalizer is not None:
                        state = self.normalizer.normalize(state)
                        subgoal = normalize_subgoal(subgoal, self.normalizer)
                    examples.append(
                        WorkerExample(
                            state=state,
                            subgoal=subgoal,
                            action=action,
                            offset=k,
                        )
                    )
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one BC example as CPU float32 tensors."""
        return {
            "state": self.states[index],
            "subgoal": self.subgoals[index],
            "action": self.actions[index],
        }
