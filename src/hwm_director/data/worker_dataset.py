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

    state: np.ndarray  # (STATE_DIM,)
    subgoal: np.ndarray  # (ACHIEVED_GOAL_DIM,)
    action: np.ndarray  # (ACTION_DIM,)
    offset: int  # k >= 1


def normalize_subgoal(xy: np.ndarray, normalizer: StateNormalizer) -> np.ndarray:
    """Normalize x/y with the **state** normalizer's first two dimensions.

    Parameters
    ----------
    xy:
        ``(2,)`` or ``(N, 2)``.
    normalizer:
        Already ``fit`` on **training** states of shape ``(N, STATE_DIM)``.
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


def _empty_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, STATE_DIM), dtype=np.float32),
        np.zeros((0, ACHIEVED_GOAL_DIM), dtype=np.float32),
        np.zeros((0, ACTION_DIM), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
    )


def _bc_arrays_from_transitions(
    transitions: Sequence[Transition],
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized same-episode ``(state, subgoal, action, k)`` arrays.

    For each episode of length ``L`` and each ``k`` in ``1..min(K, L)``, take
    every ``t`` with ``t + k <= L`` (i.e. ``t = 0 .. L-k``).
    """
    state_chunks: list[np.ndarray] = []
    subgoal_chunks: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    offset_chunks: list[np.ndarray] = []
    for traj in group_by_episode(transitions).values():
        n_steps = len(traj)
        if n_steps == 0:
            continue
        states = np.stack([np.asarray(t.state, dtype=np.float32) for t in traj], axis=0)
        next_xy = np.stack(
            [
                np.asarray(t.next_state[:ACHIEVED_GOAL_DIM], dtype=np.float32)
                for t in traj
            ],
            axis=0,
        )
        actions = np.stack(
            [np.asarray(t.action, dtype=np.float32) for t in traj], axis=0
        )
        last_k = min(horizon_k, n_steps)
        for k in range(1, last_k + 1):
            n_t = n_steps - k + 1
            state_chunks.append(states[:n_t])
            subgoal_chunks.append(next_xy[k - 1 : k - 1 + n_t])
            action_chunks.append(actions[:n_t])
            offset_chunks.append(np.full(n_t, k, dtype=np.int32))
    if not state_chunks:
        return _empty_arrays()
    return (
        np.concatenate(state_chunks, axis=0),
        np.concatenate(subgoal_chunks, axis=0),
        np.concatenate(action_chunks, axis=0),
        np.concatenate(offset_chunks, axis=0),
    )


class WorkerDataset(Dataset):
    """Indexable ``(state, subgoal, action)`` tensors for ``pi_L``.

    Each item is a dict of float32 tensors:

    - ``state``: ``(STATE_DIM,)``
    - ``subgoal``: ``(ACHIEVED_GOAL_DIM,)``
    - ``action``: ``(ACTION_DIM,)``  (not normalized; already in [-1, 1])

    Arrays are built with a vectorized per-episode slice (not one Python
    object per ``(t, k)`` pair). ``.examples`` is a lazy view for tests.
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
        states, subgoals, actions, offsets = _bc_arrays_from_transitions(
            list(transitions), self.horizon_k
        )
        if self.normalizer is not None and states.shape[0] > 0:
            states = self.normalizer.normalize(states).astype(np.float32, copy=False)
            subgoals = normalize_subgoal(subgoals, self.normalizer).astype(
                np.float32, copy=False
            )
        self.offsets = offsets
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.subgoals = torch.as_tensor(subgoals, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self._examples: list[WorkerExample] | None = None

    @property
    def examples(self) -> list[WorkerExample]:
        """Materialize ``WorkerExample`` rows (for tests; avoid on 1e6-scale data)."""
        if self._examples is None:
            self._examples = [
                WorkerExample(
                    state=self.states[i].numpy(),
                    subgoal=self.subgoals[i].numpy(),
                    action=self.actions[i].numpy(),
                    offset=int(self.offsets[i]),
                )
                for i in range(len(self))
            ]
        return self._examples

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one BC example as CPU float32 tensors."""
        return {
            "state": self.states[index],
            "subgoal": self.subgoals[index],
            "action": self.actions[index],
        }
