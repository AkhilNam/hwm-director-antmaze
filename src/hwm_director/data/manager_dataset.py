"""Behavior-cloning examples for the Director manager ``pi_H``.

For a step ``t`` in an episode of length ``L``, one high-level example is

    input:   h_tau = s_t
             final_goal = transition.goal   (desired_goal, 2-D)
    target:  g_tau = s_{t+K}[:2]

``s_{t+K}`` is ``traj[t + K - 1].next_state``. The target **must** come from
the same episode. If fewer than ``K`` primitive steps remain (``t > L - K``),
the example is skipped. This is **exactly-K** BC, not the worker's
``k in 1..K`` scheme.

This is the simple first Director manager. The unified hierarchy later
allows training ``pi_H`` with offline RL / behavior-regularized actor-critic;
that objective is not implemented here.

Normalization convention (applied by ``DirectorManagerDataset``):
- state and final-goal x/y are **train-only** ``StateNormalizer``-scaled
- target subgoal x/y uses the same first two dimensions
- the MLP predicts a relative displacement in that normalized x/y space
- raw meters are recovered with ``denormalize_subgoal``
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import ACHIEVED_GOAL_DIM, GOAL_DIM, STATE_DIM
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K, normalize_subgoal


class ManagerExample(NamedTuple):
    """One BC sample for ``pi_H(s_t, g*) -> s_{t+K}[:2]``."""

    state: np.ndarray
    final_goal: np.ndarray
    target_subgoal: np.ndarray
    episode_id: int
    t: int


def _empty_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, STATE_DIM), dtype=np.float32),
        np.zeros((0, GOAL_DIM), dtype=np.float32),
        np.zeros((0, ACHIEVED_GOAL_DIM), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
    )


def manager_arrays_from_transitions(
    transitions: Sequence[Transition],
    horizon_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exactly-K same-episode manager pairs as arrays.

    For episode length ``L`` and horizon ``K``, times ``t = 0, ..., L-K``.
    """
    if horizon_k < 1:
        raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
    state_chunks: list[np.ndarray] = []
    goal_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    episode_chunks: list[np.ndarray] = []
    t_chunks: list[np.ndarray] = []
    for episode_id, traj in group_by_episode(transitions).items():
        n_steps = len(traj)
        n_t = n_steps - horizon_k + 1
        if n_t <= 0:
            continue
        states = np.stack([np.asarray(step.state, dtype=np.float32) for step in traj])
        goals = np.stack([np.asarray(step.goal, dtype=np.float32) for step in traj])
        next_xy = np.stack(
            [
                np.asarray(step.next_state[:ACHIEVED_GOAL_DIM], dtype=np.float32)
                for step in traj
            ]
        )
        t_idx = np.arange(n_t, dtype=np.int64)
        state_chunks.append(states[:n_t])
        goal_chunks.append(goals[:n_t])
        target_chunks.append(next_xy[horizon_k - 1 : horizon_k - 1 + n_t])
        episode_chunks.append(np.full(n_t, int(episode_id), dtype=np.int64))
        t_chunks.append(t_idx)
    if not state_chunks:
        return _empty_arrays()
    return (
        np.concatenate(state_chunks, axis=0),
        np.concatenate(goal_chunks, axis=0),
        np.concatenate(target_chunks, axis=0),
        np.concatenate(episode_chunks, axis=0),
        np.concatenate(t_chunks, axis=0),
    )


class DirectorManagerDataset(Dataset):
    """Indexable ``(state, final_goal, target_subgoal)`` tensors for ``pi_H``.

    When a ``normalizer`` is provided, all three x/y-bearing fields are
    scaled with **that** (train-fit) normalizer. The network then predicts
    relative displacement in normalized coordinates.
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
        states, goals, targets, episode_ids, times = manager_arrays_from_transitions(
            list(transitions), self.horizon_k
        )
        self.raw_states = states
        self.raw_final_goals = goals
        self.raw_target_subgoals = targets
        self.episode_ids = episode_ids
        self.times = times
        if self.normalizer is not None and states.shape[0] > 0:
            states = self.normalizer.normalize(states).astype(np.float32, copy=False)
            goals = normalize_subgoal(goals, self.normalizer).astype(
                np.float32, copy=False
            )
            targets = normalize_subgoal(targets, self.normalizer).astype(
                np.float32, copy=False
            )
        self.states = torch.as_tensor(states, dtype=torch.float32)
        self.final_goals = torch.as_tensor(goals, dtype=torch.float32)
        self.target_subgoals = torch.as_tensor(targets, dtype=torch.float32)
        self._examples: list[ManagerExample] | None = None

    @property
    def examples(self) -> list[ManagerExample]:
        if self._examples is None:
            self._examples = [
                ManagerExample(
                    state=self.raw_states[i],
                    final_goal=self.raw_final_goals[i],
                    target_subgoal=self.raw_target_subgoals[i],
                    episode_id=int(self.episode_ids[i]),
                    t=int(self.times[i]),
                )
                for i in range(len(self))
            ]
        return self._examples

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.states[index],
            "final_goal": self.final_goals[index],
            "target_subgoal": self.target_subgoals[index],
        }
