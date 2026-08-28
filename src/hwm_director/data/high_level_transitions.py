"""Exactly-K high-level offline transitions, shared by Director and (later) HWM.

For a primitive step ``t`` in an episode of length ``L``, with horizon ``K``:

    h_tau      = s_t
    g_tau      = s_{t+K}[:2]
    h_next     = s_{t+K}
    final_goal = desired_goal

``s_{t+K}`` is ``traj[t + K - 1].next_state``. Examples with fewer than ``K``
remaining primitive steps are skipped. Transitions never cross episode
boundaries.

This table is **not** a high-level world model. It is the shared offline
dataset later used by:

- Director-BC (imitate ``g_tau``)
- Director-Value / HWM value training (score ``(h_tau, g_tau, g*)``)
- candidate retrieval (nearby ``h_tau`` x/y -> recorded ``g_tau``)
- explicit ``f_H_phi`` (supervised ``(h_tau, g_tau) -> h_next``)

Director and HWM must use the same builder. Their scientific difference is
only how ``f_H`` is realized, not this dataset.

``f_H_phi`` is trained on **recorded** K-step transitions
``(s_t, s_{t+K}[:2]) -> s_{t+K}``. That is not a rollout of the current
learned ``pi_L``.

Trajectory-aware value target
-----------------------------
An episode is successful only if some primitive ``next_state`` x/y is within
``success_threshold`` of ``g*``. Reaching the episode time limit is **not**
success.

Let ``t_success`` be the first primitive index ``i`` with

    ||traj[i].next_state[:2] - g*|| < success_threshold

For a high-level example starting at ``t``:

    remaining_high_level_steps = max(0, (t_success - t + K) // K)

i.e. the number of K-step hops until that first success (0 if ``t`` is
already after success). Then

    value_target = gamma ** remaining_high_level_steps

for successful episodes. For unsuccessful episodes every example gets
``unsuccessful_value`` (default ``0.0``). That is strictly below
``gamma ** n`` for every finite ``n``, so timeout is not treated as a
short successful path.
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

DEFAULT_VALUE_GAMMA = 0.99
DEFAULT_UNSUCCESSFUL_VALUE = 0.0
DEFAULT_SUCCESS_THRESHOLD = 0.5


class HighLevelTransition(NamedTuple):
    """One exactly-K high-level tuple plus trajectory-aware value target."""

    h_tau: np.ndarray
    g_tau: np.ndarray
    h_next: np.ndarray
    final_goal: np.ndarray
    episode_id: int
    t: int
    high_level_index: int
    episode_succeeded: bool
    remaining_high_level_steps: int
    value_target: float
    reached_goal_at_next: bool


def remaining_high_level_steps(
    t: int,
    t_success: int | None,
    horizon_k: int,
) -> int:
    """K-step hops from primitive time ``t`` until first success.

    ``t_success`` is the first primitive index whose ``next_state`` is at
    the goal, or ``None`` if the episode never succeeds.
    """
    if t_success is None:
        raise ValueError("remaining_high_level_steps requires a successful episode")
    if horizon_k < 1:
        raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
    return int(max(0, (int(t_success) - int(t) + int(horizon_k)) // int(horizon_k)))


def first_success_index(
    traj: Sequence[Transition],
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> int | None:
    """First primitive ``i`` with ``||next_xy - g*|| < threshold``, else None."""
    thresh = float(success_threshold)
    for i, step in enumerate(traj):
        nxt = np.asarray(step.next_state[:ACHIEVED_GOAL_DIM], dtype=np.float64)
        goal = np.asarray(step.goal, dtype=np.float64)
        if float(np.linalg.norm(nxt - goal)) < thresh:
            return int(i)
    return None


def value_target_from_remaining(
    remaining_steps: int,
    *,
    gamma: float,
    episode_succeeded: bool,
    unsuccessful_value: float = DEFAULT_UNSUCCESSFUL_VALUE,
) -> float:
    if not episode_succeeded:
        return float(unsuccessful_value)
    return float(gamma ** int(remaining_steps))


def high_level_transitions_from_episode(
    traj: Sequence[Transition],
    horizon_k: int,
    *,
    gamma: float = DEFAULT_VALUE_GAMMA,
    unsuccessful_value: float = DEFAULT_UNSUCCESSFUL_VALUE,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> list[HighLevelTransition]:
    """Exactly-K high-level examples from one episode. Never crosses the end."""
    if horizon_k < 1:
        raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
    traj = list(traj)
    n_steps = len(traj)
    n_t = n_steps - horizon_k + 1
    if n_t <= 0:
        return []
    t_success = first_success_index(traj, success_threshold=success_threshold)
    episode_succeeded = t_success is not None
    out: list[HighLevelTransition] = []
    for t in range(n_t):
        start = traj[t]
        nxt = traj[t + horizon_k - 1]
        h_tau = np.asarray(start.state, dtype=np.float64)
        h_next = np.asarray(nxt.next_state, dtype=np.float64)
        g_tau = h_next[:ACHIEVED_GOAL_DIM].copy()
        final_goal = np.asarray(start.goal, dtype=np.float64)
        if episode_succeeded:
            remaining = remaining_high_level_steps(t, t_success, horizon_k)
        else:
            remaining = -1
        reached = (
            float(np.linalg.norm(g_tau - final_goal)) < float(success_threshold)
        )
        out.append(
            HighLevelTransition(
                h_tau=h_tau,
                g_tau=g_tau,
                h_next=h_next,
                final_goal=final_goal,
                episode_id=int(start.episode_id),
                t=int(t),
                high_level_index=int(t),
                episode_succeeded=bool(episode_succeeded),
                remaining_high_level_steps=int(remaining),
                value_target=value_target_from_remaining(
                    max(remaining, 0),
                    gamma=gamma,
                    episode_succeeded=episode_succeeded,
                    unsuccessful_value=unsuccessful_value,
                ),
                reached_goal_at_next=bool(reached),
            )
        )
    return out


def build_high_level_transitions(
    transitions: Sequence[Transition],
    horizon_k: int = DEFAULT_HORIZON_K,
    *,
    gamma: float = DEFAULT_VALUE_GAMMA,
    unsuccessful_value: float = DEFAULT_UNSUCCESSFUL_VALUE,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
) -> list[HighLevelTransition]:
    """All exactly-K high-level tuples from a transition list."""
    out: list[HighLevelTransition] = []
    for traj in group_by_episode(transitions).values():
        out.extend(
            high_level_transitions_from_episode(
                traj,
                horizon_k,
                gamma=gamma,
                unsuccessful_value=unsuccessful_value,
                success_threshold=success_threshold,
            )
        )
    return out


def high_level_arrays(
    examples: Sequence[HighLevelTransition],
) -> dict[str, np.ndarray]:
    """Stack high-level examples into arrays (empty-safe)."""
    n = len(examples)
    if n == 0:
        return {
            "h_tau": np.zeros((0, STATE_DIM), dtype=np.float32),
            "g_tau": np.zeros((0, ACHIEVED_GOAL_DIM), dtype=np.float32),
            "h_next": np.zeros((0, STATE_DIM), dtype=np.float32),
            "final_goal": np.zeros((0, GOAL_DIM), dtype=np.float32),
            "episode_id": np.zeros((0,), dtype=np.int64),
            "t": np.zeros((0,), dtype=np.int64),
            "high_level_index": np.zeros((0,), dtype=np.int64),
            "episode_succeeded": np.zeros((0,), dtype=np.bool_),
            "remaining_high_level_steps": np.zeros((0,), dtype=np.int64),
            "value_target": np.zeros((0,), dtype=np.float32),
            "reached_goal_at_next": np.zeros((0,), dtype=np.bool_),
        }
    return {
        "h_tau": np.stack([e.h_tau for e in examples]).astype(np.float32),
        "g_tau": np.stack([e.g_tau for e in examples]).astype(np.float32),
        "h_next": np.stack([e.h_next for e in examples]).astype(np.float32),
        "final_goal": np.stack([e.final_goal for e in examples]).astype(np.float32),
        "episode_id": np.asarray([e.episode_id for e in examples], dtype=np.int64),
        "t": np.asarray([e.t for e in examples], dtype=np.int64),
        "high_level_index": np.asarray(
            [e.high_level_index for e in examples], dtype=np.int64
        ),
        "episode_succeeded": np.asarray(
            [e.episode_succeeded for e in examples], dtype=np.bool_
        ),
        "remaining_high_level_steps": np.asarray(
            [e.remaining_high_level_steps for e in examples], dtype=np.int64
        ),
        "value_target": np.asarray(
            [e.value_target for e in examples], dtype=np.float32
        ),
        "reached_goal_at_next": np.asarray(
            [e.reached_goal_at_next for e in examples], dtype=np.bool_
        ),
    }


class HighLevelValueDataset(Dataset):
    """Indexable ``(h_tau, g_tau, g*, value_target)`` for training ``Q_H``.

    When a ``normalizer`` is provided (train-fit only), state and both x/y
    fields are scaled with that normalizer. Value targets stay in raw
    ``gamma ** remaining`` / unsuccessful units.
    """

    def __init__(
        self,
        examples: Sequence[HighLevelTransition],
        normalizer: StateNormalizer | None = None,
    ) -> None:
        arrays = high_level_arrays(examples)
        self.raw_h_tau = arrays["h_tau"]
        self.raw_g_tau = arrays["g_tau"]
        self.raw_h_next = arrays["h_next"]
        self.raw_final_goals = arrays["final_goal"]
        self.episode_ids = arrays["episode_id"]
        self.times = arrays["t"]
        self.high_level_index = arrays["high_level_index"]
        self.episode_succeeded = arrays["episode_succeeded"]
        self.remaining_high_level_steps = arrays["remaining_high_level_steps"]
        self.value_targets = arrays["value_target"]
        self.reached_goal_at_next = arrays["reached_goal_at_next"]
        self.normalizer = normalizer
        h_tau = self.raw_h_tau
        g_tau = self.raw_g_tau
        goals = self.raw_final_goals
        if self.normalizer is not None and h_tau.shape[0] > 0:
            h_tau = self.normalizer.normalize(h_tau).astype(np.float32, copy=False)
            g_tau = normalize_subgoal(g_tau, self.normalizer).astype(
                np.float32, copy=False
            )
            goals = normalize_subgoal(goals, self.normalizer).astype(
                np.float32, copy=False
            )
        self.h_tau = torch.as_tensor(h_tau, dtype=torch.float32)
        self.g_tau = torch.as_tensor(g_tau, dtype=torch.float32)
        self.final_goals = torch.as_tensor(goals, dtype=torch.float32)
        self.values = torch.as_tensor(self.value_targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.h_tau.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.h_tau[index],
            "subgoal": self.g_tau[index],
            "final_goal": self.final_goals[index],
            "value": self.values[index],
        }


class HighLevelDynamicsDataset(Dataset):
    """Indexable ``(h_tau, g_tau, h_next)`` for training explicit ``f_H_phi``.

    Normalization (train-fit ``StateNormalizer`` only):

        h_tau_n  = normalize(h_tau)
        g_tau_n  = normalize_subgoal(g_tau)   # first two state dims
        h_next_n = normalize(h_next)
        delta_n  = h_next_n - h_tau_n

    The MLP is trained so ``h_tau_n + predicted_delta = h_next_n``.
    """

    def __init__(
        self,
        examples: Sequence[HighLevelTransition],
        normalizer: StateNormalizer,
    ) -> None:
        if normalizer.mean is None or normalizer.std is None:
            raise ValueError("normalizer must be fit before HighLevelDynamicsDataset")
        arrays = high_level_arrays(examples)
        self.raw_h_tau = arrays["h_tau"]
        self.raw_g_tau = arrays["g_tau"]
        self.raw_h_next = arrays["h_next"]
        self.episode_ids = arrays["episode_id"]
        self.times = arrays["t"]
        self.normalizer = normalizer
        if self.raw_h_tau.shape[0] == 0:
            h_n = np.zeros((0, STATE_DIM), dtype=np.float32)
            g_n = np.zeros((0, ACHIEVED_GOAL_DIM), dtype=np.float32)
            nxt_n = np.zeros((0, STATE_DIM), dtype=np.float32)
        else:
            h_n = np.asarray(normalizer.normalize(self.raw_h_tau), dtype=np.float32)
            nxt_n = np.asarray(normalizer.normalize(self.raw_h_next), dtype=np.float32)
            g_n = np.asarray(
                normalize_subgoal(self.raw_g_tau, normalizer), dtype=np.float32
            )
        self.h_tau = torch.as_tensor(h_n, dtype=torch.float32)
        self.g_tau = torch.as_tensor(g_n, dtype=torch.float32)
        self.h_next = torch.as_tensor(nxt_n, dtype=torch.float32)
        self.delta = self.h_next - self.h_tau

    def __len__(self) -> int:
        return int(self.h_tau.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "state": self.h_tau[index],
            "subgoal": self.g_tau[index],
            "next_state": self.h_next[index],
            "delta": self.delta[index],
        }
