"""Director: shared ``pi_L`` / ``f_L`` plus manager ``pi_H``.

There is **no** independently trained high-level world model. The high-level
transition is the implicit composition

    f_H^Director(h_tau, g_tau) = (f_L, pi_L)^K (h_tau, g_tau)

i.e. ``K`` closed-loop model steps. At imagined step ``i``:

    a_i = pi_L(s_i, g_tau)
    s_{i+1} = f_L(s_i, a_i)

``pi_L`` is queried again after every predicted state. No environment
``step`` occurs inside ``high_level_transition``.

Normalization boundary
----------------------
Public methods take and return **raw** 29-D states and **raw** x/y.
Internally:

- ``pi_L`` sees worker-normalized state and subgoal (same as BC training).
- ``f_L`` sees dynamics-normalized state and raw actions in ``[-1, 1]``
  (same as one-step dynamics training).
- ``pi_H`` sees manager-normalized state and final goal; its output is
  denormalized to raw x/y and optionally clamped in meters.
"""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import ACHIEVED_GOAL_DIM, STATE_DIM
from hwm_director.data.worker_dataset import (
    DEFAULT_HORIZON_K,
    denormalize_subgoal,
    normalize_subgoal,
)
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.hierarchy import HierarchicalController
from hwm_director.models.worker import GoalConditionedWorker


def _as_batch(array: np.ndarray, dim: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(array, dtype=np.float64)
    if arr.shape == (dim,):
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[-1] == dim:
        return arr, False
    raise ValueError(f"expected shape ({dim},) or (N, {dim}), got {arr.shape}")


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32)


class Director:
    """``M_hier`` with implicit ``f_H = (f_L, pi_L)^K``.

    ``explicit_f_h`` is always ``None``: Director does not own a learned
    high-level dynamics network.
    """

    explicit_f_h = None

    def __init__(
        self,
        manager: DirectorManager,
        worker: GoalConditionedWorker,
        dynamics: LowLevelDynamicsModel,
        manager_normalizer: StateNormalizer,
        worker_normalizer: StateNormalizer,
        dynamics_normalizer: StateNormalizer,
        horizon_k: int = DEFAULT_HORIZON_K,
        high_level_policy: object | None = None,
    ) -> None:
        if horizon_k < 1:
            raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
        self.manager = manager
        self.worker = worker
        self.dynamics = dynamics
        self.manager_normalizer = manager_normalizer
        self.worker_normalizer = worker_normalizer
        self.dynamics_normalizer = dynamics_normalizer
        self.horizon_k = int(horizon_k)
        self.high_level_policy = high_level_policy
        self.n_worker_calls = 0
        self.n_dynamics_calls = 0
        self.manager.eval()
        self.worker.eval()
        self.dynamics.eval()

    def reset_call_counts(self) -> None:
        self.n_worker_calls = 0
        self.n_dynamics_calls = 0

    def select_high_level_command(
        self, state: np.ndarray, final_goal: np.ndarray
    ) -> np.ndarray:
        """Raw ``pi_H(s_t, g*) -> g_tau`` (meters).

        If ``high_level_policy`` is set (Director-Value), that callable is
        used. Otherwise this is Director-BC. ``f_H`` is never involved.
        """
        if self.high_level_policy is not None:
            return np.asarray(
                self.high_level_policy(state, final_goal), dtype=np.float64
            )
        return self._select_bc(state, final_goal)

    def _select_bc(self, state: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
        """Director-BC: MLP imitation of recorded ``s_{t+K}[:2]``."""
        state_b, squeezed = _as_batch(state, STATE_DIM)
        goal_b, _ = _as_batch(final_goal, ACHIEVED_GOAL_DIM)
        state_n = self.manager_normalizer.normalize(state_b)
        goal_n = normalize_subgoal(goal_b, self.manager_normalizer)
        with torch.no_grad():
            subgoal_n = self.manager(
                _to_tensor(state_n),
                _to_tensor(goal_n),
                clamp=False,
            )
        subgoal_raw = denormalize_subgoal(
            subgoal_n.cpu().numpy(), self.manager_normalizer
        )
        current_xy = torch.as_tensor(state_b[:, :2], dtype=torch.float32)
        target_xy = torch.as_tensor(subgoal_raw, dtype=torch.float32)
        from hwm_director.models.director_manager import clamp_xy_displacement

        clamped = clamp_xy_displacement(
            current_xy, target_xy, self.manager.max_subgoal_distance
        ).numpy()
        if squeezed:
            return np.asarray(clamped[0], dtype=np.float64)
        return np.asarray(clamped, dtype=np.float64)

    def low_level_action(
        self, state: np.ndarray, subgoal: np.ndarray
    ) -> np.ndarray:
        """Raw ``pi_L(s_t, g_tau) -> a_t`` in ``[-1, 1]``.

        State and subgoal are worker-normalized internally.
        """
        state_b, squeezed = _as_batch(state, STATE_DIM)
        subgoal_b, _ = _as_batch(subgoal, ACHIEVED_GOAL_DIM)
        state_n = self.worker_normalizer.normalize(state_b)
        subgoal_n = normalize_subgoal(subgoal_b, self.worker_normalizer)
        with torch.no_grad():
            action = self.worker(_to_tensor(state_n), _to_tensor(subgoal_n))
        self.n_worker_calls += int(state_b.shape[0])
        out = np.clip(action.cpu().numpy(), -1.0, 1.0).astype(np.float32)
        if squeezed:
            return out[0]
        return out

    def _dynamics_step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """One ``f_L`` step: raw state + raw action -> raw next state."""
        state_b, squeezed = _as_batch(state, STATE_DIM)
        action_b, _ = _as_batch(np.asarray(action, dtype=np.float64), int(self.dynamics.action_dim))
        state_n = self.dynamics_normalizer.normalize(state_b)
        with torch.no_grad():
            next_n = self.dynamics.predict_next_state(
                _to_tensor(state_n), _to_tensor(action_b)
            )
        self.n_dynamics_calls += int(state_b.shape[0])
        next_raw = self.dynamics_normalizer.denormalize(next_n.cpu().numpy())
        if squeezed:
            return np.asarray(next_raw[0], dtype=np.float64)
        return np.asarray(next_raw, dtype=np.float64)

    def high_level_transition(
        self, state: np.ndarray, subgoal: np.ndarray, horizon_k: int | None = None
    ) -> np.ndarray:
        """Implicit ``f_H``: ``K`` closed-loop ``(pi_L, f_L)`` model steps.

        Recomputes the worker action from the **new** predicted state every
        step. Returns a raw 29-D state. No env interaction.
        """
        k = self.horizon_k if horizon_k is None else int(horizon_k)
        if k < 1:
            raise ValueError(f"horizon_k must be >= 1, got {k}")
        predicted = np.asarray(state, dtype=np.float64).copy()
        for _ in range(k):
            action = self.low_level_action(predicted, subgoal)
            predicted = self._dynamics_step(predicted, action)
        return predicted


def assert_director_has_no_learned_f_h(controller: HierarchicalController) -> None:
    """Director must not expose an independently trained ``f_H`` network."""
    explicit = getattr(controller, "explicit_f_h", None)
    if explicit is not None:
        raise AssertionError("Director must not have an independently trained f_H")
