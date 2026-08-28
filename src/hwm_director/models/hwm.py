"""HWM: shared ``E``, ``pi_L``, ``f_L`` plus explicit ``f_H_phi``.

    f_H^HWM(h_tau, g_tau) = f_H_phi(h_tau, g_tau)

This is **not** Director. Director's high-level transition remains

    f_H^Director = (f_L, pi_L)^K

HWM reuses the same worker and one-step dynamics **instances**. It does
not own a second ``pi_L`` or ``f_L``. Real environment execution still
calls ``pi_L``; ``f_H_phi`` is used only for high-level prediction /
candidate reachability scoring.

Public methods take and return **raw** 29-D states and **raw** x/y.
Internally ``f_H_phi`` sees the train-only high-level-dynamics normalizer.
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
from hwm_director.models.director import Director
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_high import ExplicitHighLevelDynamics
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.encoder import IdentityEncoder
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


class HierarchicalWorldModel:
    """``M_hier`` with explicit learned ``f_H_phi``.

    ``explicit_f_h`` is the trained coarse model. ``pi_L`` / ``f_L`` are
    the shared low-level objects (typically the same instances as Director).
    """

    def __init__(
        self,
        manager: DirectorManager,
        worker: GoalConditionedWorker,
        dynamics: LowLevelDynamicsModel,
        explicit_f_h: ExplicitHighLevelDynamics,
        manager_normalizer: StateNormalizer,
        worker_normalizer: StateNormalizer,
        dynamics_normalizer: StateNormalizer,
        fh_normalizer: StateNormalizer,
        horizon_k: int = DEFAULT_HORIZON_K,
        high_level_policy: object | None = None,
        encoder: IdentityEncoder | None = None,
    ) -> None:
        if horizon_k < 1:
            raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
        if explicit_f_h is None:
            raise ValueError("HWM requires an explicit f_H_phi")
        self.manager = manager
        self.worker = worker
        self.dynamics = dynamics
        self.explicit_f_h = explicit_f_h
        self.manager_normalizer = manager_normalizer
        self.worker_normalizer = worker_normalizer
        self.dynamics_normalizer = dynamics_normalizer
        self.fh_normalizer = fh_normalizer
        self.horizon_k = int(horizon_k)
        self.high_level_policy = high_level_policy
        self.encoder = encoder if encoder is not None else IdentityEncoder()
        self.n_worker_calls = 0
        self.n_dynamics_calls = 0
        self.n_explicit_fh_calls = 0
        self.manager.eval()
        self.worker.eval()
        self.dynamics.eval()
        self.explicit_f_h.eval()

    @classmethod
    def from_director(
        cls,
        director: Director,
        explicit_f_h: ExplicitHighLevelDynamics,
        fh_normalizer: StateNormalizer,
        high_level_policy: object | None = None,
    ) -> HierarchicalWorldModel:
        """Share Director's ``pi_L``, ``f_L``, manager, and normalizers."""
        return cls(
            manager=director.manager,
            worker=director.worker,
            dynamics=director.dynamics,
            explicit_f_h=explicit_f_h,
            manager_normalizer=director.manager_normalizer,
            worker_normalizer=director.worker_normalizer,
            dynamics_normalizer=director.dynamics_normalizer,
            fh_normalizer=fh_normalizer,
            horizon_k=director.horizon_k,
            high_level_policy=high_level_policy,
        )

    def reset_call_counts(self) -> None:
        self.n_worker_calls = 0
        self.n_dynamics_calls = 0
        self.n_explicit_fh_calls = 0

    def select_high_level_command(
        self, state: np.ndarray, final_goal: np.ndarray
    ) -> np.ndarray:
        if self.high_level_policy is not None:
            return np.asarray(
                self.high_level_policy(state, final_goal), dtype=np.float64
            )
        return self._select_bc(state, final_goal)

    def _select_bc(self, state: np.ndarray, final_goal: np.ndarray) -> np.ndarray:
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
        """Shared ``pi_L``. Same code path as Director."""
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

    def high_level_transition(
        self, state: np.ndarray, subgoal: np.ndarray, horizon_k: int | None = None
    ) -> np.ndarray:
        """Explicit ``f_H_phi``: one coarse step, no ``pi_L`` / ``f_L``.

        ``horizon_k`` must match the trained ``K`` (or be omitted). The
        network is not a variable-horizon model.
        """
        if horizon_k is not None and int(horizon_k) != self.horizon_k:
            raise ValueError(
                f"HWM f_H_phi is trained for K={self.horizon_k}, got {horizon_k}"
            )
        state_b, squeezed = _as_batch(state, STATE_DIM)
        subgoal_b, _ = _as_batch(subgoal, ACHIEVED_GOAL_DIM)
        state_n = self.fh_normalizer.normalize(state_b)
        subgoal_n = normalize_subgoal(subgoal_b, self.fh_normalizer)
        with torch.no_grad():
            next_n = self.explicit_f_h.predict_next_state(
                _to_tensor(state_n), _to_tensor(subgoal_n)
            )
        self.n_explicit_fh_calls += int(state_b.shape[0])
        next_raw = self.fh_normalizer.denormalize(next_n.cpu().numpy())
        if squeezed:
            return np.asarray(next_raw[0], dtype=np.float64)
        return np.asarray(next_raw, dtype=np.float64)


def assert_hwm_has_explicit_f_h(controller: object) -> None:
    explicit = getattr(controller, "explicit_f_h", None)
    if explicit is None:
        raise AssertionError("HWM must have an independently trained f_H")
