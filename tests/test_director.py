"""Tests for Director implicit f_H, protocol, and env-free rollout counts."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.models.director import Director, assert_director_has_no_learned_f_h
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.hierarchy import HierarchicalController
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.data.worker_dataset import normalize_subgoal


def _fitted_normalizer() -> StateNormalizer:
    rng = np.random.default_rng(0)
    states = rng.normal(loc=1.0, scale=2.0, size=(32, STATE_DIM))
    return StateNormalizer().fit(states)


def _director(horizon_k: int = 10, hidden: int = 8) -> Director:
    normalizer = _fitted_normalizer()
    return Director(
        manager=DirectorManager(hidden_dims=(hidden,), max_subgoal_distance=2.0),
        worker=GoalConditionedWorker(hidden_dims=(hidden,)),
        dynamics=LowLevelDynamicsModel(hidden_dims=(hidden,)),
        manager_normalizer=normalizer,
        worker_normalizer=normalizer,
        dynamics_normalizer=normalizer,
        horizon_k=horizon_k,
    )


def test_director_is_hierarchical_controller() -> None:
    director = _director(horizon_k=3)
    assert isinstance(director, HierarchicalController)


def test_director_has_no_independent_f_h() -> None:
    director = _director(horizon_k=3)
    assert director.explicit_f_h is None
    assert_director_has_no_learned_f_h(director)


def test_implicit_rollout_does_exactly_k_dynamics_and_worker_calls() -> None:
    k = 7
    director = _director(horizon_k=k)
    state = np.zeros(STATE_DIM)
    subgoal = np.array([1.0, 0.5], dtype=np.float64)
    director.reset_call_counts()
    out = director.high_level_transition(state, subgoal)
    assert out.shape == (STATE_DIM,)
    assert director.n_dynamics_calls == k
    assert director.n_worker_calls == k


def test_worker_is_called_on_updated_predicted_state() -> None:
    k = 4
    director = _director(horizon_k=k)
    seen: list[torch.Tensor] = []
    original_forward = director.worker.forward

    def wrapped(state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        seen.append(state.detach().clone())
        return original_forward(state, subgoal)

    director.worker.forward = wrapped  # type: ignore[method-assign]
    director.high_level_transition(np.zeros(STATE_DIM), np.ones(2))
    assert len(seen) == k
    # After the first f_L step the worker input should change.
    assert not torch.allclose(seen[0], seen[-1])


def test_raw_normalized_boundaries_match_manual_worker_call() -> None:
    director = _director(horizon_k=1)
    rng = np.random.default_rng(1)
    state = rng.normal(size=STATE_DIM)
    subgoal = rng.normal(size=2)
    action = director.low_level_action(state, subgoal)
    state_n = director.worker_normalizer.normalize(state)
    subgoal_n = normalize_subgoal(subgoal, director.worker_normalizer)
    with torch.no_grad():
        expected = director.worker(
            torch.as_tensor(state_n, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(subgoal_n, dtype=torch.float32).unsqueeze(0),
        ).squeeze(0).numpy()
    np.testing.assert_allclose(action, np.clip(expected, -1.0, 1.0), atol=1e-6)
    # Raw state is not already normalized, so skipping normalize would differ.
    with torch.no_grad():
        wrong = director.worker(
            torch.as_tensor(state, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(subgoal, dtype=torch.float32).unsqueeze(0),
        ).squeeze(0).numpy()
    assert np.linalg.norm(action - np.clip(wrong, -1.0, 1.0)) > 1e-4
