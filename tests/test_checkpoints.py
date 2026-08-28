"""Checkpoint save/load reproduces predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.checkpoints import (
    load_dynamics_checkpoint,
    load_manager_checkpoint,
    load_worker_checkpoint,
    save_dynamics_checkpoint,
    save_manager_checkpoint,
    save_worker_checkpoint,
)


def _normalizer() -> StateNormalizer:
    rng = np.random.default_rng(0)
    return StateNormalizer().fit(rng.normal(size=(16, STATE_DIM)))


def test_dynamics_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = LowLevelDynamicsModel(hidden_dims=(8, 8))
    normalizer = _normalizer()
    path = tmp_path / "f_l.pt"
    state = torch.randn(3, STATE_DIM)
    action = torch.randn(3, 8)
    with torch.no_grad():
        before = model.predict_next_state(state, action)
    save_dynamics_checkpoint(path, model, normalizer)
    loaded, loaded_norm = load_dynamics_checkpoint(path)
    with torch.no_grad():
        after = loaded.predict_next_state(state, action)
    torch.testing.assert_close(after, before)
    np.testing.assert_allclose(loaded_norm.mean, normalizer.mean)
    np.testing.assert_allclose(loaded_norm.std, normalizer.std)


def test_worker_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = GoalConditionedWorker(hidden_dims=(8,))
    normalizer = _normalizer()
    path = tmp_path / "pi_l.pt"
    state = torch.randn(5, STATE_DIM)
    subgoal = torch.randn(5, 2)
    with torch.no_grad():
        before = model(state, subgoal)
    save_worker_checkpoint(path, model, normalizer)
    loaded, _ = load_worker_checkpoint(path)
    with torch.no_grad():
        after = loaded(state, subgoal)
    torch.testing.assert_close(after, before)


def test_manager_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(2)
    model = DirectorManager(hidden_dims=(8,), max_subgoal_distance=1.5)
    normalizer = _normalizer()
    path = tmp_path / "pi_h.pt"
    state = torch.randn(2, STATE_DIM)
    goal = torch.randn(2, 2)
    with torch.no_grad():
        before = model(state, goal, clamp=False)
    save_manager_checkpoint(path, model, normalizer)
    loaded, _ = load_manager_checkpoint(path)
    with torch.no_grad():
        after = loaded(state, goal, clamp=False)
    torch.testing.assert_close(after, before)
    assert loaded.max_subgoal_distance == 1.5
