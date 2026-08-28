"""High-level transitions, Q_H scorer, candidate retrieval, value pi_H."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.high_level_transitions import (
    DEFAULT_UNSUCCESSFUL_VALUE,
    HighLevelValueDataset,
    build_high_level_transitions,
    first_success_index,
    remaining_high_level_steps,
    value_target_from_remaining,
)
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.data.subgoal_candidates import SubgoalCandidateIndex
from hwm_director.models.director import Director, assert_director_has_no_learned_f_h
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.high_level_value import VALUE_INPUT_DIM, HighLevelValueModel
from hwm_director.models.value_manager import ValueHighLevelPolicy
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.checkpoints import (
    load_high_level_value_checkpoint,
    save_high_level_value_checkpoint,
)
from hwm_director.training.train_high_level_value import train_high_level_value
from tests.helpers import make_transition


def _episode_along_x(
    episode_id: int,
    length: int,
    *,
    goal: np.ndarray,
    success_at: int | None = None,
) -> list:
    steps = []
    for t in range(length):
        state = np.zeros(STATE_DIM)
        nxt = np.zeros(STATE_DIM)
        state[0] = float(t)
        nxt[0] = float(t + 1)
        if success_at is not None and t == success_at:
            nxt[:2] = np.asarray(goal, dtype=np.float64)
        steps.append(
            make_transition(
                episode_id=episode_id,
                state=state,
                next_state=nxt,
                goal=np.asarray(goal, dtype=np.float64),
            )
        )
    return steps


def test_high_level_transition_never_crosses_episode_boundary() -> None:
    k = 4
    a = _episode_along_x(0, 12, goal=np.array([100.0, 0.0]))
    b = _episode_along_x(1, 12, goal=np.array([100.0, 0.0]))
    for step in a:
        step.state[1] = 0.0
        step.next_state[1] = 0.0
    for step in b:
        step.state[1] = 7.0
        step.next_state[1] = 7.0
    examples = build_high_level_transitions(a + b, horizon_k=k)
    assert examples
    for ex in examples:
        assert ex.h_tau[1] == ex.h_next[1]
        assert ex.t + k <= 12
        np.testing.assert_allclose(ex.g_tau, ex.h_next[:2])
        np.testing.assert_allclose(ex.h_next[0], ex.h_tau[0] + k)


def test_successful_remaining_steps_target() -> None:
    k = 10
    goal = np.array([50.0, 0.0])
    traj = _episode_along_x(0, 60, goal=goal, success_at=49)
    assert first_success_index(traj, success_threshold=0.5) == 49
    assert remaining_high_level_steps(0, 49, k) == 5
    examples = build_high_level_transitions(
        traj, horizon_k=k, gamma=0.99, success_threshold=0.5
    )
    by_t = {e.t: e for e in examples}
    assert by_t[0].episode_succeeded
    assert by_t[0].remaining_high_level_steps == 5
    np.testing.assert_allclose(by_t[0].value_target, 0.99 ** 5)
    assert by_t[40].remaining_high_level_steps == 1
    np.testing.assert_allclose(by_t[40].value_target, 0.99 ** 1)


def test_unsuccessful_episodes_get_documented_low_target() -> None:
    traj = _episode_along_x(0, 20, goal=np.array([100.0, 0.0]), success_at=None)
    examples = build_high_level_transitions(
        traj,
        horizon_k=10,
        gamma=0.99,
        unsuccessful_value=DEFAULT_UNSUCCESSFUL_VALUE,
        success_threshold=0.5,
    )
    assert examples
    assert all(not e.episode_succeeded for e in examples)
    assert all(e.value_target == DEFAULT_UNSUCCESSFUL_VALUE for e in examples)
    assert value_target_from_remaining(
        3, gamma=0.99, episode_succeeded=False, unsuccessful_value=0.0
    ) == 0.0
    assert first_success_index(traj, success_threshold=0.5) is None


def test_qh_input_is_33_and_output_is_scalar() -> None:
    model = HighLevelValueModel(hidden_dims=(8, 8))
    assert model.input_dim == VALUE_INPUT_DIM == 33
    first = model.net[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.in_features == 33
    state = torch.zeros(4, STATE_DIM)
    subgoal = torch.zeros(4, 2)
    goal = torch.zeros(4, 2)
    out = model(state, subgoal, goal)
    assert out.shape == (4,)
    single = model(state[0], subgoal[0], goal[0])
    assert single.shape == ()
    assert model.is_high_level_dynamics is False


def test_candidates_are_local_recorded_futures_within_max_distance() -> None:
    current = np.array([[0.0, 0.0], [0.2, 0.0], [8.0, 8.0]], dtype=np.float64)
    future = np.array([[0.4, 0.0], [3.5, 0.0], [8.2, 8.0]], dtype=np.float64)
    index = SubgoalCandidateIndex(current, future, cell_size=0.25)
    cand = index.retrieve(
        np.array([0.0, 0.0]),
        radius=0.75,
        n_candidates=32,
        max_subgoal_distance=2.0,
    )
    assert cand.shape[0] == 1
    np.testing.assert_allclose(cand[0], [0.4, 0.0])
    assert all(np.linalg.norm(row - np.array([0.0, 0.0])) <= 2.0 + 1e-9 for row in cand)


def test_highest_value_candidate_is_selected() -> None:
    current = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64)
    future = np.array([[0.5, 0.0], [0.2, 0.3]], dtype=np.float64)
    index = SubgoalCandidateIndex(current, future)
    normalizer = StateNormalizer().fit(np.zeros((4, STATE_DIM)))
    model = HighLevelValueModel(hidden_dims=(4,))

    def fallback(state, goal):
        return np.array([-9.0, -9.0])

    policy = ValueHighLevelPolicy(
        model, normalizer, index, fallback, n_candidates=8, candidate_state_radius=1.0
    )
    policy.score_candidates = lambda state, cand, goal: np.array([0.1, 2.0])  # type: ignore[method-assign]
    chosen = policy.select_subgoal(np.zeros(STATE_DIM), np.array([10.0, 0.0]))
    np.testing.assert_allclose(chosen, [0.2, 0.3])
    assert policy.last_used_fallback is False


def test_bc_fallback_when_no_candidates() -> None:
    index = SubgoalCandidateIndex(np.zeros((0, 2)), np.zeros((0, 2)))
    normalizer = StateNormalizer().fit(np.zeros((4, STATE_DIM)))
    model = HighLevelValueModel(hidden_dims=(4,))
    sentinel = np.array([1.5, -0.25])

    def fallback(state, goal):
        return sentinel

    policy = ValueHighLevelPolicy(model, normalizer, index, fallback)
    out = policy(np.zeros(STATE_DIM), np.ones(2))
    np.testing.assert_allclose(out, sentinel)
    assert policy.last_used_fallback is True
    assert policy.n_fallback == 1


def _director(policy=None) -> Director:
    rng = np.random.default_rng(0)
    normalizer = StateNormalizer().fit(rng.normal(size=(16, STATE_DIM)))
    return Director(
        manager=DirectorManager(hidden_dims=(8,), max_subgoal_distance=2.0),
        worker=GoalConditionedWorker(hidden_dims=(8,)),
        dynamics=LowLevelDynamicsModel(hidden_dims=(8,)),
        manager_normalizer=normalizer,
        worker_normalizer=normalizer,
        dynamics_normalizer=normalizer,
        horizon_k=5,
        high_level_policy=policy,
    )


def test_value_policy_does_not_change_implicit_f_h() -> None:
    calls = {"n": 0}

    def counting_policy(state, goal):
        calls["n"] += 1
        return np.array([1.0, 0.0])

    director = _director(policy=counting_policy)
    assert director.explicit_f_h is None
    assert_director_has_no_learned_f_h(director)
    director.reset_call_counts()
    out = director.high_level_transition(np.zeros(STATE_DIM), np.array([0.5, 0.1]))
    assert out.shape == (STATE_DIM,)
    assert director.n_dynamics_calls == director.horizon_k
    assert director.n_worker_calls == director.horizon_k
    assert calls["n"] == 0
    director.select_high_level_command(np.zeros(STATE_DIM), np.ones(2))
    assert calls["n"] == 1


def test_pi_l_and_f_l_objects_are_the_ones_passed_in() -> None:
    director = _director()
    worker = director.worker
    dynamics = director.dynamics
    director.high_level_policy = lambda s, g: np.zeros(2)
    director.select_high_level_command(np.zeros(STATE_DIM), np.zeros(2))
    director.high_level_transition(np.zeros(STATE_DIM), np.zeros(2))
    assert director.worker is worker
    assert director.dynamics is dynamics
    assert director.explicit_f_h is None


def test_train_high_level_value_reports_success_counts() -> None:
    goal = np.array([50.0, 0.0])
    transitions = _episode_along_x(
        0, 30, goal=goal, success_at=20
    ) + _episode_along_x(1, 30, goal=goal, success_at=None)
    metrics = train_high_level_value(
        transitions,
        model=HighLevelValueModel(hidden_dims=(8,)),
        horizon_k=10,
        val_fraction=0.5,
        seed=0,
        batch_size=8,
        epochs=2,
        gamma=0.99,
    )
    assert metrics["n_train_examples"] > 0
    assert metrics["n_val_examples"] > 0
    assert (
        metrics["n_train_success_examples"]
        + metrics["n_train_unsuccessful_examples"]
        == metrics["n_train_examples"]
    )
    assert np.isfinite(metrics["train_mse"])
    assert np.isfinite(metrics["val_mse"])
    ds = HighLevelValueDataset(
        metrics["train_high_level"], normalizer=metrics["normalizer"]
    )
    assert len(ds) == metrics["n_train_examples"]


def test_high_level_value_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    model = HighLevelValueModel(hidden_dims=(8, 8))
    normalizer = StateNormalizer().fit(
        np.random.default_rng(0).normal(size=(8, STATE_DIM))
    )
    path = tmp_path / "high_level_value.pt"
    state = torch.randn(3, STATE_DIM)
    sub = torch.randn(3, 2)
    goal = torch.randn(3, 2)
    with torch.no_grad():
        before = model(state, sub, goal)
    save_high_level_value_checkpoint(
        path, model, normalizer, gamma=0.99, horizon_k=10, n_candidates=16
    )
    loaded, loaded_norm, cfg = load_high_level_value_checkpoint(path)
    with torch.no_grad():
        after = loaded(state, sub, goal)
    torch.testing.assert_close(after, before)
    assert cfg["gamma"] == 0.99
    assert cfg["horizon_k"] == 10
    assert cfg["n_candidates"] == 16
    assert cfg["is_high_level_dynamics"] is False
    np.testing.assert_allclose(loaded_norm.mean, normalizer.mean)
