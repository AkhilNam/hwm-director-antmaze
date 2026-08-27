"""Tests for worker closed-loop eval helpers."""

from __future__ import annotations

import numpy as np

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM, extract_state_and_goal
from hwm_director.envs.antmaze import DEFAULT_ENV_ID, make_antmaze
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.train_worker import (
    choose_unique_trial_indices,
    controller_action,
    evaluate_worker_on_recorded_subgoals,
    restore_ant_state,
    restored_state_diagnostics,
    subgoal_candidates,
    summarize_subgoal_eval,
)
from tests.helpers import make_transition


def _xy_step(
    episode_id: int, t: int, x: float, next_x: float, y: float = 0.0
):
    state = np.zeros(STATE_DIM)
    next_state = np.zeros(STATE_DIM)
    state[0] = x
    state[1] = y
    next_state[0] = next_x
    next_state[1] = y
    return make_transition(
        episode_id=episode_id,
        state=state,
        next_state=next_state,
        qpos=np.zeros(15),
        qvel=np.zeros(14),
    )


def _walk(episode_id: int, n_steps: int, step: float):
    return [
        _xy_step(episode_id, t, t * step, (t + 1) * step) for t in range(n_steps)
    ]


def test_subgoal_candidates_same_episode_k_and_distance() -> None:
    # 0.4 m/step so k=1 is 0.4 (too close), k=2 is 0.8 (in 0.5–2.0).
    transitions = _walk(0, 8, 0.4) + _walk(1, 8, 0.4)
    cands = subgoal_candidates(
        transitions, horizon_k=5, min_distance=0.5, max_distance=2.0
    )
    assert cands
    for cand in cands:
        assert 1 <= cand.k <= 5
        start_ep = cand.traj[cand.t].episode_id
        future_ep = cand.traj[cand.t + cand.k - 1].episode_id
        assert start_ep == future_ep
        dist = float(np.linalg.norm(cand.start_xy - cand.subgoal_xy))
        assert 0.5 <= dist <= 2.0
        assert abs(dist - cand.initial_distance) < 1e-9


def test_subgoal_candidates_exclude_trivial_distance() -> None:
    transitions = _walk(0, 6, 0.2)
    cands = subgoal_candidates(
        transitions, horizon_k=10, min_distance=0.5, max_distance=2.0
    )
    assert cands
    assert all(c.initial_distance >= 0.5 for c in cands)
    assert all(c.k >= 3 for c in cands)


def test_subgoal_candidates_empty_raises_context_for_eval_path() -> None:
    transitions = _walk(0, 4, 0.05)
    cands = subgoal_candidates(
        transitions, horizon_k=3, min_distance=0.5, max_distance=2.0
    )
    assert cands == []


def test_choose_unique_trial_indices_no_duplicates() -> None:
    rng = np.random.default_rng(0)
    picks = choose_unique_trial_indices(10, n_trials=10, rng=rng)
    assert len(picks) == 10
    assert len(set(picks.tolist())) == 10
    rng = np.random.default_rng(1)
    small = choose_unique_trial_indices(5, n_trials=20, rng=rng)
    assert len(small) == 5
    assert len(set(small.tolist())) == 5


def test_summarize_already_successful_and_reductions() -> None:
    initials = np.array([0.2, 1.0, 1.0])
    finals = np.array([0.1, 0.8, 1.2])
    summary = summarize_subgoal_eval(initials, finals, success_threshold=0.5)
    assert summary["n_trials"] == 3
    assert abs(summary["fraction_already_successful_at_start"] - (1 / 3)) < 1e-9
    reductions = initials - finals
    assert abs(summary["mean_distance_reduction"] - float(np.mean(reductions))) < 1e-9
    assert abs(summary["median_distance_reduction"] - float(np.median(reductions))) < 1e-9
    assert abs(summary["fraction_positive_reduction"] - (2 / 3)) < 1e-9
    assert abs(summary["success_rate"] - (1 / 3)) < 1e-9
    relative = reductions / initials
    assert abs(summary["fraction_relative_progress_10"] - float(np.mean(relative >= 0.1))) < 1e-9


def test_restore_ant_state_matches_recorded_xy() -> None:
    env = make_antmaze()
    try:
        assert env.spec is None or env.spec.id == DEFAULT_ENV_ID
        observation, _info = env.reset(seed=0)
        ant = env.unwrapped.ant_env
        qpos = np.array(ant.data.qpos, copy=True)
        qvel = np.array(ant.data.qvel, copy=True)
        recorded, _ = extract_state_and_goal(observation)
        env.reset(seed=1)
        restored_obs = restore_ant_state(env, qpos, qvel)
        restored, _ = extract_state_and_goal(restored_obs)
        np.testing.assert_allclose(restored[:2], recorded[:2], atol=1e-5)
        diag = restored_state_diagnostics(restored, recorded)
        assert diag["xy_abs_err"] < 1e-4
        assert diag["proprio_abs_err"] < 1e-4
        assert "contact_abs_err" not in diag
    finally:
        env.close()


def test_zero_controller_is_zeros() -> None:
    action = controller_action("zero")
    assert action.shape == (8,)
    assert action.dtype == np.float32
    np.testing.assert_array_equal(action, np.zeros(8, dtype=np.float32))


def test_random_controller_shape_bounds_and_seed() -> None:
    a = controller_action("random", rng=np.random.default_rng(0))
    b = controller_action("random", rng=np.random.default_rng(0))
    c = controller_action("random", rng=np.random.default_rng(1))
    assert a.shape == (8,)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_empty_eval_has_three_controller_metrics() -> None:
    model = GoalConditionedWorker(hidden_dims=(8,))
    normalizer = StateNormalizer().fit(np.zeros((2, STATE_DIM)))
    result = evaluate_worker_on_recorded_subgoals(
        [], model, normalizer, n_trials=0, dataset_id=None
    )
    for name in ("worker", "zero", "random"):
        assert "mean_final_distance" in result[name]
        assert "success_rate" in result[name]
    assert result["n_trials"] == 0
    assert result["candidate_indices"] == []
    assert result["skipped"] is False


def test_eval_skips_when_qpos_qvel_missing() -> None:
    state = np.zeros(STATE_DIM)
    next_state = np.zeros(STATE_DIM)
    next_state[0] = 0.7
    transition = make_transition(state=state, next_state=next_state)
    model = GoalConditionedWorker(hidden_dims=(8,))
    normalizer = StateNormalizer().fit(np.stack([state, next_state]))
    result = evaluate_worker_on_recorded_subgoals(
        [transition],
        model,
        normalizer,
        horizon_k=1,
        n_trials=5,
        min_distance=0.5,
        max_distance=2.0,
        dataset_id=None,
    )
    assert result["skipped"] is True
    assert result["n_trials"] == 0
    assert "qpos" in result["skip_reason"] or "qvel" in result["skip_reason"]
    assert "TODO" in result["todo"]


def test_three_controllers_share_candidates_and_restore_each_rollout() -> None:
    env = make_antmaze()
    try:
        observation, _info = env.reset(seed=0)
        ant = env.unwrapped.ant_env
        qpos = np.array(ant.data.qpos, copy=True)
        qvel = np.array(ant.data.qvel, copy=True)
        recorded, goal = extract_state_and_goal(observation)
    finally:
        env.close()

    next_state = np.array(recorded, copy=True)
    next_state[0] = recorded[0] + 0.7
    transition = make_transition(
        state=recorded,
        next_state=next_state,
        goal=goal,
        qpos=qpos,
        qvel=qvel,
    )
    model = GoalConditionedWorker(hidden_dims=(8,))
    normalizer = StateNormalizer().fit(np.stack([recorded, next_state]))
    result = evaluate_worker_on_recorded_subgoals(
        [transition],
        model,
        normalizer,
        horizon_k=1,
        n_trials=5,
        min_distance=0.5,
        max_distance=2.0,
        seed=0,
        dataset_id=None,
    )
    assert result["skipped"] is False
    assert result["n_trials"] == 1
    assert result["n_restores"] == 3
    assert result["candidate_indices"] == [0]
    for name in ("worker", "zero", "random"):
        assert result[name]["n_trials"] == 1
        assert np.isfinite(result[name]["mean_final_distance"])
    trial = result["trials"][0]
    assert trial["candidate_index"] == 0
    assert "worker" in trial and "zero" in trial and "random" in trial
