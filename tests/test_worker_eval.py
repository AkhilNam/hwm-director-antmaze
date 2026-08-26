"""Tests for worker closed-loop eval helpers."""

from __future__ import annotations

import numpy as np

from hwm_director.data.state import extract_state_and_goal
from hwm_director.data.transitions import Transition
from hwm_director.envs.antmaze import make_antmaze
from hwm_director.training.train_worker import (
    choose_unique_trial_indices,
    restore_ant_state,
    restored_state_diagnostics,
    subgoal_candidates,
    summarize_subgoal_eval,
)


def _xy_step(
    episode_id: int, t: int, x: float, next_x: float, y: float = 0.0
) -> Transition:
    state = np.zeros(107)
    next_state = np.zeros(107)
    state[0] = x
    state[1] = y
    next_state[0] = next_x
    next_state[1] = y
    return Transition(
        state=state,
        action=np.zeros(8, dtype=np.float32),
        next_state=next_state,
        goal=np.zeros(2),
        episode_id=episode_id,
        qpos=np.zeros(15),
        qvel=np.zeros(14),
    )


def _walk(episode_id: int, n_steps: int, step: float) -> list[Transition]:
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
    finally:
        env.close()
