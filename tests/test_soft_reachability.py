"""Soft reachability scoring: penalty, not a hard filter."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.data.subgoal_candidates import SubgoalCandidateIndex
from hwm_director.models.director import Director, assert_director_has_no_learned_f_h
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.high_level_value import HighLevelValueModel
from hwm_director.models.value_manager import (
    SCORE_EPS,
    ValueHighLevelPolicy,
    candidate_set_zscore,
    combined_reachability_scores,
    director_reachability_error,
)
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.director_diagnostics import format_pareto_table


def test_candidate_set_q_and_error_zscore() -> None:
    q = np.array([1.0, 3.0, 5.0])
    z = candidate_set_zscore(q)
    np.testing.assert_allclose(np.mean(z), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(z), np.std(q) / (np.std(q) + SCORE_EPS), atol=1e-6)
    constant = candidate_set_zscore(np.array([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(constant, 0.0)


def test_combined_score_formula_zscore_and_raw() -> None:
    q = np.array([0.0, 2.0])
    err = np.array([4.0, 0.0])
    scores, q_n, err_n = combined_reachability_scores(
        q, err, lambda_reach=1.0, normalization="candidate_zscore"
    )
    expected = q_n - 1.0 * err_n
    np.testing.assert_allclose(scores, expected)
    raw, q_raw, err_raw = combined_reachability_scores(
        q, err, lambda_reach=0.5, normalization="raw"
    )
    np.testing.assert_allclose(q_raw, q)
    np.testing.assert_allclose(err_raw, err)
    np.testing.assert_allclose(raw, q - 0.5 * err)


def _two_candidate_index() -> tuple[SubgoalCandidateIndex, StateNormalizer]:
    source = np.zeros((2, STATE_DIM))
    source[0, 0] = 0.0
    source[1, 0] = 0.1
    future = np.array([[0.4, 0.0], [1.2, 0.0]])
    normalizer = StateNormalizer().fit(np.zeros((8, STATE_DIM)))
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    return index, normalizer


def test_lambda_zero_ignores_reachability_penalty() -> None:
    index, normalizer = _two_candidate_index()
    policy = ValueHighLevelPolicy(
        HighLevelValueModel(hidden_dims=(4,)),
        normalizer,
        index,
        bc_fallback=lambda s, g: np.array([-9.0, -9.0]),
        retrieval_mode="xy",
        candidate_state_radius=1.0,
        reachability_fn=lambda s, g: 10.0 if g[0] > 1.0 else 0.1,
        use_soft_reachability=True,
        reachability_score_weight=0.0,
    )
    policy.score_candidates = lambda state, cand, goal: np.array([0.1, 5.0])  # type: ignore[method-assign]
    chosen = policy.select_subgoal(np.zeros(STATE_DIM), np.ones(2))
    np.testing.assert_allclose(chosen, [1.2, 0.0])
    assert policy.last_diagnostics["reachability_score_weight"] == 0.0
    assert policy.last_n_candidates == 2


def test_larger_lambda_prefers_more_reachable_candidate() -> None:
    index, normalizer = _two_candidate_index()
    errors = {0.4: 0.05, 1.2: 2.0}

    def make_policy(lam: float) -> ValueHighLevelPolicy:
        policy = ValueHighLevelPolicy(
            HighLevelValueModel(hidden_dims=(4,)),
            normalizer,
            index,
            bc_fallback=lambda s, g: np.array([-9.0, -9.0]),
            retrieval_mode="xy",
            candidate_state_radius=1.0,
            reachability_fn=lambda s, g: errors[round(float(g[0]), 1)],
            use_soft_reachability=True,
            reachability_score_weight=lam,
        )
        policy.score_candidates = lambda state, cand, goal: np.array([1.0, 1.2])  # type: ignore[method-assign]
        return policy

    low = make_policy(0.0)
    np.testing.assert_allclose(
        low.select_subgoal(np.zeros(STATE_DIM), np.ones(2)), [1.2, 0.0]
    )
    high = make_policy(8.0)
    np.testing.assert_allclose(
        high.select_subgoal(np.zeros(STATE_DIM), np.ones(2)), [0.4, 0.0]
    )


def test_soft_mode_does_not_hard_reject_high_error() -> None:
    index, normalizer = _two_candidate_index()
    policy = ValueHighLevelPolicy(
        HighLevelValueModel(hidden_dims=(4,)),
        normalizer,
        index,
        bc_fallback=lambda s, g: np.array([-9.0, -9.0]),
        retrieval_mode="xy",
        candidate_state_radius=1.0,
        reachability_fn=lambda s, g: 9.0,
        max_predicted_subgoal_error=0.5,
        use_soft_reachability=True,
        reachability_score_weight=0.0,
    )
    policy.score_candidates = lambda state, cand, goal: np.array([0.2, 3.0])  # type: ignore[method-assign]
    chosen = policy.select_subgoal(np.zeros(STATE_DIM), np.ones(2))
    np.testing.assert_allclose(chosen, [1.2, 0.0])
    assert policy.last_used_fallback is False
    assert policy.last_n_candidates == 2


def test_soft_records_score_components() -> None:
    index, normalizer = _two_candidate_index()
    policy = ValueHighLevelPolicy(
        HighLevelValueModel(hidden_dims=(4,)),
        normalizer,
        index,
        bc_fallback=lambda s, g: np.zeros(2),
        retrieval_mode="xy",
        candidate_state_radius=1.0,
        reachability_fn=lambda s, g: 0.2 if g[0] < 1.0 else 1.5,
        use_soft_reachability=True,
        reachability_score_weight=1.0,
    )
    policy.score_candidates = lambda state, cand, goal: np.array([2.0, 1.0])  # type: ignore[method-assign]
    policy.select_subgoal(np.zeros(STATE_DIM), np.ones(2))
    diag = policy.last_diagnostics
    for key in (
        "qh_score",
        "qh_score_normalized",
        "predicted_subgoal_error",
        "reach_error_normalized",
        "reachability_score_weight",
        "combined_score",
        "qh_max",
        "qh_min",
        "qh_mean",
        "reach_error_max",
        "reach_error_min",
        "reach_error_mean",
    ):
        assert key in diag
        assert np.isfinite(diag[key])
    assert diag["qh_max"] == 2.0
    assert diag["qh_min"] == 1.0
    assert diag["reach_error_max"] == 1.5
    assert diag["reach_error_min"] == 0.2


def _tiny_director() -> Director:
    rng = np.random.default_rng(1)
    normalizer = StateNormalizer().fit(rng.normal(size=(16, STATE_DIM)))
    return Director(
        manager=DirectorManager(hidden_dims=(8,)),
        worker=GoalConditionedWorker(hidden_dims=(8,)),
        dynamics=LowLevelDynamicsModel(hidden_dims=(8,)),
        manager_normalizer=normalizer,
        worker_normalizer=normalizer,
        dynamics_normalizer=normalizer,
        horizon_k=5,
    )


def test_soft_rollout_recomputes_worker_each_step_and_qh_unchanged() -> None:
    director = _tiny_director()
    model = HighLevelValueModel(hidden_dims=(4,))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    source = np.zeros((1, STATE_DIM))
    index = SubgoalCandidateIndex(
        source[:, :2],
        np.array([[0.3, 0.0]]),
        source_states=source,
        normalizer=director.manager_normalizer,
    )
    policy = ValueHighLevelPolicy(
        model,
        director.manager_normalizer,
        index,
        director._select_bc,
        retrieval_mode="xy",
        reachability_fn=director_reachability_error(director),
        use_soft_reachability=True,
        reachability_score_weight=1.0,
    )
    director.high_level_policy = policy
    director.reset_call_counts()
    director.select_high_level_command(np.zeros(STATE_DIM), np.ones(2))
    assert director.n_dynamics_calls == director.horizon_k
    assert director.n_worker_calls == director.horizon_k
    after = model.state_dict()
    for key, tensor in before.items():
        torch.testing.assert_close(after[key], tensor)
    assert director.explicit_f_h is None
    assert_director_has_no_learned_f_h(director)


def test_pareto_table_picks_balanced_above_bc() -> None:
    rows = [
        {
            "manager_name": "BC",
            "success_rate": 0.20,
            "subgoal_reach_rate": 0.97,
            "mean_final_distance": 1.1,
            "mean_predicted_subgoal_error": 0.2,
            "mean_worker_subgoal_error": 0.2,
            "wall_region": {"mean_high_level_steps_in_region": 37.0},
        },
        {
            "manager_name": "Value-State",
            "success_rate": 0.34,
            "subgoal_reach_rate": 0.44,
            "mean_final_distance": 1.3,
            "mean_predicted_subgoal_error": 0.56,
            "mean_worker_subgoal_error": 0.56,
            "wall_region": {"mean_high_level_steps_in_region": 13.0},
        },
        {
            "manager_name": "Soft-l1",
            "success_rate": 0.28,
            "subgoal_reach_rate": 0.70,
            "mean_final_distance": 1.4,
            "mean_predicted_subgoal_error": 0.4,
            "mean_worker_subgoal_error": 0.4,
            "wall_region": {"mean_high_level_steps_in_region": 18.0},
        },
    ]
    text = format_pareto_table(rows)
    assert "highest success: Value-State" in text
    assert "highest reach: BC" in text
    assert "Soft-l1" in text and "0.700" in text
