"""State-aware candidate retrieval and reachability filtering."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.data.subgoal_candidates import (
    SubgoalCandidateIndex,
    estimate_source_state_distance_threshold,
)
from hwm_director.models.director import Director, assert_director_has_no_learned_f_h
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.high_level_value import HighLevelValueModel
from hwm_director.models.value_manager import (
    ValueHighLevelPolicy,
    director_reachability_error,
)
from hwm_director.models.worker import GoalConditionedWorker
from tests.helpers import make_transition


def _normalizer_from_states(states: np.ndarray) -> StateNormalizer:
    return StateNormalizer().fit(states)


def test_state_retrieval_uses_normalized_29d() -> None:
    n = 4
    source = np.zeros((n, STATE_DIM))
    source[:, 0] = [0.0, 0.05, 0.05, 8.0]
    source[0, 10] = 50.0
    source[1, 10] = 50.0
    source[2, 10] = 0.0
    future = np.array([[0.4, 0.0], [0.5, 0.0], [0.6, 0.1], [8.2, 8.0]])
    train = np.vstack([source, np.zeros((8, STATE_DIM))])
    normalizer = _normalizer_from_states(train)
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    query = np.zeros(STATE_DIM)
    query[0] = 0.04
    query[10] = 0.0
    recs = index.retrieve_records(
        query, mode="state", radius=0.75, n_candidates=1, max_subgoal_distance=2.0
    )
    assert recs
    # xy-nearest would be source[1] (0.05 vs 0.04) but dim 10 is huge after normalize
    np.testing.assert_allclose(recs[0]["future_xy"], [0.6, 0.1])
    called = {"n": 0}
    orig = normalizer.normalize

    def wrapped(states):
        called["n"] += 1
        out = orig(states)
        assert np.asarray(out).shape[-1] == STATE_DIM
        return out

    normalizer.normalize = wrapped  # type: ignore[method-assign]
    index.retrieve_records(
        query, mode="state", radius=0.75, n_candidates=2, max_subgoal_distance=2.0
    )
    assert called["n"] >= 1


def test_state_normalizer_is_train_only() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(loc=0.0, scale=1.0, size=(32, STATE_DIM))
    val = rng.normal(loc=10.0, scale=1.0, size=(32, STATE_DIM))
    train_norm = StateNormalizer().fit(train)
    val_norm = StateNormalizer().fit(val)
    assert np.linalg.norm(train_norm.mean - val_norm.mean) > 1.0
    source = train[:4].copy()
    source[:, :2] = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
    future = source[:, :2] + 0.2
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=train_norm
    )
    np.testing.assert_allclose(index.normalizer.mean, train_norm.mean)
    np.testing.assert_allclose(index.source_states_n, train_norm.normalize(source))


def test_hybrid_retrieval_uses_state_and_xy_weights() -> None:
    source = np.zeros((2, STATE_DIM))
    source[0, :2] = [0.0, 0.0]
    source[1, :2] = [0.4, 0.0]
    source[1, 5] = 20.0
    future = np.array([[0.2, 0.0], [0.5, 0.0]])
    normalizer = _normalizer_from_states(np.vstack([source, np.zeros((8, STATE_DIM))]))
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    query = np.zeros(STATE_DIM)
    query[0] = 0.35
    only_xy = index.retrieve_records(
        query,
        mode="hybrid",
        radius=1.0,
        n_candidates=1,
        state_distance_weight=0.0,
        xy_distance_weight=1.0,
    )
    np.testing.assert_allclose(only_xy[0]["future_xy"], [0.5, 0.0])
    only_state = index.retrieve_records(
        query,
        mode="hybrid",
        radius=1.0,
        n_candidates=1,
        state_distance_weight=1.0,
        xy_distance_weight=0.0,
    )
    np.testing.assert_allclose(only_state[0]["future_xy"], [0.2, 0.0])


def test_source_state_threshold_filtering() -> None:
    source = np.zeros((2, STATE_DIM))
    source[0, :2] = [0.0, 0.0]
    source[1, :2] = [0.1, 0.0]
    source[1, 8] = 40.0
    future = np.array([[0.3, 0.0], [0.4, 0.0]])
    normalizer = _normalizer_from_states(np.vstack([source, np.zeros((8, STATE_DIM))]))
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    recs = index.retrieve_records(
        np.zeros(STATE_DIM),
        mode="state",
        radius=0.75,
        n_candidates=8,
        max_source_state_distance=0.5,
    )
    assert len(recs) == 1
    np.testing.assert_allclose(recs[0]["future_xy"], [0.3, 0.0])
    assert recs[0]["source_state_distance"] <= 0.5


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


def test_reachability_rollout_is_exactly_k_and_recomputes_worker() -> None:
    director = _tiny_director()
    k = director.horizon_k
    seen: list[torch.Tensor] = []
    original = director.worker.forward

    def wrapped(state: torch.Tensor, subgoal: torch.Tensor) -> torch.Tensor:
        seen.append(state.detach().clone())
        return original(state, subgoal)

    director.worker.forward = wrapped  # type: ignore[method-assign]
    director.reset_call_counts()
    err_fn = director_reachability_error(director)
    err = err_fn(np.zeros(STATE_DIM), np.array([0.5, 0.0]))
    assert np.isfinite(err)
    assert director.n_dynamics_calls == k
    assert director.n_worker_calls == k
    assert len(seen) == k
    assert not torch.allclose(seen[0], seen[-1])
    assert director.explicit_f_h is None
    assert_director_has_no_learned_f_h(director)


def test_reachability_filter_rejects_large_predicted_error() -> None:
    source = np.zeros((2, STATE_DIM))
    source[0, 0] = 0.0
    source[1, 0] = 0.1
    future = np.array([[0.4, 0.0], [1.5, 0.0]])
    normalizer = _normalizer_from_states(np.zeros((8, STATE_DIM)))
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    errors = {0.4: 0.1, 1.5: 0.9}

    def reach(state, subgoal):
        return errors[round(float(subgoal[0]), 1)]

    policy = ValueHighLevelPolicy(
        HighLevelValueModel(hidden_dims=(4,)),
        normalizer,
        index,
        bc_fallback=lambda s, g: np.array([-1.0, -1.0]),
        retrieval_mode="xy",
        candidate_state_radius=1.0,
        reachability_fn=reach,
        max_predicted_subgoal_error=0.5,
    )
    chosen = policy.select_subgoal(np.zeros(STATE_DIM), np.ones(2))
    np.testing.assert_allclose(chosen, [0.4, 0.0])
    assert policy.last_used_fallback is False


def test_bc_fallback_when_reachability_filters_all() -> None:
    source = np.zeros((1, STATE_DIM))
    future = np.array([[0.4, 0.0]])
    normalizer = _normalizer_from_states(np.zeros((8, STATE_DIM)))
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    sentinel = np.array([3.0, 4.0])
    policy = ValueHighLevelPolicy(
        HighLevelValueModel(hidden_dims=(4,)),
        normalizer,
        index,
        bc_fallback=lambda s, g: sentinel,
        retrieval_mode="xy",
        reachability_fn=lambda s, g: 9.0,
        max_predicted_subgoal_error=0.5,
    )
    out = policy(np.zeros(STATE_DIM), np.zeros(2))
    np.testing.assert_allclose(out, sentinel)
    assert policy.last_used_fallback is True


def test_qh_unchanged_and_director_has_no_explicit_fh() -> None:
    director = _tiny_director()
    model = HighLevelValueModel(hidden_dims=(4,))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    normalizer = director.manager_normalizer
    source = np.zeros((1, STATE_DIM))
    index = SubgoalCandidateIndex(
        source[:, :2],
        np.array([[0.3, 0.0]]),
        source_states=source,
        normalizer=normalizer,
    )
    policy = ValueHighLevelPolicy(
        model, normalizer, index, director._select_bc, retrieval_mode="xy"
    )
    director.high_level_policy = policy
    director.select_high_level_command(np.zeros(STATE_DIM), np.ones(2))
    after = model.state_dict()
    for key, tensor in before.items():
        torch.testing.assert_close(after[key], tensor)
    assert director.explicit_f_h is None
    director.reset_call_counts()
    director.high_level_transition(np.zeros(STATE_DIM), np.array([0.3, 0.0]))
    assert director.n_dynamics_calls == director.horizon_k
    assert director.n_worker_calls == director.horizon_k


def test_threshold_estimator_uses_train_normalizer() -> None:
    steps = []
    for t in range(12):
        state = np.zeros(STATE_DIM)
        nxt = np.zeros(STATE_DIM)
        state[0] = float(t) * 0.05
        nxt[0] = float(t + 1) * 0.05
        state[3] = 0.01 * t
        nxt[3] = 0.01 * (t + 1)
        steps.append(make_transition(episode_id=0, state=state, next_state=nxt))
    for t in range(12):
        state = np.zeros(STATE_DIM)
        nxt = np.zeros(STATE_DIM)
        state[0] = 10.0 + 0.05 * t
        nxt[0] = 10.0 + 0.05 * (t + 1)
        steps.append(make_transition(episode_id=1, state=state, next_state=nxt))
    train_states = np.stack([s.state for s in steps if s.episode_id == 0])
    normalizer = StateNormalizer().fit(train_states)
    info = estimate_source_state_distance_threshold(
        steps, normalizer, xy_radius=0.75, percentile=90.0, seed=0
    )
    assert info["n_lag1_pairs"] > 0
    assert np.isfinite(info["chosen_threshold"])
    assert info["percentile"] == 90.0
