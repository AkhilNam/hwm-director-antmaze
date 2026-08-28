"""Explicit f_H_phi, HWM sharing, and matched SoftReach scoring."""

from __future__ import annotations

import numpy as np
import torch

from hwm_director.data.high_level_transitions import (
    HighLevelDynamicsDataset,
    build_high_level_transitions,
)
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.state import STATE_DIM
from hwm_director.data.subgoal_candidates import SubgoalCandidateIndex
from hwm_director.models.director import Director, assert_director_has_no_learned_f_h
from hwm_director.models.director_manager import DirectorManager
from hwm_director.models.dynamics_high import (
    HIGH_LEVEL_DYNAMICS_INPUT_DIM,
    ExplicitHighLevelDynamics,
)
from hwm_director.models.dynamics_low import LowLevelDynamicsModel
from hwm_director.models.encoder import IdentityEncoder
from hwm_director.models.high_level_value import HighLevelValueModel
from hwm_director.models.hierarchy import HierarchicalController
from hwm_director.models.hwm import HierarchicalWorldModel, assert_hwm_has_explicit_f_h
from hwm_director.models.value_manager import (
    ValueHighLevelPolicy,
    combined_reachability_scores,
    director_reachability_error,
)
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.checkpoints import (
    load_high_level_dynamics_checkpoint,
    load_high_level_value_checkpoint,
    save_high_level_dynamics_checkpoint,
    save_high_level_value_checkpoint,
)
from hwm_director.training.director_diagnostics import format_fairness_summary
from hwm_director.training.train_high_level_dynamics import train_high_level_dynamics
from tests.helpers import make_transition


def _episode_along_x(episode_id: int, length: int) -> list:
    steps = []
    for t in range(length):
        state = np.zeros(STATE_DIM)
        nxt = np.zeros(STATE_DIM)
        state[0] = float(t) * 0.1
        nxt[0] = float(t + 1) * 0.1
        steps.append(
            make_transition(episode_id=episode_id, state=state, next_state=nxt)
        )
    return steps


def test_high_level_dataset_is_exactly_k_and_same_episode() -> None:
    k = 4
    examples = build_high_level_transitions(
        _episode_along_x(0, 12) + _episode_along_x(1, 12), horizon_k=k
    )
    assert examples
    ids = {ex.episode_id for ex in examples}
    assert ids == {0, 1}
    for ex in examples:
        np.testing.assert_allclose(ex.g_tau, ex.h_next[:2])
        np.testing.assert_allclose(ex.h_next[0], ex.h_tau[0] + 0.1 * k)
        assert ex.t + k <= 12
        # Crossing an episode boundary would reset x to the other episode's start.
        assert ex.h_next[0] > ex.h_tau[0]


def test_fh_input_31_output_29_delta_reconstructs() -> None:
    model = ExplicitHighLevelDynamics(hidden_dims=(8, 8))
    assert model.input_dim == HIGH_LEVEL_DYNAMICS_INPUT_DIM == 31
    first = model.net[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.in_features == 31
    assert model.net[-1].out_features == STATE_DIM == 29
    state = torch.zeros(3, STATE_DIM)
    subgoal = torch.zeros(3, 2)
    delta = model(state, subgoal)
    nxt = model.predict_next_state(state, subgoal)
    torch.testing.assert_close(nxt, state + delta)
    assert nxt.shape == (3, STATE_DIM)
    assert model.is_high_level_dynamics is True


def test_train_val_episodes_disjoint_and_normalizer_train_only() -> None:
    transitions = _episode_along_x(0, 20) + _episode_along_x(1, 20)
    metrics = train_high_level_dynamics(
        transitions,
        model=ExplicitHighLevelDynamics(hidden_dims=(8,)),
        horizon_k=5,
        val_fraction=0.5,
        seed=0,
        batch_size=8,
        epochs=2,
    )
    assert set(metrics["train_episode_ids"]).isdisjoint(metrics["val_episode_ids"])
    assert metrics["n_train_examples"] > 0
    assert metrics["n_val_examples"] > 0
    train_states = np.stack(
        [s.state for s in transitions if s.episode_id in metrics["train_episode_ids"]]
    )
    np.testing.assert_allclose(metrics["normalizer"].mean, train_states.mean(axis=0))
    ds = HighLevelDynamicsDataset(
        metrics["train_high_level"], normalizer=metrics["normalizer"]
    )
    assert len(ds) == metrics["n_train_examples"]
    torch.testing.assert_close(ds.h_next, ds.h_tau + ds.delta)


def _pair():
    rng = np.random.default_rng(0)
    normalizer = StateNormalizer().fit(rng.normal(size=(16, STATE_DIM)))
    director = Director(
        manager=DirectorManager(hidden_dims=(8,)),
        worker=GoalConditionedWorker(hidden_dims=(8,)),
        dynamics=LowLevelDynamicsModel(hidden_dims=(8,)),
        manager_normalizer=normalizer,
        worker_normalizer=normalizer,
        dynamics_normalizer=normalizer,
        horizon_k=5,
    )
    fh = ExplicitHighLevelDynamics(hidden_dims=(8,))
    hwm = HierarchicalWorldModel.from_director(director, fh, normalizer)
    return director, hwm, normalizer


def test_hwm_shares_pi_l_and_f_l_and_has_explicit_fh() -> None:
    director, hwm, _ = _pair()
    assert hwm.worker is director.worker
    assert hwm.dynamics is director.dynamics
    assert hwm.manager is director.manager
    assert director.explicit_f_h is None
    assert_director_has_no_learned_f_h(director)
    assert hwm.explicit_f_h is not None
    assert_hwm_has_explicit_f_h(hwm)
    assert isinstance(hwm, HierarchicalController)
    assert isinstance(hwm.encoder, IdentityEncoder)
    np.testing.assert_allclose(hwm.encoder.encode(np.arange(STATE_DIM)), np.arange(STATE_DIM))
    summary = format_fairness_summary(director=director, hwm=hwm, lambda_reach=1.0)
    assert "Director f_H = (f_L, pi_L)^K" in summary
    assert "HWM f_H = explicit learned f_H_phi" in summary
    assert "pi_L identity: True" in summary
    assert "f_L identity: True" in summary


def test_hwm_transition_is_one_fh_call_not_k_worker_steps() -> None:
    director, hwm, _ = _pair()
    k = director.horizon_k
    director.reset_call_counts()
    director.high_level_transition(np.zeros(STATE_DIM), np.array([0.4, 0.0]))
    assert director.n_worker_calls == k
    assert director.n_dynamics_calls == k
    hwm.reset_call_counts()
    out = hwm.high_level_transition(np.zeros(STATE_DIM), np.array([0.4, 0.0]))
    assert out.shape == (STATE_DIM,)
    assert hwm.n_explicit_fh_calls == 1
    assert hwm.n_worker_calls == 0
    assert hwm.n_dynamics_calls == 0
    try:
        hwm.high_level_transition(np.zeros(STATE_DIM), np.array([0.4, 0.0]), horizon_k=k + 1)
    except ValueError as err:
        assert "trained for K=" in str(err)
    else:
        raise AssertionError("HWM f_H_phi must reject a mismatched K")


def test_candidate_generation_and_lambda_match() -> None:
    director, hwm, normalizer = _pair()
    source = np.zeros((2, STATE_DIM))
    source[0, 0] = 0.0
    source[1, 0] = 0.1
    future = np.array([[0.4, 0.0], [0.5, 0.1]])
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    qh = HighLevelValueModel(hidden_dims=(4,))
    d_pol = ValueHighLevelPolicy(
        qh,
        normalizer,
        index,
        director._select_bc,
        retrieval_mode="state",
        candidate_state_radius=1.0,
        reachability_fn=director_reachability_error(director),
        use_soft_reachability=True,
        reachability_score_weight=1.0,
    )
    h_pol = ValueHighLevelPolicy(
        qh,
        normalizer,
        index,
        hwm._select_bc,
        retrieval_mode="state",
        candidate_state_radius=1.0,
        reachability_fn=director_reachability_error(hwm),
        use_soft_reachability=True,
        reachability_score_weight=1.0,
    )
    query = np.zeros(STATE_DIM)
    d_recs = d_pol.retrieve_candidate_records(query)
    h_recs = h_pol.retrieve_candidate_records(query)
    assert len(d_recs) == len(h_recs) == 2
    np.testing.assert_allclose(d_recs[0]["future_xy"], h_recs[0]["future_xy"])
    assert d_pol.reachability_score_weight == h_pol.reachability_score_weight == 1.0
    director.high_level_policy = d_pol
    hwm.high_level_policy = h_pol
    assert d_pol.candidate_index is h_pol.candidate_index


def test_scoring_formula_same_except_transition_source() -> None:
    q = np.array([1.0, 1.0])
    err_d = np.array([0.8, 0.1])
    err_h = np.array([0.2, 0.9])
    s_d, _, _ = combined_reachability_scores(q, err_d, lambda_reach=1.0)
    s_h, _, _ = combined_reachability_scores(q, err_h, lambda_reach=1.0)
    assert int(np.argmax(s_d)) != int(np.argmax(s_h))
    director, hwm, normalizer = _pair()
    source = np.zeros((2, STATE_DIM))
    source[1, 0] = 0.1
    future = np.array([[0.4, 0.0], [1.2, 0.0]])
    index = SubgoalCandidateIndex(
        source[:, :2], future, source_states=source, normalizer=normalizer
    )
    qh = HighLevelValueModel(hidden_dims=(4,))

    def policy(reach_fn):
        pol = ValueHighLevelPolicy(
            qh,
            normalizer,
            index,
            lambda s, g: np.zeros(2),
            retrieval_mode="xy",
            candidate_state_radius=1.0,
            reachability_fn=reach_fn,
            use_soft_reachability=True,
            reachability_score_weight=1.0,
        )
        pol.score_candidates = lambda state, cand, goal: np.array([1.0, 1.0])  # type: ignore[method-assign]
        return pol

    pick_d = policy(lambda s, g: 0.05 if g[0] < 1.0 else 2.0)
    pick_h = policy(lambda s, g: 2.0 if g[0] < 1.0 else 0.05)
    a = pick_d.select_subgoal(np.zeros(STATE_DIM), np.ones(2))
    b = pick_h.select_subgoal(np.zeros(STATE_DIM), np.ones(2))
    np.testing.assert_allclose(a, [0.4, 0.0])
    np.testing.assert_allclose(b, [1.2, 0.0])


def test_qh_checkpoint_loads_for_hwm_policy(tmp_path) -> None:
    torch.manual_seed(0)
    model = HighLevelValueModel(hidden_dims=(8, 8))
    rng = np.random.default_rng(0)
    normalizer = StateNormalizer().fit(rng.normal(size=(8, STATE_DIM)))
    path = tmp_path / "qh.pt"
    save_high_level_value_checkpoint(path, model, normalizer, gamma=0.99, horizon_k=10)
    loaded, loaded_norm, cfg = load_high_level_value_checkpoint(path)
    assert cfg["is_high_level_dynamics"] is False
    director, hwm, _ = _pair()
    source = np.zeros((1, STATE_DIM))
    index = SubgoalCandidateIndex(
        source[:, :2], np.array([[0.3, 0.0]]), source_states=source, normalizer=loaded_norm
    )
    policy = ValueHighLevelPolicy(
        loaded,
        loaded_norm,
        index,
        hwm._select_bc,
        retrieval_mode="xy",
        use_soft_reachability=True,
        reachability_score_weight=1.0,
        reachability_fn=director_reachability_error(hwm),
    )
    hwm.high_level_policy = policy
    out = hwm.select_high_level_command(np.zeros(STATE_DIM), np.ones(2))
    assert out.shape == (2,)


def test_fh_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    model = ExplicitHighLevelDynamics(hidden_dims=(8, 8))
    rng = np.random.default_rng(0)
    normalizer = StateNormalizer().fit(rng.normal(size=(8, STATE_DIM)))
    path = tmp_path / "f_h.pt"
    state = torch.randn(2, STATE_DIM)
    sub = torch.randn(2, 2)
    with torch.no_grad():
        before = model.predict_next_state(state, sub)
    save_high_level_dynamics_checkpoint(path, model, normalizer, horizon_k=10, dataset_id="x")
    loaded, loaded_norm, cfg = load_high_level_dynamics_checkpoint(path)
    with torch.no_grad():
        after = loaded.predict_next_state(state, sub)
    torch.testing.assert_close(after, before)
    assert cfg["horizon_k"] == 10
    assert cfg["is_high_level_dynamics"] is True
    np.testing.assert_allclose(loaded_norm.mean, normalizer.mean)
