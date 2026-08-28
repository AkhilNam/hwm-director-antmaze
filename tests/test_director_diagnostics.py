"""Director diagnostics: interval records, stuck detector, CSV, support index."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hwm_director.data.state import STATE_DIM
from hwm_director.training.director_diagnostics import (
    DatasetSupportIndex,
    classify_failed_trials,
    detect_stuck,
    longest_no_progress_run,
    summarize_interval_records,
    write_eval_csv,
)
from hwm_director.training.eval_director import (
    INTERVAL_RECORD_KEYS,
    build_interval_record,
)
from tests.helpers import make_transition


def _interval(
    tau: int,
    current: tuple[float, float],
    goal: tuple[float, float],
    subgoal: tuple[float, float],
    worker: tuple[float, float],
    nn: float = 0.1,
    threshold: float = 0.5,
) -> dict:
    return build_interval_record(
        tau=tau,
        current_xy=np.array(current, dtype=np.float64),
        final_goal=np.array(goal, dtype=np.float64),
        subgoal_xy=np.array(subgoal, dtype=np.float64),
        worker_final_xy=np.array(worker, dtype=np.float64),
        subgoal_success_threshold=threshold,
        nearest_dataset_subgoal_distance=nn,
    )


def test_interval_record_has_required_keys() -> None:
    rec = _interval(0, (0.0, 0.0), (4.0, 0.0), (0.4, 0.0), (0.3, 0.0))
    assert set(INTERVAL_RECORD_KEYS) <= set(rec)
    assert rec["tau"] == 0
    assert rec["subgoal_reached"] is True
    np.testing.assert_allclose(rec["goal_progress"], 0.3)
    np.testing.assert_allclose(rec["subgoal_distance"], 0.4)
    np.testing.assert_allclose(rec["worker_subgoal_error"], 0.1)


def test_subgoal_not_reached_when_after_k_error_exceeds_threshold() -> None:
    rec = _interval(
        0, (0.0, 0.0), (4.0, 0.0), (1.0, 0.0), (0.2, 0.0), threshold=0.5
    )
    assert rec["subgoal_reached"] is False
    np.testing.assert_allclose(rec["worker_subgoal_error"], 0.8)


def test_progress_positive_when_closer_to_final_goal() -> None:
    rec = _interval(1, (0.0, 0.0), (5.0, 0.0), (1.0, 0.0), (1.2, 0.0))
    assert rec["goal_progress"] > 0
    rec_away = _interval(1, (3.0, 0.0), (5.0, 0.0), (1.0, 0.0), (1.0, 0.0))
    assert rec_away["goal_progress"] < 0


def test_summarize_interval_records() -> None:
    intervals = [
        _interval(0, (0.0, 0.0), (5.0, 0.0), (1.0, 0.0), (0.9, 0.0), nn=0.2),
        _interval(1, (0.9, 0.0), (5.0, 0.0), (0.4, 0.0), (0.3, 0.0), nn=0.8),
        _interval(2, (0.3, 0.0), (5.0, 0.0), (1.3, 0.0), (1.2, 0.0), nn=1.2),
    ]
    summary = summarize_interval_records(intervals, max_subgoal_distance=2.0)
    assert summary["n_intervals"] == 3
    assert 0.0 <= summary["subgoal_reach_rate"] <= 1.0
    assert summary["fraction_positive_progress"] > 0
    assert summary["fraction_dataset_distance_gt_0_5"] > 0
    assert summary["longest_no_progress_run"] >= 0


def test_detect_stuck_and_longest_no_progress() -> None:
    stuck_intervals = [
        _interval(tau, (0.0, 0.0), (5.0, 0.0), (0.1, 0.0), (0.0, 0.0))
        for tau in range(6)
    ]
    stuck = detect_stuck(stuck_intervals, window=5, distance=0.25)
    assert stuck["stuck"] is True
    assert stuck["first_stuck_tau"] == 0
    assert stuck["stuck_xy"] == [0.0, 0.0]
    assert longest_no_progress_run(stuck_intervals) == 6

    moving = [
        _interval(
            tau,
            (float(tau), 0.0),
            (10.0, 0.0),
            (float(tau + 1), 0.0),
            (float(tau + 1), 0.0),
        )
        for tau in range(6)
    ]
    free = detect_stuck(moving, window=5, distance=0.25)
    assert free["stuck"] is False
    assert longest_no_progress_run(moving) == 0


def test_dataset_support_nearest_future_xy() -> None:
    index = DatasetSupportIndex(
        current_xy=np.array([[0.0, 0.0], [10.0, 10.0]]),
        future_xy=np.array([[0.2, 0.0], [10.5, 10.0]]),
        neighbor_radius=0.5,
        min_neighbors=1,
        max_index=100,
    )
    near = index.nearest_dataset_subgoal_distance(
        np.array([0.0, 0.0]), np.array([0.25, 0.0])
    )
    np.testing.assert_allclose(near, 0.05, atol=1e-6)
    far = index.nearest_dataset_subgoal_distance(
        np.array([0.0, 0.0]), np.array([5.0, 0.0])
    )
    assert far > 4.0


def test_dataset_support_from_transitions() -> None:
    steps = []
    for t in range(12):
        state = np.zeros(STATE_DIM)
        nxt = np.zeros(STATE_DIM)
        state[0] = float(t)
        nxt[0] = float(t + 1)
        steps.append(
            make_transition(episode_id=0, state=state, next_state=nxt, goal=np.array([20.0, 0.0]))
        )
    index = DatasetSupportIndex.from_transitions(steps, horizon_k=10, min_neighbors=1)
    assert index.current_xy.shape[0] > 0
    dist = index.nearest_dataset_subgoal_distance(
        np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert np.isfinite(dist)


def test_write_eval_csv_columns(tmp_path: Path) -> None:
    rec = _interval(0, (0.1, 0.2), (4.0, 1.0), (0.5, 0.2), (0.4, 0.2))
    trial = {
        "success": False,
        "intervals": [rec],
    }
    path = tmp_path / "director_eval.csv"
    write_eval_csv(path, [trial])
    text = path.read_text()
    header = text.splitlines()[0]
    for col in (
        "trial",
        "tau",
        "current_x",
        "current_y",
        "goal_x",
        "goal_y",
        "subgoal_x",
        "subgoal_y",
        "worker_final_x",
        "worker_final_y",
        "subgoal_reached",
        "subgoal_error",
        "goal_distance_before",
        "goal_distance_after",
        "goal_progress",
        "nearest_dataset_subgoal_distance",
        "success",
    ):
        assert col in header
    assert "0.1" in text


def test_classify_failed_trials_counts() -> None:
    failed_worker = {
        "success": False,
        "stuck": False,
        "subgoal_reach_rate": 0.2,
        "mean_progress_to_final": 0.1,
        "fraction_subgoals_closer_to_final_goal": 0.8,
    }
    failed_manager = {
        "success": False,
        "stuck": True,
        "subgoal_reach_rate": 0.9,
        "mean_progress_to_final": -0.05,
        "fraction_subgoals_closer_to_final_goal": 0.2,
    }
    success = {
        "success": True,
        "stuck": False,
        "subgoal_reach_rate": 1.0,
        "mean_progress_to_final": 0.4,
        "fraction_subgoals_closer_to_final_goal": 1.0,
    }
    out = classify_failed_trials(
        [failed_worker, failed_manager, success],
        multi_horizon={
            "by_horizon": {
                1: {"mean_xy_error": 0.1},
                5: {"mean_xy_error": 0.8},
            }
        },
    )
    assert out["n_failed"] == 2
    assert out["n_failed_stuck"] == 1
    assert out["n_failed_low_subgoal_reach"] == 1
    assert out["n_failed_weak_final_progress"] == 1
    assert "compounds" in out["model_rollout_note"]


def test_analyze_wall_region_counts_entry_and_direction() -> None:
    from hwm_director.training.director_diagnostics import analyze_wall_region

    in_region = _interval(0, (-2.9, 4.1), (-4.0, 4.0), (-3.4, 4.1), (-3.3, 4.1))
    outside = _interval(0, (0.0, 0.0), (-4.0, 4.0), (0.4, 0.0), (0.3, 0.0))
    entered_fail = {"success": False, "intervals": [in_region]}
    entered_ok = {"success": True, "intervals": [in_region]}
    never = {"success": False, "intervals": [outside]}
    stats = analyze_wall_region(
        [entered_fail, entered_ok, never],
        center=(-2.9, 4.1),
        radius=1.0,
    )
    assert stats["n_trials_entered"] == 2
    assert stats["n_entered_then_success"] == 1
    assert stats["mean_high_level_steps_in_region"] == 1.0
    assert stats["mean_subgoal_direction"][0] < 0
