"""Diagnostics for end-to-end Director failures (no training changes)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from hwm_director.data.manager_dataset import (
    DirectorManagerDataset,
    manager_arrays_from_transitions,
)
from hwm_director.data.normalization import StateNormalizer
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.director_manager import (
    DEFAULT_MAX_SUBGOAL_DISTANCE,
    DirectorManager,
)
from hwm_director.training.train_manager import (
    _predict_subgoals_raw,
    xy_euclidean_error,
    xy_mse,
)

DEFAULT_SUBGOAL_SUCCESS = 0.5
DEFAULT_STUCK_WINDOW = 5
DEFAULT_STUCK_DISTANCE = 0.25
DEFAULT_NEAR_MAX_FRACTION = 0.9
DEFAULT_SUPPORT_RADIUS = 0.5
DEFAULT_SUPPORT_MIN_NEIGHBORS = 8
DEFAULT_SUPPORT_INDEX_CAP = 20000

GOAL_DISTANCE_BINS = (0.0, 2.0, 4.0, 8.0, np.inf)
DISPLACEMENT_BINS = (0.0, 0.5, 1.0, 1.5, 2.5, np.inf)


class DatasetSupportIndex:
    """Nearest observed K-step future x/y among states near the query x/y.

    Diagnostic only. Not used as a training regularizer.
    """

    def __init__(
        self,
        current_xy: np.ndarray,
        future_xy: np.ndarray,
        neighbor_radius: float = DEFAULT_SUPPORT_RADIUS,
        min_neighbors: int = DEFAULT_SUPPORT_MIN_NEIGHBORS,
        max_index: int = DEFAULT_SUPPORT_INDEX_CAP,
        seed: int = 0,
    ) -> None:
        current_xy = np.asarray(current_xy, dtype=np.float64)
        future_xy = np.asarray(future_xy, dtype=np.float64)
        if current_xy.shape[0] > max_index:
            rng = np.random.default_rng(seed)
            picks = rng.choice(current_xy.shape[0], size=max_index, replace=False)
            current_xy = current_xy[picks]
            future_xy = future_xy[picks]
        self.current_xy = current_xy
        self.future_xy = future_xy
        self.neighbor_radius = float(neighbor_radius)
        self.min_neighbors = int(min_neighbors)

    @classmethod
    def from_transitions(
        cls,
        transitions: Sequence[Transition],
        horizon_k: int = DEFAULT_HORIZON_K,
        **kwargs: Any,
    ) -> DatasetSupportIndex:
        states, _goals, targets, _episode_ids, _times = manager_arrays_from_transitions(
            transitions, horizon_k
        )
        return cls(states[:, :2], targets, **kwargs)

    def nearest_dataset_subgoal_distance(
        self, current_xy: np.ndarray, subgoal_xy: np.ndarray
    ) -> float:
        if self.current_xy.shape[0] == 0:
            return float("nan")
        current_xy = np.asarray(current_xy, dtype=np.float64).reshape(2)
        subgoal_xy = np.asarray(subgoal_xy, dtype=np.float64).reshape(2)
        d_cur = np.linalg.norm(self.current_xy - current_xy, axis=1)
        nearby = np.flatnonzero(d_cur <= self.neighbor_radius)
        if nearby.size < self.min_neighbors:
            n = min(self.min_neighbors, int(d_cur.size))
            nearby = np.argpartition(d_cur, n - 1)[:n]
        return float(
            np.min(np.linalg.norm(self.future_xy[nearby] - subgoal_xy, axis=1))
        )


def detect_stuck(
    intervals: Sequence[dict],
    window: int = DEFAULT_STUCK_WINDOW,
    distance: float = DEFAULT_STUCK_DISTANCE,
) -> dict:
    """First window of ``window`` high-level intervals with net motion < ``distance``."""
    if window < 1 or len(intervals) < window:
        return {
            "stuck": False,
            "first_stuck_tau": None,
            "stuck_xy": None,
        }
    for start in range(0, len(intervals) - window + 1):
        origin = np.asarray(intervals[start]["current_xy"], dtype=np.float64)
        end = np.asarray(
            intervals[start + window - 1]["worker_final_xy"], dtype=np.float64
        )
        if float(np.linalg.norm(end - origin)) < float(distance):
            return {
                "stuck": True,
                "first_stuck_tau": int(intervals[start]["tau"]),
                "stuck_xy": [float(origin[0]), float(origin[1])],
            }
    return {"stuck": False, "first_stuck_tau": None, "stuck_xy": None}


def longest_no_progress_run(intervals: Sequence[dict]) -> int:
    longest = 0
    run = 0
    for rec in intervals:
        if float(rec.get("goal_progress", 0.0)) <= 0.0:
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return longest


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _corr_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(mask)) < 3:
        return float("nan"), float("nan")
    aa = a[mask]
    bb = b[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(aa, bb)[0, 1])
    spearman = float(np.corrcoef(_ranks(aa), _ranks(bb))[0, 1])
    return pearson, spearman


def summarize_interval_records(
    intervals: Sequence[dict],
    *,
    max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
    near_max_fraction: float = DEFAULT_NEAR_MAX_FRACTION,
) -> dict:
    if not intervals:
        nan = float("nan")
        return {
            "n_intervals": 0,
            "subgoal_reach_rate": nan,
            "mean_worker_subgoal_error": nan,
            "median_worker_subgoal_error": nan,
            "mean_progress_to_final": nan,
            "median_progress_to_final": nan,
            "fraction_positive_progress": nan,
            "fraction_negative_progress": nan,
            "longest_no_progress_run": 0,
            "mean_subgoal_displacement": nan,
            "fraction_subgoals_closer_to_final_goal": nan,
            "fraction_subgoals_farther_from_final_goal": nan,
            "fraction_near_max_subgoal_distance": nan,
            "mean_nearest_dataset_subgoal_distance": nan,
            "median_nearest_dataset_subgoal_distance": nan,
            "fraction_dataset_distance_gt_0_5": nan,
            "fraction_dataset_distance_gt_1_0": nan,
            "mean_source_state_distance": nan,
            "mean_source_xy_distance": nan,
            "mean_predicted_subgoal_error": nan,
            "mean_qh_score": nan,
            "mean_qh_score_normalized": nan,
            "mean_reach_error_normalized": nan,
            "mean_combined_score": nan,
            "fallback_rate": nan,
            "predicted_vs_actual_pearson": nan,
            "predicted_vs_actual_spearman": nan,
        }
    reached = np.asarray([bool(r["subgoal_reached"]) for r in intervals])
    sub_err = _finite([r["worker_subgoal_error"] for r in intervals])
    progress = _finite([r["goal_progress"] for r in intervals])
    disp = _finite([r["subgoal_distance"] for r in intervals])
    toward = np.asarray(
        [
            float(r["final_goal_distance_before"]) > float(r["subgoal_to_final_distance"])
            for r in intervals
        ]
    )
    away = np.asarray(
        [
            float(r["subgoal_to_final_distance"]) > float(r["final_goal_distance_before"])
            for r in intervals
        ]
    )
    near_max = disp >= (near_max_fraction * float(max_subgoal_distance))
    nn = _finite([r["nearest_dataset_subgoal_distance"] for r in intervals])
    src_state = _finite([r.get("source_state_distance", float("nan")) for r in intervals])
    src_xy = _finite([r.get("source_xy_distance", float("nan")) for r in intervals])
    pred_err = _finite([r.get("predicted_subgoal_error", float("nan")) for r in intervals])
    qh = _finite([r.get("qh_score", float("nan")) for r in intervals])
    qh_n = _finite([r.get("qh_score_normalized", float("nan")) for r in intervals])
    reach_n = _finite(
        [r.get("reach_error_normalized", float("nan")) for r in intervals]
    )
    combined = _finite([r.get("combined_score", float("nan")) for r in intervals])
    fallback = np.asarray([bool(r.get("used_fallback", False)) for r in intervals])
    pred_all = np.asarray(
        [r.get("predicted_subgoal_error", float("nan")) for r in intervals],
        dtype=np.float64,
    )
    act_all = np.asarray(
        [r.get("worker_subgoal_error", float("nan")) for r in intervals],
        dtype=np.float64,
    )
    pearson, spearman = _corr_pair(pred_all, act_all)
    return {
        "n_intervals": len(intervals),
        "subgoal_reach_rate": float(np.mean(reached)),
        "mean_worker_subgoal_error": float(np.mean(sub_err)) if sub_err.size else float("nan"),
        "median_worker_subgoal_error": float(np.median(sub_err)) if sub_err.size else float("nan"),
        "mean_progress_to_final": float(np.mean(progress)) if progress.size else float("nan"),
        "median_progress_to_final": float(np.median(progress)) if progress.size else float("nan"),
        "fraction_positive_progress": float(np.mean(progress > 0.0)) if progress.size else float("nan"),
        "fraction_negative_progress": float(np.mean(progress < 0.0)) if progress.size else float("nan"),
        "longest_no_progress_run": longest_no_progress_run(intervals),
        "mean_subgoal_displacement": float(np.mean(disp)) if disp.size else float("nan"),
        "fraction_subgoals_closer_to_final_goal": float(np.mean(toward)),
        "fraction_subgoals_farther_from_final_goal": float(np.mean(away)),
        "fraction_near_max_subgoal_distance": float(np.mean(near_max)) if disp.size else float("nan"),
        "mean_nearest_dataset_subgoal_distance": float(np.mean(nn)) if nn.size else float("nan"),
        "median_nearest_dataset_subgoal_distance": float(np.median(nn)) if nn.size else float("nan"),
        "fraction_dataset_distance_gt_0_5": float(np.mean(nn > 0.5)) if nn.size else float("nan"),
        "fraction_dataset_distance_gt_1_0": float(np.mean(nn > 1.0)) if nn.size else float("nan"),
        "mean_source_state_distance": float(np.mean(src_state)) if src_state.size else float("nan"),
        "mean_source_xy_distance": float(np.mean(src_xy)) if src_xy.size else float("nan"),
        "mean_predicted_subgoal_error": float(np.mean(pred_err)) if pred_err.size else float("nan"),
        "mean_qh_score": float(np.mean(qh)) if qh.size else float("nan"),
        "mean_qh_score_normalized": float(np.mean(qh_n)) if qh_n.size else float("nan"),
        "mean_reach_error_normalized": (
            float(np.mean(reach_n)) if reach_n.size else float("nan")
        ),
        "mean_combined_score": (
            float(np.mean(combined)) if combined.size else float("nan")
        ),
        "fallback_rate": float(np.mean(fallback)),
        "predicted_vs_actual_pearson": pearson,
        "predicted_vs_actual_spearman": spearman,
    }


def classify_failed_trials(
    trials: Sequence[dict],
    *,
    multi_horizon: dict | None = None,
) -> dict:
    """Transparent heuristic counts over failed trials. Not a learned classifier."""
    failed = [t for t in trials if not t.get("success")]
    n_failed = len(failed)
    n_stuck = sum(1 for t in failed if t.get("stuck"))
    n_worker = 0
    n_manager = 0
    for trial in failed:
        reach = float(trial.get("subgoal_reach_rate", float("nan")))
        prog = float(trial.get("mean_progress_to_final", float("nan")))
        toward = float(trial.get("fraction_subgoals_closer_to_final_goal", float("nan")))
        if np.isfinite(reach) and reach < 0.5:
            n_worker += 1
        if (np.isfinite(prog) and prog <= 0.0) or (
            np.isfinite(toward) and toward < 0.5
        ):
            n_manager += 1
    model_note = ""
    if multi_horizon and multi_horizon.get("by_horizon"):
        by_h = multi_horizon["by_horizon"]
        e1 = by_h.get(1, {}).get("mean_xy_error", float("nan"))
        e5 = by_h.get(5, {}).get("mean_xy_error", float("nan"))
        compounds = bool(
            np.isfinite(e1) and np.isfinite(e5) and (e5 > max(0.5, 2.0 * e1))
        )
        model_note = (
            f"1K mean xy error={e1:.3f} m, 5K={e5:.3f} m; "
            f"{'compounds' if compounds else 'does not strongly compound'} "
            "relative to the 0.5 m / 2x heuristic."
        )
    return {
        "n_failed": n_failed,
        "n_failed_stuck": n_stuck,
        "n_failed_low_subgoal_reach": n_worker,
        "n_failed_weak_final_progress": n_manager,
        "stuck_rate_among_failed": (
            float(n_stuck / n_failed) if n_failed else float("nan")
        ),
        "model_rollout_note": model_note,
    }


DEFAULT_WALL_REGION_XY = (-2.9, 4.1)
DEFAULT_WALL_REGION_RADIUS = 1.0


def analyze_wall_region(
    trials: Sequence[dict],
    *,
    center: Sequence[float] = DEFAULT_WALL_REGION_XY,
    radius: float = DEFAULT_WALL_REGION_RADIUS,
) -> dict:
    """Eval-only U-wall neighborhood stats. Not used by ``pi_H``."""
    center_xy = np.asarray(center, dtype=np.float64).reshape(2)
    radius = float(radius)
    n_entered = 0
    n_entered_then_success = 0
    steps_in: list[int] = []
    directions: list[np.ndarray] = []
    for trial in trials:
        entered = False
        n_here = 0
        for rec in trial.get("intervals", []):
            cur = np.asarray(rec["current_xy"], dtype=np.float64)
            if float(np.linalg.norm(cur - center_xy)) <= radius:
                entered = True
                n_here += 1
                delta = np.asarray(rec["subgoal_xy"], dtype=np.float64) - cur
                nrm = float(np.linalg.norm(delta))
                if nrm > 1e-8:
                    directions.append(delta / nrm)
        if entered:
            n_entered += 1
            steps_in.append(n_here)
            if trial.get("success"):
                n_entered_then_success += 1
    if directions:
        mean_dir = np.mean(np.stack(directions, axis=0), axis=0)
        mean_subgoal_direction = [float(mean_dir[0]), float(mean_dir[1])]
    else:
        mean_subgoal_direction = [float("nan"), float("nan")]
    return {
        "center": [float(center_xy[0]), float(center_xy[1])],
        "radius": radius,
        "n_trials_entered": n_entered,
        "n_entered_then_success": n_entered_then_success,
        "mean_high_level_steps_in_region": (
            float(np.mean(steps_in)) if steps_in else float("nan")
        ),
        "mean_subgoal_direction": mean_subgoal_direction,
        "n_direction_samples": len(directions),
    }


def format_manager_comparison(
    bc_eval: dict | None,
    value_eval: dict | None,
    *,
    value_train: dict | None = None,
) -> str:
    """Side-by-side Director-BC vs Director-Value summary."""
    lines = ["=== Director-BC vs Director-Value ==="]
    if value_train:
        lines.extend(
            [
                f"  Q_H train MSE: {_fmt(value_train.get('train_mse', float('nan')))}",
                f"  Q_H val MSE: {_fmt(value_train.get('val_mse', float('nan')))}",
                f"  Q_H is not f_H; Director f_H remains (f_L, pi_L)^K",
            ]
        )
    keys = [
        ("success_rate", "success rate"),
        ("mean_final_distance", "mean final distance (m)"),
        ("subgoal_reach_rate", "subgoal reach rate"),
        ("stuck_rate_among_failed", "stuck rate among failed"),
        ("mean_progress_to_final", "mean progress per HL step (m)"),
        ("median_progress_to_final", "median progress (m)"),
        ("fraction_positive_progress", "fraction positive progress"),
        ("mean_subgoal_displacement", "mean subgoal displacement (m)"),
        ("mean_nearest_dataset_subgoal_distance", "dataset-support distance mean (m)"),
        ("median_nearest_dataset_subgoal_distance", "dataset-support distance median (m)"),
    ]

    def _get(block: dict | None, key: str) -> float:
        if not block:
            return float("nan")
        return float(block.get(key, float("nan")))

    for key, label in keys:
        lines.append(
            f"  {label}:  BC={_fmt(_get(bc_eval, key))}  "
            f"Value={_fmt(_get(value_eval, key))}"
        )
    for name, block in (("BC", bc_eval), ("Value", value_eval)):
        wall = (block or {}).get("wall_region")
        if not wall:
            continue
        lines.append(
            f"  U-wall {name}: entered={wall.get('n_trials_entered')} "
            f"later_success={wall.get('n_entered_then_success')} "
            f"mean_steps_in={_fmt(wall.get('mean_high_level_steps_in_region', float('nan')), 2)} "
            f"mean_subgoal_dir={wall.get('mean_subgoal_direction')}"
        )
    return "\n".join(lines)


def analyze_failed_stuck(trials: Sequence[dict]) -> dict:
    """Reach / fallback stats before vs after first stuck tau (failed trials)."""
    failed = [t for t in trials if not t.get("success")]
    n_failed = len(failed)
    n_stuck = sum(1 for t in failed if t.get("stuck"))
    before_reach: list[float] = []
    after_reach: list[float] = []
    fallback_near: list[float] = []
    src_near: list[float] = []
    first_taus: list[float] = []
    qh_before: list[float] = []
    qh_after: list[float] = []
    pen_before: list[float] = []
    pen_after: list[float] = []
    for trial in failed:
        intervals = trial.get("intervals") or []
        if not intervals:
            continue
        stuck_tau = trial.get("first_stuck_tau")
        if stuck_tau is None:
            continue
        first_taus.append(float(stuck_tau))
        before = [r for r in intervals if int(r["tau"]) < int(stuck_tau)]
        after = [r for r in intervals if int(r["tau"]) >= int(stuck_tau)]
        if before:
            before_reach.append(float(np.mean([bool(r["subgoal_reached"]) for r in before])))
            qb = _finite([r.get("qh_score", float("nan")) for r in before])
            if qb.size:
                qh_before.append(float(np.mean(qb)))
            pb = _finite(
                [r.get("reach_error_normalized", float("nan")) for r in before]
            )
            if pb.size:
                pen_before.append(float(np.mean(pb)))
        if after:
            after_reach.append(float(np.mean([bool(r["subgoal_reached"]) for r in after])))
            fallback_near.append(
                float(np.mean([bool(r.get("used_fallback", False)) for r in after]))
            )
            src = _finite([r.get("source_state_distance", float("nan")) for r in after])
            if src.size:
                src_near.append(float(np.mean(src)))
            qa = _finite([r.get("qh_score", float("nan")) for r in after])
            if qa.size:
                qh_after.append(float(np.mean(qa)))
            pa = _finite(
                [r.get("reach_error_normalized", float("nan")) for r in after]
            )
            if pa.size:
                pen_after.append(float(np.mean(pa)))
    return {
        "n_failed": n_failed,
        "n_failed_stuck": n_stuck,
        "stuck_rate_among_failed": (
            float(n_stuck / n_failed) if n_failed else float("nan")
        ),
        "mean_first_stuck_tau": (
            float(np.mean(first_taus)) if first_taus else float("nan")
        ),
        "mean_reach_before_stuck": (
            float(np.mean(before_reach)) if before_reach else float("nan")
        ),
        "mean_reach_after_stuck": (
            float(np.mean(after_reach)) if after_reach else float("nan")
        ),
        "mean_qh_before_stuck": (
            float(np.mean(qh_before)) if qh_before else float("nan")
        ),
        "mean_qh_after_stuck": (
            float(np.mean(qh_after)) if qh_after else float("nan")
        ),
        "mean_reach_penalty_before_stuck": (
            float(np.mean(pen_before)) if pen_before else float("nan")
        ),
        "mean_reach_penalty_after_stuck": (
            float(np.mean(pen_after)) if pen_after else float("nan")
        ),
        "mean_fallback_after_stuck": (
            float(np.mean(fallback_near)) if fallback_near else float("nan")
        ),
        "mean_source_state_distance_after_stuck": (
            float(np.mean(src_near)) if src_near else float("nan")
        ),
    }


def format_candidate_quality(eval_metrics: dict) -> str:
    """Distributions used to compare retrieval modes. Eval-only."""
    intervals = [
        rec
        for trial in eval_metrics.get("trials") or []
        for rec in trial.get("intervals") or []
    ]
    lines = ["-- candidate quality --"]
    fields = (
        ("source_xy_distance", "source xy distance (m)"),
        ("source_state_distance", "source full-state distance (norm)"),
        ("subgoal_distance", "candidate displacement (m)"),
        ("predicted_subgoal_error", "predicted subgoal error (m)"),
        ("worker_subgoal_error", "actual subgoal error after K (m)"),
        ("qh_score", "Q_H score"),
        ("qh_score_normalized", "normalized Q_H"),
        ("reach_error_normalized", "normalized reach error"),
        ("combined_score", "combined score"),
    )
    if not intervals:
        lines.append("  (no intervals)")
        return "\n".join(lines)
    for key, label in fields:
        arr = _finite([r.get(key, float("nan")) for r in intervals])
        if arr.size == 0:
            lines.append(f"  {label}: n=0")
            continue
        lines.append(
            f"  {label}: n={arr.size} mean={_fmt(float(np.mean(arr)), 4)} "
            f"p50={_fmt(float(np.median(arr)), 4)} "
            f"p90={_fmt(float(np.percentile(arr, 90)), 4)}"
        )
    lines.append(
        f"  predicted vs actual Pearson={_fmt(eval_metrics.get('predicted_vs_actual_pearson', float('nan')), 3)} "
        f"Spearman={_fmt(eval_metrics.get('predicted_vs_actual_spearman', float('nan')), 3)}"
    )
    lines.append(
        f"  fallback_rate={_fmt(eval_metrics.get('fallback_rate', float('nan')), 3)}"
    )
    return "\n".join(lines)


def format_fairness_summary(*, director, hwm, lambda_reach: float, extra: dict | None = None) -> str:
    """Print what Director and HWM share vs what differs."""
    lines = [
        "=== Director vs HWM fairness ===",
        "Shared:",
        f"  pi_L identity: {director.worker is hwm.worker}",
        f"  f_L identity: {director.dynamics is hwm.dynamics}",
        f"  manager identity: {director.manager is hwm.manager}",
        f"  K: Director={director.horizon_k} HWM={hwm.horizon_k}",
        f"  lambda_reach: {lambda_reach}",
        f"  Director explicit_f_h is None: {getattr(director, 'explicit_f_h', 'missing') is None}",
        f"  HWM explicit_f_h is not None: {getattr(hwm, 'explicit_f_h', None) is not None}",
        "Differ:",
        "  Director f_H = (f_L, pi_L)^K",
        "  HWM f_H = explicit learned f_H_phi",
        "  SoftReach predicted_subgoal_error uses that f_H",
    ]
    if extra:
        for key, value in extra.items():
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def format_ablation_table(rows: Sequence[dict]) -> str:
    lines = [
        "=== candidate-retrieval / soft-reach ablation ===",
        "manager                   success  final_d  reach  stuck  fallback  "
        "pred_err  actual_err  qh  wall_steps  progress",
    ]
    for row in rows:
        wall = row.get("wall_region") or {}
        lines.append(
            f"{row.get('manager_name', '?'):<26} "
            f"{_fmt(row.get('success_rate', float('nan')), 3)}  "
            f"{_fmt(row.get('mean_final_distance', float('nan')), 3)}  "
            f"{_fmt(row.get('subgoal_reach_rate', float('nan')), 3)}  "
            f"{_fmt(row.get('stuck_rate_among_failed', float('nan')), 3)}  "
            f"{_fmt(row.get('fallback_rate', float('nan')), 3)}  "
            f"{_fmt(row.get('mean_predicted_subgoal_error', float('nan')), 3)}  "
            f"{_fmt(row.get('mean_worker_subgoal_error', float('nan')), 3)}  "
            f"{_fmt(row.get('mean_qh_score', float('nan')), 3)}  "
            f"{_fmt(wall.get('mean_high_level_steps_in_region', float('nan')), 2)}  "
            f"{_fmt(row.get('mean_progress_to_final', float('nan')), 3)}"
        )
    return "\n".join(lines)


def format_pareto_table(rows: Sequence[dict]) -> str:
    """Success vs subgoal-reach tradeoff. Eval-only."""
    lines = [
        "=== Pareto: success vs subgoal reach ===",
        "method/lambda             success  reach  final_d  wall_steps  pred_err  actual_err",
    ]
    for row in rows:
        wall = row.get("wall_region") or {}
        lines.append(
            f"{row.get('manager_name', '?'):<26} "
            f"{_fmt(row.get('success_rate', float('nan')), 3)}  "
            f"{_fmt(row.get('subgoal_reach_rate', float('nan')), 3)}  "
            f"{_fmt(row.get('mean_final_distance', float('nan')), 3)}  "
            f"{_fmt(wall.get('mean_high_level_steps_in_region', float('nan')), 2)}  "
            f"{_fmt(row.get('mean_predicted_subgoal_error', float('nan')), 3)}  "
            f"{_fmt(row.get('mean_worker_subgoal_error', float('nan')), 3)}"
        )
    finite = [r for r in rows if np.isfinite(r.get("success_rate", float("nan")))]
    if not finite:
        return "\n".join(lines)
    best_success = max(finite, key=lambda r: float(r.get("success_rate", float("-inf"))))
    best_reach = max(
        finite, key=lambda r: float(r.get("subgoal_reach_rate", float("-inf")))
    )
    bc = next((r for r in finite if r.get("manager_name") == "BC"), None)
    bc_success = float(bc["success_rate"]) if bc is not None else 0.0
    above_bc = [
        r
        for r in finite
        if r.get("manager_name") != "BC"
        and float(r.get("success_rate", float("nan"))) > bc_success
    ]
    if above_bc:
        balanced = max(
            above_bc, key=lambda r: float(r.get("subgoal_reach_rate", float("-inf")))
        )
        balanced_note = (
            f"among configs with success > BC ({_fmt(bc_success, 3)}), "
            f"highest reach: {balanced.get('manager_name')} "
            f"(success={_fmt(balanced.get('success_rate', float('nan')), 3)}, "
            f"reach={_fmt(balanced.get('subgoal_reach_rate', float('nan')), 3)})"
        )
    else:
        balanced_note = "no non-BC config has success above the BC baseline"
    lines.extend(
        [
            f"highest success: {best_success.get('manager_name')} "
            f"({_fmt(best_success.get('success_rate', float('nan')), 3)})",
            f"highest reach: {best_reach.get('manager_name')} "
            f"({_fmt(best_reach.get('subgoal_reach_rate', float('nan')), 3)})",
            f"best balanced: {balanced_note}",
        ]
    )
    return "\n".join(lines)


def write_eval_csv(path: str | Path, trials: Sequence[dict]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "used_fallback",
        "source_state_distance",
        "source_xy_distance",
        "predicted_subgoal_error",
        "qh_score",
        "qh_score_normalized",
        "reach_error_normalized",
        "combined_score",
        "reachability_score_weight",
        "qh_max",
        "qh_min",
        "qh_mean",
        "reach_error_max",
        "reach_error_min",
        "reach_error_mean",
    ]
    with dest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial_i, trial in enumerate(trials):
            success = bool(trial.get("success"))
            for rec in trial.get("intervals", []):
                cur = rec["current_xy"]
                goal = rec["final_goal"]
                sub = rec["subgoal_xy"]
                wxy = rec["worker_final_xy"]
                writer.writerow(
                    {
                        "trial": trial_i,
                        "tau": rec["tau"],
                        "current_x": cur[0],
                        "current_y": cur[1],
                        "goal_x": goal[0],
                        "goal_y": goal[1],
                        "subgoal_x": sub[0],
                        "subgoal_y": sub[1],
                        "worker_final_x": wxy[0],
                        "worker_final_y": wxy[1],
                        "subgoal_reached": int(bool(rec["subgoal_reached"])),
                        "subgoal_error": rec["worker_subgoal_error"],
                        "goal_distance_before": rec["final_goal_distance_before"],
                        "goal_distance_after": rec["final_goal_distance_after"],
                        "goal_progress": rec["goal_progress"],
                        "nearest_dataset_subgoal_distance": rec[
                            "nearest_dataset_subgoal_distance"
                        ],
                        "success": int(success),
                        "used_fallback": int(bool(rec.get("used_fallback", False))),
                        "source_state_distance": rec.get(
                            "source_state_distance", ""
                        ),
                        "source_xy_distance": rec.get("source_xy_distance", ""),
                        "predicted_subgoal_error": rec.get(
                            "predicted_subgoal_error", ""
                        ),
                        "qh_score": rec.get("qh_score", ""),
                        "qh_score_normalized": rec.get("qh_score_normalized", ""),
                        "reach_error_normalized": rec.get(
                            "reach_error_normalized", ""
                        ),
                        "combined_score": rec.get("combined_score", ""),
                        "reachability_score_weight": rec.get(
                            "reachability_score_weight", ""
                        ),
                        "qh_max": rec.get("qh_max", ""),
                        "qh_min": rec.get("qh_min", ""),
                        "qh_mean": rec.get("qh_mean", ""),
                        "reach_error_max": rec.get("reach_error_max", ""),
                        "reach_error_min": rec.get("reach_error_min", ""),
                        "reach_error_mean": rec.get("reach_error_mean", ""),
                    }
                )


def _bin_means(
    values: np.ndarray, errors: np.ndarray, edges: Sequence[float]
) -> list[dict]:
    out: list[dict] = []
    edges = list(edges)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values < hi)
        n = int(np.sum(mask))
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "mean_euclidean": (
                    float(np.mean(errors[mask])) if n else float("nan")
                ),
            }
        )
    return out


def evaluate_manager_bc_error(
    model: DirectorManager,
    normalizer: StateNormalizer,
    train_transitions: Sequence[Transition],
    val_transitions: Sequence[Transition],
    horizon_k: int = DEFAULT_HORIZON_K,
    batch_size: int = 4096,
) -> dict:
    """Predicted vs recorded ``s_{t+K}[:2]`` on train and val (raw meters)."""
    train_ds = DirectorManagerDataset(
        train_transitions, horizon_k=horizon_k, normalizer=normalizer
    )
    val_ds = DirectorManagerDataset(
        val_transitions, horizon_k=horizon_k, normalizer=normalizer
    )
    train_pred = _predict_subgoals_raw(
        model, train_ds, normalizer, batch_size, clamp=False
    )
    val_pred = _predict_subgoals_raw(
        model, val_ds, normalizer, batch_size, clamp=False
    )
    result = {
        "n_train_examples": len(train_ds),
        "n_val_examples": len(val_ds),
        "train_mse": (
            xy_mse(train_pred, train_ds.raw_target_subgoals)
            if len(train_ds)
            else float("nan")
        ),
        "val_mse": (
            xy_mse(val_pred, val_ds.raw_target_subgoals)
            if len(val_ds)
            else float("nan")
        ),
        "train_euclidean": (
            xy_euclidean_error(train_pred, train_ds.raw_target_subgoals)
            if len(train_ds)
            else float("nan")
        ),
        "val_euclidean": (
            xy_euclidean_error(val_pred, val_ds.raw_target_subgoals)
            if len(val_ds)
            else float("nan")
        ),
        "val_by_goal_distance": [],
        "val_by_displacement": [],
    }
    if len(val_ds) == 0:
        return result
    err = np.linalg.norm(val_pred - val_ds.raw_target_subgoals, axis=1)
    goal_d = np.linalg.norm(
        val_ds.raw_states[:, :2] - val_ds.raw_final_goals, axis=1
    )
    disp = np.linalg.norm(
        val_ds.raw_target_subgoals - val_ds.raw_states[:, :2], axis=1
    )
    result["val_by_goal_distance"] = _bin_means(goal_d, err, GOAL_DISTANCE_BINS)
    result["val_by_displacement"] = _bin_means(disp, err, DISPLACEMENT_BINS)
    return result


def _fmt(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _fmt_bin(bin_row: dict) -> str:
    hi = bin_row["hi"]
    hi_s = "inf" if not np.isfinite(hi) else f"{hi:g}"
    return (
        f"[{bin_row['lo']:g}, {hi_s}) n={bin_row['n']} "
        f"mean={_fmt(bin_row['mean_euclidean'], 4)} m"
    )


def format_director_diagnostic_report(
    *,
    env_eval: dict | None = None,
    manager_bc: dict | None = None,
    implicit_1k: dict | None = None,
    multi_horizon: dict | None = None,
) -> str:
    """Human-readable diagnostic summary. No algorithm changes."""
    lines: list[str] = ["=== Director diagnostic report ==="]
    if env_eval:
        lines.extend(
            [
                "",
                "-- end-to-end --",
                f"  n_trials: {env_eval.get('n_trials', 0)}",
                f"  success_rate: {_fmt(env_eval.get('success_rate', float('nan')))}",
                f"  mean_final_distance: {_fmt(env_eval.get('mean_final_distance', float('nan')))} m",
                f"  mean_initial_distance: {_fmt(env_eval.get('mean_initial_distance', float('nan')))} m",
                f"  subgoal_reach_rate: {_fmt(env_eval.get('subgoal_reach_rate', float('nan')))}",
                f"  mean worker->subgoal after K: {_fmt(env_eval.get('mean_worker_subgoal_error', float('nan')))} m",
                f"  median worker->subgoal after K: {_fmt(env_eval.get('median_worker_subgoal_error', float('nan')))} m",
                f"  mean progress to g* per high-level step: {_fmt(env_eval.get('mean_progress_to_final', float('nan')))} m",
                f"  median progress: {_fmt(env_eval.get('median_progress_to_final', float('nan')))} m",
                f"  fraction positive progress: {_fmt(env_eval.get('fraction_positive_progress', float('nan')))}",
                f"  fraction negative progress: {_fmt(env_eval.get('fraction_negative_progress', float('nan')))}",
                f"  max no-progress run: {env_eval.get('max_longest_no_progress_run', 0)}",
                f"  mean subgoal displacement: {_fmt(env_eval.get('mean_subgoal_displacement', float('nan')))} m",
                f"  fraction subgoals closer to g*: {_fmt(env_eval.get('fraction_subgoals_closer_to_final_goal', float('nan')))}",
                f"  fraction subgoals farther from g*: {_fmt(env_eval.get('fraction_subgoals_farther_from_final_goal', float('nan')))}",
                f"  fraction near max subgoal distance: {_fmt(env_eval.get('fraction_near_max_subgoal_distance', float('nan')))}",
                f"  nearest_dataset_subgoal_distance mean: {_fmt(env_eval.get('mean_nearest_dataset_subgoal_distance', float('nan')))} m",
                f"  nearest_dataset_subgoal_distance median: {_fmt(env_eval.get('median_nearest_dataset_subgoal_distance', float('nan')))} m",
                f"  fraction > 0.5 m: {_fmt(env_eval.get('fraction_dataset_distance_gt_0_5', float('nan')))}",
                f"  fraction > 1.0 m: {_fmt(env_eval.get('fraction_dataset_distance_gt_1_0', float('nan')))}",
                f"  stuck_rate among failed: {_fmt(env_eval.get('stuck_rate_among_failed', float('nan')))}",
            ]
        )
        locs = env_eval.get("stuck_locations") or []
        if locs:
            lines.append("  stuck locations (failed trials):")
            for loc in locs[:20]:
                xy = loc.get("stuck_xy")
                lines.append(
                    f"    trial={loc.get('trial')} tau={loc.get('first_stuck_tau')} "
                    f"xy={xy}"
                )
            if len(locs) > 20:
                lines.append(f"    ... {len(locs) - 20} more")
        fail = env_eval.get("failure_summary") or {}
        if fail:
            lines.extend(
                [
                    "",
                    "-- failure heuristics (counts, not a classifier) --",
                    f"  n_failed: {fail.get('n_failed', 0)}",
                    f"  A worker (reach_rate < 0.5): {fail.get('n_failed_low_subgoal_reach', 0)}",
                    f"  B manager (mean progress<=0 or <50% subgoals toward g*): {fail.get('n_failed_weak_final_progress', 0)}",
                    f"  D stuck (window={env_eval.get('stuck_window')}, "
                    f"{env_eval.get('stuck_distance')} m): {fail.get('n_failed_stuck', 0)}",
                    f"  C model-rollout: {fail.get('model_rollout_note') or 'not evaluated'}",
                ]
            )
    if manager_bc:
        lines.extend(
            [
                "",
                "-- manager BC vs recorded s_{t+K}[:2] --",
                f"  train Euclidean: {_fmt(manager_bc.get('train_euclidean', float('nan')))} m",
                f"  val Euclidean: {_fmt(manager_bc.get('val_euclidean', float('nan')))} m",
                f"  train MSE: {_fmt(manager_bc.get('train_mse', float('nan')))}",
                f"  val MSE: {_fmt(manager_bc.get('val_mse', float('nan')))}",
            ]
        )
        if manager_bc.get("val_by_goal_distance"):
            lines.append("  val error by distance to g*:")
            for row in manager_bc["val_by_goal_distance"]:
                lines.append(f"    {_fmt_bin(row)}")
        if manager_bc.get("val_by_displacement"):
            lines.append("  val error by recorded subgoal displacement:")
            for row in manager_bc["val_by_displacement"]:
                lines.append(f"    {_fmt_bin(row)}")
    if implicit_1k and not implicit_1k.get("skipped"):
        lines.extend(
            [
                "",
                "-- implicit f_H vs real worker, fixed recorded subgoal, 1K --",
                f"  mean x/y error: {_fmt(implicit_1k.get('mean_xy_error', float('nan')))} m",
                f"  median x/y error: {_fmt(implicit_1k.get('median_xy_error', float('nan')))} m",
                f"  no-change mean x/y error: {_fmt(implicit_1k.get('no_change_mean_xy_error', float('nan')))} m",
            ]
        )
    if multi_horizon and not multi_horizon.get("skipped"):
        lines.extend(
            [
                "",
                "-- implicit f_H multi-horizon (new pi_H subgoal each K from predicted/real state) --",
                f"  protocol: {multi_horizon.get('protocol', '')}",
            ]
        )
        for h in multi_horizon.get("horizons") or []:
            row = (multi_horizon.get("by_horizon") or {}).get(h, {})
            lines.append(
                f"  {h}K  n={row.get('n_trials', 0)}  "
                f"mean={_fmt(row.get('mean_xy_error', float('nan')))} m  "
                f"median={_fmt(row.get('median_xy_error', float('nan')))} m"
            )
    return "\n".join(lines)
