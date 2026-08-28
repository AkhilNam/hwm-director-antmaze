"""Evaluate Director's implicit ``f_H`` and end-to-end env rollouts."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from hwm_director.data.state import extract_state_and_goal
from hwm_director.data.trajectories import group_by_episode
from hwm_director.data.transitions import Transition
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.envs.antmaze import DEFAULT_DATASET_ID, DEFAULT_ENV_ID, make_antmaze
from hwm_director.models.director import Director
from hwm_director.models.director_manager import DEFAULT_MAX_SUBGOAL_DISTANCE
from hwm_director.training.director_diagnostics import (
    DEFAULT_STUCK_DISTANCE,
    DEFAULT_STUCK_WINDOW,
    DEFAULT_SUBGOAL_SUCCESS,
    DatasetSupportIndex,
    classify_failed_trials,
    detect_stuck,
    summarize_interval_records,
)
from hwm_director.training.train_worker import (
    _assert_xy_restored,
    _reset_and_restore,
    _worker_action,
)

MULTI_HORIZONS = (1, 2, 3, 5)
INTERVAL_RECORD_KEYS = (
    "tau",
    "current_xy",
    "final_goal",
    "subgoal_xy",
    "subgoal_distance",
    "final_goal_distance_before",
    "subgoal_to_final_distance",
    "worker_final_xy",
    "worker_subgoal_error",
    "subgoal_reached",
    "final_goal_distance_after",
    "goal_progress",
    "nearest_dataset_subgoal_distance",
)


def _future_state(traj: Sequence[Transition], t: int, k: int) -> np.ndarray:
    return np.asarray(traj[t + k - 1].next_state, dtype=np.float64)


def implicit_fh_candidates(
    transitions: Sequence[Transition],
    horizon_k: int,
) -> list[tuple[Transition, np.ndarray, np.ndarray]]:
    """``(start_transition, g_tau, recorded_s_{t+K})`` with exactly ``K`` steps left."""
    out: list[tuple[Transition, np.ndarray, np.ndarray]] = []
    for traj in group_by_episode(transitions).values():
        traj = list(traj)
        n_steps = len(traj)
        if n_steps < horizon_k:
            continue
        for t in range(n_steps - horizon_k + 1):
            start = traj[t]
            future = _future_state(traj, t, horizon_k)
            out.append((start, future[:2].copy(), future))
    return out


def evaluate_implicit_high_level_transition(
    transitions: Sequence[Transition],
    director: Director,
    horizon_k: int = DEFAULT_HORIZON_K,
    n_trials: int = 20,
    seed: int = 0,
    *,
    dataset_id: str | None = None,
    env_id: str = DEFAULT_ENV_ID,
) -> dict:
    """Compare ``(f_L, pi_L)^K`` to a real worker rollout from the same restore.

    A. model: Director ``high_level_transition`` (no env.step inside ``f_H``)
    B. real: restore qpos/qvel, run ``pi_L`` for ``K`` env steps
    Also report the no-change predictor ``s_{t+K} = s_t``.
    """
    del dataset_id  # env is constructed locally; recovered env is optional at call site
    candidates = implicit_fh_candidates(transitions, horizon_k)
    usable = [
        c
        for c in candidates
        if c[0].qpos is not None and c[0].qvel is not None
    ]
    empty = {
        "n_candidates": len(candidates),
        "n_trials": 0,
        "horizon_k": int(horizon_k),
        "mean_xy_error": float("nan"),
        "median_xy_error": float("nan"),
        "mean_state_mse": float("nan"),
        "no_change_mean_xy_error": float("nan"),
        "mean_model_vs_recorded_xy_error": float("nan"),
        "skipped": False,
        "skip_reason": "",
    }
    if n_trials <= 0:
        return empty
    if not usable:
        empty["skipped"] = True
        empty["skip_reason"] = (
            "No transitions with qpos/qvel and K remaining steps; "
            "refusing to approximate simulator restore."
        )
        return empty

    rng = np.random.default_rng(seed)
    n = min(int(n_trials), len(usable))
    picks = rng.choice(len(usable), size=n, replace=False)
    director.worker.eval()
    director.dynamics.eval()

    model_xy_err: list[float] = []
    model_state_mse: list[float] = []
    no_change_xy_err: list[float] = []
    vs_recorded_xy_err: list[float] = []

    env = make_antmaze(env_id)
    try:
        for trial_i, idx in enumerate(picks):
            start, subgoal_xy, _recorded_future = usable[int(idx)]
            director.reset_call_counts()
            predicted = director.high_level_transition(
                start.state, subgoal_xy, horizon_k=horizon_k
            )
            if director.n_dynamics_calls != horizon_k:
                raise AssertionError(
                    f"expected {horizon_k} f_L calls, got {director.n_dynamics_calls}"
                )
            if director.n_worker_calls != horizon_k:
                raise AssertionError(
                    f"expected {horizon_k} pi_L calls, got {director.n_worker_calls}"
                )

            observation = _reset_and_restore(
                env,
                seed=int(seed + trial_i),
                qpos=start.qpos,
                qvel=start.qvel,
            )
            reconstructed, _ = extract_state_and_goal(observation)
            _assert_xy_restored(reconstructed, start.state)
            for _ in range(horizon_k):
                state, _ = extract_state_and_goal(observation)
                action = _worker_action(
                    director.worker,
                    director.worker_normalizer,
                    state,
                    subgoal_xy,
                )
                observation, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    break
            real_state, _ = extract_state_and_goal(observation)
            model_xy_err.append(float(np.linalg.norm(predicted[:2] - real_state[:2])))
            model_state_mse.append(float(np.mean((predicted - real_state) ** 2)))
            no_change_xy_err.append(
                float(np.linalg.norm(start.state[:2] - real_state[:2]))
            )
            vs_recorded_xy_err.append(
                float(np.linalg.norm(predicted[:2] - subgoal_xy))
            )
    finally:
        env.close()

    arr = np.asarray(model_xy_err, dtype=np.float64)
    return {
        "n_candidates": len(candidates),
        "n_trials": int(arr.size),
        "horizon_k": int(horizon_k),
        "mean_xy_error": float(np.mean(arr)),
        "median_xy_error": float(np.median(arr)),
        "mean_state_mse": float(np.mean(model_state_mse)),
        "no_change_mean_xy_error": float(np.mean(no_change_xy_err)),
        "mean_model_vs_recorded_xy_error": float(np.mean(vs_recorded_xy_err)),
        "skipped": False,
        "skip_reason": "",
    }


def _xy_list(xy: np.ndarray) -> list[float]:
    arr = np.asarray(xy, dtype=np.float64).reshape(2)
    return [float(arr[0]), float(arr[1])]


def build_interval_record(
    *,
    tau: int,
    current_xy: np.ndarray,
    final_goal: np.ndarray,
    subgoal_xy: np.ndarray,
    worker_final_xy: np.ndarray,
    subgoal_success_threshold: float,
    nearest_dataset_subgoal_distance: float,
    extra: dict | None = None,
) -> dict:
    """One high-level decision ``tau`` with worker outcome after up to ``K`` steps."""
    current_xy = np.asarray(current_xy, dtype=np.float64).reshape(2)
    final_goal = np.asarray(final_goal, dtype=np.float64).reshape(2)
    subgoal_xy = np.asarray(subgoal_xy, dtype=np.float64).reshape(2)
    worker_final_xy = np.asarray(worker_final_xy, dtype=np.float64).reshape(2)
    dist_before = float(np.linalg.norm(current_xy - final_goal))
    dist_after = float(np.linalg.norm(worker_final_xy - final_goal))
    worker_err = float(np.linalg.norm(worker_final_xy - subgoal_xy))
    rec = {
        "tau": int(tau),
        "current_xy": _xy_list(current_xy),
        "final_goal": _xy_list(final_goal),
        "subgoal_xy": _xy_list(subgoal_xy),
        "subgoal_distance": float(np.linalg.norm(subgoal_xy - current_xy)),
        "final_goal_distance_before": dist_before,
        "subgoal_to_final_distance": float(np.linalg.norm(subgoal_xy - final_goal)),
        "worker_final_xy": _xy_list(worker_final_xy),
        "worker_subgoal_error": worker_err,
        "subgoal_reached": bool(worker_err < float(subgoal_success_threshold)),
        "final_goal_distance_after": dist_after,
        "goal_progress": dist_before - dist_after,
        "nearest_dataset_subgoal_distance": float(nearest_dataset_subgoal_distance),
    }
    if extra:
        rec.update(extra)
    return rec


def _trial_from_intervals(
    intervals: list[dict],
    *,
    success: bool,
    initial_distance: float,
    final_distance: float,
    n_primitive_steps: int,
    terminated: bool,
    truncated: bool,
    stuck_window: int,
    stuck_distance: float,
    max_subgoal_distance: float,
) -> dict:
    stuck = detect_stuck(
        intervals, window=stuck_window, distance=stuck_distance
    )
    summary = summarize_interval_records(
        intervals, max_subgoal_distance=max_subgoal_distance
    )
    total_progress = float(initial_distance - final_distance)
    return {
        "success": bool(success),
        "initial_distance": float(initial_distance),
        "final_distance": float(final_distance),
        "n_primitive_steps": int(n_primitive_steps),
        "n_high_level_decisions": len(intervals),
        "total_progress": total_progress,
        "distance_moved_toward_final_goal": total_progress,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "intervals": intervals,
        "subgoals": [rec["subgoal_xy"] for rec in intervals],
        "subgoal_reached_within_k": [bool(rec["subgoal_reached"]) for rec in intervals],
        **stuck,
        **summary,
    }


def evaluate_implicit_multi_horizon(
    transitions: Sequence[Transition],
    director,
    horizons: Sequence[int] = MULTI_HORIZONS,
    n_trials: int = 20,
    seed: int = 0,
    *,
    env_id: str = DEFAULT_ENV_ID,
) -> dict:
    """Implicit ``f_H`` x/y error after 1, 2, 3, and 5 high-level intervals.

    At each interval a **new** manager subgoal is chosen from the current
    state. The model uses ``pi_H(predicted_state, g*)`` then
    ``(f_L, pi_L)^K``. The real Ant uses ``pi_H(real_state, g*)`` then
    ``K`` env worker steps. That matches end-to-end Director, not a
    fixed recorded subgoal.
    """
    horizons = tuple(int(h) for h in horizons)
    max_h = max(horizons) if horizons else 0
    usable = [
        t
        for t in transitions
        if t.qpos is not None and t.qvel is not None
    ]
    empty_h = {
        h: {"n_trials": 0, "mean_xy_error": float("nan"), "median_xy_error": float("nan")}
        for h in horizons
    }
    empty = {
        "n_candidates": len(usable),
        "n_trials": 0,
        "horizon_k": int(director.horizon_k),
        "horizons": list(horizons),
        "protocol": (
            "new pi_H subgoal each K steps from the current predicted "
            "(model) or real (env) state; recursive error is not one-step val MSE"
        ),
        "by_horizon": empty_h,
        "skipped": False,
        "skip_reason": "",
    }
    if n_trials <= 0 or max_h < 1:
        return empty
    if not usable:
        empty["skipped"] = True
        empty["skip_reason"] = (
            "No transitions with qpos/qvel; refusing to approximate simulator restore."
        )
        return empty

    rng = np.random.default_rng(seed)
    n = min(int(n_trials), len(usable))
    picks = rng.choice(len(usable), size=n, replace=False)
    director.worker.eval()
    director.dynamics.eval()
    director.manager.eval()
    explicit = getattr(director, "explicit_f_h", None)
    if explicit is not None:
        explicit.eval()

    errors: dict[int, list[float]] = {h: [] for h in horizons}
    env = make_antmaze(env_id)
    try:
        for trial_i, idx in enumerate(picks):
            start = usable[int(idx)]
            predicted = np.asarray(start.state, dtype=np.float64).copy()
            observation = _reset_and_restore(
                env,
                seed=int(seed + trial_i),
                qpos=start.qpos,
                qvel=start.qvel,
            )
            real_state, final_goal = extract_state_and_goal(observation)
            _assert_xy_restored(real_state, start.state)
            terminated = False
            truncated = False
            for h in range(1, max_h + 1):
                g_model = director.select_high_level_command(predicted, final_goal)
                predicted = director.high_level_transition(
                    predicted, g_model, horizon_k=director.horizon_k
                )
                g_real = director.select_high_level_command(real_state, final_goal)
                for _ in range(director.horizon_k):
                    action = director.low_level_action(real_state, g_real)
                    observation, _reward, terminated, truncated, _info = env.step(
                        action
                    )
                    real_state, final_goal = extract_state_and_goal(observation)
                    if terminated or truncated:
                        break
                if h in errors:
                    errors[h].append(
                        float(np.linalg.norm(predicted[:2] - real_state[:2]))
                    )
                if terminated or truncated:
                    break
    finally:
        env.close()

    by_horizon = {}
    for h in horizons:
        arr = np.asarray(errors[h], dtype=np.float64)
        by_horizon[h] = {
            "n_trials": int(arr.size),
            "mean_xy_error": float(np.mean(arr)) if arr.size else float("nan"),
            "median_xy_error": float(np.median(arr)) if arr.size else float("nan"),
        }
    return {
        "n_candidates": len(usable),
        "n_trials": n,
        "horizon_k": int(director.horizon_k),
        "horizons": list(horizons),
        "protocol": (
            "new pi_H subgoal each K steps from the current predicted "
            "(model) or real (env) state; recursive error is not one-step val MSE"
        ),
        "by_horizon": by_horizon,
        "skipped": False,
        "skip_reason": "",
    }


def evaluate_matched_high_level_models(
    transitions: Sequence[Transition],
    director: Director,
    hwm,
    horizon_k: int = DEFAULT_HORIZON_K,
    n_trials: int = 20,
    seed: int = 0,
    *,
    env_id: str = DEFAULT_ENV_ID,
) -> dict:
    """Same starts and subgoals: Director ``(f_L, pi_L)^K`` vs HWM ``f_H_phi`` vs real.

    Ground truth is a real ``pi_L`` env rollout of ``K`` steps (shared worker).
    Also reports error vs the recorded ``s_{t+K}`` (the explicit model's
    training target, which is not a current-``pi_L`` rollout).
    """
    candidates = implicit_fh_candidates(transitions, horizon_k)
    usable = [
        c for c in candidates if c[0].qpos is not None and c[0].qvel is not None
    ]
    empty = {
        "n_candidates": len(candidates),
        "n_trials": 0,
        "horizon_k": int(horizon_k),
        "skipped": False,
        "skip_reason": "",
    }
    if n_trials <= 0:
        return empty
    if not usable:
        empty["skipped"] = True
        empty["skip_reason"] = (
            "No transitions with qpos/qvel and K remaining steps; "
            "refusing to approximate simulator restore."
        )
        return empty

    rng = np.random.default_rng(seed)
    n = min(int(n_trials), len(usable))
    picks = rng.choice(len(usable), size=n, replace=False)
    director.worker.eval()
    director.dynamics.eval()
    hwm.explicit_f_h.eval()

    director_xy: list[float] = []
    hwm_xy: list[float] = []
    no_change_xy: list[float] = []
    director_mse: list[float] = []
    hwm_mse: list[float] = []
    no_change_mse: list[float] = []
    director_vs_rec: list[float] = []
    hwm_vs_rec: list[float] = []

    env = make_antmaze(env_id)
    try:
        for trial_i, idx in enumerate(picks):
            start, subgoal_xy, recorded_future = usable[int(idx)]
            director.reset_call_counts()
            pred_d = director.high_level_transition(
                start.state, subgoal_xy, horizon_k=horizon_k
            )
            hwm.reset_call_counts()
            pred_h = hwm.high_level_transition(
                start.state, subgoal_xy, horizon_k=horizon_k
            )
            if director.n_dynamics_calls != horizon_k:
                raise AssertionError(
                    f"Director expected {horizon_k} f_L calls, "
                    f"got {director.n_dynamics_calls}"
                )
            if getattr(hwm, "n_explicit_fh_calls", 0) != 1:
                raise AssertionError(
                    f"HWM expected 1 f_H_phi call, got {hwm.n_explicit_fh_calls}"
                )
            if hwm.n_dynamics_calls != 0 or hwm.n_worker_calls != 0:
                raise AssertionError("HWM f_H must not call pi_L or f_L")

            observation = _reset_and_restore(
                env,
                seed=int(seed + trial_i),
                qpos=start.qpos,
                qvel=start.qvel,
            )
            reconstructed, _ = extract_state_and_goal(observation)
            _assert_xy_restored(reconstructed, start.state)
            for _ in range(horizon_k):
                state, _ = extract_state_and_goal(observation)
                action = director.low_level_action(state, subgoal_xy)
                observation, _reward, terminated, truncated, _info = env.step(
                    action
                )
                if terminated or truncated:
                    break
            real_state, _ = extract_state_and_goal(observation)
            director_xy.append(float(np.linalg.norm(pred_d[:2] - real_state[:2])))
            hwm_xy.append(float(np.linalg.norm(pred_h[:2] - real_state[:2])))
            no_change_xy.append(
                float(np.linalg.norm(start.state[:2] - real_state[:2]))
            )
            director_mse.append(float(np.mean((pred_d - real_state) ** 2)))
            hwm_mse.append(float(np.mean((pred_h - real_state) ** 2)))
            no_change_mse.append(float(np.mean((start.state - real_state) ** 2)))
            rec_xy = np.asarray(recorded_future[:2], dtype=np.float64)
            director_vs_rec.append(float(np.linalg.norm(pred_d[:2] - rec_xy)))
            hwm_vs_rec.append(float(np.linalg.norm(pred_h[:2] - rec_xy)))
    finally:
        env.close()

    def _summ(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(arr)), float(np.median(arr))

    d_mean, d_med = _summ(director_xy)
    h_mean, h_med = _summ(hwm_xy)
    n_mean, n_med = _summ(no_change_xy)
    return {
        "n_candidates": len(candidates),
        "n_trials": n,
        "horizon_k": int(horizon_k),
        "director_mean_xy_error": d_mean,
        "director_median_xy_error": d_med,
        "director_mean_state_mse": float(np.mean(director_mse)) if director_mse else float("nan"),
        "hwm_mean_xy_error": h_mean,
        "hwm_median_xy_error": h_med,
        "hwm_mean_state_mse": float(np.mean(hwm_mse)) if hwm_mse else float("nan"),
        "no_change_mean_xy_error": n_mean,
        "no_change_median_xy_error": n_med,
        "no_change_mean_state_mse": (
            float(np.mean(no_change_mse)) if no_change_mse else float("nan")
        ),
        "director_mean_xy_error_vs_recorded": (
            float(np.mean(director_vs_rec)) if director_vs_rec else float("nan")
        ),
        "hwm_mean_xy_error_vs_recorded": (
            float(np.mean(hwm_vs_rec)) if hwm_vs_rec else float("nan")
        ),
        "skipped": False,
        "skip_reason": "",
        "real_outcome": "shared pi_L env rollout for K steps from restored state",
        "hwm_training_target": "recorded s_{t+K}, not current pi_L",
    }


def evaluate_director_env_rollouts(
    director,
    n_trials: int = 5,
    max_high_level_steps: int = 70,
    success_threshold: float = 0.5,
    seed: int = 0,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    use_recovered_env: bool = True,
    subgoal_success_threshold: float = DEFAULT_SUBGOAL_SUCCESS,
    stuck_window: int = DEFAULT_STUCK_WINDOW,
    stuck_distance: float = DEFAULT_STUCK_DISTANCE,
    support_index: DatasetSupportIndex | None = None,
) -> dict:
    """End-to-end hierarchy: manager every ``K`` env steps, per-interval records.

    Real execution always uses ``pi_L`` + ``env.step``. ``f_H`` is used only
    to record predicted subgoal error (Director: ``(f_L, pi_L)^K``; HWM:
    ``f_H_phi``). Imagined states are never stepped in MuJoCo.
    """
    from hwm_director.envs.antmaze import recover_minari_environment

    if n_trials <= 0:
        return {
            "n_trials": 0,
            "success_rate": float("nan"),
            "mean_final_distance": float("nan"),
            "trials": [],
        }

    if use_recovered_env:
        env = recover_minari_environment(dataset_id)
    else:
        env = make_antmaze()
    trials: list[dict] = []
    max_subgoal_distance = float(director.manager.max_subgoal_distance)
    try:
        for trial_i in range(n_trials):
            observation, info = env.reset(seed=int(seed + trial_i))
            state, final_goal = extract_state_and_goal(observation)
            start_xy = state[:2].copy()
            initial_distance = float(np.linalg.norm(start_xy - final_goal))
            intervals: list[dict] = []
            n_primitive = 0
            success = False
            terminated = False
            truncated = False
            for tau in range(max_high_level_steps):
                current_xy = state[:2].copy()
                g_star = np.asarray(final_goal, dtype=np.float64).copy()
                g_tau = director.select_high_level_command(state, final_goal)
                extra: dict = {}
                policy = getattr(director, "high_level_policy", None)
                if policy is not None and hasattr(policy, "last_diagnostics"):
                    extra = dict(policy.last_diagnostics)
                predicted = director.high_level_transition(state, g_tau)
                extra["predicted_subgoal_error"] = float(
                    np.linalg.norm(np.asarray(predicted[:2], dtype=np.float64) - g_tau)
                )
                if support_index is None:
                    nn_dist = float("nan")
                else:
                    nn_dist = support_index.nearest_dataset_subgoal_distance(
                        current_xy, g_tau
                    )
                for _step in range(director.horizon_k):
                    action = director.low_level_action(state, g_tau)
                    observation, _reward, terminated, truncated, info = env.step(
                        action
                    )
                    n_primitive += 1
                    state, final_goal = extract_state_and_goal(observation)
                    success = bool(info.get("success", False)) or (
                        float(np.linalg.norm(state[:2] - final_goal))
                        < success_threshold
                    )
                    if terminated or truncated or success:
                        break
                intervals.append(
                    build_interval_record(
                        tau=tau,
                        current_xy=current_xy,
                        final_goal=g_star,
                        subgoal_xy=g_tau,
                        worker_final_xy=state[:2],
                        subgoal_success_threshold=subgoal_success_threshold,
                        nearest_dataset_subgoal_distance=nn_dist,
                        extra=extra,
                    )
                )
                if terminated or truncated or success:
                    break
            final_dist = float(np.linalg.norm(state[:2] - final_goal))
            trials.append(
                _trial_from_intervals(
                    intervals,
                    success=bool(success),
                    initial_distance=initial_distance,
                    final_distance=final_dist,
                    n_primitive_steps=n_primitive,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    stuck_window=stuck_window,
                    stuck_distance=stuck_distance,
                    max_subgoal_distance=max_subgoal_distance,
                )
            )
    finally:
        env.close()

    return summarize_director_env_eval(
        trials,
        max_subgoal_distance=max_subgoal_distance,
        subgoal_success_threshold=subgoal_success_threshold,
        stuck_window=stuck_window,
        stuck_distance=stuck_distance,
    )


def summarize_director_env_eval(
    trials: Sequence[dict],
    *,
    max_subgoal_distance: float = DEFAULT_MAX_SUBGOAL_DISTANCE,
    subgoal_success_threshold: float = DEFAULT_SUBGOAL_SUCCESS,
    stuck_window: int = DEFAULT_STUCK_WINDOW,
    stuck_distance: float = DEFAULT_STUCK_DISTANCE,
    multi_horizon: dict | None = None,
) -> dict:
    """Aggregate per-trial records plus pooled high-level interval stats."""
    successes = [bool(t["success"]) for t in trials]
    finals = [float(t["final_distance"]) for t in trials]
    all_intervals = [rec for trial in trials for rec in trial.get("intervals", [])]
    pooled = summarize_interval_records(
        all_intervals, max_subgoal_distance=max_subgoal_distance
    )
    failed = [t for t in trials if not t.get("success")]
    stuck_failed = [t for t in failed if t.get("stuck")]
    longest_runs = [int(t.get("longest_no_progress_run", 0)) for t in trials]
    stuck_locations = [
        {
            "trial": i,
            "first_stuck_tau": t.get("first_stuck_tau"),
            "stuck_xy": t.get("stuck_xy"),
        }
        for i, t in enumerate(trials)
        if (not t.get("success")) and t.get("stuck")
    ]
    return {
        "n_trials": len(trials),
        "success_rate": float(np.mean(successes)) if trials else float("nan"),
        "mean_final_distance": float(np.mean(finals)) if trials else float("nan"),
        "mean_initial_distance": (
            float(np.mean([t["initial_distance"] for t in trials]))
            if trials
            else float("nan")
        ),
        "subgoal_success_threshold": float(subgoal_success_threshold),
        "stuck_window": int(stuck_window),
        "stuck_distance": float(stuck_distance),
        "max_longest_no_progress_run": (
            int(max(longest_runs)) if longest_runs else 0
        ),
        "stuck_rate_among_failed": (
            float(len(stuck_failed) / len(failed)) if failed else float("nan")
        ),
        "n_failed": len(failed),
        "n_failed_stuck": len(stuck_failed),
        "stuck_locations": stuck_locations,
        "failure_summary": classify_failed_trials(
            trials, multi_horizon=multi_horizon
        ),
        "trials": list(trials),
        **pooled,
    }
