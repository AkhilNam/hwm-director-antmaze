#!/usr/bin/env python3
"""Train explicit ``f_H_phi`` and evaluate HWM vs frozen Director.

Director (frozen, not retuned here):

    f_H = (f_L, pi_L)^K
    SoftReach lambda = 1.0

HWM:

    f_H = explicit learned f_H_phi
    same Q_H, candidates, lambda, pi_L, f_L, K

``f_H_phi`` is trained on recorded K-step tuples
``(s_t, s_{t+K}[:2]) -> s_{t+K}``, not a rollout of the current ``pi_L``.
"""

from __future__ import annotations

import argparse

from hwm_director.data.minari_antmaze import (
    DEFAULT_MINARI_DATASET_ID,
    load_minari_transitions,
)
from hwm_director.data.subgoal_candidates import (
    DEFAULT_CANDIDATE_RADIUS,
    DEFAULT_N_CANDIDATES,
    SubgoalCandidateIndex,
    estimate_source_state_distance_threshold,
)
from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.director import Director, assert_director_has_no_learned_f_h
from hwm_director.models.director_manager import DEFAULT_MAX_SUBGOAL_DISTANCE
from hwm_director.models.dynamics_high import ExplicitHighLevelDynamics
from hwm_director.models.hwm import HierarchicalWorldModel, assert_hwm_has_explicit_f_h
from hwm_director.models.value_manager import (
    REACHABILITY_NORM_ZSCORE,
    ValueHighLevelPolicy,
    director_reachability_error,
)
from hwm_director.training.checkpoints import (
    load_dynamics_checkpoint,
    load_high_level_dynamics_checkpoint,
    load_high_level_value_checkpoint,
    load_manager_checkpoint,
    load_worker_checkpoint,
    save_high_level_dynamics_checkpoint,
)
from hwm_director.training.director_diagnostics import (
    DEFAULT_STUCK_DISTANCE,
    DEFAULT_STUCK_WINDOW,
    DEFAULT_SUBGOAL_SUCCESS,
    DEFAULT_WALL_REGION_RADIUS,
    DEFAULT_WALL_REGION_XY,
    DatasetSupportIndex,
    analyze_failed_stuck,
    analyze_wall_region,
    classify_failed_trials,
    format_ablation_table,
    format_candidate_quality,
    format_director_diagnostic_report,
    format_fairness_summary,
    write_eval_csv,
)
from hwm_director.training.eval_director import (
    evaluate_director_env_rollouts,
    evaluate_implicit_multi_horizon,
    evaluate_matched_high_level_models,
)
from hwm_director.training.train_dynamics import split_episode_indices
from hwm_director.training.train_high_level_dynamics import train_high_level_dynamics
from hwm_director.training.train_worker import _select

SEED = 0
FROZEN_DIRECTOR_LAMBDA = 1.0


def _csv_path_for_manager(path: str, manager_name: str) -> str:
    if not path:
        return ""
    if path.endswith(".csv"):
        return f"{path[:-4]}_{manager_name}.csv"
    return f"{path}_{manager_name}"


def _make_softreach_policy(
    *,
    controller,
    value_model,
    value_normalizer,
    candidate_index,
    max_source_d: float,
    lambda_reach: float,
    n_candidates: int,
    candidate_state_radius: float,
    max_subgoal_distance: float,
    normalization: str,
) -> ValueHighLevelPolicy:
    return ValueHighLevelPolicy(
        value_model,
        value_normalizer,
        candidate_index,
        controller._select_bc,
        candidate_state_radius=candidate_state_radius,
        n_candidates=n_candidates,
        max_subgoal_distance=max_subgoal_distance,
        retrieval_mode="state",
        max_source_state_distance=max_source_d,
        reachability_fn=director_reachability_error(controller),
        max_predicted_subgoal_error=None,
        reachability_score_weight=float(lambda_reach),
        reachability_score_normalization=normalization,
        use_soft_reachability=True,
    )


def _run_env_eval(controller, args, support, *, manager_name: str, csv_path: str):
    print(f"=== end-to-end {manager_name} env evaluation ===", flush=True)
    env_metrics = evaluate_director_env_rollouts(
        controller,
        n_trials=args.n_env_eval_trials,
        max_high_level_steps=args.max_high_level_steps,
        seed=args.seed,
        dataset_id=args.dataset_id,
        subgoal_success_threshold=args.subgoal_success_threshold,
        stuck_window=args.stuck_window,
        stuck_distance=args.stuck_distance,
        support_index=support,
    )
    env_metrics["failure_summary"] = classify_failed_trials(env_metrics["trials"])
    env_metrics["wall_region"] = analyze_wall_region(
        env_metrics["trials"],
        center=(args.wall_region_x, args.wall_region_y),
        radius=args.wall_region_radius,
    )
    env_metrics["manager_name"] = manager_name
    env_metrics["failed_stuck"] = analyze_failed_stuck(env_metrics["trials"])
    policy = getattr(controller, "high_level_policy", None)
    if policy is not None and hasattr(policy, "stats"):
        stats = policy.stats()
        env_metrics["policy_fallback_rate"] = stats.get("fallback_rate", float("nan"))
        print(f"  fallback_rate: {stats.get('fallback_rate')}")
    print(f"  n_trials: {env_metrics['n_trials']}")
    print(f"  success_rate: {env_metrics['success_rate']:.6f}")
    print(f"  mean_final_distance: {env_metrics['mean_final_distance']:.6f}")
    print(f"  subgoal_reach_rate: {env_metrics['subgoal_reach_rate']:.6f}")
    print(
        f"  stuck_rate among failed: {env_metrics['stuck_rate_among_failed']:.6f}"
    )
    print(
        f"  n_high_level_decisions (pooled): {env_metrics.get('n_intervals')}"
    )
    wall = env_metrics["wall_region"]
    print(
        f"  pred_err={env_metrics.get('mean_predicted_subgoal_error')} "
        f"actual_err={env_metrics.get('mean_worker_subgoal_error')} "
        f"pearson={env_metrics.get('predicted_vs_actual_pearson')} "
        f"spearman={env_metrics.get('predicted_vs_actual_spearman')}"
    )
    stuck = env_metrics["failed_stuck"]
    print(
        f"  failed stuck: n={stuck['n_failed_stuck']}/{stuck['n_failed']} "
        f"reach_before={stuck['mean_reach_before_stuck']} "
        f"reach_after={stuck['mean_reach_after_stuck']}"
    )
    print(
        f"  U-wall entered={wall['n_trials_entered']} "
        f"later_success={wall['n_entered_then_success']} "
        f"mean_steps={wall['mean_high_level_steps_in_region']:.3f} "
        f"mean_dir={wall['mean_subgoal_direction']}"
    )
    print(format_candidate_quality(env_metrics), flush=True)
    for i, trial in enumerate(env_metrics["trials"]):
        print(
            f"  trial {i}: success={trial['success']} "
            f"final_d={trial['final_distance']:.4f} "
            f"decisions={trial['n_high_level_decisions']} "
            f"reach={trial['subgoal_reach_rate']:.3f} "
            f"stuck={trial['stuck']}"
        )
    if csv_path:
        write_eval_csv(csv_path, env_metrics["trials"])
        print(f"wrote {csv_path}", flush=True)
    return env_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default=DEFAULT_MINARI_DATASET_ID)
    parser.add_argument("--horizon-k", type=int, default=DEFAULT_HORIZON_K)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--worker-checkpoint", default="checkpoints/pi_l.pt")
    parser.add_argument("--dynamics-checkpoint", default="checkpoints/f_l.pt")
    parser.add_argument("--manager-checkpoint", default="checkpoints/pi_h_director.pt")
    parser.add_argument(
        "--value-checkpoint", default="checkpoints/high_level_value.pt"
    )
    parser.add_argument(
        "--save-high-level-dynamics-checkpoint",
        default="checkpoints/f_h_explicit.pt",
    )
    parser.add_argument(
        "--load-high-level-dynamics-checkpoint",
        default="checkpoints/f_h_explicit.pt",
    )
    parser.add_argument(
        "--skip-high-level-dynamics-train",
        action="store_true",
        help="load --load-high-level-dynamics-checkpoint instead of training f_H_phi",
    )
    parser.add_argument(
        "--lambda-reach",
        type=float,
        default=FROZEN_DIRECTOR_LAMBDA,
        help="SoftReach lambda (frozen Director comparison uses 1.0)",
    )
    parser.add_argument(
        "--reachability-score-normalization",
        choices=("candidate_zscore", "raw"),
        default=REACHABILITY_NORM_ZSCORE,
    )
    parser.add_argument("--n-env-eval-trials", type=int, default=0)
    parser.add_argument("--n-model-rollout-trials", type=int, default=20)
    parser.add_argument("--n-multi-horizon-trials", type=int, default=20)
    parser.add_argument("--max-high-level-steps", type=int, default=70)
    parser.add_argument(
        "--subgoal-success-threshold",
        type=float,
        default=DEFAULT_SUBGOAL_SUCCESS,
    )
    parser.add_argument("--stuck-window", type=int, default=DEFAULT_STUCK_WINDOW)
    parser.add_argument("--stuck-distance", type=float, default=DEFAULT_STUCK_DISTANCE)
    parser.add_argument("--save-eval-csv", default="")
    parser.add_argument(
        "--max-subgoal-distance",
        type=float,
        default=DEFAULT_MAX_SUBGOAL_DISTANCE,
    )
    parser.add_argument(
        "--candidate-state-radius",
        type=float,
        default=DEFAULT_CANDIDATE_RADIUS,
    )
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument("--wall-region-x", type=float, default=DEFAULT_WALL_REGION_XY[0])
    parser.add_argument("--wall-region-y", type=float, default=DEFAULT_WALL_REGION_XY[1])
    parser.add_argument(
        "--wall-region-radius", type=float, default=DEFAULT_WALL_REGION_RADIUS
    )
    parser.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 256]
    )
    args = parser.parse_args()

    print("loading shared pi_L, f_L, pi_H, Q_H (not retrained)...", flush=True)
    worker, worker_normalizer = load_worker_checkpoint(args.worker_checkpoint)
    dynamics, dynamics_normalizer = load_dynamics_checkpoint(
        args.dynamics_checkpoint
    )
    manager, manager_normalizer = load_manager_checkpoint(args.manager_checkpoint)
    value_model, value_normalizer, value_cfg = load_high_level_value_checkpoint(
        args.value_checkpoint
    )
    manager.eval()
    worker.eval()
    dynamics.eval()
    value_model.eval()

    transitions = load_minari_transitions(
        args.dataset_id,
        max_episodes=args.max_episodes,
        max_transitions=args.max_transitions,
    )
    n_episodes = len({t.episode_id for t in transitions})
    print(
        f"loaded {len(transitions)} transitions across {n_episodes} episodes "
        f"from {args.dataset_id}",
        flush=True,
    )
    train_idx, val_idx = split_episode_indices(
        transitions, val_fraction=args.val_fraction, seed=args.seed
    )
    train_raw = _select(transitions, train_idx)
    val_raw = _select(transitions, val_idx)

    if args.skip_high_level_dynamics_train:
        print(
            f"loading f_H_phi from {args.load_high_level_dynamics_checkpoint}...",
            flush=True,
        )
        fh_model, fh_normalizer, fh_cfg = load_high_level_dynamics_checkpoint(
            args.load_high_level_dynamics_checkpoint
        )
        fh_metrics = {
            "model": fh_model,
            "normalizer": fh_normalizer,
            "horizon_k": int(fh_cfg.get("horizon_k", args.horizon_k)),
            "train_mse": float("nan"),
            "val_mse": float("nan"),
            "val_xy_mse": float("nan"),
            "val_xy_euclidean": float("nan"),
            "no_change_val_xy_euclidean": float("nan"),
            "no_change_val_mse": float("nan"),
            "n_train_episodes": float("nan"),
            "n_val_episodes": float("nan"),
            "n_train_examples": float("nan"),
            "n_val_examples": float("nan"),
            "target": fh_cfg.get("training_target", "recorded s_{t+K}"),
        }
    else:
        print("=== train explicit f_H_phi on recorded K-step transitions ===", flush=True)
        print(
            "target: (s_t, s_{t+K}[:2]) -> s_{t+K}; not a current pi_L rollout",
            flush=True,
        )
        fh_model = ExplicitHighLevelDynamics(hidden_dims=tuple(args.hidden_dims))
        fh_metrics = train_high_level_dynamics(
            transitions,
            model=fh_model,
            horizon_k=args.horizon_k,
            val_fraction=args.val_fraction,
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            log=lambda message: print(message, flush=True),
        )
        print(f"  n_train_episodes: {fh_metrics['n_train_episodes']}")
        print(f"  n_val_episodes: {fh_metrics['n_val_episodes']}")
        print(f"  n_train_examples: {fh_metrics['n_train_examples']}")
        print(f"  n_val_examples: {fh_metrics['n_val_examples']}")
        print(f"  train MSE: {fh_metrics['train_mse']:.6f}")
        print(f"  val MSE: {fh_metrics['val_mse']:.6f}")
        print(f"  val x/y MSE: {fh_metrics['val_xy_mse']:.6f}")
        print(f"  val x/y Euclidean (m): {fh_metrics['val_xy_euclidean']:.6f}")
        print(
            f"  no-change val x/y Euclidean (m): "
            f"{fh_metrics['no_change_val_xy_euclidean']:.6f}"
        )
        print(f"  no-change val full-state MSE: {fh_metrics['no_change_val_mse']:.6f}")
        if args.save_high_level_dynamics_checkpoint:
            save_high_level_dynamics_checkpoint(
                args.save_high_level_dynamics_checkpoint,
                fh_metrics["model"],
                fh_metrics["normalizer"],
                horizon_k=args.horizon_k,
                dataset_id=args.dataset_id,
                seed=args.seed,
                val_fraction=args.val_fraction,
            )
            print(
                f"saved f_H_phi to {args.save_high_level_dynamics_checkpoint}",
                flush=True,
            )

    director = Director(
        manager=manager,
        worker=worker,
        dynamics=dynamics,
        manager_normalizer=manager_normalizer,
        worker_normalizer=worker_normalizer,
        dynamics_normalizer=dynamics_normalizer,
        horizon_k=args.horizon_k,
    )
    assert_director_has_no_learned_f_h(director)
    hwm = HierarchicalWorldModel.from_director(
        director, fh_metrics["model"], fh_metrics["normalizer"]
    )
    assert_hwm_has_explicit_f_h(hwm)
    if hwm.worker is not director.worker or hwm.dynamics is not director.dynamics:
        raise AssertionError("HWM must share Director pi_L and f_L instances")

    print("estimating source-state distance threshold (train episodes)...", flush=True)
    source_threshold_info = estimate_source_state_distance_threshold(
        train_raw,
        value_normalizer,
        xy_radius=args.candidate_state_radius,
        percentile=90.0,
        seed=args.seed,
    )
    max_source_d = float(source_threshold_info["chosen_threshold"])
    print(
        f"  chosen max_source_state_distance={max_source_d:.4f} "
        f"from {source_threshold_info['source']} "
        f"p{source_threshold_info['percentile']:g}",
        flush=True,
    )
    print("building shared subgoal candidate index...", flush=True)
    candidate_index = SubgoalCandidateIndex.from_transitions(
        transitions,
        horizon_k=args.horizon_k,
        normalizer=value_normalizer,
    )
    print(
        f"  indexed {candidate_index.current_xy.shape[0]} recorded K-step futures",
        flush=True,
    )

    director.high_level_policy = _make_softreach_policy(
        controller=director,
        value_model=value_model,
        value_normalizer=value_normalizer,
        candidate_index=candidate_index,
        max_source_d=max_source_d,
        lambda_reach=args.lambda_reach,
        n_candidates=args.n_candidates,
        candidate_state_radius=args.candidate_state_radius,
        max_subgoal_distance=args.max_subgoal_distance,
        normalization=args.reachability_score_normalization,
    )
    hwm.high_level_policy = _make_softreach_policy(
        controller=hwm,
        value_model=value_model,
        value_normalizer=value_normalizer,
        candidate_index=candidate_index,
        max_source_d=max_source_d,
        lambda_reach=args.lambda_reach,
        n_candidates=args.n_candidates,
        candidate_state_radius=args.candidate_state_radius,
        max_subgoal_distance=args.max_subgoal_distance,
        normalization=args.reachability_score_normalization,
    )
    print(
        format_fairness_summary(
            director=director,
            hwm=hwm,
            lambda_reach=args.lambda_reach,
            extra={
                "Q_H checkpoint": args.value_checkpoint,
                "Q_H is_high_level_dynamics": getattr(
                    value_model, "is_high_level_dynamics", None
                ),
                "candidate_index identity": (
                    director.high_level_policy.candidate_index
                    is hwm.high_level_policy.candidate_index
                ),
                "n_candidates": args.n_candidates,
                "retrieval_mode": "state",
                "value_cfg_horizon_k": value_cfg.get("horizon_k"),
            },
        ),
        flush=True,
    )

    matched_1k = None
    if args.n_model_rollout_trials > 0:
        print(
            "=== matched 1K: Director (f_L,pi_L)^K vs HWM f_H_phi vs real pi_L ===",
            flush=True,
        )
        matched_1k = evaluate_matched_high_level_models(
            val_raw,
            director,
            hwm,
            horizon_k=args.horizon_k,
            n_trials=args.n_model_rollout_trials,
            seed=args.seed,
        )
        if matched_1k.get("skipped"):
            print(f"  SKIPPED: {matched_1k['skip_reason']}")
        else:
            print(f"  n_trials: {matched_1k['n_trials']}")
            print(
                f"  Director vs real xy mean/median: "
                f"{matched_1k['director_mean_xy_error']:.6f} / "
                f"{matched_1k['director_median_xy_error']:.6f} m"
            )
            print(
                f"  HWM vs real xy mean/median: "
                f"{matched_1k['hwm_mean_xy_error']:.6f} / "
                f"{matched_1k['hwm_median_xy_error']:.6f} m"
            )
            print(
                f"  no-change vs real xy mean: "
                f"{matched_1k['no_change_mean_xy_error']:.6f} m"
            )
            print(
                f"  Director / HWM / no-change full-state MSE: "
                f"{matched_1k['director_mean_state_mse']:.6f} / "
                f"{matched_1k['hwm_mean_state_mse']:.6f} / "
                f"{matched_1k['no_change_mean_state_mse']:.6f}"
            )
            print(
                f"  vs recorded s_{{t+K}} xy: Director="
                f"{matched_1k['director_mean_xy_error_vs_recorded']:.6f} "
                f"HWM={matched_1k['hwm_mean_xy_error_vs_recorded']:.6f}"
            )

    director_multi = None
    hwm_multi = None
    if args.n_multi_horizon_trials > 0:
        print(
            "=== recursive 1K/2K/3K/5K (new pi_H each interval; not val MSE) ===",
            flush=True,
        )
        director_multi = evaluate_implicit_multi_horizon(
            val_raw, director, n_trials=args.n_multi_horizon_trials, seed=args.seed
        )
        hwm_multi = evaluate_implicit_multi_horizon(
            val_raw, hwm, n_trials=args.n_multi_horizon_trials, seed=args.seed
        )
        for name, block in (("Director", director_multi), ("HWM", hwm_multi)):
            if block.get("skipped"):
                print(f"  {name} SKIPPED: {block['skip_reason']}")
                continue
            print(f"  {name}: {block['protocol']}")
            for h in block["horizons"]:
                row = block["by_horizon"][h]
                print(
                    f"    {h}K  n={row['n_trials']}  "
                    f"mean={row['mean_xy_error']:.6f} m  "
                    f"median={row['median_xy_error']:.6f} m"
                )

    env_rows: list = []
    if args.n_env_eval_trials > 0:
        print("building dataset-support index (diagnostic only)...", flush=True)
        support = DatasetSupportIndex.from_transitions(
            transitions, horizon_k=args.horizon_k, seed=args.seed
        )
        csv_d = (
            _csv_path_for_manager(args.save_eval_csv, "director_softreach_l1")
            if args.save_eval_csv
            else ""
        )
        csv_h = (
            _csv_path_for_manager(args.save_eval_csv, "hwm_softreach_l1")
            if args.save_eval_csv
            else ""
        )
        env_rows.append(
            _run_env_eval(
                director,
                args,
                support,
                manager_name="Director-SoftReach-l1",
                csv_path=csv_d,
            )
        )
        env_rows.append(
            _run_env_eval(
                hwm, args, support, manager_name="HWM-SoftReach-l1", csv_path=csv_h
            )
        )

    if env_rows:
        for row in env_rows:
            print(format_director_diagnostic_report(env_eval=row), flush=True)
        print(format_ablation_table(env_rows), flush=True)
    print(
        "Director f_H = (f_L, pi_L)^K; HWM f_H = explicit f_H_phi; "
        "Q_H is a scorer in both.",
        flush=True,
    )


if __name__ == "__main__":
    main()
