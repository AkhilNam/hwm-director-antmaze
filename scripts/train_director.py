#!/usr/bin/env python3
"""Train Director ``pi_H`` and evaluate implicit ``(f_L, pi_L)^K``.

Loads existing ``pi_L`` and ``f_L`` checkpoints. Does not retrain them.
Use ``--skip-manager-train`` to diagnose a saved ``pi_H`` without BC.

``--manager-objective value`` trains / loads ``Q_H`` (a subgoal scorer, not
``f_H``) and selects among data-supported local futures. Director still uses

    f_H = (f_L, pi_L)^K
"""

from __future__ import annotations

import argparse

from hwm_director.data.high_level_transitions import (
    DEFAULT_UNSUCCESSFUL_VALUE,
    DEFAULT_VALUE_GAMMA,
)
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
from hwm_director.models.director_manager import (
    DEFAULT_MAX_SUBGOAL_DISTANCE,
    DirectorManager,
)
from hwm_director.models.high_level_value import HighLevelValueModel
from hwm_director.models.value_manager import (
    REACHABILITY_NORM_ZSCORE,
    ValueHighLevelPolicy,
    director_reachability_error,
)
from hwm_director.training.checkpoints import (
    load_dynamics_checkpoint,
    load_high_level_value_checkpoint,
    load_manager_checkpoint,
    load_worker_checkpoint,
    save_high_level_value_checkpoint,
    save_manager_checkpoint,
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
    evaluate_manager_bc_error,
    format_ablation_table,
    format_candidate_quality,
    format_director_diagnostic_report,
    format_manager_comparison,
    format_pareto_table,
    write_eval_csv,
)
from hwm_director.training.eval_director import (
    evaluate_director_env_rollouts,
    evaluate_implicit_high_level_transition,
    evaluate_implicit_multi_horizon,
)
from hwm_director.training.train_dynamics import split_episode_indices
from hwm_director.training.train_high_level_value import train_high_level_value
from hwm_director.training.train_manager import train_director_manager
from hwm_director.training.train_worker import _select

SEED = 0


def _csv_path_for_manager(path: str, manager_name: str) -> str:
    if not path:
        return ""
    if path.endswith(".csv"):
        return f"{path[:-4]}_{manager_name}.csv"
    return f"{path}_{manager_name}"


def _run_env_eval(director, args, support, *, manager_name: str, csv_path: str):
    print(f"=== end-to-end Director-{manager_name} env evaluation ===", flush=True)
    env_metrics = evaluate_director_env_rollouts(
        director,
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
    policy = getattr(director, "high_level_policy", None)
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
    wall = env_metrics["wall_region"]
    print(
        f"  mean_source_state_d={env_metrics.get('mean_source_state_distance')} "
        f"pred_err={env_metrics.get('mean_predicted_subgoal_error')} "
        f"actual_err={env_metrics.get('mean_worker_subgoal_error')} "
        f"pearson={env_metrics.get('predicted_vs_actual_pearson')} "
        f"spearman={env_metrics.get('predicted_vs_actual_spearman')}"
    )
    stuck = env_metrics["failed_stuck"]
    print(
        f"  failed stuck: n={stuck['n_failed_stuck']}/{stuck['n_failed']} "
        f"first_tau={stuck.get('mean_first_stuck_tau')} "
        f"reach_before={stuck['mean_reach_before_stuck']} "
        f"reach_after={stuck['mean_reach_after_stuck']} "
        f"qh_before={stuck.get('mean_qh_before_stuck')} "
        f"qh_after={stuck.get('mean_qh_after_stuck')} "
        f"pen_before={stuck.get('mean_reach_penalty_before_stuck')} "
        f"pen_after={stuck.get('mean_reach_penalty_after_stuck')} "
        f"fallback_after={stuck['mean_fallback_after_stuck']}"
    )
    print(
        f"  mean_qh={env_metrics.get('mean_qh_score')} "
        f"mean_progress={env_metrics.get('mean_progress_to_final')} "
        f"median_progress={env_metrics.get('median_progress_to_final')} "
        f"frac_pos={env_metrics.get('fraction_positive_progress')}"
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
            f"init_d={trial['initial_distance']:.4f} "
            f"final_d={trial['final_distance']:.4f} "
            f"steps={trial['n_primitive_steps']} "
            f"decisions={trial['n_high_level_decisions']} "
            f"progress={trial['total_progress']:.4f} "
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
    parser.add_argument(
        "--worker-checkpoint",
        default="checkpoints/pi_l.pt",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        default="checkpoints/f_l.pt",
    )
    parser.add_argument(
        "--manager-checkpoint",
        default="checkpoints/pi_h_director.pt",
        help="load this pi_H when --skip-manager-train is set",
    )
    parser.add_argument(
        "--save-manager-checkpoint",
        default="checkpoints/pi_h_director.pt",
        help="path to write pi_H (empty string to skip)",
    )
    parser.add_argument(
        "--skip-manager-train",
        action="store_true",
        help="load --manager-checkpoint instead of training pi_H",
    )
    parser.add_argument("--n-model-rollout-trials", type=int, default=20)
    parser.add_argument("--n-multi-horizon-trials", type=int, default=20)
    parser.add_argument("--n-env-eval-trials", type=int, default=0)
    parser.add_argument("--max-high-level-steps", type=int, default=70)
    parser.add_argument(
        "--subgoal-success-threshold",
        type=float,
        default=DEFAULT_SUBGOAL_SUCCESS,
    )
    parser.add_argument("--stuck-window", type=int, default=DEFAULT_STUCK_WINDOW)
    parser.add_argument(
        "--stuck-distance",
        type=float,
        default=DEFAULT_STUCK_DISTANCE,
    )
    parser.add_argument(
        "--save-eval-csv",
        default="",
        help="write one CSV row per high-level decision",
    )
    parser.add_argument(
        "--max-subgoal-distance",
        type=float,
        default=DEFAULT_MAX_SUBGOAL_DISTANCE,
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
    )
    parser.add_argument(
        "--manager-objective",
        choices=("bc", "value", "both", "ablation", "soft-reach"),
        default="bc",
        help=(
            "bc / value / both / ablation (BC, Value-XY, Value-State, State+hard-reach) "
            "/ soft-reach (BC, Value-State, hard-reach, SoftReach lambdas)"
        ),
    )
    parser.add_argument("--gamma", type=float, default=DEFAULT_VALUE_GAMMA)
    parser.add_argument(
        "--unsuccessful-value",
        type=float,
        default=DEFAULT_UNSUCCESSFUL_VALUE,
        help="value target for episodes that never reach g* (not treated as success)",
    )
    parser.add_argument(
        "--candidate-state-radius",
        type=float,
        default=DEFAULT_CANDIDATE_RADIUS,
    )
    parser.add_argument("--n-candidates", type=int, default=DEFAULT_N_CANDIDATES)
    parser.add_argument(
        "--candidate-retrieval",
        choices=("xy", "state", "hybrid"),
        default="xy",
        help="Director-Value candidate source matching (ignored for ablation)",
    )
    parser.add_argument(
        "--state-distance-weight",
        type=float,
        default=1.0,
        help="hybrid alpha on normalized 29-D distance",
    )
    parser.add_argument(
        "--xy-distance-weight",
        type=float,
        default=1.0,
        help="hybrid beta on normalized xy distance",
    )
    parser.add_argument(
        "--max-source-state-distance",
        type=float,
        default=-1.0,
        help="reject sources above this normalized 29-D distance; <0 auto percentile",
    )
    parser.add_argument(
        "--source-state-distance-percentile",
        type=float,
        default=90.0,
    )
    parser.add_argument(
        "--use-reachability-filter",
        action="store_true",
        help="drop candidates with (f_L, pi_L)^K predicted xy error above threshold",
    )
    parser.add_argument(
        "--max-predicted-subgoal-error",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--reachability-score-weight",
        type=float,
        default=0.0,
        help="lambda for soft score = z(Q_H) - lambda * z(reach_error)",
    )
    parser.add_argument(
        "--reachability-score-normalization",
        choices=("candidate_zscore", "raw"),
        default=REACHABILITY_NORM_ZSCORE,
        help="candidate_zscore (main) or raw Q_H minus lambda * meters",
    )
    parser.add_argument(
        "--reachability-lambdas",
        type=float,
        nargs="*",
        default=None,
        help="soft-reach lambda sweep; default 0.25 0.5 1.0 2.0 for --manager-objective soft-reach",
    )
    parser.add_argument("--value-epochs", type=int, default=20)
    parser.add_argument("--value-batch-size", type=int, default=4096)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument(
        "--save-value-checkpoint",
        default="checkpoints/high_level_value.pt",
    )
    parser.add_argument(
        "--load-value-checkpoint",
        default="checkpoints/high_level_value.pt",
    )
    parser.add_argument(
        "--skip-value-train",
        action="store_true",
        help="load --load-value-checkpoint instead of training Q_H",
    )
    parser.add_argument(
        "--wall-region-x",
        type=float,
        default=DEFAULT_WALL_REGION_XY[0],
    )
    parser.add_argument(
        "--wall-region-y",
        type=float,
        default=DEFAULT_WALL_REGION_XY[1],
    )
    parser.add_argument(
        "--wall-region-radius",
        type=float,
        default=DEFAULT_WALL_REGION_RADIUS,
    )
    args = parser.parse_args()

    print("loading pi_L and f_L checkpoints...", flush=True)
    worker, worker_normalizer = load_worker_checkpoint(args.worker_checkpoint)
    dynamics, dynamics_normalizer = load_dynamics_checkpoint(
        args.dynamics_checkpoint
    )

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

    if args.skip_manager_train:
        print(f"loading pi_H from {args.manager_checkpoint} (no BC)...", flush=True)
        manager, manager_normalizer = load_manager_checkpoint(args.manager_checkpoint)
        manager.eval()
    else:
        manager = DirectorManager(
            hidden_dims=tuple(args.hidden_dims),
            max_subgoal_distance=args.max_subgoal_distance,
        )
        metrics = train_director_manager(
            transitions,
            model=manager,
            horizon_k=args.horizon_k,
            val_fraction=args.val_fraction,
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            log=lambda message: print(message, flush=True),
        )
        manager = metrics["model"]
        manager_normalizer = metrics["normalizer"]
        print("=== pi_H BC report (training run) ===")
        print(f"  n_episodes: {metrics['n_episodes']}")
        print(f"  n_train_episodes: {metrics['n_train_episodes']}")
        print(f"  n_val_episodes: {metrics['n_val_episodes']}")
        print(f"  n_train_examples: {metrics['n_train_examples']}")
        print(f"  n_val_examples: {metrics['n_val_examples']}")
        print(f"  train subgoal MSE: {metrics['train_mse']:.6f}")
        print(f"  val subgoal MSE: {metrics['val_mse']:.6f}")
        print(f"  val x/y Euclidean (m): {metrics['val_xy_euclidean']:.6f}")
        print(
            "  current-position val MSE / Euclidean: "
            f"{metrics['current_position_val_mse']:.6f} / "
            f"{metrics['current_position_val_euclidean']:.6f}"
        )
        print(
            "  final-goal val MSE / Euclidean: "
            f"{metrics['final_goal_val_mse']:.6f} / "
            f"{metrics['final_goal_val_euclidean']:.6f}"
        )
        vs = (
            "beats"
            if metrics["val_mse"] < metrics["current_position_val_mse"]
            else "does not beat"
        )
        print(f"  learned val MSE {vs} current-position baseline")
        if args.save_manager_checkpoint:
            save_manager_checkpoint(
                args.save_manager_checkpoint,
                metrics["model"],
                metrics["normalizer"],
            )
            print(
                f"saved pi_H checkpoint to {args.save_manager_checkpoint}",
                flush=True,
            )

    print("=== manager BC vs recorded s_{t+K}[:2] ===", flush=True)
    manager_bc = evaluate_manager_bc_error(
        manager,
        manager_normalizer,
        train_raw,
        val_raw,
        horizon_k=args.horizon_k,
        batch_size=args.batch_size,
    )
    print(f"  n_train_examples: {manager_bc['n_train_examples']}")
    print(f"  n_val_examples: {manager_bc['n_val_examples']}")
    print(f"  train Euclidean (m): {manager_bc['train_euclidean']:.6f}")
    print(f"  val Euclidean (m): {manager_bc['val_euclidean']:.6f}")
    print(f"  train MSE: {manager_bc['train_mse']:.6f}")
    print(f"  val MSE: {manager_bc['val_mse']:.6f}")

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
    if getattr(director, "explicit_f_h", None) is not None:
        raise AssertionError("Director must not have an independently trained f_H")

    implicit_1k = None
    if args.n_model_rollout_trials > 0:
        print("=== implicit f_H = (f_L, pi_L)^K vs real worker (fixed subgoal) ===", flush=True)
        implicit_1k = evaluate_implicit_high_level_transition(
            val_raw,
            director,
            horizon_k=args.horizon_k,
            n_trials=args.n_model_rollout_trials,
            seed=args.seed,
        )
        if implicit_1k.get("skipped"):
            print(f"  SKIPPED: {implicit_1k['skip_reason']}")
        else:
            print(f"  K: {implicit_1k['horizon_k']}")
            print(f"  n_trials: {implicit_1k['n_trials']}")
            print(f"  mean K-step x/y error (m): {implicit_1k['mean_xy_error']:.6f}")
            print(f"  median K-step x/y error (m): {implicit_1k['median_xy_error']:.6f}")
            print(f"  mean full-state MSE: {implicit_1k['mean_state_mse']:.6f}")
            print(
                "  no-change mean x/y error (m): "
                f"{implicit_1k['no_change_mean_xy_error']:.6f}"
            )

    multi_horizon = None
    if args.n_multi_horizon_trials > 0:
        print(
            "=== implicit f_H multi-horizon (new pi_H subgoal each K) ===",
            flush=True,
        )
        multi_horizon = evaluate_implicit_multi_horizon(
            val_raw,
            director,
            n_trials=args.n_multi_horizon_trials,
            seed=args.seed,
        )
        if multi_horizon.get("skipped"):
            print(f"  SKIPPED: {multi_horizon['skip_reason']}")
        else:
            print(f"  protocol: {multi_horizon['protocol']}")
            for h in multi_horizon["horizons"]:
                row = multi_horizon["by_horizon"][h]
                print(
                    f"  {h}K  n={row['n_trials']}  "
                    f"mean={row['mean_xy_error']:.6f} m  "
                    f"median={row['median_xy_error']:.6f} m"
                )

    value_metrics = None
    use_value = args.manager_objective in ("value", "both", "ablation", "soft-reach")
    source_threshold_info = None
    candidate_index = None
    if use_value:
        if args.skip_value_train:
            print(
                f"loading Q_H from {args.load_value_checkpoint} (not f_H)...",
                flush=True,
            )
            value_model, value_normalizer, value_cfg = load_high_level_value_checkpoint(
                args.load_value_checkpoint
            )
            value_metrics = {
                "train_mse": float("nan"),
                "val_mse": float("nan"),
                "gamma": float(value_cfg.get("gamma", args.gamma)),
                "horizon_k": int(value_cfg.get("horizon_k", args.horizon_k)),
                "model": value_model,
                "normalizer": value_normalizer,
            }
        else:
            print("=== train Q_H (value scorer, NOT f_H) ===", flush=True)
            value_model = HighLevelValueModel(hidden_dims=tuple(args.hidden_dims))
            value_metrics = train_high_level_value(
                transitions,
                model=value_model,
                horizon_k=args.horizon_k,
                val_fraction=args.val_fraction,
                seed=args.seed,
                batch_size=args.value_batch_size,
                epochs=args.value_epochs,
                lr=args.value_lr,
                gamma=args.gamma,
                unsuccessful_value=args.unsuccessful_value,
                success_threshold=args.subgoal_success_threshold,
                log=lambda message: print(message, flush=True),
            )
            print(f"  train value MSE: {value_metrics['train_mse']:.6f}")
            print(f"  val value MSE: {value_metrics['val_mse']:.6f}")
            print(
                f"  train hl success/unsuccessful: "
                f"{value_metrics['n_train_success_examples']}/"
                f"{value_metrics['n_train_unsuccessful_examples']}"
            )
            print(
                f"  val hl success/unsuccessful: "
                f"{value_metrics['n_val_success_examples']}/"
                f"{value_metrics['n_val_unsuccessful_examples']}"
            )
            if args.save_value_checkpoint:
                save_high_level_value_checkpoint(
                    args.save_value_checkpoint,
                    value_metrics["model"],
                    value_metrics["normalizer"],
                    gamma=args.gamma,
                    horizon_k=args.horizon_k,
                    unsuccessful_value=args.unsuccessful_value,
                    success_threshold=args.subgoal_success_threshold,
                    candidate_state_radius=args.candidate_state_radius,
                    n_candidates=args.n_candidates,
                    max_subgoal_distance=args.max_subgoal_distance,
                )
                print(
                    f"saved Q_H checkpoint to {args.save_value_checkpoint}",
                    flush=True,
                )
        print("estimating source-state distance threshold on train episodes...", flush=True)
        source_threshold_info = estimate_source_state_distance_threshold(
            train_raw,
            value_metrics["normalizer"],
            xy_radius=args.candidate_state_radius,
            percentile=args.source_state_distance_percentile,
            seed=args.seed,
        )
        auto_thresh = float(source_threshold_info["chosen_threshold"])
        if args.max_source_state_distance < 0:
            max_source_d = auto_thresh
        else:
            max_source_d = float(args.max_source_state_distance)
        print(
            f"  lag1 p50/p90/p95: "
            f"{source_threshold_info['lag1_p50']:.4f}/"
            f"{source_threshold_info['lag1_p90']:.4f}/"
            f"{source_threshold_info['lag1_p95']:.4f} "
            f"(n={source_threshold_info['n_lag1_pairs']})"
        )
        print(
            f"  xy-nearby p50/p90/p95: "
            f"{source_threshold_info['nearby_p50']:.4f}/"
            f"{source_threshold_info['nearby_p90']:.4f}/"
            f"{source_threshold_info['nearby_p95']:.4f} "
            f"(n={source_threshold_info['n_xy_nearby_pairs']})"
        )
        print(
            f"  chosen max_source_state_distance={max_source_d:.4f} "
            f"from {source_threshold_info['source']} "
            f"p{source_threshold_info['percentile']:g}",
            flush=True,
        )
        print("building subgoal candidate index...", flush=True)
        candidate_index = SubgoalCandidateIndex.from_transitions(
            transitions,
            horizon_k=args.horizon_k,
            normalizer=value_metrics["normalizer"],
        )
        print(
            f"  indexed {candidate_index.current_xy.shape[0]} recorded K-step futures",
            flush=True,
        )

        def _make_policy(
            *,
            mode: str,
            use_reach: bool,
            lambda_reach: float = 0.0,
            use_soft: bool = False,
        ) -> ValueHighLevelPolicy:
            need_rollout = use_reach or use_soft or lambda_reach != 0.0
            reach_fn = director_reachability_error(director) if need_rollout else None
            max_pred = args.max_predicted_subgoal_error if use_reach else None
            src_cap = max_source_d if mode in ("state", "hybrid") else None
            policy = ValueHighLevelPolicy(
                value_metrics["model"],
                value_metrics["normalizer"],
                candidate_index,
                director._select_bc,
                candidate_state_radius=args.candidate_state_radius,
                n_candidates=args.n_candidates,
                max_subgoal_distance=args.max_subgoal_distance,
                retrieval_mode=mode,
                max_source_state_distance=src_cap,
                state_distance_weight=args.state_distance_weight,
                xy_distance_weight=args.xy_distance_weight,
                reachability_fn=reach_fn,
                max_predicted_subgoal_error=max_pred,
                reachability_score_weight=float(lambda_reach),
                reachability_score_normalization=args.reachability_score_normalization,
                use_soft_reachability=use_soft,
            )
            return policy

        value_policy = _make_policy(
            mode=args.candidate_retrieval,
            use_reach=args.use_reachability_filter
            and args.reachability_score_weight == 0.0,
            lambda_reach=args.reachability_score_weight,
            use_soft=args.reachability_score_weight != 0.0,
        )

    env_metrics = None
    value_env_metrics = None
    ablation_rows: list = []
    if args.n_env_eval_trials > 0:
        print("building dataset-support index (diagnostic only)...", flush=True)
        support = DatasetSupportIndex.from_transitions(
            transitions, horizon_k=args.horizon_k, seed=args.seed
        )
        print(
            f"  indexed {support.current_xy.shape[0]} (current_xy, future_xy) pairs",
            flush=True,
        )
        run_bc = args.manager_objective in (
            "bc",
            "both",
            "value",
            "ablation",
            "soft-reach",
        )
        if run_bc:
            director.high_level_policy = None
            assert_director_has_no_learned_f_h(director)
            csv_bc = (
                _csv_path_for_manager(args.save_eval_csv, "bc")
                if args.manager_objective != "bc"
                else args.save_eval_csv
            )
            env_metrics = _run_env_eval(
                director, args, support, manager_name="BC", csv_path=csv_bc
            )
            ablation_rows.append(env_metrics)
        if args.manager_objective in ("ablation", "soft-reach"):
            if args.manager_objective == "ablation":
                configs: list[tuple] = [
                    ("Value-XY", "xy", False, 0.0, False),
                    ("Value-State", "state", False, 0.0, False),
                    ("Value-State-Reach", "state", True, 0.0, False),
                ]
            else:
                configs = [
                    ("Value-State", "state", False, 0.0, False),
                    ("Value-State-Reach", "state", True, 0.0, False),
                ]
            lambdas = args.reachability_lambdas
            if lambdas is None:
                lambdas = (
                    [0.0, 0.25, 0.5, 1.0, 2.0]
                    if args.manager_objective == "soft-reach"
                    else []
                )
            for lam in lambdas:
                configs.append(
                    (
                        f"Value-State-SoftReach-l{lam:g}",
                        "state",
                        False,
                        float(lam),
                        True,
                    )
                )
            for name, mode, use_reach, lam, use_soft in configs:
                policy = _make_policy(
                    mode=mode,
                    use_reach=use_reach,
                    lambda_reach=lam,
                    use_soft=use_soft,
                )
                director.high_level_policy = policy
                assert_director_has_no_learned_f_h(director)
                csv_path = (
                    _csv_path_for_manager(
                        args.save_eval_csv, name.lower().replace("-", "_")
                    )
                    if args.save_eval_csv
                    else ""
                )
                row = _run_env_eval(
                    director, args, support, manager_name=name, csv_path=csv_path
                )
                ablation_rows.append(row)
                if name == "Value-State":
                    value_env_metrics = row
        elif use_value:
            director.high_level_policy = value_policy
            assert_director_has_no_learned_f_h(director)
            csv_val = (
                _csv_path_for_manager(args.save_eval_csv, "value")
                if args.save_eval_csv
                else ""
            )
            value_env_metrics = _run_env_eval(
                director,
                args,
                support,
                manager_name="Value",
                csv_path=csv_val,
            )
            ablation_rows.append(value_env_metrics)
            print(f"  Q_H fallback rate: {value_policy.stats()['fallback_rate']}")

    print(
        format_director_diagnostic_report(
            env_eval=env_metrics,
            manager_bc=manager_bc,
            implicit_1k=implicit_1k,
            multi_horizon=multi_horizon,
        ),
        flush=True,
    )
    if value_env_metrics is not None:
        print(
            format_director_diagnostic_report(
                env_eval=value_env_metrics,
                manager_bc=None,
                implicit_1k=None,
                multi_horizon=None,
            ).replace(
                "=== Director diagnostic report ===",
                "=== Director-Value diagnostic report ===",
            ),
            flush=True,
        )
        print(
            format_manager_comparison(
                env_metrics, value_env_metrics, value_train=value_metrics
            ),
            flush=True,
        )
    print(
        "Director f_H is still (f_L, pi_L)^K; Q_H is a scorer, not high-level dynamics.",
        flush=True,
    )
    if len(ablation_rows) > 1:
        print(format_ablation_table(ablation_rows), flush=True)
        print(format_pareto_table(ablation_rows), flush=True)


if __name__ == "__main__":
    main()
