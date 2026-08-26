#!/usr/bin/env python3
"""Train the shared low-level worker ``pi_L(s_t, g_tau) -> a_t``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hwm_director.data.worker_dataset import DEFAULT_HORIZON_K
from hwm_director.models.worker import GoalConditionedWorker
from hwm_director.training.train_worker import (
    DEFAULT_MAX_SUBGOAL_DISTANCE,
    DEFAULT_MIN_SUBGOAL_DISTANCE,
    evaluate_worker_on_recorded_subgoals,
    train_goal_conditioned_worker,
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from collect_random_transitions import collect_random_transitions  # noqa: E402

SEED = 0
N_TRANSITIONS = 5000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-transitions",
        type=int,
        default=N_TRANSITIONS,
        help="one-step tuples to collect (need enough for >= 2 episodes; "
        "UMaze episodes are often ~700 steps)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--horizon-k",
        type=int,
        default=DEFAULT_HORIZON_K,
        help="low-level horizon K (future offset and eval rollout cap)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[256, 256],
        help="MLP hidden layer widths",
    )
    parser.add_argument(
        "--n-eval-trials",
        type=int,
        default=20,
        help="subgoal-reaching trials after BC (0 to skip)",
    )
    parser.add_argument(
        "--min-subgoal-distance",
        type=float,
        default=DEFAULT_MIN_SUBGOAL_DISTANCE,
        help="meters; skip eval targets closer than this",
    )
    parser.add_argument(
        "--max-subgoal-distance",
        type=float,
        default=DEFAULT_MAX_SUBGOAL_DISTANCE,
        help="meters; skip eval targets farther than this",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="meters; eval success if final distance is below this",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print episode IDs and per-trial eval details",
    )
    args = parser.parse_args()

    transitions = collect_random_transitions(args.n_transitions, args.seed)
    n_episodes = len({t.episode_id for t in transitions})
    print(f"collected {len(transitions)} transitions across {n_episodes} episodes")

    model = GoalConditionedWorker(hidden_dims=tuple(args.hidden_dims))
    metrics = train_goal_conditioned_worker(
        transitions,
        model=model,
        horizon_k=args.horizon_k,
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )

    train_mse = metrics["train_mse"]
    val_mse = metrics["val_mse"]
    zero_mse = metrics["zero_action_val_mse"]
    vs_baseline = "beats" if val_mse < zero_mse else "does not beat"

    print("=== pi_L BC report ===")
    print(f"  n_episodes: {metrics['n_episodes']}")
    print(f"  n_train_episodes: {metrics['n_train_episodes']}")
    print(f"  n_val_episodes: {metrics['n_val_episodes']}")
    print(f"  n_train_transitions: {metrics['n_train_transitions']}")
    print(f"  n_val_transitions: {metrics['n_val_transitions']}")
    print(f"  n_train_examples: {metrics['n_train_examples']}")
    print(f"  n_val_examples: {metrics['n_val_examples']}")
    if args.verbose:
        print(f"  train_episode_ids: {metrics['train_episode_ids']}")
        print(f"  val_episode_ids: {metrics['val_episode_ids']}")
    print(f"  train action MSE: {train_mse:.6f}")
    print(f"  val action MSE: {val_mse:.6f}")
    print(f"  zero-action val MSE: {zero_mse:.6f}")
    print(
        f"  learned val MSE {vs_baseline} zero-action "
        f"({val_mse:.6f} vs {zero_mse:.6f})"
    )

    if args.n_eval_trials > 0:
        val_ids = set(metrics["val_episode_ids"])
        val_transitions = [t for t in transitions if t.episode_id in val_ids]
        eval_metrics = evaluate_worker_on_recorded_subgoals(
            val_transitions,
            metrics["model"],
            metrics["normalizer"],
            horizon_k=args.horizon_k,
            n_trials=args.n_eval_trials,
            success_threshold=args.success_threshold,
            min_distance=args.min_subgoal_distance,
            max_distance=args.max_subgoal_distance,
            seed=args.seed,
            verbose=args.verbose,
        )
        print("=== pi_L subgoal eval ===")
        print(f"  n_candidates: {eval_metrics['n_candidates']}")
        print(f"  n_trials: {eval_metrics['n_trials']}")
        print(f"  mean initial distance (m): {eval_metrics['mean_initial_distance']:.6f}")
        print(f"  mean final distance (m): {eval_metrics['mean_final_distance']:.6f}")
        print(f"  mean distance reduction (m): {eval_metrics['mean_distance_reduction']:.6f}")
        print(f"  median distance reduction (m): {eval_metrics['median_distance_reduction']:.6f}")
        print(f"  progress fraction: {eval_metrics['progress_fraction']:.6f}")
        print(
            "  fraction positive reduction: "
            f"{eval_metrics['fraction_positive_reduction']:.6f}"
        )
        print(
            "  fraction >=10% relative progress: "
            f"{eval_metrics['fraction_relative_progress_10']:.6f}"
        )
        print(
            "  fraction already successful at start: "
            f"{eval_metrics['fraction_already_successful_at_start']:.6f}"
        )
        print(
            f"  success rate (final < {args.success_threshold} m): "
            f"{eval_metrics['success_rate']:.6f}"
        )


if __name__ == "__main__":
    main()
